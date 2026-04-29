"""
Stage 11 — Sparsity and Buffer Sweep.

Mirrors benchmark_federation_ablations.py sections A (sparsity) and C (buffer size).

11.1  Sparsity sweep: drop 0/20/40/60% of years uniformly at random,
      compare local vs federated MAE degradation curves.
      Target: federated MAE degrades more gracefully than local (smaller slope).

11.2  Buffer size sweep: buffer_size in [25, 50, 100, 200],
      measure discovery quality (n_active_hypotheses) and prediction MAE.
      Target: MAE monotonically improves (or plateaus) as buffer grows.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stages.utils import (
    ARTIFACTS_DIR, fail_result, make_result, save_artifact, skip_result,
)

logger = logging.getLogger(__name__)

START_YEAR = 1990
END_YEAR = 2023
MIN_TRAIN = 5
COUNTRIES = ["KEN", "TZA", "UGA"]
PRIMARY = "KEN"

SPARSITY_LEVELS = [0.0, 0.20, 0.40, 0.60]
BUFFER_SIZES = [25, 50, 100, 200]


# ---------------------------------------------------------------------------
# Synthetic data (same pattern as stage9/10 for consistency)
# ---------------------------------------------------------------------------

def _mock_country_data(country: str, start: int, end: int, seed: int = 0) -> Dict[int, Dict[str, float]]:
    import random
    rng = random.Random(seed + hash(country) % 1000)
    base = {k: rng.uniform(lo, hi) for k, (lo, hi) in {
        "gdp_growth": (2.0, 7.0), "inflation": (3.0, 12.0),
        "unemployment": (2.0, 8.0), "exports_gdp": (10.0, 25.0),
        "imports_gdp": (12.0, 30.0), "current_account": (-8.0, 2.0),
        "govt_consumption": (8.0, 18.0), "tax_revenue": (10.0, 18.0),
        "govt_debt": (20.0, 65.0), "real_interest_rate": (1.0, 10.0),
        "broad_money": (20.0, 40.0), "private_credit": (10.0, 30.0),
        "urban_population": (15.0, 45.0), "school_enrollment": (70.0, 110.0),
        "life_expectancy": (52.0, 68.0), "electricity_access": (10.0, 70.0),
        "internet_users": (0.1, 40.0),
    }.items()}
    result: Dict[int, Dict[str, float]] = {}
    for year in range(start, end + 1):
        row = {}
        for k, v in base.items():
            noise = rng.gauss(0, abs(v) * 0.05)
            trend = (year - start) * rng.uniform(-0.1, 0.2)
            row[k] = v + noise + trend
        result[year] = row
    return result


def _apply_sparsity(yearly: Dict[int, Dict], drop_frac: float, rng: np.random.Generator) -> Dict[int, Dict]:
    years = sorted(yearly.keys())
    n_drop = int(len(years) * drop_frac)
    if n_drop == 0:
        return yearly
    drop_set = set(rng.choice(years, size=n_drop, replace=False).tolist())
    return {y: v for y, v in yearly.items() if y not in drop_set}


def _build_schema(fields: List[str]) -> Dict:
    return {"fields": [{"name": n, "type": "numeric"} for n in fields]}


# ---------------------------------------------------------------------------
# Shared MAE utilities
# ---------------------------------------------------------------------------

def _field_stats(rows: List[Dict]) -> Dict[str, Tuple[float, float]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if math.isfinite(v):
                buckets[k].append(v)
    return {
        k: (
            sum(v) / len(v),
            max(math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / max(1, len(v))), 1e-9),
        )
        for k, v in buckets.items()
    }


def _normalised_mae(preds: List[Dict], actuals: List[Dict],
                    norm: Dict[str, Tuple[float, float]]) -> float:
    common = sorted({k for p in preds for k in p} & {k for a in actuals for k in a} & set(norm))
    mae_vals = []
    for field in common:
        mu, sigma = norm[field]
        pairs = [
            (
                (p.get(field, float("nan")) - mu) / sigma,
                (a.get(field, float("nan")) - mu) / sigma,
            )
            for p, a in zip(preds, actuals)
            if math.isfinite(p.get(field, float("nan")))
            and math.isfinite(a.get(field, float("nan")))
        ]
        if len(pairs) >= 2:
            mae_vals.append(sum(abs(p - a) for p, a in pairs) / len(pairs))
    return sum(mae_vals) / len(mae_vals) if mae_vals else float("nan")


def _ar1_fit(values: List[float]) -> Tuple[float, float]:
    pairs = [(values[i - 1], values[i]) for i in range(1, len(values))
             if math.isfinite(values[i - 1]) and math.isfinite(values[i])]
    if len(pairs) < 2:
        mu = sum(v for v in values if math.isfinite(v)) / max(1, sum(1 for v in values if math.isfinite(v)))
        return mu, 0.0
    x, y = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    var = sum((xi - mx) ** 2 for xi in x)
    beta = cov / var if var > 1e-12 else 0.0
    return my - beta * mx, beta


def _fit_ar1(rows: List[Dict]) -> Dict[str, Tuple[float, float]]:
    history: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if math.isfinite(v):
                history[k].append(v)
    return {k: _ar1_fit(vs) for k, vs in history.items()}


def _predict_ar1(params: Dict, last_row: Dict) -> Dict[str, float]:
    return {
        k: (a + b * last_row[k])
        for k, (a, b) in params.items()
        if k in last_row and math.isfinite(last_row.get(k, float("nan")))
    }


# ---------------------------------------------------------------------------
# 11.1 Sparsity sweep
# ---------------------------------------------------------------------------

def _sparsity_one_seed(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    full_data = {c: _mock_country_data(c, START_YEAR, END_YEAR, seed=seed) for c in COUNTRIES}
    all_fields = sorted({f for row in full_data[PRIMARY].values() for f in row})

    scarcity_available = False
    try:
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
        scarcity_available = True
    except Exception:
        pass

    results_by_sparsity = {}

    for drop_frac in SPARSITY_LEVELS:
        # Apply sparsity to primary only (local scenario) or all (federated)
        sparse_primary = _apply_sparsity(full_data[PRIMARY], drop_frac, rng)
        sparse_peers = {c: _apply_sparsity(full_data[c], drop_frac, rng) for c in COUNTRIES if c != PRIMARY}

        years_avail = sorted(sparse_primary.keys())
        if len(years_avail) < MIN_TRAIN + 2:
            results_by_sparsity[f"drop_{int(drop_frac*100)}pct"] = {"skipped": "too few years"}
            continue

        # Full set years for evaluation (always evaluate on non-dropped years)
        eval_years = sorted(full_data[PRIMARY].keys())[MIN_TRAIN:]
        actuals_local, actuals_fed = [], []
        local_sc_preds, fed_sc_preds = [], []
        local_ar1_preds, fed_ar1_preds = [], []

        # Build engines if available
        if scarcity_available:
            try:
                eng_local = OnlineDiscoveryEngine(explore_interval=5, mode="balanced", buffer_size=50)
                eng_local.initialize(_build_schema(all_fields))
                eng_fed = OnlineDiscoveryEngine(explore_interval=5, mode="balanced", buffer_size=50)
                eng_fed.initialize(_build_schema(all_fields))

                # Pre-stream sparse data
                for y in sorted(sparse_primary.keys())[:MIN_TRAIN]:
                    eng_local.process_row(sparse_primary[y])
                    eng_fed.process_row(sparse_primary[y])
                    for c, pd in sparse_peers.items():
                        if y in pd:
                            eng_fed.process_row(pd[y])
            except Exception:
                scarcity_available = False

        seen_local = [sparse_primary[y] for y in sorted(sparse_primary.keys())[:MIN_TRAIN]]
        seen_all = list(seen_local)
        for c, pd in sparse_peers.items():
            seen_all.extend(pd[y] for y in sorted(pd.keys())[:MIN_TRAIN] if y in pd)

        for t_idx in range(len(eval_years)):
            year = eval_years[t_idx]
            if year not in full_data[PRIMARY]:
                continue

            prev_year = eval_years[t_idx - 1] if t_idx > 0 else sorted(sparse_primary.keys())[MIN_TRAIN - 1]
            last_row = full_data[PRIMARY].get(prev_year, {})
            if not last_row:
                continue

            actual = full_data[PRIMARY][year]

            # Local AR1
            ar1_params = _fit_ar1(seen_local)
            local_ar1_preds.append(_predict_ar1(ar1_params, last_row))
            actuals_local.append(actual)

            # Fed AR1: avg across seen_all
            fed_params = _fit_ar1(seen_all)
            fed_ar1_preds.append(_predict_ar1(fed_params, last_row))
            actuals_fed.append(actual)

            # Scarcity engines
            if scarcity_available:
                try:
                    local_sc_preds.append(eng_local.predict(last_row))
                    eng_local.process_row(actual)
                    # Feed peer data if available
                    if year in sparse_primary:
                        for c, pd in sparse_peers.items():
                            if year in pd:
                                eng_fed.process_row(pd[year])
                    fed_sc_preds.append(eng_fed.predict(last_row))
                    eng_fed.process_row(actual)
                except Exception:
                    pass

            # Update rolling history with available sparse data
            if year in sparse_primary:
                seen_local.append(sparse_primary[year])
                seen_all.append(sparse_primary[year])
            for c, pd in sparse_peers.items():
                if year in pd:
                    seen_all.append(pd[year])

        norm = _field_stats(actuals_local) if actuals_local else {}
        entry: Dict[str, Any] = {"n_eval": len(actuals_local)}

        if actuals_local and local_ar1_preds:
            entry["local_ar1_mae"] = round(_normalised_mae(local_ar1_preds, actuals_local, norm), 4)
        if actuals_fed and fed_ar1_preds:
            entry["fed_ar1_mae"] = round(_normalised_mae(fed_ar1_preds, actuals_fed, norm), 4)
        if scarcity_available and local_sc_preds:
            entry["local_sc_mae"] = round(_normalised_mae(
                local_sc_preds[:len(actuals_local)], actuals_local[:len(local_sc_preds)], norm), 4)
        if scarcity_available and fed_sc_preds:
            entry["fed_sc_mae"] = round(_normalised_mae(
                fed_sc_preds[:len(actuals_fed)], actuals_fed[:len(fed_sc_preds)], norm), 4)

        results_by_sparsity[f"drop_{int(drop_frac * 100)}pct"] = entry

    return results_by_sparsity


# ---------------------------------------------------------------------------
# 11.2 Buffer size sweep
# ---------------------------------------------------------------------------

def _buffer_one_seed(seed: int) -> Dict[str, Any]:
    data = _mock_country_data(PRIMARY, START_YEAR, END_YEAR, seed=seed)
    all_fields = sorted({f for row in data.values() for f in row})
    years = sorted(data.keys())

    try:
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    except Exception:
        return {}

    results_by_buffer: Dict[str, Any] = {}

    for buf in BUFFER_SIZES:
        try:
            eng = OnlineDiscoveryEngine(explore_interval=5, mode="balanced", buffer_size=buf)
            eng.initialize(_build_schema(all_fields))

            preds, actuals = [], []
            for t_idx, year in enumerate(years):
                if t_idx < MIN_TRAIN:
                    eng.process_row(data[year])
                    continue
                last_row = data[years[t_idx - 1]]
                actual = data[year]
                actuals.append(actual)
                try:
                    preds.append(eng.predict(last_row))
                    eng.process_row(actual)
                except Exception:
                    eng.process_row(actual)

            norm = _field_stats(actuals)
            mae = _normalised_mae(preds[:len(actuals)], actuals[:len(preds)], norm) if preds else float("nan")

            # Count active hypotheses if accessible
            n_active = None
            try:
                hyps = getattr(eng, "_hypotheses", None) or getattr(eng, "hypotheses", None)
                if hyps is not None:
                    if isinstance(hyps, dict):
                        n_active = sum(1 for h in hyps.values()
                                       if getattr(h, "state", None) not in ("DEAD", "DECAYING"))
                    elif isinstance(hyps, list):
                        n_active = sum(1 for h in hyps
                                       if getattr(h, "state", None) not in ("DEAD", "DECAYING"))
            except Exception:
                pass

            results_by_buffer[f"buf_{buf}"] = {
                "buffer_size": buf,
                "mae": round(mae, 4) if math.isfinite(mae) else None,
                "n_active_hypotheses": n_active,
            }
        except Exception as e:
            results_by_buffer[f"buf_{buf}"] = {"buffer_size": buf, "error": str(e)}

    return results_by_buffer


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage_11_1(n_seeds: int = 3) -> Dict[str, Any]:
    start = time.time()
    seed_results = [_sparsity_one_seed(s) for s in range(n_seeds)]

    # Aggregate across seeds per sparsity level
    all_keys = sorted({k for r in seed_results for k in r})
    metrics = ["local_ar1_mae", "fed_ar1_mae", "local_sc_mae", "fed_sc_mae"]
    agg: Dict[str, Any] = {}

    for key in all_keys:
        entry_agg: Dict[str, Any] = {}
        for metric in metrics:
            vals = [r[key][metric] for r in seed_results
                    if key in r and metric in r[key] and isinstance(r[key].get(metric), float)
                    and math.isfinite(r[key][metric])]
            if vals:
                entry_agg[metric] = round(float(np.mean(vals)), 4)
        agg[key] = entry_agg

    # Compute degradation slopes (MAE increase per 10pp sparsity)
    def _slope(metric: str) -> Optional[float]:
        x, y = [], []
        for lvl, frac in [(0, 0.0), (20, 0.20), (40, 0.40), (60, 0.60)]:
            k = f"drop_{lvl}pct"
            if k in agg and metric in agg[k]:
                x.append(frac)
                y.append(agg[k][metric])
        if len(x) < 2:
            return None
        n, mx, my = len(x), sum(x) / len(x), sum(y) / len(y)
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        var = sum((xi - mx) ** 2 for xi in x)
        return round(cov / var if var > 1e-12 else 0.0, 4)

    slopes = {m: _slope(m) for m in metrics}

    # Target: federated MAE slope <= local MAE slope (fed degrades more gracefully)
    local_slope = slopes.get("local_sc_mae") or slopes.get("local_ar1_mae")
    fed_slope = slopes.get("fed_sc_mae") or slopes.get("fed_ar1_mae")

    if local_slope is not None and fed_slope is not None:
        status = "PASS" if fed_slope <= local_slope else "WARN"
    else:
        status = "WARN"

    return make_result(
        stage="11.1", name="sparsity_sweep", status=status,
        target="Federated MAE degrades more slowly than local as data is dropped",
        result={
            "n_seeds": n_seeds,
            "sparsity_levels": [f"{int(f * 100)}%" for f in SPARSITY_LEVELS],
            "mae_by_sparsity": agg,
            "degradation_slopes": slopes,
            "local_slope": local_slope,
            "fed_slope": fed_slope,
            "interpretation": (
                "Slope = MAE increase per unit sparsity fraction. "
                "Smaller (or negative) slope = more graceful degradation. "
                "Fed should degrade more slowly because peer data compensates."
            ),
        },
        wallclock_s=time.time() - start,
    )


def run_stage_11_2(n_seeds: int = 3) -> Dict[str, Any]:
    start = time.time()

    try:
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine  # noqa: F401
    except Exception as e:
        return skip_result("11.2", "buffer_sweep", f"OnlineDiscoveryEngine not importable: {e}")

    seed_results = [_buffer_one_seed(s) for s in range(n_seeds)]

    # Aggregate across seeds per buffer size
    agg: Dict[str, Any] = {}
    for buf in BUFFER_SIZES:
        key = f"buf_{buf}"
        maes = [r[key]["mae"] for r in seed_results
                if key in r and r[key].get("mae") is not None
                and math.isfinite(r[key]["mae"])]
        n_actives = [r[key]["n_active_hypotheses"] for r in seed_results
                     if key in r and r[key].get("n_active_hypotheses") is not None]
        entry: Dict[str, Any] = {"buffer_size": buf}
        if maes:
            entry["mean_mae"] = round(float(np.mean(maes)), 4)
            entry["std_mae"] = round(float(np.std(maes)), 4)
        if n_actives:
            entry["mean_active_hyps"] = round(float(np.mean(n_actives)), 1)
        agg[key] = entry

    # Check monotonicity: MAE should not increase with larger buffer
    mae_vals = [(buf, agg.get(f"buf_{buf}", {}).get("mean_mae")) for buf in BUFFER_SIZES]
    mae_vals = [(b, m) for b, m in mae_vals if m is not None]
    monotone = all(mae_vals[i][1] >= mae_vals[i + 1][1] - 0.01
                   for i in range(len(mae_vals) - 1))  # allow 0.01 tolerance

    status = "PASS" if monotone and len(mae_vals) >= 2 else "WARN"

    return make_result(
        stage="11.2", name="buffer_sweep", status=status,
        target="MAE monotonically improves (or plateaus) as buffer_size grows",
        result={
            "n_seeds": n_seeds,
            "buffer_sizes": BUFFER_SIZES,
            "results": agg,
            "monotone_improvement": monotone,
            "mae_sequence": [(b, m) for b, m in mae_vals],
            "interpretation": (
                "Larger buffer = more history for hypothesis fitting. "
                "MAE should decrease or plateau; a rise signals overfitting or noise. "
                "n_active_hypotheses tracks discovery richness vs buffer size."
            ),
        },
        wallclock_s=time.time() - start,
    )


def run_all(fast: bool = False) -> List[Dict[str, Any]]:
    n_seeds = 1 if fast else 3
    results = [
        run_stage_11_1(n_seeds=n_seeds),
        run_stage_11_2(n_seeds=n_seeds),
    ]
    for r in results:
        save_artifact(f"stage11_{r['name']}.json", r)
    return results


if __name__ == "__main__":
    for r in run_all(fast=True):
        print(f"  [{r['status']}] {r['stage']}: {r['name']}")
        import json
        print(json.dumps(r["result"], indent=2))

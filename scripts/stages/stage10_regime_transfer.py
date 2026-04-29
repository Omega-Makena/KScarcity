"""
Stage 10 — Regime Transfer.

Train models on pre-2008 data, evaluate on 2008-2023 data.
Mirrors benchmark_reviewer.py R2: does ScarcityEngine adapt to structural breaks
faster than fixed AR(1)?

Methods compared:
  - AR1-Fixed: fit once on pre-2008, never update
  - AR1-Rolling: refit on all data seen so far (expanding window)
  - ScarcityEngine: OnlineDiscoveryEngine with online updates

Target: ScarcityEngine rolling MAE in post-2008 <= AR1-Fixed rolling MAE
(adaptive system no worse than frozen baseline).
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

PRE_BREAK = 2007   # train up to and including this year
POST_BREAK = 2008  # evaluate from this year onward
START_YEAR = 1990
END_YEAR = 2023
COUNTRIES = ["KEN", "TZA", "UGA"]
PRIMARY = "KEN"

# Magnitude of structural break injected at POST_BREAK
BREAK_MAGNITUDE = 0.30  # 30% shift in level for half the indicators


# ---------------------------------------------------------------------------
# Synthetic data with an injected structural break
# ---------------------------------------------------------------------------

def _mock_data_with_break(country: str, start: int, end: int,
                           seed: int = 0, break_year: int = POST_BREAK,
                           break_magnitude: float = BREAK_MAGNITUDE) -> Dict[int, Dict[str, float]]:
    import random
    rng = random.Random(seed + hash(country) % 1000)
    fields = [
        "gdp_growth", "inflation", "unemployment", "exports_gdp",
        "imports_gdp", "current_account", "govt_consumption", "tax_revenue",
        "govt_debt", "real_interest_rate", "broad_money", "private_credit",
        "urban_population", "school_enrollment", "life_expectancy",
        "electricity_access", "internet_users",
    ]
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

    # Fields that shift at break_year (every other field)
    break_fields = set(fields[::2])

    result: Dict[int, Dict[str, float]] = {}
    for year in range(start, end + 1):
        row = {}
        shift = break_magnitude if year >= break_year else 0.0
        for k, v in base.items():
            noise = rng.gauss(0, abs(v) * 0.05)
            trend = (year - start) * rng.uniform(-0.1, 0.15)
            level_shift = v * shift if k in break_fields else 0.0
            row[k] = v + noise + trend + level_shift
        result[year] = row
    return result


def _build_schema(fields: List[str]) -> Dict:
    return {"fields": [{"name": n, "type": "numeric"} for n in fields]}


# ---------------------------------------------------------------------------
# Shared MAE helpers (duplicated from stage9 to keep stages independent)
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
    n, mx, my = len(x), sum(x) / len(x), sum(y) / len(y)
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
# Regime transfer evaluation for one seed
# ---------------------------------------------------------------------------

def _regime_eval(seed: int, break_year: int = POST_BREAK) -> Dict[str, Any]:
    data = {c: _mock_data_with_break(c, START_YEAR, END_YEAR, seed=seed, break_year=break_year)
            for c in COUNTRIES}
    years_ken = sorted(data[PRIMARY].keys())
    pre_years = [y for y in years_ken if y <= PRE_BREAK]
    post_years = [y for y in years_ken if y > PRE_BREAK]

    all_fields = sorted({f for row in data[PRIMARY].values() for f in row})

    # Pre-break training rows
    pre_rows = [data[PRIMARY][y] for y in pre_years]

    # Fit fixed AR(1) once on pre-break data
    ar1_fixed_params = _fit_ar1(pre_rows)

    # Scarcity engine: stream pre-break data
    scarcity_available = False
    try:
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
        eng = OnlineDiscoveryEngine(explore_interval=5, mode="balanced", buffer_size=50)
        eng.initialize(_build_schema(all_fields))
        for y in pre_years:
            eng.process_row(data[PRIMARY][y])
        scarcity_available = True
    except Exception:
        pass

    # Rolling evaluation over post-break years (expanding window from pre-break)
    ar1_fixed_preds, ar1_rolling_preds, sc_preds = [], [], []
    actuals = []

    all_seen = list(pre_rows)

    for t_idx, year in enumerate(post_years):
        if t_idx == 0:
            last_row = pre_rows[-1]
        else:
            last_row = data[PRIMARY][post_years[t_idx - 1]]

        actual = data[PRIMARY][year]
        actuals.append(actual)

        # AR1-Fixed: frozen pre-break params
        ar1_fixed_preds.append(_predict_ar1(ar1_fixed_params, last_row))

        # AR1-Rolling: refit on all seen data
        rolling_params = _fit_ar1(all_seen)
        ar1_rolling_preds.append(_predict_ar1(rolling_params, last_row))

        # ScarcityEngine
        if scarcity_available:
            try:
                sc_preds.append(eng.predict(last_row))
                eng.process_row(actual)
            except Exception:
                pass

        all_seen.append(actual)

    norm = _field_stats(actuals)
    results: Dict[str, Any] = {}

    for method, preds in [("ar1_fixed", ar1_fixed_preds), ("ar1_rolling", ar1_rolling_preds)]:
        if preds:
            results[method] = round(_normalised_mae(preds[:len(actuals)], actuals, norm), 4)

    if scarcity_available and sc_preds:
        results["scarcity_engine"] = round(
            _normalised_mae(sc_preds[:len(actuals)], actuals[:len(sc_preds)], norm), 4
        )

    # Adaptation speed: MAE in first 3 post-break years vs last 3
    def _window_mae(preds, n):
        if len(preds) < n:
            return float("nan")
        return round(_normalised_mae(preds[:n], actuals[:n], norm), 4)

    def _tail_mae(preds, n):
        if len(preds) < n:
            return float("nan")
        return round(_normalised_mae(preds[-n:], actuals[-n:], norm), 4)

    adaptation = {}
    for method, preds in [("ar1_fixed", ar1_fixed_preds), ("ar1_rolling", ar1_rolling_preds),
                           ("scarcity_engine", sc_preds if scarcity_available else [])]:
        if preds:
            adaptation[method] = {
                "early_mae": _window_mae(preds, 3),
                "late_mae": _tail_mae(preds, 3),
            }

    return {
        "n_pre": len(pre_years),
        "n_post": len(post_years),
        "mae": results,
        "adaptation": adaptation,
    }


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage_10(n_seeds: int = 5) -> Dict[str, Any]:
    start = time.time()
    seed_results = []
    for seed in range(n_seeds):
        try:
            r = _regime_eval(seed)
            seed_results.append(r)
        except Exception as e:
            seed_results.append({"error": str(e)})

    # Aggregate
    methods = set()
    for r in seed_results:
        methods.update(r.get("mae", {}).keys())

    agg: Dict[str, Any] = {}
    for method in sorted(methods):
        vals = [r["mae"][method] for r in seed_results
                if "mae" in r and method in r["mae"] and math.isfinite(r["mae"][method])]
        if vals:
            agg[method] = {
                "mean_mae": round(float(np.mean(vals)), 4),
                "std_mae": round(float(np.std(vals)), 4),
                "n_seeds": len(vals),
            }

    # Adaptation speed: early vs late MAE for scarcity_engine
    sc_adapt = defaultdict(list)
    for r in seed_results:
        for method, d in r.get("adaptation", {}).items():
            if math.isfinite(d.get("early_mae", float("nan"))):
                sc_adapt[f"{method}_early"].append(d["early_mae"])
            if math.isfinite(d.get("late_mae", float("nan"))):
                sc_adapt[f"{method}_late"].append(d["late_mae"])

    adapt_summary = {k: round(float(np.mean(v)), 4) for k, v in sc_adapt.items() if v}

    # Target: ScarcityEngine MAE <= AR1-Fixed MAE in post-break period
    sc_mae = agg.get("scarcity_engine", {}).get("mean_mae")
    fixed_mae = agg.get("ar1_fixed", {}).get("mean_mae")
    rolling_mae = agg.get("ar1_rolling", {}).get("mean_mae")

    if sc_mae is not None and fixed_mae is not None:
        status = "PASS" if sc_mae <= fixed_mae else "WARN"
        gap_vs_fixed = round(fixed_mae - sc_mae, 4)
    else:
        status = "WARN"
        gap_vs_fixed = None

    gap_vs_rolling = round(rolling_mae - sc_mae, 4) if (rolling_mae is not None and sc_mae is not None) else None

    return make_result(
        stage="10", name="regime_transfer", status=status,
        target="ScarcityEngine post-break MAE <= AR1-Fixed MAE (adaptive >= frozen)",
        result={
            "n_seeds": n_seeds,
            "break_year": POST_BREAK,
            "pre_years": f"{START_YEAR}-{PRE_BREAK}",
            "post_years": f"{POST_BREAK}-{END_YEAR}",
            "methods": agg,
            "adaptation": adapt_summary,
            "scarcity_engine_mae": sc_mae,
            "ar1_fixed_mae": fixed_mae,
            "ar1_rolling_mae": rolling_mae,
            "gap_vs_fixed_pp": gap_vs_fixed,
            "gap_vs_rolling_pp": gap_vs_rolling,
            "interpretation": (
                "Positive gap_vs_fixed = Scarcity adapts better than frozen AR(1). "
                "Compare early_mae vs late_mae to see adaptation speed. "
                "Scarcity should converge faster after the structural break."
            ),
        },
        wallclock_s=time.time() - start,
    )


def run_all(fast: bool = False) -> List[Dict[str, Any]]:
    results = [run_stage_10(n_seeds=2 if fast else 5)]
    for r in results:
        save_artifact(f"stage10_{r['name']}.json", r)
    return results


if __name__ == "__main__":
    for r in run_all(fast=True):
        print(f"  [{r['status']}] {r['stage']}: {r['name']}")
        import json
        print(json.dumps(r["result"]["methods"], indent=2))
        print("Adaptation:", json.dumps(r["result"]["adaptation"], indent=2))

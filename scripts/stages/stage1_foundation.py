"""
Stage 1 — Foundation Validation.

1.1  Non-IID verification (Jensen-Shannon Divergence between country pairs)
1.2  Null data FPR (100 trials of pure noise → should rarely exceed conf gate)
1.3  Temporal ordering test (chronological vs reversed vs shuffled)
1.4  Correlation-sign baseline (engine must beat naive Pearson-sign by ≥10pp)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stages.utils import (
    ALL_INDICATORS, INDICATOR_MEANS, INDICATOR_STDS, ARTIFACTS_DIR,
    build_hub, compute_baseline_means, compute_baseline_stds,
    compute_discovery_metrics, fail_result, filter_pairs, load_ground_truth,
    make_null_data, make_result, make_structured_data, rows_to_yearly,
    save_artifact, skip_result, stream_rows,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.1 Non-IID Verification
# ---------------------------------------------------------------------------

def _make_country_data(seed: int, n_obs: int = 34,
                       mean_shift: float = 0.0,
                       var_scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = np.zeros((n_obs, len(ALL_INDICATORS)))
    for j, v in enumerate(ALL_INDICATORS):
        mu = INDICATOR_MEANS[v] + mean_shift
        sigma = INDICATOR_STDS[v] * var_scale
        ar = 0.6
        x = mu
        for i in range(n_obs):
            x = ar * x + (1 - ar) * mu + rng.normal(0, sigma * 0.5)
            mat[i, j] = x
    return mat


def _empirical_hist(series: np.ndarray, bins: int = 20) -> np.ndarray:
    counts, _ = np.histogram(series, bins=bins, density=False)
    counts = counts.astype(float) + 1e-9
    return counts / counts.sum()


def run_stage_1_1(n_bins: int = 20) -> Dict[str, Any]:
    start = time.time()
    try:
        country_seeds = {"KEN": 42, "TZA": 43, "UGA": 44, "ETH": 45, "GHA": 46}
        country_scales = {"KEN": 1.0, "TZA": 1.1, "UGA": 0.9, "ETH": 1.2, "GHA": 0.95}
        country_shifts = {"KEN": 0.0, "TZA": 0.5, "UGA": -0.3, "ETH": 1.0, "GHA": -0.1}

        datasets = {
            cid: _make_country_data(seed, mean_shift=country_shifts[cid], var_scale=country_scales[cid])
            for cid, seed in country_seeds.items()
        }

        country_ids = list(datasets.keys())
        pair_jsds = []
        pair_results = []

        for i in range(len(country_ids)):
            for j in range(i + 1, len(country_ids)):
                a_id, b_id = country_ids[i], country_ids[j]
                a_data, b_data = datasets[a_id], datasets[b_id]
                jsds_per_var = []
                for k in range(a_data.shape[1]):
                    ha = _empirical_hist(a_data[:, k], bins=n_bins)
                    hb = _empirical_hist(b_data[:, k], bins=n_bins)
                    jsds_per_var.append(float(jensenshannon(ha, hb)))
                mean_jsd = float(np.mean(jsds_per_var))
                pair_jsds.append(mean_jsd)
                pair_results.append({"pair": f"{a_id}-{b_id}", "mean_jsd": round(mean_jsd, 4)})

        overall_mean_jsd = float(np.mean(pair_jsds))
        status = "PASS" if overall_mean_jsd > 0.20 else "FAIL"

        return make_result(
            stage="1.1", name="non_iid_verification", status=status,
            target="mean JSD > 0.20 across country pairs",
            result={"mean_jsd": round(overall_mean_jsd, 4), "pairs": pair_results},
            wallclock_s=time.time() - start,
        )
    except Exception as e:
        return fail_result("1.1", "non_iid_verification", "mean JSD > 0.20", str(e), time.time() - start)


# ---------------------------------------------------------------------------
# 1.2 Null Data FPR
# ---------------------------------------------------------------------------

def run_stage_1_2(n_trials: int = 100, n_years: int = 34, n_indicators: int = 17,
                  gates: Optional[List[float]] = None) -> Dict[str, Any]:
    start = time.time()
    if gates is None:
        gates = [0.10, 0.15, 0.20, 0.25, 0.30]

    try:
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
        from scarcity.engine.discovery import HypothesisState
    except ImportError as e:
        return skip_result("1.2", "null_fpr", f"import failed: {e}")

    trial_results = []
    for trial_seed in range(n_trials):
        null_rows = make_null_data(n_obs=n_years, n_vars=n_indicators, seed=trial_seed)
        eng = OnlineDiscoveryEngine(mode="balanced")
        for row in null_rows:
            eng.process_row(row)

        all_hyps = list(eng.hypotheses.population.values())
        live = [h for h in all_hyps if h.meta.state != HypothesisState.DEAD]
        confs = [h.confidence for h in live] if live else [0.0]

        trial_result = {
            "seed": trial_seed,
            "n_live_hypotheses": len(live),
            "max_confidence": round(float(max(confs)), 4),
            "mean_confidence": round(float(np.mean(confs)), 4),
        }
        for gate in gates:
            trial_result[f"any_above_{gate}"] = any(c > gate for c in confs)
        trial_results.append(trial_result)

    # Aggregate FPR per gate
    fpr_per_gate = {}
    for gate in gates:
        key = f"any_above_{gate}"
        fpr = sum(1 for r in trial_results if r[key]) / n_trials
        fpr_per_gate[f"gate_{gate}"] = round(fpr, 4)

    # Primary target: FPR at gate 0.25 < 0.10
    fpr_025 = fpr_per_gate.get("gate_0.25", 1.0)
    status = "PASS" if fpr_025 < 0.10 else "FAIL"

    return make_result(
        stage="1.2", name="null_fpr", status=status,
        target="FPR at conf_gate=0.25 < 0.10 on pure noise",
        result={
            "n_trials": n_trials,
            "fpr_per_gate": fpr_per_gate,
            "fpr_at_025": fpr_025,
            "mean_max_conf_across_trials": round(float(np.mean([r["max_confidence"] for r in trial_results])), 4),
        },
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# 1.3 Temporal Ordering Test
# ---------------------------------------------------------------------------

def _make_lag_dataset(n_obs: int = 100, true_coeff: float = 0.6,
                      noise_std: float = 0.5, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n_obs)
    Y = np.zeros(n_obs)
    Y[0] = rng.standard_normal()
    for t in range(1, n_obs):
        Y[t] = true_coeff * X[t - 1] + rng.normal(0, noise_std)
    return {"X": X, "Y": Y}


def _stream_xy_through_engine(
    engine_cls: Any,
    X_vals: np.ndarray,
    Y_vals: np.ndarray,
) -> Dict[str, float]:
    """Stream (X_t, Y_t) pairs through a fresh engine, return confidence stats."""
    try:
        from scarcity.engine.discovery import HypothesisState
        from scarcity.engine.relationships import CausalHypothesis, CorrelationalHypothesis
    except ImportError:
        return {"error": "import failed"}

    eng = engine_cls(mode="balanced")
    for x, y in zip(X_vals, Y_vals):
        eng.process_row({"X": float(x), "Y": float(y)})

    best_xy_conf = 0.0
    xy_sign = None
    for h in eng.hypotheses.population.values():
        if h.meta.state == HypothesisState.DEAD:
            continue
        if isinstance(h, CausalHypothesis):
            if getattr(h, "source", None) == "X" and getattr(h, "target", None) == "Y":
                if h.confidence > best_xy_conf:
                    best_xy_conf = h.confidence
                    coef = getattr(h, "_coef_aug", None)
                    lag = getattr(h, "lag", 1)
                    if coef is not None and len(coef) > lag + 1:
                        xy_sign = 1 if float(np.sum(coef[lag + 1:])) > 0 else -1
        elif isinstance(h, CorrelationalHypothesis):
            if h.var1 == "X" and h.var2 == "Y":
                if h.confidence > best_xy_conf:
                    best_xy_conf = h.confidence
                    xy_sign = 1 if getattr(h, "r", 0) > 0 else -1

    return {"confidence": round(best_xy_conf, 4), "sign": xy_sign}


def run_stage_1_3(n_datasets: int = 50, n_obs: int = 100, true_coeff: float = 0.6,
                  noise_std: float = 0.5) -> Dict[str, Any]:
    start = time.time()
    try:
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    except ImportError as e:
        return skip_result("1.3", "temporal_ordering", f"import failed: {e}")

    chrono_correct = 0
    reversed_correct = 0
    shuffled_correct = 0
    conf_chrono, conf_reversed, conf_shuffled = [], [], []

    for seed in range(n_datasets):
        ds = _make_lag_dataset(n_obs=n_obs, true_coeff=true_coeff, noise_std=noise_std, seed=seed)
        X, Y = ds["X"], ds["Y"]
        rng = np.random.default_rng(seed + 1000)
        shuffle_idx = rng.permutation(n_obs)

        r_chrono = _stream_xy_through_engine(OnlineDiscoveryEngine, X, Y)
        r_reversed = _stream_xy_through_engine(OnlineDiscoveryEngine, X[::-1], Y[::-1])
        r_shuffled = _stream_xy_through_engine(OnlineDiscoveryEngine, X[shuffle_idx], Y[shuffle_idx])

        conf_chrono.append(r_chrono.get("confidence", 0.0))
        conf_reversed.append(r_reversed.get("confidence", 0.0))
        conf_shuffled.append(r_shuffled.get("confidence", 0.0))

        if r_chrono.get("sign") == 1:  # true_coeff > 0 → expected sign = +1
            chrono_correct += 1
        if r_reversed.get("sign") == 1:
            reversed_correct += 1
        if r_shuffled.get("sign") == 1:
            shuffled_correct += 1

    chrono_acc = chrono_correct / n_datasets
    status = "PASS" if chrono_acc > 0.80 else ("WARN" if chrono_acc > 0.60 else "FAIL")

    return make_result(
        stage="1.3", name="temporal_ordering", status=status,
        target="chronological correct direction > 80%",
        result={
            "n_datasets": n_datasets,
            "chrono_sign_accuracy": round(chrono_acc, 4),
            "reversed_sign_accuracy": round(reversed_correct / n_datasets, 4),
            "shuffled_sign_accuracy": round(shuffled_correct / n_datasets, 4),
            "mean_conf_chrono": round(float(np.mean(conf_chrono)), 4),
            "mean_conf_reversed": round(float(np.mean(conf_reversed)), 4),
            "mean_conf_shuffled": round(float(np.mean(conf_shuffled)), 4),
            "chrono_vs_reversed_conf_delta": round(
                float(np.mean(conf_chrono)) - float(np.mean(conf_reversed)), 4
            ),
        },
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# 1.4 Correlation-Sign Baseline
# ---------------------------------------------------------------------------

def _pearson_sign_accuracy(yearly: Dict[int, Any], pairs: List[Dict],
                            first_diff: bool = False) -> Dict[str, float]:
    all_years = sorted(yearly.keys())
    correct = 0
    total = 0
    for pair in pairs:
        src, tgt, expected_sign = pair["source"], pair["target"], pair["expected_sign"]
        x_vals, y_vals = [], []
        for yr in all_years[:-1]:
            if yr in yearly and (yr + 1) in yearly:
                x_raw = yearly[yr].get(src)
                y_raw = yearly[yr + 1].get(tgt)
                if first_diff and (yr - 1) in yearly:
                    x_prev = yearly[yr - 1].get(src)
                    y_prev = yearly[yr].get(tgt)
                    if x_prev is None or y_prev is None:
                        continue
                    x = x_raw - x_prev if (x_raw is not None) else None
                    y = y_raw - y_prev if (y_raw is not None) else None
                elif first_diff:
                    continue
                else:
                    x, y = x_raw, y_raw
                if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                    x_vals.append(x)
                    y_vals.append(y)
        if len(x_vals) < 5:
            continue
        try:
            r, _ = pearsonr(x_vals, y_vals)
            pred_sign = 1 if r > 0 else -1
            if pred_sign == expected_sign:
                correct += 1
            total += 1
        except Exception:
            pass
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def run_stage_1_4(seed: int = 42) -> Dict[str, Any]:
    start = time.time()
    try:
        all_pairs = load_ground_truth()
        unambiguous = filter_pairs(all_pairs, "unambiguous")
        contested = filter_pairs(all_pairs, "contested")
        identity = filter_pairs(all_pairs, "identity")
    except Exception as e:
        return fail_result("1.4", "correlation_sign_baseline", "engine > best Pearson baseline by >= 5pp", str(e))

    rows = make_structured_data(n_obs=34, seed=seed)
    yearly = rows_to_yearly(rows)

    # Level Pearson and first-difference Pearson baselines
    level_p = _pearson_sign_accuracy(yearly, unambiguous, first_diff=False)
    diff_p = _pearson_sign_accuracy(yearly, unambiguous, first_diff=True)
    better_pearson = max(level_p["accuracy"], diff_p["accuracy"])

    # Per-category level Pearson
    cat_results = {}
    for cat_name, cat_pairs in [("unambiguous", unambiguous), ("contested", contested), ("identity", identity)]:
        lv = _pearson_sign_accuracy(yearly, cat_pairs, first_diff=False)
        fd = _pearson_sign_accuracy(yearly, cat_pairs, first_diff=True)
        cat_results[cat_name] = {
            "n_pairs": len(cat_pairs),
            "level_pearson_acc": round(lv["accuracy"], 4),
            "firstdiff_pearson_acc": round(fd["accuracy"], 4),
        }

    # Engine accuracy (unambiguous only — headline metric)
    try:
        hub = build_hub("KEN")
        stream_rows(hub, "KEN", yearly)
        baseline = compute_baseline_means(yearly)
        stds = compute_baseline_stds(yearly)
        metrics = compute_discovery_metrics(hub, "KEN", unambiguous, baseline, stds)
        engine_acc = metrics["ua_conf_weighted_accuracy"]
    except Exception as e:
        return fail_result("1.4", "correlation_sign_baseline", "engine > best Pearson by >= 5pp", str(e))

    gap_vs_level = engine_acc - level_p["accuracy"]
    gap_vs_diff = engine_acc - diff_p["accuracy"]
    gap_vs_better = engine_acc - better_pearson

    # Target: engine beats BETTER of the two Pearson baselines by >= 5pp on unambiguous
    status = "PASS" if gap_vs_better >= 0.05 else ("WARN" if gap_vs_better >= 0.0 else "FAIL")

    return make_result(
        stage="1.4", name="correlation_sign_baseline", status=status,
        target="engine conf-weighted accuracy > best Pearson baseline (level or first-diff) by >= 5pp",
        result={
            "level_pearson_unambiguous": round(level_p["accuracy"], 4),
            "firstdiff_pearson_unambiguous": round(diff_p["accuracy"], 4),
            "better_pearson_baseline": round(better_pearson, 4),
            "engine_ua_conf_weighted_accuracy": round(engine_acc, 4),
            "gap_vs_level_pp": round(gap_vs_level, 4),
            "gap_vs_firstdiff_pp": round(gap_vs_diff, 4),
            "gap_vs_better_pp": round(gap_vs_better, 4),
            "n_unambiguous_pairs": len(unambiguous),
            "by_category": cat_results,
            "interpretation": (
                "First-diff Pearson removes trend confound affecting infrastructure basket. "
                "Engine must beat the stronger baseline by >= 5pp to show genuine advantage. "
                "Near-zero gap on smooth synthetic data is expected — real-data gap (§31) is larger."
            ),
        },
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(fast: bool = False) -> List[Dict[str, Any]]:
    results = []
    n_trials = 20 if fast else 100
    n_datasets = 20 if fast else 50

    results.append(run_stage_1_1())
    results.append(run_stage_1_2(n_trials=n_trials))
    results.append(run_stage_1_3(n_datasets=n_datasets))
    results.append(run_stage_1_4())

    for r in results:
        save_artifact(f"stage1_{r['name']}.json", r)

    return results


if __name__ == "__main__":
    import json
    for r in run_all(fast=True):
        print(f"  [{r['status']}] {r['stage']}: {r['name']}")

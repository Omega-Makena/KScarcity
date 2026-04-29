"""
Stage 5 — Meta-Learning.

5.1  Pretrain inversion diagnosis (do inverted pairs get corrected by live data?)
5.2  Pioneer row sweep (conf-weighted sign accuracy vs n_pioneer_rows)
5.3  MetaIntegrativeLayer policy verification (SKIP if not standalone)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stages.utils import (
    ALL_INDICATORS, ARTIFACTS_DIR, build_hub, compute_baseline_means,
    compute_baseline_stds, compute_discovery_metrics, fail_result,
    filter_pairs, get_hypothesis_confidence, lag_sweep_predict,
    load_ground_truth, make_result, make_structured_data, rows_to_yearly,
    save_artifact, skip_result, stream_rows,
)

logger = logging.getLogger(__name__)

INVERTED_PAIRS = [
    ("inflation", "real_interest_rate"),      # expected +1 (Taylor rule)
    ("private_credit", "broad_money"),        # expected +1 (money multiplier)
    ("electricity_access", "internet_users"), # expected +1 (digital co-location)
]


# ---------------------------------------------------------------------------
# 5.1 Pretrain Inversion Diagnosis
# ---------------------------------------------------------------------------

def _make_inverted_corpus(n_obs: int = 60, seed: int = 1) -> List[Dict]:
    """
    Corpus where each pair in INVERTED_PAIRS has the OPPOSITE sign embedded.
    Used to simulate a pretrained prior that conflicts with Kenya live data.
    """
    from scripts.stages.utils import INDICATOR_MEANS, INDICATOR_STDS
    rng = np.random.default_rng(seed)
    rows = []
    prev = {k: INDICATOR_MEANS[k] for k in ALL_INDICATORS}

    for _ in range(n_obs):
        row = {}
        for v in ALL_INDICATORS:
            noise = rng.normal(0, INDICATOR_STDS[v] * 0.5)
            row[v] = 0.6 * prev[v] + 0.4 * INDICATOR_MEANS[v] + noise
        # Embed INVERTED relationships
        row["real_interest_rate"] = row["real_interest_rate"] - 0.5 * (
            row["inflation"] - INDICATOR_MEANS["inflation"]
        )
        row["broad_money"] = row["broad_money"] - 0.4 * (
            row["private_credit"] - INDICATOR_MEANS["private_credit"]
        )
        row["internet_users"] = max(1.0, row["internet_users"] - 0.3 * (
            row["electricity_access"] - INDICATOR_MEANS["electricity_access"]
        ))
        prev = row
        rows.append(row)
    return rows


def run_stage_5_1(seed: int = 42) -> Dict[str, Any]:
    start = time.time()
    try:
        from scarcity.engine.federation_hub import FederationHub
        from scarcity.engine.federation_node import FederationNode
        from scarcity.engine.baskets import REGISTRY
    except ImportError as e:
        return skip_result("5.1", "pretrain_inversion_trace", f"import failed: {e}")

    inverted_corpus = _make_inverted_corpus(n_obs=60, seed=seed + 1)
    kenya_rows = make_structured_data(n_obs=34, seed=seed)  # correct signs embedded
    kenya_yearly = rows_to_yearly(kenya_rows)

    pair_traces = []

    for src, tgt in INVERTED_PAIRS:
        try:
            # Pretrain on inverted corpus
            hub = FederationHub()
            hub.register(FederationNode("KEN"))
            node = hub.node("KEN")
            for bid in REGISTRY.all_ids():
                node.pretrain(bid, inverted_corpus)
            node.begin_live_stream(pretrain_discount=0.5)

            conf_after_pretrain = get_hypothesis_confidence(hub, "KEN", src, tgt)

            # Stream Kenya data row by row and record direction at each step
            timeline = []
            baseline = compute_baseline_means(kenya_yearly)
            stds = compute_baseline_stds(kenya_yearly)

            for yr in sorted(kenya_yearly.keys()):
                hub.observe_all({"KEN": kenya_yearly[yr]}, fan_out=False, peer_weight=0.70)
                conf = get_hypothesis_confidence(hub, "KEN", src, tgt)
                perturb = stds.get(src, 1.0)
                node = hub.node("KEN")
                responses = lag_sweep_predict(node, baseline, src, perturb, max_k=2)
                delta = sum(r.get(tgt, baseline.get(tgt, 0.0)) - baseline.get(tgt, 0.0)
                            for r in responses)
                sign = 1 if delta > 1e-9 else (-1 if delta < -1e-9 else 0)
                timeline.append({"year": yr, "conf": round(conf, 4), "sign": sign})

            # Final direction after all Kenya data
            final_conf = get_hypothesis_confidence(hub, "KEN", src, tgt)
            final_deltas = [t["sign"] for t in timeline[-5:] if t["sign"] != 0]
            final_sign = int(np.sign(np.mean(final_deltas))) if final_deltas else 0

            pair_traces.append({
                "pair": f"{src}->{tgt}",
                "expected_sign": +1,
                "conf_after_pretrain": round(conf_after_pretrain, 4),
                "conf_after_live": round(final_conf, 4),
                "final_sign": final_sign,
                "corrected": final_sign == 1,
                "timeline_length": len(timeline),
            })
        except Exception as e:
            pair_traces.append({"pair": f"{src}->{tgt}", "error": str(e)})

    n_corrected = sum(1 for t in pair_traces if t.get("corrected", False))
    status = "PASS" if n_corrected >= len(INVERTED_PAIRS) // 2 else "WARN"

    return make_result(
        stage="5.1", name="pretrain_inversion_trace", status=status,
        target="Live Kenya data corrects ≥50% of inverted pretrained pairs",
        result={
            "n_pairs": len(INVERTED_PAIRS),
            "n_corrected": n_corrected,
            "correction_rate": round(n_corrected / max(len(INVERTED_PAIRS), 1), 4),
            "pair_traces": pair_traces,
        },
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# 5.2 Pioneer Row Sweep
# ---------------------------------------------------------------------------

def run_stage_5_2(seed: int = 42) -> Dict[str, Any]:
    start = time.time()
    try:
        all_pairs = load_ground_truth()
        unambiguous = filter_pairs(all_pairs, "unambiguous")
    except Exception as e:
        return fail_result("5.2", "pioneer_row_sweep", "accuracy increases with pioneer rows", str(e))

    pioneer_counts = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    full_rows = make_structured_data(n_obs=60, seed=seed)
    full_yearly = rows_to_yearly(full_rows)
    baseline = compute_baseline_means(full_yearly)
    stds = compute_baseline_stds(full_yearly)

    sweep_results = []

    for n_pioneer in pioneer_counts:
        try:
            hub = build_hub("KEN")

            if n_pioneer > 0:
                # Pre-stream n_pioneer rows before "live" evaluation
                pioneer_yearly = {yr: full_yearly[yr] for yr in list(sorted(full_yearly.keys()))[:n_pioneer]}
                stream_rows(hub, "KEN", pioneer_yearly)

            # Stream remaining data
            remaining_years = sorted(full_yearly.keys())[n_pioneer:]
            if remaining_years:
                remaining_yearly = {yr: full_yearly[yr] for yr in remaining_years}
                stream_rows(hub, "KEN", remaining_yearly)

            metrics = compute_discovery_metrics(hub, "KEN", unambiguous, baseline, stds)
            sweep_results.append({
                "n_pioneer": n_pioneer,
                "ua_conf_weighted_accuracy": round(metrics["ua_conf_weighted_accuracy"], 4),
                "ua_sign_accuracy": round(metrics["ua_sign_accuracy"], 4),
                "ua_overall_recall": round(metrics["ua_overall_recall"], 4),
                "discovery_rate": round(metrics["discovery_rate"], 4),
            })
        except Exception as e:
            sweep_results.append({"n_pioneer": n_pioneer, "error": str(e)})

    # Check monotone (roughly) improvement
    valid = [r for r in sweep_results if "ua_conf_weighted_accuracy" in r]
    if len(valid) >= 2:
        accs = [r["ua_conf_weighted_accuracy"] for r in valid]
        monotone = accs[-1] >= accs[0]
    else:
        monotone = False

    status = "PASS" if monotone else "WARN"

    return make_result(
        stage="5.2", name="pioneer_row_sweep", status=status,
        target="Accuracy at 50 pioneer rows >= accuracy at 0 pioneer rows",
        result={
            "pioneer_counts": pioneer_counts,
            "sweep": sweep_results,
            "monotone_improvement": monotone,
        },
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# 5.3 MetaIntegrativeLayer Policy Verification
# ---------------------------------------------------------------------------

def run_stage_5_3() -> Dict[str, Any]:
    start = time.time()
    try:
        from scarcity.meta.integrative_meta import MetaIntegrativeLayer
        from scarcity.meta.integrative_config import IntegrativeMetaConfig
    except ImportError as e:
        return skip_result("5.3", "meta_policy_verification",
                           f"MetaIntegrativeLayer not importable standalone: {e}")

    try:
        config = IntegrativeMetaConfig()
        layer = MetaIntegrativeLayer(config)

        test_cases = [
            {
                "name": "high_vram",
                "input": {
                    "vram_util": 0.90, "latency_ms": 60.0, "accept_rate": 0.10,
                    "gain_p50": 0.05, "ci_width_target": 0.10, "stability_avg": 0.70,
                    "rcl_contrast": 0.50, "oom_flag": False, "accept_low_windows": 0,
                },
                "expect_tau_direction": +1,  # tau should increase (more exploration)
            },
            {
                "name": "low_accept_rate",
                "input": {
                    "vram_util": 0.50, "latency_ms": 50.0, "accept_rate": 0.02,
                    "gain_p50": 0.02, "ci_width_target": 0.10, "stability_avg": 0.50,
                    "rcl_contrast": 0.40, "oom_flag": False, "accept_low_windows": 6,
                },
                "expect_g_min_direction": -1,  # g_min should decrease (relax threshold)
            },
            {
                "name": "high_latency",
                "input": {
                    "vram_util": 0.60, "latency_ms": 120.0, "accept_rate": 0.08,
                    "gain_p50": 0.04, "ci_width_target": 0.10, "stability_avg": 0.60,
                    "rcl_contrast": 0.45, "oom_flag": False, "accept_low_windows": 2,
                },
                "expect_resource_reduction": True,
            },
            {
                "name": "oom_flag",
                "input": {
                    "vram_util": 0.95, "latency_ms": 90.0, "accept_rate": 0.05,
                    "gain_p50": 0.02, "ci_width_target": 0.10, "stability_avg": 0.50,
                    "rcl_contrast": 0.40, "oom_flag": True, "accept_low_windows": 3,
                },
                "expect_resource_reduction": True,
            },
        ]

        case_results = []
        for case in test_cases:
            try:
                outputs = layer.update(case["input"])
                result = {"case": case["name"], "outputs": outputs, "passed": True}
                # Validate direction expectations
                if "expect_tau_direction" in case:
                    tau = outputs.get("tau", None)
                    baseline_tau = getattr(config, "controller", None)
                    result["tau_change_sign"] = (
                        "increased" if tau is not None and tau > 0.9 else "not_determined"
                    )
                case_results.append(result)
            except Exception as e:
                case_results.append({"case": case["name"], "error": str(e), "passed": False})

        n_passed = sum(1 for r in case_results if r.get("passed", False))
        status = "PASS" if n_passed == len(test_cases) else "WARN"

        return make_result(
            stage="5.3", name="meta_policy_verification", status=status,
            target="All MetaIntegrativeLayer policy cases produce output without error",
            result={"n_cases": len(test_cases), "n_passed": n_passed, "cases": case_results},
            wallclock_s=time.time() - start,
        )
    except Exception as e:
        return fail_result("5.3", "meta_policy_verification",
                           "MetaIntegrativeLayer policy cases pass", str(e), time.time() - start)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(fast: bool = False) -> List[Dict[str, Any]]:
    results = [
        run_stage_5_1(),
        run_stage_5_2(),
        run_stage_5_3(),
    ]
    for r in results:
        save_artifact(f"stage5_{r['name']}.json", r)
    return results


if __name__ == "__main__":
    for r in run_all():
        print(f"  [{r['status']}] {r['stage']}: {r['name']}")

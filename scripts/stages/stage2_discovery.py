"""
Stage 2 — Discovery Quality.

2.1  Four-condition matrix (cold/pretrain × no-fed/fed)
2.2  Discovery baselines (Pearson, batch Granger, VAR)
2.3  Comparison table across all conditions and baselines
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
    filter_pairs, load_ground_truth, make_result, make_structured_data,
    rows_to_yearly, save_artifact, skip_result, stream_rows,
    stream_rows_hub,
)

logger = logging.getLogger(__name__)

SSA_PRETRAIN_COUNTRIES = ["ETH", "NGA", "GHA", "ZAF", "MOZ"]


def _make_multi_country_data(seeds: Dict[str, int], n_obs: int = 34) -> Dict[str, Dict]:
    from scripts.stages.utils import INDICATOR_MEANS, INDICATOR_STDS
    import numpy as np
    result = {}
    for cid, seed in seeds.items():
        rng = np.random.default_rng(seed)
        rows = []
        prev = {k: INDICATOR_MEANS[k] for k in ALL_INDICATORS}
        for _ in range(n_obs):
            row = {}
            for v in ALL_INDICATORS:
                noise = rng.normal(0, INDICATOR_STDS[v] * 0.5)
                row[v] = 0.6 * prev[v] + 0.4 * INDICATOR_MEANS[v] + noise
            prev = row
            rows.append(row)
        result[cid] = rows_to_yearly(rows)
    return result


def _run_condition(
    primary_id: str,
    primary_yearly: Dict,
    pairs: List[Dict],
    pretrain_corpus: Optional[List[Dict]],
    peer_data: Optional[Dict],
    do_pretrain: bool,
    do_federate: bool,
) -> Dict[str, Any]:
    try:
        from scarcity.engine.federation_hub import FederationHub
        from scarcity.engine.federation_node import FederationNode
        from scarcity.engine.baskets import REGISTRY
    except ImportError as e:
        return {"error": f"import failed: {e}"}

    hub = FederationHub()
    hub.register(FederationNode(primary_id))
    if do_federate and peer_data:
        for pid in peer_data:
            hub.register(FederationNode(pid))

    if do_pretrain and pretrain_corpus:
        for nid in hub.node_ids():
            node = hub.node(nid)
            for bid in REGISTRY.all_ids():
                node.pretrain(bid, pretrain_corpus)
        for nid in hub.node_ids():
            hub.node(nid).begin_live_stream(pretrain_discount=0.5)

    if do_federate and peer_data:
        stream_rows_hub(hub, primary_id, primary_yearly, peer_data=peer_data, fan_out=True)
    else:
        stream_rows(hub, primary_id, primary_yearly)

    baseline = compute_baseline_means(primary_yearly)
    stds = compute_baseline_stds(primary_yearly)
    metrics = compute_discovery_metrics(hub, primary_id, pairs, baseline, stds)
    return metrics


# ---------------------------------------------------------------------------
# 2.1 Four-Condition Matrix
# ---------------------------------------------------------------------------

def run_stage_2_1(seed: int = 42, pretrain_n: int = 60) -> Dict[str, Any]:
    start = time.time()
    try:
        all_pairs = load_ground_truth()
        unambiguous = filter_pairs(all_pairs, "unambiguous")
    except Exception as e:
        return fail_result("2.1", "four_condition_matrix", "cond D best on unambiguous", str(e))

    # Primary country synthetic data
    primary_rows = make_structured_data(n_obs=34, seed=seed)
    primary_yearly = rows_to_yearly(primary_rows)

    # Peer data (TZA, UGA)
    peer_seeds = {"TZA": seed + 1, "UGA": seed + 2}
    peer_data = _make_multi_country_data(peer_seeds, n_obs=34)

    # Pretrain corpus: 5 SSA countries
    ssa_seeds = {"ETH": 10, "NGA": 11, "GHA": 12, "ZAF": 13, "MOZ": 14}
    ssa_data = _make_multi_country_data(ssa_seeds, n_obs=pretrain_n)
    pretrain_corpus = []
    for cdata in ssa_data.values():
        pretrain_corpus.extend(list(cdata.values()))

    conditions = {
        "A_cold_no_fed": (False, False),
        "B_cold_fed": (False, True),
        "C_pretrain_no_fed": (True, False),
        "D_pretrain_fed": (True, True),
    }

    condition_results = {}
    for cname, (do_pretrain, do_federate) in conditions.items():
        t0 = time.time()
        try:
            metrics = _run_condition(
                primary_id="KEN",
                primary_yearly=primary_yearly,
                pairs=unambiguous,
                pretrain_corpus=pretrain_corpus if do_pretrain else None,
                peer_data=peer_data if do_federate else None,
                do_pretrain=do_pretrain,
                do_federate=do_federate,
            )
            metrics["wallclock_s"] = round(time.time() - t0, 2)
        except Exception as e:
            metrics = {"error": str(e)}
        condition_results[cname] = metrics

    # Determine best condition by ua_conf_weighted_accuracy
    best_cond = max(
        condition_results.items(),
        key=lambda kv: kv[1].get("ua_conf_weighted_accuracy", 0.0),
    )
    status = "PASS" if best_cond[0] in ("C_pretrain_no_fed", "D_pretrain_fed") else "WARN"

    return make_result(
        stage="2.1", name="four_condition_matrix", status=status,
        target="Condition C or D has highest ua_conf_weighted_accuracy",
        result={
            "conditions": condition_results,
            "best_condition": best_cond[0],
            "best_ua_conf_weighted_accuracy": round(
                best_cond[1].get("ua_conf_weighted_accuracy", 0.0), 4
            ),
        },
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# 2.2 Discovery Baselines
# ---------------------------------------------------------------------------

def baseline_correlation_sign(yearly: Dict, pairs: List[Dict]) -> Dict[str, Any]:
    from scipy.stats import pearsonr
    correct, total = 0, 0
    all_years = sorted(yearly.keys())
    for pair in pairs:
        src, tgt, exp = pair["source"], pair["target"], pair["expected_sign"]
        xs, ys = [], []
        for yr in all_years[:-1]:
            x = yearly.get(yr, {}).get(src)
            y = yearly.get(yr + 1, {}).get(tgt)
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                xs.append(x); ys.append(y)
        if len(xs) < 5:
            continue
        try:
            r, _ = pearsonr(xs, ys)
            if (1 if r > 0 else -1) == exp:
                correct += 1
            total += 1
        except Exception:
            pass
    return {
        "sign_accuracy": round(correct / max(total, 1), 4),
        "n_evaluated": total,
    }


def baseline_batch_granger(yearly: Dict, pairs: List[Dict], maxlag: int = 2) -> Dict[str, Any]:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return {"skipped": "statsmodels not available"}

    all_years = sorted(yearly.keys())
    n = len(all_years)
    all_data = np.array([[yearly[yr].get(v, 0.0) for v in ALL_INDICATORS] for yr in all_years])

    correct, total = 0, 0
    for pair in pairs:
        src_idx = ALL_INDICATORS.index(pair["source"]) if pair["source"] in ALL_INDICATORS else -1
        tgt_idx = ALL_INDICATORS.index(pair["target"]) if pair["target"] in ALL_INDICATORS else -1
        if src_idx < 0 or tgt_idx < 0 or n < maxlag + 10:
            continue
        try:
            xy = np.column_stack([all_data[:, tgt_idx], all_data[:, src_idx]])
            gresult = grangercausalitytests(xy, maxlag=maxlag, verbose=False)
            # Use p-value from lag-1 F-test
            p_ftest = gresult[1][0]["params_ftest"][1]
            # Sign from OLS coefficient
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm
            y_vec = all_data[1:, tgt_idx]
            x_mat = sm.add_constant(np.column_stack([all_data[:-1, tgt_idx], all_data[:-1, src_idx]]))
            ols = OLS(y_vec, x_mat).fit()
            coef_src = float(ols.params[-1])
            if p_ftest < 0.10:
                pred_sign = 1 if coef_src > 0 else -1
                if pred_sign == pair["expected_sign"]:
                    correct += 1
                total += 1
        except Exception:
            pass
    return {
        "sign_accuracy_when_significant": round(correct / max(total, 1), 4),
        "n_significant": total,
        "n_pairs": len(pairs),
    }


def baseline_var_sign(yearly: Dict, pairs: List[Dict], lag: int = 1) -> Dict[str, Any]:
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
    except ImportError:
        return {"skipped": "statsmodels not available"}

    all_years = sorted(yearly.keys())
    all_data = np.array([[yearly[yr].get(v, 0.0) for v in ALL_INDICATORS] for yr in all_years])

    # Fit single VAR on all data
    try:
        model = VAR(all_data)
        fitted = model.fit(maxlags=lag)
        coef_matrix = fitted.coefs[0]  # (n_vars × n_vars), [j,i] = effect of var_i on var_j
    except Exception as e:
        return {"error": str(e)}

    correct, total = 0, 0
    for pair in pairs:
        src = pair["source"]; tgt = pair["target"]
        if src not in ALL_INDICATORS or tgt not in ALL_INDICATORS:
            continue
        i = ALL_INDICATORS.index(src)
        j = ALL_INDICATORS.index(tgt)
        coef = float(coef_matrix[j, i])  # effect of src(t-1) on tgt(t)
        pred_sign = 1 if coef > 0 else -1
        if pred_sign == pair["expected_sign"]:
            correct += 1
        total += 1

    return {
        "sign_accuracy": round(correct / max(total, 1), 4),
        "n_evaluated": total,
    }


def run_stage_2_2(seed: int = 42) -> Dict[str, Any]:
    start = time.time()
    try:
        all_pairs = load_ground_truth()
        unambiguous = filter_pairs(all_pairs, "unambiguous")
    except Exception as e:
        return fail_result("2.2", "discovery_baselines", "baselines computed", str(e))

    rows = make_structured_data(n_obs=34, seed=seed)
    yearly = rows_to_yearly(rows)

    results = {}
    results["pearson_lag1"] = baseline_correlation_sign(yearly, unambiguous)
    results["batch_granger"] = baseline_batch_granger(yearly, unambiguous)
    results["var_sign"] = baseline_var_sign(yearly, unambiguous)

    return make_result(
        stage="2.2", name="discovery_baselines", status="PASS",
        target="Baselines computed for comparison",
        result={"n_unambiguous_pairs": len(unambiguous), "baselines": results},
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# 2.3 Comparison Table
# ---------------------------------------------------------------------------

def run_stage_2_3(seed: int = 42) -> Dict[str, Any]:
    start = time.time()
    try:
        all_pairs = load_ground_truth()
        unambiguous = filter_pairs(all_pairs, "unambiguous")
    except Exception as e:
        return fail_result("2.3", "discovery_comparison", "comparison table produced", str(e))

    rows = make_structured_data(n_obs=34, seed=seed)
    yearly = rows_to_yearly(rows)

    table = []

    # Baselines
    for label, fn in [
        ("pearson_lag1", lambda: baseline_correlation_sign(yearly, unambiguous)),
        ("batch_granger", lambda: baseline_batch_granger(yearly, unambiguous)),
        ("var_sign", lambda: baseline_var_sign(yearly, unambiguous)),
    ]:
        try:
            r = fn()
            table.append({
                "method": label,
                "ua_conf_weighted_accuracy": None,
                "ua_sign_accuracy": r.get("sign_accuracy") or r.get("sign_accuracy_when_significant"),
                "discovery_rate": None,
                "n_evaluated": r.get("n_evaluated") or r.get("n_significant"),
                "skipped": r.get("skipped"),
                "error": r.get("error"),
            })
        except Exception as e:
            table.append({"method": label, "error": str(e)})

    # Engine conditions
    peer_seeds = {"TZA": seed + 1, "UGA": seed + 2}
    peer_data = _make_multi_country_data(peer_seeds, n_obs=34)
    ssa_data = _make_multi_country_data({"ETH": 10, "NGA": 11, "GHA": 12}, n_obs=60)
    pretrain_corpus = [row for cdata in ssa_data.values() for row in cdata.values()]

    for cname, (do_pretrain, do_federate) in [
        ("engine_A_cold_no_fed", (False, False)),
        ("engine_B_cold_fed", (False, True)),
        ("engine_C_pretrain_no_fed", (True, False)),
        ("engine_D_pretrain_fed", (True, True)),
    ]:
        try:
            m = _run_condition(
                "KEN", yearly, unambiguous,
                pretrain_corpus if do_pretrain else None,
                peer_data if do_federate else None,
                do_pretrain, do_federate,
            )
            table.append({
                "method": cname,
                "ua_conf_weighted_accuracy": round(m.get("ua_conf_weighted_accuracy", 0.0), 4),
                "ua_sign_accuracy": round(m.get("ua_sign_accuracy", 0.0), 4),
                "ua_overall_recall": round(m.get("ua_overall_recall", 0.0), 4),
                "discovery_rate": round(m.get("discovery_rate", 0.0), 4),
                "n_discovered": m.get("n_discovered"),
                "n_sign_correct": m.get("n_sign_correct"),
            })
        except Exception as e:
            table.append({"method": cname, "error": str(e)})

    save_artifact("discovery_comparison.json", {"table": table, "n_pairs": len(unambiguous)})

    status = "PASS"
    return make_result(
        stage="2.3", name="discovery_comparison", status=status,
        target="Comparison table produced across baselines and engine conditions",
        result={"n_methods": len(table), "table": table},
        wallclock_s=time.time() - start,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(fast: bool = False) -> List[Dict[str, Any]]:
    results = [
        run_stage_2_1(),
        run_stage_2_2(),
        run_stage_2_3(),
    ]
    for r in results:
        save_artifact(f"stage2_{r['name']}.json", r)
    return results


if __name__ == "__main__":
    import json
    for r in run_all(fast=True):
        print(f"  [{r['status']}] {r['stage']}: {r['name']}")

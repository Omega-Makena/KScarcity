"""
utils.py — shared helpers for the benchmark harness.
"""
from __future__ import annotations

import json
import time
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "harness"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "ground_truth.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract helpers
# ---------------------------------------------------------------------------

def make_result(
    stage: str,
    name: str,
    status: str,
    target: str,
    result: Dict[str, Any],
    wallclock_s: float,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "name": name,
        "status": status,
        "target": target,
        "result": result,
        "wallclock_s": round(wallclock_s, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_artifact(filename: str, data: Any) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def skip_result(stage: str, name: str, reason: str, wallclock_s: float = 0.0) -> Dict[str, Any]:
    return make_result(stage, name, "SKIP", "N/A", {"reason": reason}, wallclock_s)


def fail_result(stage: str, name: str, target: str, error: str, wallclock_s: float = 0.0) -> Dict[str, Any]:
    return make_result(stage, name, "FAIL", target, {"error": error}, wallclock_s)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth() -> List[Dict[str, Any]]:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_PATH}")
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["pairs"]


def filter_pairs(pairs: List[Dict], category: Optional[str] = None) -> List[Dict]:
    if category is None:
        return pairs
    return [p for p in pairs if p.get("category") == category]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

ALL_INDICATORS = [
    "gdp_growth", "inflation", "unemployment", "exports_gdp", "imports_gdp",
    "current_account", "govt_consumption", "tax_revenue", "govt_debt",
    "real_interest_rate", "broad_money", "private_credit",
    "urban_population", "school_enrollment", "life_expectancy",
    "electricity_access", "internet_users",
]

INDICATOR_MEANS = {
    "gdp_growth": 5.0,
    "inflation": 7.0,
    "unemployment": 10.0,
    "exports_gdp": 15.0,
    "imports_gdp": 20.0,
    "current_account": -5.0,
    "govt_consumption": 15.0,
    "tax_revenue": 18.0,
    "govt_debt": 55.0,
    "real_interest_rate": 4.0,
    "broad_money": 40.0,
    "private_credit": 30.0,
    "urban_population": 28.0,
    "school_enrollment": 90.0,
    "life_expectancy": 67.0,
    "electricity_access": 55.0,
    "internet_users": 22.0,
}

INDICATOR_STDS = {
    "gdp_growth": 2.5,
    "inflation": 5.0,
    "unemployment": 3.0,
    "exports_gdp": 4.0,
    "imports_gdp": 4.0,
    "current_account": 3.0,
    "govt_consumption": 2.0,
    "tax_revenue": 2.5,
    "govt_debt": 10.0,
    "real_interest_rate": 3.5,
    "broad_money": 8.0,
    "private_credit": 6.0,
    "urban_population": 3.0,
    "school_enrollment": 5.0,
    "life_expectancy": 2.0,
    "electricity_access": 8.0,
    "internet_users": 6.0,
}


def make_structured_data(n_obs: int = 34, seed: int = 42) -> List[Dict[str, float]]:
    """
    Generate synthetic data with embedded economic relationships for testing.
    All relationships conform to GROUND_TRUTH signs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    prev = {k: INDICATOR_MEANS[k] for k in ALL_INDICATORS}

    for _ in range(n_obs):
        row = {}
        # AR(1) base
        for v in ALL_INDICATORS:
            noise = rng.normal(0, INDICATOR_STDS[v] * 0.5)
            row[v] = 0.6 * prev[v] + 0.4 * INDICATOR_MEANS[v] + noise

        # Embed key relationships
        row["unemployment"] = max(3.0, row["unemployment"] - 0.3 * (row["gdp_growth"] - INDICATOR_MEANS["gdp_growth"]))
        row["tax_revenue"] = row["tax_revenue"] + 0.4 * (row["gdp_growth"] - INDICATOR_MEANS["gdp_growth"])
        row["real_interest_rate"] = row["real_interest_rate"] + 0.5 * (row["inflation"] - INDICATOR_MEANS["inflation"])
        row["private_credit"] = row["private_credit"] - 0.3 * (row["real_interest_rate"] - INDICATOR_MEANS["real_interest_rate"])
        row["broad_money"] = row["broad_money"] + 0.4 * (row["private_credit"] - INDICATOR_MEANS["private_credit"])
        row["internet_users"] = max(1.0, row["internet_users"] + 0.3 * (row["electricity_access"] - INDICATOR_MEANS["electricity_access"]))
        row["gdp_growth"] = (
            row["gdp_growth"]
            + 0.2 * (row["private_credit"] - INDICATOR_MEANS["private_credit"]) / INDICATOR_STDS["private_credit"]
            + 0.15 * (row["electricity_access"] - INDICATOR_MEANS["electricity_access"]) / INDICATOR_STDS["electricity_access"]
            - 0.15 * (row["unemployment"] - INDICATOR_MEANS["unemployment"]) / INDICATOR_STDS["unemployment"]
        )

        prev = row
        rows.append(row)
    return rows


def make_null_data(n_obs: int = 34, n_vars: int = 17, seed: int = 0) -> List[Dict[str, float]]:
    """Pure i.i.d. Gaussian noise — no relationships."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n_obs, n_vars))
    return [
        {ALL_INDICATORS[j]: float(mat[i, j]) for j in range(n_vars)}
        for i in range(n_obs)
    ]


def rows_to_yearly(rows: List[Dict[str, float]], start_year: int = 1990) -> Dict[int, Dict[str, float]]:
    return {start_year + i: row for i, row in enumerate(rows)}


# ---------------------------------------------------------------------------
# Engine builder (mirrors benchmark_discovery.py exactly)
# ---------------------------------------------------------------------------

def build_engine_node(node_id: str = "KEN") -> Any:
    from scarcity.engine.federation_node import FederationNode
    return FederationNode(node_id)


def build_hub(primary_id: str = "KEN", peer_ids: Optional[List[str]] = None) -> Any:
    from scarcity.engine.federation_hub import FederationHub
    from scarcity.engine.federation_node import FederationNode
    hub = FederationHub()
    hub.register(FederationNode(primary_id))
    if peer_ids:
        for pid in peer_ids:
            hub.register(FederationNode(pid))
    return hub


def stream_rows(hub: Any, primary_id: str, yearly: Dict[int, Dict[str, float]],
                fan_out: bool = False) -> None:
    for yr in sorted(yearly.keys()):
        hub.observe_all({primary_id: yearly[yr]}, fan_out=fan_out, peer_weight=0.70)


def stream_rows_hub(hub: Any, primary_id: str, yearly: Dict[int, Dict[str, float]],
                    peer_data: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
                    fan_out: bool = True) -> None:
    all_years = sorted(set(yearly.keys()) | (set().union(*[set(d) for d in (peer_data or {}).values()]) if peer_data else set()))
    for yr in all_years:
        rows: Dict[str, Dict[str, float]] = {}
        if yr in yearly:
            rows[primary_id] = {k: v for k, v in yearly[yr].items() if k in set(ALL_INDICATORS)}
        if peer_data and fan_out:
            for pid, pdata in peer_data.items():
                if yr in pdata:
                    rows[pid] = {k: v for k, v in pdata[yr].items() if k in set(ALL_INDICATORS)}
        if rows:
            hub.observe_all(rows, fan_out=fan_out, peer_weight=0.70)


# ---------------------------------------------------------------------------
# Discovery evaluation helpers
# ---------------------------------------------------------------------------

def get_hypothesis_confidence(hub: Any, node_id: str, source: str, target: str) -> float:
    """Best confidence score for any live hypothesis covering source->target."""
    try:
        from scarcity.engine.discovery import HypothesisState
        from scarcity.engine.relationships import (
            CorrelationalHypothesis, CausalHypothesis, FunctionalHypothesis,
        )
        node = hub.node(node_id)
        best = 0.0
        for bid in node.basket_ids:
            eng = node._engines.get(bid)
            if eng is None:
                continue
            for h in eng.hypotheses.population.values():
                if h.meta.state == HypothesisState.DEAD:
                    continue
                if isinstance(h, CorrelationalHypothesis):
                    if h.var1 == source and h.var2 == target:
                        best = max(best, h.confidence)
                elif isinstance(h, (CausalHypothesis, FunctionalHypothesis)):
                    if getattr(h, "source", None) == source and getattr(h, "target", None) == target:
                        best = max(best, h.confidence)
    except Exception:
        pass
    return best


def low_threshold_predict(node: Any, row: Dict[str, float], threshold: float = 0.10) -> Dict[str, float]:
    """Confidence-weighted ensemble prediction (mirrors _low_threshold_predict in benchmark_discovery)."""
    try:
        from scarcity.engine.discovery import HypothesisState
        from scarcity.engine.baskets import REGISTRY
    except ImportError:
        return dict(row)

    weighted_sum: Dict[str, float] = {}
    weight_total: Dict[str, float] = {}

    for bid in node.basket_ids:
        eng = node._engines.get(bid)
        if eng is None:
            continue
        basket = REGISTRY.get(bid)
        filtered = basket.filter_row(row)
        if not filtered:
            continue
        for h in eng.hypotheses.population.values():
            if h.meta.state == HypothesisState.DEAD:
                continue
            if h.confidence < threshold:
                continue
            result = h.predict_value(filtered)
            if result is None:
                continue
            var, val = result
            if not np.isfinite(val):
                continue
            w = h.confidence
            weighted_sum[var] = weighted_sum.get(var, 0.0) + w * val
            weight_total[var] = weight_total.get(var, 0.0) + w

    output = {v: weighted_sum[v] / weight_total[v] for v in weighted_sum if weight_total[v] > 0}
    for var, val in row.items():
        if var not in output and np.isfinite(val):
            output[var] = val
    return output


def lag_sweep_predict(node: Any, baseline: Dict[str, float], source: str, perturb: float,
                      max_k: int = 4, threshold: float = 0.10) -> List[Dict[str, float]]:
    """Step-function perturbation propagated for max_k steps."""
    current = dict(baseline)
    current[source] = baseline.get(source, 0.0) + perturb
    responses = []
    for _ in range(max_k):
        try:
            pred = low_threshold_predict(node, current, threshold=threshold)
        except Exception:
            pred = dict(current)
        responses.append(dict(pred))
        current = {v: pred.get(v, baseline.get(v, 0.0)) for v in baseline}
        current[source] = baseline.get(source, 0.0) + perturb
    return responses


def evaluate_pair_discovery(hub: Any, node_id: str, source: str, target: str,
                             baseline: Dict[str, float], stds: Dict[str, float],
                             max_k: int = 4) -> Dict[str, Any]:
    """
    Evaluate one source->target pair using lag sweep perturbation.
    Returns dict with: discovered (bool), sign (int or None), confidence (float), delta (float).
    """
    perturb = stds.get(source, 1.0)
    if perturb == 0:
        perturb = 1.0

    try:
        node = hub.node(node_id)
        responses = lag_sweep_predict(node, baseline, source, perturb, max_k=max_k)
    except Exception as e:
        return {"discovered": False, "sign": None, "confidence": 0.0, "delta": 0.0, "error": str(e)}

    # Aggregate delta across steps
    target_base = baseline.get(target, 0.0)
    deltas = [r.get(target, target_base) - target_base for r in responses]
    total_delta = sum(deltas)

    discovered = abs(total_delta) > 1e-9
    sign = None
    if discovered:
        sign = 1 if total_delta > 0 else -1

    conf = get_hypothesis_confidence(hub, node_id, source, target)

    return {
        "discovered": discovered,
        "sign": sign,
        "confidence": conf,
        "delta": round(total_delta, 6),
    }


def compute_discovery_metrics(hub: Any, node_id: str, pairs: List[Dict],
                              baseline: Dict[str, float],
                              stds: Dict[str, float]) -> Dict[str, Any]:
    """Compute discovery rate, sign accuracy, conf-weighted accuracy."""
    results = []
    for pair in pairs:
        src, tgt, expected_sign = pair["source"], pair["target"], pair["expected_sign"]
        ev = evaluate_pair_discovery(hub, node_id, src, tgt, baseline, stds)
        ev["source"] = src
        ev["target"] = tgt
        ev["expected_sign"] = expected_sign
        ev["category"] = pair.get("category", "unambiguous")
        if ev["discovered"] and ev["sign"] is not None:
            ev["sign_correct"] = ev["sign"] == expected_sign
        else:
            ev["sign_correct"] = False
        results.append(ev)

    discovered = [r for r in results if r["discovered"]]
    sign_correct = [r for r in results if r["sign_correct"]]
    unambiguous = [r for r in results if r["category"] == "unambiguous"]
    ua_correct = [r for r in unambiguous if r["sign_correct"]]
    ua_discovered = [r for r in unambiguous if r["discovered"]]

    total_conf = sum(r["confidence"] for r in results)
    correct_conf = sum(r["confidence"] for r in results if r["sign_correct"])
    conf_weighted_acc = correct_conf / max(total_conf, 1e-9)

    ua_conf_total = sum(r["confidence"] for r in unambiguous)
    ua_conf_correct = sum(r["confidence"] for r in unambiguous if r["sign_correct"])
    ua_conf_weighted_acc = ua_conf_correct / max(ua_conf_total, 1e-9)

    return {
        "n_pairs": len(results),
        "n_discovered": len(discovered),
        "n_sign_correct": len(sign_correct),
        "discovery_rate": len(discovered) / max(len(results), 1),
        "sign_accuracy": len(sign_correct) / max(len(discovered), 1),
        "overall_recall": len(sign_correct) / max(len(results), 1),
        "conf_weighted_accuracy": round(conf_weighted_acc, 4),
        "n_unambiguous": len(unambiguous),
        "n_ua_discovered": len(ua_discovered),
        "n_ua_correct": len(ua_correct),
        "ua_sign_accuracy": len(ua_correct) / max(len(ua_discovered), 1),
        "ua_overall_recall": len(ua_correct) / max(len(unambiguous), 1),
        "ua_conf_weighted_accuracy": round(ua_conf_weighted_acc, 4),
        "per_pair": results,
    }


def compute_baseline_stds(yearly: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    stds = {}
    for v in ALL_INDICATORS:
        vals = [yearly[yr][v] for yr in yearly if v in yearly.get(yr, {})]
        if len(vals) >= 2:
            stds[v] = max(float(np.std(vals)), 1e-6)
        else:
            stds[v] = 1.0
    return stds


def compute_baseline_means(yearly: Dict[int, Dict[str, float]], n_tail: int = 5) -> Dict[str, float]:
    years = sorted(yearly.keys())[-n_tail:]
    baseline: Dict[str, float] = {}
    for v in ALL_INDICATORS:
        vals = [yearly[yr][v] for yr in years if v in yearly.get(yr, {})]
        if vals:
            baseline[v] = float(np.mean(vals))
        else:
            baseline[v] = INDICATOR_MEANS.get(v, 0.0)
    return baseline

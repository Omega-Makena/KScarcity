"""
Failure-mode taxonomy — adversarial datasets with known correct answers.

Recovery on clean synthetic data shows the engine works when the ground truth is
present and the sample supports it. This suite does the opposite: it builds
datasets designed to *mislead* a discovery system — confounding, reverse
causality, colliders, feedback, nonstationarity, weak signal — and checks what
Scarcity does. Some the engine should get right (no-relationship, direction,
decay); some are known-hard (an unobserved confounder, adjusting for a collider)
and the honest result is to document the failure, not hide it.

Each scenario yields rows + a ground-truth spec + an evaluator that returns a
structured verdict. ``run_all`` runs the suite and returns a report; the
benchmark script wraps each scenario in the experiment layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rows(arrays: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    names = list(arrays)
    n = len(next(iter(arrays.values())))
    return [{k: float(arrays[k][t]) for k in names} for t in range(n)]


def _discover(rows: List[Dict[str, float]], variables: List[str],
              min_conf: float = 0.55):
    """Stream rows through the pure-Python engine; return the knowledge graph."""
    from scarcity.engine import OnlineDiscoveryEngine
    eng = OnlineDiscoveryEngine(vectorized=False, small_dataset_mode=True)
    eng.initialize({"fields": [{"name": v} for v in variables]})
    for r in rows:
        eng.process_row(r)
    kg = eng.get_knowledge_graph()
    strong = [e for e in kg if e["metrics"].get("confidence", 0) >= min_conf
              and e.get("state") != "dead"]
    return eng, kg, strong


def _causal_ate(rows, treatment, outcome, confounders):
    """Adjusted ATE + sensitivity via the offline arm (or None if unavailable)."""
    try:
        import pandas as pd
        from scarcity.causal import run_causal, EstimandSpec, EstimandType, RuntimeSpec
    except Exception:
        return None
    df = pd.DataFrame(rows)
    spec = EstimandSpec(treatment=treatment, outcome=outcome,
                        confounders=list(confounders), type=EstimandType.ATE)
    res = run_causal(df, spec, RuntimeSpec(refutation_simulations=0, parallelism="none", seed=0))
    if not res.results:
        return None
    a = res.results[0]
    return {
        "estimate": float(getattr(a.estimate, "value", a.estimate)),
        "sensitivity": (a.refuter_results or {}).get("sensitivity", {}),
    }


@dataclass
class ScenarioResult:
    name: str
    category: str
    expected: str
    observed: str
    verdict: str            # correct | misled | detected | abstained | documented
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    category: str
    expected: str
    run: Callable[[int], ScenarioResult]


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
def _s_no_relationship(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    n = 1200
    rows = _rows({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    _, _, strong = _discover(rows, ["a", "b", "c"])
    directional = [e for e in strong if e["type"] in ("causal", "temporal", "mediating")]
    ok = len(directional) == 0
    return ScenarioResult(
        "no_relationship", "false-positive",
        "no strong directional edge (all variables independent)",
        f"{len(strong)} strong edges, {len(directional)} directional",
        "correct" if ok else "misled",
        {"strong_edges": [(e["type"], tuple(e["variables"])) for e in strong]},
    )


def _s_confounding_observed(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    n = 3000
    c = rng.normal(size=n)
    a = 1.5 * c + rng.normal(scale=0.5, size=n)      # C -> A
    b = 1.5 * c + rng.normal(scale=0.5, size=n)      # C -> B, no A -> B
    rows = _rows({"a": a, "b": b, "c": c})
    res = _causal_ate(rows, "a", "b", ["c"])
    if res is None:
        return ScenarioResult("confounding_observed", "confounding",
                              "adjusted A->B effect ~ 0 (spurious via C)",
                              "causal arm unavailable", "documented")
    ate = res["estimate"]
    ok = abs(ate) < 0.15
    return ScenarioResult(
        "confounding_observed", "confounding",
        "adjusted A->B effect ~ 0 (association is spurious, via observed C)",
        f"adjusted ATE(A->B | C) = {ate:+.3f}",
        "correct" if ok else "misled",
        {"adjusted_ate": ate},
    )


def _s_reverse_causality(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    n = 3000
    b = rng.normal(size=n)                                    # white: no contemporaneous confound
    a = np.zeros(n)
    a[1:] = 0.9 * b[:-1] + rng.normal(scale=0.3, size=n - 1)  # B_{t-1} -> A_t (clean lag-1)
    rows = _rows({"a": a, "b": b})
    _, kg, _ = _discover(rows, ["a", "b"])
    directional = [e for e in kg if e["type"] in ("causal", "temporal")
                   and e["metrics"]["confidence"] >= 0.55]

    def conf(src, tgt):
        return max((e["metrics"]["confidence"] for e in directional
                    if e["variables"][0] == src and e["variables"][-1] == tgt), default=0.0)

    ba, ab = conf("b", "a"), conf("a", "b")
    top = max(kg, key=lambda e: e["metrics"].get("confidence", 0.0), default=None)
    top_desc = f"{top['type']}{tuple(top['variables'])}@{top['metrics'].get('confidence',0):.2f}" if top else "none"
    if ba >= 0.55 and ba > ab:
        verdict, obs = "correct", f"directional B->A (conf {ba:.2f})"
    elif ab >= 0.55 and ab > ba:
        verdict, obs = "misled", f"wrong direction A->B (conf {ab:.2f})"
    else:
        verdict, obs = "missed", f"no confirmed directional edge (strongest: {top_desc})"
    return ScenarioResult(
        "reverse_causality", "direction",
        "recover the directional edge B -> A (A depends on lagged B, r=0.95)",
        obs, verdict, {"conf_B_to_A": ba, "conf_A_to_B": ab, "strongest": top_desc},
    )


def _s_nonstationary_decay(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    half = 1500
    x1 = rng.normal(size=half)
    y1 = 0.9 * x1 + rng.normal(scale=0.2, size=half)          # regime 1: x -> y
    x2 = rng.normal(size=half)
    y2 = rng.normal(size=half)                                # regime 2: independent
    from scarcity.engine import OnlineDiscoveryEngine
    eng = OnlineDiscoveryEngine(vectorized=False, small_dataset_mode=True)
    eng.initialize({"fields": [{"name": "x"}, {"name": "y"}]})

    def _conf():
        return max((e["metrics"]["confidence"] for e in eng.get_knowledge_graph()
                    if "x" in e["variables"] and "y" in e["variables"]
                    and e.get("state") != "dead"), default=0.0)

    for t in range(half):
        eng.process_row({"x": float(x1[t]), "y": float(y1[t])})
    conf_regime1 = _conf()
    for t in range(half):
        eng.process_row({"x": float(x2[t]), "y": float(y2[t])})
    conf_regime2 = _conf()
    decayed = conf_regime1 >= 0.55 and conf_regime2 < conf_regime1 - 0.10
    verdict = "correct" if decayed else "missed"             # stale edge survived
    return ScenarioResult(
        "nonstationary_decay", "drift",
        "x-y edge active in regime 1, decays after the regime break",
        f"conf regime1={conf_regime1:.2f} -> regime2={conf_regime2:.2f}"
        + ("" if decayed else " (stale edge did not decay)"),
        verdict, {"conf_regime1": conf_regime1, "conf_regime2": conf_regime2},
    )


def _s_collider_bias(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    n = 3000
    a = rng.normal(size=n)
    b = rng.normal(size=n)                            # A, B independent
    col = 1.2 * a + 1.2 * b + rng.normal(scale=0.5, size=n)   # A -> C <- B (collider)
    rows = _rows({"a": a, "b": b, "col": col})
    # (i) online engine on A,B alone should find no edge (marginally independent)
    _, _, strong = _discover(rows, ["a", "b"])
    online_ab = [e for e in strong if set(e["variables"]) == {"a", "b"}]
    # (ii) WRONGLY adjusting for the collider induces a spurious A->B effect
    res_adj = _causal_ate(rows, "a", "b", ["col"])
    res_raw = _causal_ate(rows, "a", "b", [])
    spurious = abs(res_adj["estimate"]) if res_adj else float("nan")
    unadjusted = abs(res_raw["estimate"]) if res_raw else float("nan")
    return ScenarioResult(
        "collider_bias", "collider",
        "no A-B edge marginally; conditioning on the collider induces a spurious one",
        f"online A-B edges={len(online_ab)}; |ATE| unadjusted={unadjusted:.3f} "
        f"vs adjusting-for-collider={spurious:.3f}",
        # correct marginal behavior; the induced bias is documented, not a pass/fail
        "documented",
        {"online_ab_edges": len(online_ab), "ate_unadjusted": unadjusted,
         "ate_conditioned_on_collider": spurious},
    )


def _s_weak_signal(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    n = 3000
    x = rng.normal(size=n)
    y = 0.12 * x + rng.normal(scale=1.0, size=n)     # weak: SNR ~ 0.12
    rows = _rows({"x": x, "y": y})
    _, _, strong = _discover(rows, ["x", "y"])
    xy = [e for e in strong if set(e["variables"]) == {"x", "y"}]
    detected = len(xy) > 0
    return ScenarioResult(
        "weak_signal", "sensitivity",
        "a weak true x-y relation: detect if evidence allows, do not false-fire otherwise",
        f"{'detected' if detected else 'abstained'} "
        f"(max conf {max((e['metrics']['confidence'] for e in xy), default=0.0):.2f})",
        "detected" if detected else "abstained",
        {"detected": detected},
    )


def _s_feedback_loop(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    n = 3000
    a = np.zeros(n)
    b = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.6 * b[t - 1] + rng.normal(scale=0.4)     # A_t <- B_{t-1}
        b[t] = 0.6 * a[t - 1] + rng.normal(scale=0.4)     # B_t <- A_{t-1}
    rows = _rows({"a": a, "b": b})
    _, kg, _ = _discover(rows, ["a", "b"])
    directional = [e for e in kg if e["type"] in ("causal", "temporal")
                   and e["metrics"]["confidence"] >= 0.55]
    dirs = {(e["variables"][0], e["variables"][-1]) for e in directional}
    both = ("a", "b") in dirs and ("b", "a") in dirs
    verdict = "correct" if both else ("detected" if dirs else "missed")
    return ScenarioResult(
        "feedback_loop", "feedback",
        "recover bidirectional coupling A<->B (each depends on the other's lag)",
        f"directional edges found: {sorted(dirs) or 'none'}",
        verdict, {"directions": sorted(dirs)},
    )


def _s_simpsons_paradox(seed: int) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    per, groups = 800, 3
    xs, ys, gs = [], [], []
    for g in range(groups):
        base = g * 5.0
        x = rng.normal(loc=base, scale=1.0, size=per)
        y = base * 2.0 - 0.8 * (x - base) + rng.normal(scale=0.3, size=per)  # within: negative
        xs.append(x); ys.append(y); gs.append(np.full(per, float(g)))
    x = np.concatenate(xs); y = np.concatenate(ys); grp = np.concatenate(gs)
    agg_corr = float(np.corrcoef(x, y)[0, 1])                       # aggregate: positive
    within = float(np.mean([np.corrcoef(xs[g], ys[g])[0, 1] for g in range(groups)]))
    rows = _rows({"x": x, "y": y, "grp": grp})
    _, _, strong = _discover(rows, ["x", "y", "grp"])
    has_struct = any(e["type"] == "structural" for e in strong)
    return ScenarioResult(
        "simpsons_paradox", "conditioning",
        "aggregate x-y sign reverses within groups; a group structure exists",
        f"aggregate corr={agg_corr:+.2f}, within-group corr={within:+.2f}; "
        f"structural edge found={has_struct}",
        "documented",
        {"aggregate_corr": agg_corr, "within_corr": within, "structural_found": has_struct},
    )


SCENARIOS: List[Scenario] = [
    Scenario("no_relationship", "false-positive", "no spurious edges", _s_no_relationship),
    Scenario("confounding_observed", "confounding", "adjusted effect ~ 0", _s_confounding_observed),
    Scenario("reverse_causality", "direction", "B -> A", _s_reverse_causality),
    Scenario("nonstationary_decay", "drift", "edge decays after break", _s_nonstationary_decay),
    Scenario("collider_bias", "collider", "no marginal edge; bias if conditioned", _s_collider_bias),
    Scenario("weak_signal", "sensitivity", "detect or abstain, never false-fire", _s_weak_signal),
    Scenario("feedback_loop", "feedback", "A <-> B", _s_feedback_loop),
    Scenario("simpsons_paradox", "conditioning", "sign reverses within groups", _s_simpsons_paradox),
]


def run_scenario(name: str, seed: int = 0) -> ScenarioResult:
    scen = next(s for s in SCENARIOS if s.name == name)
    return scen.run(seed)


def run_all(seed: int = 0) -> List[ScenarioResult]:
    return [s.run(seed) for s in SCENARIOS]

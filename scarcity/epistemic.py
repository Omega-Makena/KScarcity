"""The epistemic ladder — an explicit account of how much a relationship has earned.

Scarcity's core discipline is that a discovered edge is not a causal claim. An
edge climbs a ladder of increasingly demanding evidence, one rung at a time, and
each rung has a gate it must pass:

  1 DISCOVERED    a pattern was found in the stream (some confidence)
  2 HYPOTHESIZED  it was typed as one of the relationship kinds
  3 PREDICTIVE    it holds out-of-sample (beats a baseline, significantly)
  4 IDENTIFIED    a valid adjustment set exists (causally identifiable)
  5 ESTIMATED     the effect is estimated with a confidence interval clear of 0
  6 ROBUST        it survives refuters and an omitted-variable-bias sensitivity

You may only advance from the rung immediately below, and only if the evidence
passes the gate; evidence that fails is recorded but does not promote the edge.
Refutation is the one rung that can also *demote*: an estimated edge that fails
its refuters is marked not-robust rather than silently kept.

This module records epistemic state; it does not run the analyses. Feed it the
result of each stage (the discovery confidence, a holdout score, an identification
result, an estimate + CI, refuter outcomes + a robustness value) and it tracks
where every edge honestly stands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class Rung(IntEnum):
    UNTESTED = 0
    DISCOVERED = 1
    HYPOTHESIZED = 2
    PREDICTIVE = 3
    IDENTIFIED = 4
    ESTIMATED = 5
    ROBUST = 6


Edge = Tuple[str, str]


@dataclass
class EdgeEpistemics:
    source: str
    target: str
    rung: Rung = Rung.UNTESTED
    refuted: bool = False                       # failed a refutation gate
    evidence: Dict[str, dict] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    @property
    def edge(self) -> Edge:
        return (self.source, self.target)


class EpistemicLadder:
    """Registry of edges and the rung each has earned."""

    def __init__(self, predictive_alpha: float = 0.05, rv_threshold: float = 0.10):
        self._edges: Dict[Edge, EdgeEpistemics] = {}
        self.predictive_alpha = predictive_alpha
        self.rv_threshold = rv_threshold

    # -- internals -----------------------------------------------------------
    def _get(self, s: str, t: str) -> EdgeEpistemics:
        return self._edges.setdefault((s, t), EdgeEpistemics(s, t))

    def _advance(self, e: EdgeEpistemics, to: Rung, ev: dict, passed: bool, why: str) -> bool:
        rung_name = to.name
        if not passed:
            e.log.append(f"{rung_name}: gate failed ({why})")
            e.evidence[rung_name] = {**ev, "passed": False}
            return False
        if e.rung < to - 1:
            e.log.append(f"{rung_name}: skipped — edge only at {e.rung.name}")
            return False
        e.rung = max(e.rung, to)
        e.evidence[rung_name] = {**ev, "passed": True}
        e.log.append(f"{rung_name}: reached ({why})")
        return True

    # -- rung recorders ------------------------------------------------------
    def record_discovery(self, s: str, t: str, confidence: float) -> bool:
        e = self._get(s, t)
        return self._advance(e, Rung.DISCOVERED, {"confidence": confidence},
                             passed=confidence > 0.0, why=f"confidence={confidence:.3f}")

    def record_hypothesis(self, s: str, t: str, rel_type: str) -> bool:
        e = self._get(s, t)
        return self._advance(e, Rung.HYPOTHESIZED, {"rel_type": rel_type},
                             passed=bool(rel_type), why=f"type={rel_type}")

    def record_predictive_validation(self, s: str, t: str, skill: float,
                                     baseline: float, p_value: float) -> bool:
        e = self._get(s, t)
        ok = (skill > baseline) and (p_value < self.predictive_alpha)
        return self._advance(e, Rung.PREDICTIVE,
                             {"skill": skill, "baseline": baseline, "p_value": p_value},
                             passed=ok, why=f"skill {skill:.3f} vs {baseline:.3f}, p={p_value:.3g}")

    def record_identification(self, s: str, t: str, identifiable: bool,
                              adjustment_set: Optional[List[str]] = None) -> bool:
        e = self._get(s, t)
        return self._advance(e, Rung.IDENTIFIED,
                             {"identifiable": identifiable, "adjustment_set": adjustment_set or []},
                             passed=bool(identifiable), why=f"adjust={adjustment_set or []}")

    def record_estimation(self, s: str, t: str, effect: float,
                          ci_low: float, ci_high: float) -> bool:
        e = self._get(s, t)
        excludes_zero = not (ci_low <= 0.0 <= ci_high)
        return self._advance(e, Rung.ESTIMATED,
                             {"effect": effect, "ci": [ci_low, ci_high]},
                             passed=excludes_zero, why=f"effect={effect:.3g}, CI=[{ci_low:.3g},{ci_high:.3g}]")

    def record_refutation(self, s: str, t: str, refuters_passed: bool,
                          robustness_value: float) -> bool:
        """Advance to ROBUST iff refuters pass AND the OVB robustness value clears
        the threshold. Failure marks the edge refuted (it stays at ESTIMATED)."""
        e = self._get(s, t)
        ok = bool(refuters_passed) and (robustness_value >= self.rv_threshold)
        ev = {"refuters_passed": refuters_passed, "robustness_value": robustness_value}
        advanced = self._advance(e, Rung.ROBUST, ev, passed=ok,
                                 why=f"refuters={refuters_passed}, RV={robustness_value:.3f}")
        if not advanced and e.rung >= Rung.ESTIMATED:
            e.refuted = True
            e.log.append("ROBUST: marked refuted (does not survive scrutiny)")
        return advanced

    # -- queries -------------------------------------------------------------
    def status(self, s: str, t: str) -> EdgeEpistemics:
        return self._get(s, t)

    def at_least(self, rung: Rung) -> List[EdgeEpistemics]:
        return [e for e in self._edges.values() if e.rung >= rung and not e.refuted]

    def explain(self, s: str, t: str) -> str:
        e = self._get(s, t)
        head = f"{s} -> {t}: rung {int(e.rung)}/{int(Rung.ROBUST)} ({e.rung.name})"
        if e.refuted:
            head += " [REFUTED]"
        return "\n".join([head] + [f"  - {line}" for line in e.log])

    def summary(self) -> Dict[str, int]:
        counts = {r.name: 0 for r in Rung}
        for e in self._edges.values():
            counts[e.rung.name] += 1
        counts["REFUTED"] = sum(1 for e in self._edges.values() if e.refuted)
        return counts

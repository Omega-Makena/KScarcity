"""Tests for the epistemic ladder."""
from scarcity.epistemic import EpistemicLadder, Rung


def _climb_to_estimated(L, s="x", t="y"):
    assert L.record_discovery(s, t, confidence=0.9)
    assert L.record_hypothesis(s, t, rel_type="causal")
    assert L.record_predictive_validation(s, t, skill=0.8, baseline=0.5, p_value=0.001)
    assert L.record_identification(s, t, identifiable=True, adjustment_set=["z"])
    assert L.record_estimation(s, t, effect=0.4, ci_low=0.2, ci_high=0.6)


def test_full_climb_to_robust():
    L = EpistemicLadder()
    _climb_to_estimated(L)
    assert L.record_refutation("x", "y", refuters_passed=True, robustness_value=0.25)
    assert L.status("x", "y").rung == Rung.ROBUST
    assert not L.status("x", "y").refuted
    assert L.at_least(Rung.ROBUST) and L.at_least(Rung.ROBUST)[0].edge == ("x", "y")


def test_cannot_skip_rungs():
    L = EpistemicLadder()
    L.record_discovery("a", "b", confidence=0.9)
    # jump straight to estimation without predictive/identification
    assert not L.record_estimation("a", "b", effect=1.0, ci_low=0.5, ci_high=1.5)
    assert L.status("a", "b").rung == Rung.DISCOVERED


def test_failed_gate_does_not_promote():
    L = EpistemicLadder()
    L.record_discovery("a", "b", confidence=0.9)
    L.record_hypothesis("a", "b", rel_type="correlational")
    # predictive gate fails: skill does not beat baseline
    assert not L.record_predictive_validation("a", "b", skill=0.4, baseline=0.5, p_value=0.2)
    assert L.status("a", "b").rung == Rung.HYPOTHESIZED
    assert L.status("a", "b").evidence["PREDICTIVE"]["passed"] is False


def test_estimation_gate_requires_ci_excluding_zero():
    L = EpistemicLadder()
    _climb_to_estimated(L, "p", "q")           # this one has a CI clear of 0
    L2 = EpistemicLadder()
    L2.record_discovery("p", "q", 0.9)
    L2.record_hypothesis("p", "q", "causal")
    L2.record_predictive_validation("p", "q", 0.8, 0.5, 0.01)
    L2.record_identification("p", "q", True, ["z"])
    # CI straddles 0 -> not estimated
    assert not L2.record_estimation("p", "q", effect=0.1, ci_low=-0.2, ci_high=0.4)
    assert L2.status("p", "q").rung == Rung.IDENTIFIED


def test_refutation_failure_marks_refuted_and_holds_rung():
    L = EpistemicLadder(rv_threshold=0.10)
    _climb_to_estimated(L)
    # refuters pass but the robustness value is below threshold -> not robust
    assert not L.record_refutation("x", "y", refuters_passed=True, robustness_value=0.02)
    st = L.status("x", "y")
    assert st.rung == Rung.ESTIMATED
    assert st.refuted is True
    assert L.at_least(Rung.ROBUST) == []       # refuted edges excluded


def test_explain_and_summary():
    L = EpistemicLadder()
    _climb_to_estimated(L)
    L.record_refutation("x", "y", refuters_passed=True, robustness_value=0.3)
    text = L.explain("x", "y")
    assert "rung 6/6" in text and "ROBUST" in text
    summ = L.summary()
    assert summ["ROBUST"] == 1 and summ["REFUTED"] == 0

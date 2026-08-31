"""Failure-mode taxonomy: registry + machinery smoke test.

The full adversarial suite is a benchmark (several thousand-row engine and causal
runs); these tests validate the framework and one fast scenario end to end. Run
the whole suite via benchmark/scripts/benchmark_failure_modes.py.
"""
import pytest

from scarcity.synthetic.failure_modes import SCENARIOS, run_scenario
from scarcity.synthetic.failure_modes import ScenarioResult


def test_registry_is_well_formed():
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))                     # unique
    assert {"no_relationship", "reverse_causality", "collider_bias",
            "nonstationary_decay", "confounding_observed"} <= set(names)
    for s in SCENARIOS:
        assert s.category and s.expected and callable(s.run)


def test_confounding_scenario_recovers_null_adjusted_effect():
    # Fast (single 3000-row causal fit): the spurious A-B association driven by an
    # observed confounder C must adjust away to ~0.
    dowhy = pytest.importorskip("dowhy")  # noqa: F841
    r = run_scenario("confounding_observed", seed=0)
    assert isinstance(r, ScenarioResult)
    assert r.name == "confounding_observed"
    if r.verdict != "documented":                            # causal arm available
        assert r.verdict == "correct"
        assert abs(r.detail["adjusted_ate"]) < 0.15

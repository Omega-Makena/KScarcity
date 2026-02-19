"""
Tests for the Causal Pipeline.
"""
import pytest
pytest.importorskip("dowhy")
import pandas as pd
from dowhy import datasets

from scarcity.causal.specs import EstimandSpec, EstimandType, RuntimeSpec
from scarcity.causal.engine import run_causal


def test_run_causal_end_to_end_ate(tmp_path):
    """
    Verifies that the pipeline can recover a known effect in a simple linear system.
    True Effect = 10.
    """
    data = datasets.linear_dataset(
        beta=10,
        num_common_causes=2,
        num_instruments=0,
        num_effect_modifiers=0,
        num_samples=500,
        treatment_is_binary=True,
        stddev_treatment_noise=1,
        stddev_outcome_noise=1,
    )
    df = data["df"]

    spec = EstimandSpec(
        treatment="v0",
        outcome="y",
        confounders=["W0", "W1"],
        type=EstimandType.ATE,
    )

    runtime = RuntimeSpec(
        refutation_simulations=10,
        seed=123,
        parallelism="none",
        artifact_root=str(tmp_path),
    )

    result = run_causal(df, spec, runtime)

    assert result is not None
    assert result.results
    effect = result.results[0]
    assert abs(effect.estimate - 10.0) < 1.0, f"Estimate {effect.estimate} too far from true 10.0"

    assert "random_common_cause" in effect.refuter_results
    assert "placebo_treatment" in effect.refuter_results


def test_config_validation():
    """Ensure invalid specs raise errors."""
    with pytest.raises(ValueError):
        EstimandSpec(treatment="", outcome="y").validate()

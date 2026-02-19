"""Production-grade causal pipeline tests."""
import json
import pytest
pytest.importorskip("dowhy")
from dowhy import datasets

from scarcity.causal.engine import run_causal
from scarcity.causal.specs import EstimandSpec, EstimandType, RuntimeSpec


def _linear_df():
    data = datasets.linear_dataset(
        beta=3,
        num_common_causes=2,
        num_instruments=0,
        num_effect_modifiers=0,
        num_samples=200,
        treatment_is_binary=True,
        stddev_treatment_noise=1,
        stddev_outcome_noise=1,
    )
    return data["df"]


def test_multi_spec_parallel_process(tmp_path):
    df = _linear_df()
    specs = [
        EstimandSpec(treatment="v0", outcome="y", confounders=["W0", "W1"], type=EstimandType.ATE),
        EstimandSpec(treatment="v0", outcome="y", confounders=["W0", "W1"], type=EstimandType.ATT),
    ]
    runtime = RuntimeSpec(
        refutation_simulations=5,
        seed=123,
        parallelism="process",
        n_jobs=2,
        chunk_size=1,
        artifact_root=str(tmp_path),
    )

    result = run_causal(df, specs, runtime)

    assert len(result.results) == 2
    assert len(result.errors) == 0

    summary_path = tmp_path / "runs" / result.summary.run_id / "summary.json"
    assert summary_path.exists()


def test_fail_policy_continue(tmp_path):
    df = _linear_df()
    specs = [
        EstimandSpec(treatment="v0", outcome="y", confounders=["W0", "W1"], type=EstimandType.ATE),
        EstimandSpec(treatment="missing", outcome="y", confounders=["W0"], type=EstimandType.ATE),
    ]
    runtime = RuntimeSpec(
        refutation_simulations=1,
        seed=1,
        parallelism="none",
        fail_policy="continue",
        artifact_root=str(tmp_path),
    )

    result = run_causal(df, specs, runtime)

    assert len(result.results) == 1
    assert len(result.errors) == 1


def test_fail_policy_fail_fast(tmp_path):
    df = _linear_df()
    specs = [
        EstimandSpec(treatment="missing", outcome="y", confounders=["W0"], type=EstimandType.ATE),
        EstimandSpec(treatment="v0", outcome="y", confounders=["W0", "W1"], type=EstimandType.ATE),
    ]
    runtime = RuntimeSpec(
        refutation_simulations=1,
        seed=1,
        parallelism="none",
        fail_policy="fail_fast",
        artifact_root=str(tmp_path),
    )

    result = run_causal(df, specs, runtime)

    assert len(result.results) == 0
    assert len(result.errors) == 1


def test_time_series_validation_strict(tmp_path):
    df = _linear_df()
    spec = EstimandSpec(
        treatment="v0",
        outcome="y",
        confounders=["W0", "W1"],
        type=EstimandType.ATE,
        time_column="time",
        lag=1,
    )
    runtime = RuntimeSpec(
        refutation_simulations=1,
        seed=5,
        parallelism="none",
        time_series_policy="strict",
        artifact_root=str(tmp_path),
    )

    result = run_causal(df, spec, runtime)

    assert len(result.results) == 0
    assert len(result.errors) == 1


def test_time_series_validation_warn(tmp_path):
    df = _linear_df()
    spec = EstimandSpec(
        treatment="v0",
        outcome="y",
        confounders=["W0", "W1"],
        type=EstimandType.ATE,
        time_column="time",
        lag=1,
    )
    runtime = RuntimeSpec(
        refutation_simulations=1,
        seed=5,
        parallelism="none",
        time_series_policy="warn",
        artifact_root=str(tmp_path),
    )

    result = run_causal(df, spec, runtime)

    assert len(result.results) == 1
    assert result.results[0].temporal_diagnostics["warnings"]


def test_deterministic_parallel(tmp_path):
    df = _linear_df()
    specs = [
        EstimandSpec(treatment="v0", outcome="y", confounders=["W0", "W1"], type=EstimandType.ATE),
        EstimandSpec(treatment="v0", outcome="y", confounders=["W0", "W1"], type=EstimandType.ATT),
    ]
    runtime = RuntimeSpec(
        refutation_simulations=2,
        seed=42,
        parallelism="process",
        n_jobs=2,
        chunk_size=1,
        artifact_root=str(tmp_path),
    )

    run_a = run_causal(df, specs, runtime)
    runtime_b = RuntimeSpec(
        refutation_simulations=2,
        seed=42,
        parallelism="process",
        n_jobs=2,
        chunk_size=1,
        artifact_root=str(tmp_path),
    )
    run_b = run_causal(df, specs, runtime_b)

    values_a = [artifact.estimate for artifact in run_a.results]
    values_b = [artifact.estimate for artifact in run_b.results]

    assert len(values_a) == len(values_b)
    for a, b in zip(values_a, values_b):
        assert abs(a - b) < 1e-6


def test_artifact_integrity(tmp_path):
    df = _linear_df()
    spec = EstimandSpec(treatment="v0", outcome="y", confounders=["W0", "W1"], type=EstimandType.ATE)
    runtime = RuntimeSpec(
        refutation_simulations=1,
        seed=7,
        parallelism="none",
        artifact_root=str(tmp_path),
    )

    result = run_causal(df, spec, runtime)

    run_dir = tmp_path / "runs" / result.summary.run_id
    summary_path = run_dir / "summary.json"
    effects_path = run_dir / "effects.jsonl"
    errors_path = run_dir / "errors.jsonl"
    input_dot = run_dir / "graphs" / "input.dot"
    learned_dot = run_dir / "graphs" / "learned.dot"

    assert summary_path.exists()
    assert effects_path.exists()
    assert errors_path.exists()
    assert input_dot.exists()
    assert learned_dot.exists()

    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
        assert "summary" in payload
        assert "metadata" in payload

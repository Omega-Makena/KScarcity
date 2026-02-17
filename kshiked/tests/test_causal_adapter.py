"""Tests for K-Shield causal adapter integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from scarcity.causal.specs import EstimandType, FailPolicy, ParallelismMode, TimeSeriesPolicy

from kshiked.causal_adapter.config import (
    AdapterConfig,
    AdapterPolicyConfig,
    AdapterRuntimeConfig,
    AdapterSelectionConfig,
)
from kshiked.causal_adapter.dataset import load_unified_dataset
from kshiked.causal_adapter.policy import select_estimands
from kshiked.causal_adapter.spec_builder import build_task_specs
from kshiked.causal_adapter.types import CausalTaskSpec, TaskWindow


def _write_dot(path: Path, edges) -> None:
    lines = ["digraph G {", "  rankdir=LR;"]
    for src, tgt in edges:
        lines.append(f'  "{src}" -> "{tgt}";')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_task_specs_generates_pairs():
    selection = AdapterSelectionConfig(
        treatments=["gdp_current", "inflation"],
        outcomes=["population"],
        confounders=["gdp_growth"],
        effect_modifiers=["population"],
        time_column="year",
        lag=1,
    )
    tasks = build_task_specs(selection)
    assert len(tasks) == 2
    treatments = {task.treatment for task in tasks}
    assert treatments == {"gdp_current", "inflation"}
    assert all(task.outcome == "population" for task in tasks)


def test_select_estimands_policy_with_dot():
    task = CausalTaskSpec(
        treatment="gdp_current",
        outcome="inflation",
        effect_modifiers=["population"],
        instrument="tax_revenue",
        mediator="unemployment",
        mediator_lag=1,
    )
    dot_text = """
    digraph G {
      "tax_revenue" -> "gdp_current";
      "gdp_current" -> "inflation";
      "gdp_current" -> "unemployment";
      "unemployment" -> "inflation";
    }
    """
    available = ["gdp_current", "inflation", "population", "tax_revenue", "unemployment"]
    decision = select_estimands(task, AdapterPolicyConfig(), dot_text, available)
    assert EstimandType.ATE in decision.estimands
    assert EstimandType.CATE in decision.estimands
    assert EstimandType.LATE in decision.estimands
    assert EstimandType.MEDIATION_NDE in decision.estimands
    assert EstimandType.MEDIATION_NIE in decision.estimands


def test_runner_executes_and_returns_edges(tmp_path: Path):
    pytest.importorskip("dowhy")
    from kshiked.causal_adapter.runner import KShieldCausalRunner

    dot_path = tmp_path / "economic.dot"
    _write_dot(
        dot_path,
        edges=[
            ("population", "gdp_current"),
            ("population", "inflation"),
            ("gdp_current", "inflation"),
        ],
    )

    df = load_unified_dataset(
        indicators=["gdp_current", "inflation", "population"],
        start_year=2000,
        end_year=2010,
        time_column="year",
    )
    df = df.dropna(subset=["gdp_current", "inflation", "population", "year"])
    if len(df) <= 3:
        pytest.skip("Insufficient data after filtering")

    runtime = AdapterRuntimeConfig(
        parallelism=ParallelismMode.NONE,
        n_jobs=1,
        chunk_size=1,
        fail_policy=FailPolicy.CONTINUE,
        time_series_policy=TimeSeriesPolicy.STRICT,
        refute_random_common_cause=False,
        refute_placebo_treatment=False,
        refute_data_subset=False,
        refutation_simulations=1,
        artifact_root=str(tmp_path / "scarcity_artifacts"),
        dot_path=str(dot_path),
    )
    selection = AdapterSelectionConfig(
        treatments=["gdp_current"],
        outcomes=["inflation"],
        confounders=["population"],
        time_column="year",
        lag=1,
        windows=[TaskWindow(start_year=2000, end_year=2010)],
        dot_path=str(dot_path),
    )
    config = AdapterConfig(
        runtime=runtime,
        selection=selection,
        kshield_artifact_root=str(tmp_path / "kshield_artifacts"),
    )

    runner = KShieldCausalRunner(config)
    result = runner.run_from_dataset(df)

    assert result.run_id
    assert result.edges
    edge = result.edges[0]
    assert edge.source == "gdp_current"
    assert edge.target == "inflation"
    assert 0.0 <= edge.confidence <= 1.0

    run_dir = Path(config.kshield_artifact_root) / "runs" / result.run_id
    assert (run_dir / "edges.jsonl").exists()
    assert (run_dir / "effects.jsonl").exists()
    assert (run_dir / "summary.json").exists()

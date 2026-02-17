"""K-Shield adapter runner for Scarcity causal pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from scarcity.causal.engine import run_causal
from scarcity.causal.reporting import CausalRunResult

from .artifacts import KShieldArtifactStore
from .config import AdapterConfig
from .dataset import DatasetSegment, load_unified_dataset, segment_dataset
from .integration import artifact_to_edge, edge_to_simulation_update
from .spec_builder import build_estimand_specs, build_task_specs
from .types import AdapterRunResult, KnowledgeGraphEdge, SimulationParameterUpdate

logger = logging.getLogger("kshield.causal.runner")


class KShieldCausalRunner:
    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    def run_from_dataset(self, df: pd.DataFrame) -> AdapterRunResult:
        tasks = build_task_specs(self.config.selection)
        return self.run_with_tasks(df, tasks)

    def run_with_tasks(self, df: pd.DataFrame, tasks: Sequence) -> AdapterRunResult:
        runtime_spec = self.config.runtime.to_runtime_spec()
        segments = segment_dataset(
            df,
            time_column=self.config.selection.time_column,
            windows=self.config.selection.windows,
            counties=self.config.selection.counties,
            sectors=self.config.selection.sectors,
        )

        all_edges: List[KnowledgeGraphEdge] = []
        all_updates: List[SimulationParameterUpdate] = []
        all_results: List[CausalRunResult] = []
        all_errors: List = []

        run_id = None

        for segment in segments:
            specs = []
            for task in tasks:
                specs.extend(
                    build_estimand_specs(
                        task,
                        available_columns=segment.data.columns,
                        policy=self.config.policy,
                        context=segment.context,
                    )
                )

            if not specs:
                logger.warning("No estimands selected for segment; skipping")
                continue

            result = run_causal(segment.data, specs, runtime_spec)
            run_id = run_id or result.summary.run_id if result.summary else None
            all_results.append(result)
            all_errors.extend(result.errors)

            segment_edges = [artifact_to_edge(a, self.config.edge) for a in result.results]
            all_edges.extend(segment_edges)

            for edge in segment_edges:
                update = edge_to_simulation_update(
                    edge,
                    mapping=self.config.simulation.mapping,
                    scale=self.config.simulation.scale,
                )
                if update:
                    all_updates.append(update)

        run_id = run_id or "kshield-causal"
        store = KShieldArtifactStore(Path(self.config.kshield_artifact_root), run_id)
        store.write_all(
            effects=[effect for r in all_results for effect in r.results],
            errors=all_errors,
            edges=all_edges,
            updates=all_updates,
            summary={
                "run_id": run_id,
                "segments": len(segments),
                "results": sum(len(r.results) for r in all_results),
                "errors": len(all_errors),
            },
        )

        return AdapterRunResult(
            run_id=run_id,
            results=[r.to_dict() for r in all_results],
            edges=all_edges,
            simulation_updates=all_updates,
            errors=all_errors,
            metadata={"segments": len(segments)},
        )


def run_kshield_causal(
    config: AdapterConfig,
    df: Optional[pd.DataFrame] = None,
) -> AdapterRunResult:
    """Convenience entrypoint for K-Shield causal adapter."""
    if df is None:
        df = load_unified_dataset(
            indicators=config.selection.treatments + config.selection.outcomes,
            start_year=min([w.start_year for w in config.selection.windows], default=1990),
            end_year=max([w.end_year for w in config.selection.windows], default=None),
            time_column=config.selection.time_column or "year",
        )
    return KShieldCausalRunner(config).run_from_dataset(df)

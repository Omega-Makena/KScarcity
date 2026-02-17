"""Build Scarcity EstimandSpecs from K-Shield task specs."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from scarcity.causal.graph import load_dot
from scarcity.causal.specs import EstimandSpec

from .config import AdapterPolicyConfig, AdapterSelectionConfig
from .policy import select_estimands
from .types import BatchContext, CausalTaskSpec, TaskWindow

logger = logging.getLogger("kshield.causal.spec_builder")


def build_task_specs(selection: AdapterSelectionConfig) -> List[CausalTaskSpec]:
    """Generate task specs for all treatment/outcome pairs."""
    tasks: List[CausalTaskSpec] = []
    for treatment in selection.treatments:
        for outcome in selection.outcomes:
            if treatment == outcome:
                continue
            tasks.append(
                CausalTaskSpec(
                    treatment=treatment,
                    outcome=outcome,
                    confounders=list(selection.confounders),
                    effect_modifiers=list(selection.effect_modifiers),
                    instrument=selection.instrument,
                    mediator=selection.mediator,
                    time_column=selection.time_column,
                    lag=selection.lag,
                    mediator_lag=selection.mediator_lag,
                    panel_keys=list(selection.panel_keys),
                    dot_path=selection.dot_path,
                    metadata=dict(selection.metadata),
                )
            )
    return tasks


def build_estimand_specs(
    task: CausalTaskSpec,
    available_columns: Sequence[str],
    policy: AdapterPolicyConfig,
    context: Optional[BatchContext] = None,
) -> List[EstimandSpec]:
    dot_text = load_dot(task.dot_path) if task.dot_path else None
    decision = select_estimands(task, policy, dot_text, available_columns)
    specs: List[EstimandSpec] = []

    for estimand_type in decision.estimands:
        specs.append(
            EstimandSpec(
                treatment=task.treatment,
                outcome=task.outcome,
                confounders=list(task.confounders),
                effect_modifiers=list(task.effect_modifiers),
                instrument=task.instrument,
                mediator=task.mediator,
                type=estimand_type,
                time_column=task.time_column,
                lag=task.lag,
                mediator_lag=task.mediator_lag,
                panel_keys=list(task.panel_keys),
                dot_path=task.dot_path,
                metadata=_merge_metadata(task, context, decision.reasons.get(estimand_type)),
            )
        )

    return specs


def _merge_metadata(task: CausalTaskSpec, context: Optional[BatchContext], reason: Optional[str]) -> dict:
    metadata = dict(task.metadata)
    if context:
        metadata.update({
            "window": context.window,
            "county": context.county,
            "sector": context.sector,
        })
    if reason:
        metadata["estimand_reason"] = reason
    return metadata

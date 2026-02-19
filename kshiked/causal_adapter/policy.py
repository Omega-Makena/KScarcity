"""Estimand selection policy for K-Shield causal adapter."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from scarcity.causal.graph import parse_dot_edges
from scarcity.causal.specs import EstimandType

from .config import AdapterPolicyConfig
from .types import CausalTaskSpec, EstimandDecision

logger = logging.getLogger("kshield.causal.policy")


def select_estimands(
    task: CausalTaskSpec,
    policy: AdapterPolicyConfig,
    dot_text: Optional[str],
    available_columns: Sequence[str],
) -> EstimandDecision:
    """Select estimands based on policy and data conditions."""
    estimands: List[EstimandType] = []
    reasons: dict = {}

    estimands.append(EstimandType.ATE)
    reasons[EstimandType.ATE] = "default headline effect"

    if task.effect_modifiers and all(col in available_columns for col in task.effect_modifiers):
        estimands.append(EstimandType.CATE)
        reasons[EstimandType.CATE] = "heterogeneity modifiers present"

    if policy.allow_late and task.instrument and task.instrument in available_columns:
        if _supports_iv(dot_text, task) or not policy.require_dot_for_iv:
            estimands.append(EstimandType.LATE)
            reasons[EstimandType.LATE] = "instrument present and DAG supports"
        else:
            logger.info("Skipping LATE: DOT missing or DAG does not support IV")

    if policy.allow_mediation and task.mediator and task.mediator_lag is not None:
        if _supports_mediation(dot_text, task) or not policy.require_dot_for_mediation:
            estimands.append(EstimandType.MEDIATION_NDE)
            estimands.append(EstimandType.MEDIATION_NIE)
            reasons[EstimandType.MEDIATION_NDE] = "mediator timing specified"
            reasons[EstimandType.MEDIATION_NIE] = "mediator timing specified"
        else:
            logger.info("Skipping mediation: DOT missing or DAG does not support mediation")

    return EstimandDecision(estimands=estimands, reasons=reasons)


def _supports_iv(dot_text: Optional[str], task: CausalTaskSpec) -> bool:
    if not dot_text or not task.instrument:
        return False
    edges = parse_dot_edges(dot_text)
    return (task.instrument, task.treatment) in edges


def _supports_mediation(dot_text: Optional[str], task: CausalTaskSpec) -> bool:
    if not dot_text or not task.mediator:
        return False
    edges = parse_dot_edges(dot_text)
    required = {(task.treatment, task.mediator), (task.mediator, task.outcome)}
    return required.issubset(set(edges))

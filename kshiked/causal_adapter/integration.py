"""Convert Scarcity effect artifacts into K-Shield knowledge graph edges and simulation updates."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from scarcity.causal.reporting import EffectArtifact
from scarcity.simulation.agents import EdgeLink

from .config import AdapterEdgeConfig
from .types import KnowledgeGraphEdge, SimulationParameterUpdate, TaskWindow

logger = logging.getLogger("kshield.causal.integration")


def artifact_to_edge(
    artifact: EffectArtifact,
    edge_config: AdapterEdgeConfig,
    default_lag: Optional[int] = None,
) -> KnowledgeGraphEdge:
    mean_effect = _summarize_effect(artifact.estimate)
    sign = "positive" if mean_effect >= 0 else "negative"
    weight = min(edge_config.max_weight, max(edge_config.min_weight, abs(mean_effect) * edge_config.weight_scale))
    confidence = max(edge_config.min_confidence, _estimate_confidence(artifact.refuter_results))

    lag = artifact.spec.lag if artifact.spec.lag is not None else default_lag
    window, county, sector = _context_from_metadata(artifact.spec.metadata)

    edge_id = _build_edge_id(artifact, window, county, sector)
    metadata = {
        "estimand_type": artifact.spec.type.value,
        "backend": artifact.backend.get("name"),
        "method_name": artifact.backend.get("method_name"),
        "diagnostics": artifact.diagnostics,
        "temporal": artifact.temporal_diagnostics,
        "provenance": artifact.provenance,
        "window": window,
        "county": county,
        "sector": sector,
    }

    return KnowledgeGraphEdge(
        edge_id=edge_id,
        source=artifact.spec.treatment,
        target=artifact.spec.outcome,
        sign=sign,
        weight=weight,
        confidence=confidence,
        lag=lag,
        window=window,
        metadata=metadata,
    )


def edge_to_simulation_update(
    edge: KnowledgeGraphEdge,
    mapping: Dict[str, str],
    scale: float = 1.0,
) -> Optional[SimulationParameterUpdate]:
    key = f"{edge.source}->{edge.target}"
    if key not in mapping:
        return None
    parameter = mapping[key]
    delta = edge.weight * scale * (1.0 if edge.sign == "positive" else -1.0)
    return SimulationParameterUpdate(
        parameter=parameter,
        delta=delta,
        reason="causal_edge_update",
        edge_id=edge.edge_id,
        metadata={"confidence": edge.confidence},
    )


def edge_to_edgelink(edge: KnowledgeGraphEdge) -> EdgeLink:
    return EdgeLink(
        edge_id=edge.edge_id,
        source=edge.source,
        target=edge.target,
        weight=edge.weight if edge.sign == "positive" else -edge.weight,
        stability=edge.confidence,
        confidence_interval=max(0.0, 1.0 - edge.confidence),
        regime=-1,
    )


def _summarize_effect(value) -> float:
    if isinstance(value, (list, tuple)):
        flattened = _flatten(value)
        if not flattened:
            return 0.0
        return float(np.mean(flattened))
    try:
        return float(value)
    except Exception:
        return 0.0


def _flatten(values) -> List[float]:
    result: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten(item))
        else:
            try:
                result.append(float(item))
            except Exception:
                continue
    return result


def _estimate_confidence(refuters: Dict[str, Dict]) -> float:
    scores: List[float] = []
    for payload in refuters.values():
        if payload.get("status") != "ok":
            continue
        p_value = payload.get("p_value")
        if isinstance(p_value, (int, float)):
            scores.append(max(0.0, min(1.0, 1.0 - float(p_value))))
    if not scores:
        return 0.5
    return float(np.mean(scores))


def _context_from_metadata(metadata: Dict) -> Tuple[Optional[TaskWindow], Optional[str], Optional[str]]:
    if not metadata:
        return None, None, None
    window = metadata.get("window")
    county = metadata.get("county")
    sector = metadata.get("sector")
    if not isinstance(window, TaskWindow):
        window = None
    if county is not None:
        county = str(county)
    if sector is not None:
        sector = str(sector)
    return window, county, sector


def _build_edge_id(
    artifact: EffectArtifact,
    window: Optional[TaskWindow],
    county: Optional[str],
    sector: Optional[str],
) -> str:
    parts = [
        artifact.spec.treatment,
        artifact.spec.outcome,
        artifact.spec.type.value,
    ]
    if window:
        parts.append(f"window={window.start_year}-{window.end_year}")
    if county:
        parts.append(f"county={county}")
    if sector:
        parts.append(f"sector={sector}")
    return "|".join(parts)

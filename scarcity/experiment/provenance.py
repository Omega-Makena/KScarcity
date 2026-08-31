"""
Knowledge-graph provenance — make every edge auditable.

When Scarcity asserts ``A -> B [causal, confidence=0.94]``, provenance answers
*why*: the discovery metrics, the calibration evidence (permutation p-value,
BH-adjusted q-value), when it was first detected, and the causal analysis if one
was run. ``build_edge_provenance`` turns the live knowledge graph into a list of
``EdgeProvenance`` records; ``explain_edge`` answers the question for one edge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scarcity.experiment.record import DatasetSpec, EdgeProvenance


def _key(rel_type: str, variables) -> Tuple[str, Tuple[Any, ...]]:
    return (rel_type, tuple(variables))


def build_edge_provenance(
    engine,
    *,
    dataset: Optional[DatasetSpec] = None,
    calibration: Optional[Dict[Tuple[str, Tuple], Dict[str, Any]]] = None,
    causal: Optional[Dict[Tuple[str, Tuple], Dict[str, Any]]] = None,
    top_k: int = 50,
) -> List[EdgeProvenance]:
    """Build per-edge provenance from an engine's knowledge graph.

    ``calibration`` and ``causal`` are optional lookups keyed by
    ``(rel_type, tuple(variables))`` — pass the calibration gate's p/q values and
    any causal-arm results to fold them into the provenance of the matching edge.
    """
    try:
        kg = engine.get_knowledge_graph()
    except Exception:
        return []
    calibration = calibration or {}
    causal = causal or {}
    out: List[EdgeProvenance] = []
    for e in kg[:top_k] if top_k else kg:
        variables = list(e.get("variables", []))
        rel = e.get("type", "?")
        k = _key(rel, variables)
        out.append(EdgeProvenance(
            source=variables[0] if variables else "?",
            target=variables[-1] if variables else "?",
            rel_type=rel,
            state=e.get("state", "tentative"),
            variables=variables,
            metrics=dict(e.get("metrics", {})),
            calibration=dict(calibration.get(k, {})),
            first_detected=e.get("created_at"),
            generation=int(e.get("generation", 0) or 0),
            causal=causal.get(k),
        ))
    return out


def explain_edge(
    engine,
    source: str,
    target: str,
    *,
    calibration: Optional[Dict[Tuple[str, Tuple], Dict[str, Any]]] = None,
    causal: Optional[Dict[Tuple[str, Tuple], Dict[str, Any]]] = None,
) -> Optional[EdgeProvenance]:
    """Return the provenance for the strongest edge between two variables.

    Answers "why does Scarcity believe source -> target?" — or ``None`` if no
    such edge is in the current knowledge graph.
    """
    provs = build_edge_provenance(engine, calibration=calibration, causal=causal, top_k=0)
    matches = [p for p in provs if source in p.variables and target in p.variables]
    if not matches:
        return None
    return max(matches, key=lambda p: p.metrics.get("confidence", 0.0))

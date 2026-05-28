"""
Graph extractor — pulls the engine's learned relationship graph from the
hypothesis pool as {target: [parents]} and a flat edge list for inspection.

Supports both strict mode (ACTIVE only) and lenient mode (ACTIVE + TENTATIVE
with conf > threshold), which matters for small annual datasets where the
min_evidence=20 barrier may not be reached in all folds.

All 15 relationship types are extracted:
  - 2-var directional (CAUSAL, FUNCTIONAL, TEMPORAL, EQUILIBRIUM, STRUCTURAL,
    PROBABILISTIC, GRAPH): variables[0] → variables[1]
  - 2-var symmetric (CORRELATIONAL, COMPETITIVE, SIMILARITY): both directions
  - 3-var multi-target (MEDIATING [X,M,Y], MODERATING [X,Z,Y], LOGICAL [A,B,C],
    SYNERGISTIC [X,Z,Y], COMPOSITIONAL [A,B,C]): variables[:-1] → variables[-1]
    The 3-var check runs FIRST so these types are never mishandled as 2-var pairs.
"""

from typing import Dict, List, Tuple, Any
from .discovery import HypothesisState, RelationshipType

# 2-variable directional types: variables[0] → variables[1]
_PAIRWISE_DIRECTIONAL = {
    RelationshipType.CAUSAL,
    RelationshipType.FUNCTIONAL,
    RelationshipType.TEMPORAL,
    RelationshipType.EQUILIBRIUM,
    RelationshipType.STRUCTURAL,
    RelationshipType.PROBABILISTIC,
    RelationshipType.GRAPH,
}

# 2-variable symmetric types: variables[0] ↔ variables[1] (both directions)
_PAIRWISE_SYMMETRIC = {
    RelationshipType.CORRELATIONAL,
    RelationshipType.COMPETITIVE,
    RelationshipType.SIMILARITY,
}

# 3-variable types: variables[-1] is the target, variables[:-1] are parents.
# These MUST be checked before the 2-var branches — previously they were caught
# by _DIRECTIONAL/_SYMMETRIC and only variables[0]↔variables[1] was extracted,
# meaning the actual target variable (variables[-1]) never got parents populated.
_MULTI_VAR = {
    RelationshipType.MEDIATING,     # [X, M, Y]  X→M→Y  →  add X→Y, M→Y
    RelationshipType.MODERATING,    # [X, Z, Y]  X×Z→Y  →  add X→Y, Z→Y
    RelationshipType.LOGICAL,       # [A, B, C]  A∧B→C  →  add A→C, B→C
    RelationshipType.SYNERGISTIC,   # [X, Z, Y]  X×Z→Y  →  add X→Y, Z→Y
    RelationshipType.COMPOSITIONAL, # [A, B, C]  A+B≈C  →  add A→C, B→C
}


def extract_graph(
    engine,
    conf_threshold: float = 0.50,
    min_evidence: int = 5,
    include_decaying: bool = True,
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """
    Extract a directed graph from the engine's hypothesis pool.

    When called on an OnlineDiscoveryEngine with vectorized=True (the default),
    automatically delegates to gpu_extract_graph() which reads from the batch-tensor
    backend.  All 15 relationship types contribute edges; multi-variable types treat
    variables[-1] as the target and variables[:-1] as parents.

    Args:
        engine: OnlineDiscoveryEngine instance (already streamed with data).
        conf_threshold: Minimum confidence to include a hypothesis edge.
        min_evidence: Minimum evidence count to include a hypothesis.
        include_decaying: Whether to include DECAYING hypotheses.

    Returns:
        graph: Dict mapping target_variable to [parent_variables].
        edges: Flat list of edge dicts with full metadata for inspection.
    """
    # Vectorized backend — delegate to tensor-based extractor
    if getattr(engine, '_vec_engine', None) is not None:
        from .gpu_engine import gpu_extract_graph
        return gpu_extract_graph(
            engine._vec_engine,
            conf_threshold=conf_threshold,
            min_evidence=min_evidence,
        )
    graph: Dict[str, List[str]] = {}
    edges: List[Dict[str, Any]] = []

    accept_states = {HypothesisState.ACTIVE}
    if include_decaying:
        accept_states.add(HypothesisState.DECAYING)
    # Also accept TENTATIVE if conf > threshold (lenient mode for sparse data)
    accept_states.add(HypothesisState.TENTATIVE)

    for h in engine.hypotheses.population.values():
        state = h.meta.state

        # Always skip dead hypotheses
        if state == HypothesisState.DEAD:
            continue

        # Lenient: TENTATIVE only accepted if it has real confidence
        if state == HypothesisState.TENTATIVE and h.confidence < conf_threshold:
            continue

        if h.confidence < conf_threshold or h.evidence < min_evidence:
            continue

        variables = h.variables
        rel_type = h.rel_type

        if len(variables) < 2:
            continue

        # ── 3-var multi-target types: check FIRST ──────────────────────────
        # variables[-1] is the outcome; all variables[:-1] are parents.
        # This corrects prior behaviour where these types were mishandled as
        # 2-var pairs and the outcome variable never received parents.
        if rel_type in _MULTI_VAR and len(variables) >= 3:
            tgt = variables[-1]
            for src in variables[:-1]:
                _add_edge(graph, src, tgt)
            edges.append(_edge_dict(variables[0], tgt, h))

        # ── 2-var directional ───────────────────────────────────────────────
        elif rel_type in _PAIRWISE_DIRECTIONAL:
            src, tgt = variables[0], variables[1]
            _add_edge(graph, src, tgt)
            edges.append(_edge_dict(src, tgt, h))

        # ── 2-var symmetric ─────────────────────────────────────────────────
        elif rel_type in _PAIRWISE_SYMMETRIC:
            src, tgt = variables[0], variables[1]
            _add_edge(graph, src, tgt)
            _add_edge(graph, tgt, src)
            edges.append(_edge_dict(src, tgt, h, symmetric=True))

        else:
            # Fallback for any future type: treat as directional 2-var
            if len(variables) >= 2:
                src, tgt = variables[0], variables[-1]
                _add_edge(graph, src, tgt)
                edges.append(_edge_dict(src, tgt, h))

    return graph, edges


def _add_edge(graph: Dict[str, List[str]], src: str, tgt: str) -> None:
    graph.setdefault(tgt, [])
    if src not in graph[tgt]:
        graph[tgt].append(src)


def _edge_dict(src: str, tgt: str, h, symmetric: bool = False) -> Dict[str, Any]:
    return {
        'source': src,
        'target': tgt,
        'variables': list(h.variables),
        'type': h.rel_type.value,
        'confidence': round(float(h.confidence), 4),
        'fit_score': round(float(h.fit_score), 4),
        'evidence': int(h.evidence),
        'stability': round(float(h.stability), 4),
        'state': h.meta.state.value,
        'symmetric': symmetric,
    }


def graph_summary(graph: Dict[str, List[str]], edges: List[Dict[str, Any]]) -> str:
    """Human-readable summary of the extracted graph."""
    n_edges = len(edges)
    n_targets = len(graph)
    if n_edges == 0:
        return "Empty graph — no edges above threshold."
    by_type = {}
    for e in edges:
        by_type.setdefault(e['type'], 0)
        by_type[e['type']] += 1
    type_str = ", ".join(f"{t}:{c}" for t, c in sorted(by_type.items()))
    return (f"{n_edges} edges across {n_targets} targets | types: {type_str} | "
            f"mean conf={sum(e['confidence'] for e in edges)/n_edges:.3f}")


def inspect_edges(edges: List[Dict[str, Any]], top_n: int = 40) -> None:
    """Print a ranked edge inspection table."""
    if not edges:
        print("  No edges to inspect.")
        return
    ranked = sorted(edges, key=lambda x: -x['confidence'])[:top_n]
    print(f"\n{'Source':<22} {'-> Target':<22} {'Type':<14} {'Conf':>6} {'Fit':>6} {'Evid':>5} {'State'}")
    print("-" * 85)
    for e in ranked:
        sym = "<>" if e.get('symmetric') else "->"
        print(f"  {e['source']:<20} {sym} {e['target']:<20} {e['type']:<14} "
              f"{e['confidence']:>6.3f} {e['fit_score']:>6.3f} {e['evidence']:>5}  {e['state']}")

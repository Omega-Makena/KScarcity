"""Connectivity-aware scoring for discovery evaluation.

Pairwise-adjacency ground truth is unfair to a discovery method: a mediator chain
X -> M -> Y makes X and Y *genuinely* marginally dependent, so flagging X-Y is a
correct detection of a real (indirect) dependence, not a false positive. Scoring
it as an error is what drives the online engine's apparent precision to ~0.06.

This module builds the schema's induced marginal-dependence structure and
partitions every candidate pair into three classes:

  direct       an adjacency edge planted by a relationship (the recall target)
  indirect     marginally dependent through a chain/shared latent, not adjacent
               (detecting it is neither rewarded as recall nor punished as error)
  independent  d-separated with no conditioning -> a detection here is a true
               false positive (the honest FPR denominator)

Collider co-parents are treated as independent: A1 -> TotalA <- A2 leaves A1, A2
marginally independent (dependent only if you condition on TotalA), likewise the
sources of a synergistic/moderating/logical target. Chains and shared latents
propagate dependence (transitive closure over the non-collider edge set).
"""
import itertools
from typing import Dict, List, Set, Tuple

Pair = frozenset


def build_dependency_sets(schema: dict, variables: List[str]):
    """Return (direct_edges, dependent_pairs, independent_pairs) as sets of
    frozenset pairs, from the schema's relationships."""
    direct: Set[Pair] = set()

    def add(a, b):
        if a != b:
            direct.add(frozenset((a, b)))

    for r in schema.get("relationships", []):
        t = r["type"]
        if t in ("causal", "functional", "probabilistic"):
            add(r["source"], r["target"])
        elif t in ("correlational", "competitive"):
            add(*r["pair"])
        elif t == "mediating":                       # chain: two adjacencies, X-Y is indirect
            add(r["source"], r["mediator"]); add(r["mediator"], r["target"])
        elif t == "moderating":                      # co-parents of target (collider), not each other
            add(r["source"], r["target"]); add(r["moderator"], r["target"])
        elif t in ("synergistic", "logical"):
            for s in r["sources"]:
                add(s, r["target"])
        elif t == "compositional":
            for c in r["components"]:
                add(c, r["total"])
        elif t == "similarity":                      # shared latent -> all mutually dependent
            for a, b in itertools.combinations(r["group"], 2):
                add(a, b)
        elif t == "graph":
            for e in r["edges"]:
                add(e["source"], e["target"])
        # temporal / equilibrium / structural: univariate, no cross-variable edge

    # transitive closure over the direct (non-collider) edges = marginally
    # dependent pairs (chains and shared latents propagate; colliders were never
    # added as co-parent edges, so they stay independent)
    adj: Dict[str, Set[str]] = {v: set() for v in variables}
    for p in direct:
        a, b = tuple(p)
        adj[a].add(b); adj[b].add(a)
    dependent: Set[Pair] = set()
    for start in variables:
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w); stack.append(w)
        for other in seen:
            if other != start:
                dependent.add(frozenset((start, other)))

    all_pairs = {frozenset(p) for p in itertools.combinations(variables, 2)}
    independent = all_pairs - dependent
    return direct, dependent, independent


def score_connectivity(pred: Set[Pair], direct: Set[Pair], dependent: Set[Pair],
                       independent: Set[Pair]) -> dict:
    """Adjacency recall + honest FPR over truly-independent pairs only.

    adjacency_recall  fraction of direct edges recovered
    false_edges       predictions landing on independent (d-separated) pairs
    indep_fpr         false_edges / |independent|   (the honest false-positive rate)
    indirect_hits     predictions on marginally-dependent-but-not-adjacent pairs
                      (real dependence, reported for context, not penalized)
    """
    tp = len(pred & direct)
    false_edges = len(pred & independent)
    indirect = len(pred & (dependent - direct))
    adjacency_recall = tp / len(direct) if direct else 0.0
    indep_fpr = false_edges / len(independent) if independent else 0.0
    # precision counting only true-independent detections as errors
    honest_prec = tp / (tp + false_edges) if (tp + false_edges) else 0.0
    return dict(adjacency_recall=round(adjacency_recall, 4),
                honest_precision=round(honest_prec, 4),
                indep_fpr=round(indep_fpr, 4),
                false_edges=false_edges, indirect_hits=indirect,
                n_direct=len(direct), n_independent=len(independent))

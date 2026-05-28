"""
Evaluation utilities for the calibrated pipeline.

Provides helpers used by both run_calibration_pipeline.py and
compare_methods_calibrated.py to evaluate ranked hypothesis lists
against ground truth.
"""
from __future__ import annotations

from typing import Optional


def match_discovery_to_gt(disc: dict, gt_entry: dict, level: str = 'strict') -> bool:
    """
    Check if a discovery matches a ground-truth entry.

    level:
        'strict'    — source, target, AND type must match
        'family'    — source, target must match; type must be in same family
        'edge_only' — source and target must match (any type)
    """
    src_eq = (disc.get('source') == gt_entry.get('source')
              and disc.get('target') == gt_entry.get('target'))
    if not src_eq:
        # Allow undirected matching for symmetric types
        symm_types = ('correlational', 'temporal', 'equilibrium', 'compositional')
        if gt_entry.get('type') in symm_types:
            src_eq = (disc.get('source') == gt_entry.get('target')
                      and disc.get('target') == gt_entry.get('source'))
    if not src_eq:
        return False

    if level == 'edge_only':
        return True

    disc_type = disc.get('test_type', '')
    gt_type = gt_entry.get('type', '')

    if level == 'strict':
        return disc_type == gt_type

    # Family matching
    _FAMILY = {
        'causal': 'dependence',
        'correlational': 'dependence',
        'competitive': 'dependence',
        'temporal': 'temporal',
        'equilibrium': 'temporal',
        'structural': 'temporal',
        'compositional': 'constraint',
        'functional': 'constraint',
        'mediating': 'interaction',
        'moderating': 'interaction',
        'synergistic': 'interaction',
        'graph': 'nonlinear',
        'similarity': 'nonlinear',
        'logical': 'nonlinear',
        'probabilistic': 'nonlinear',
    }
    return _FAMILY.get(disc_type, '_d') == _FAMILY.get(gt_type, '_g')


def precision_recall_at_k_calibrated(
    ranked_results: list[dict],
    ground_truth: list[dict],
    k_values: list[int] = None,
    level: str = 'strict',
) -> dict:
    """
    Compute precision@k and recall@k for a ranked list.

    ranked_results must already be sorted by score descending.
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 30, 50]

    n_gt = len(ground_truth)
    prec = {}
    rec = {}

    for k in sorted(k_values):
        top_k = ranked_results[:k]
        matched_gt_idxs: set[int] = set()
        for disc in top_k:
            for gi, gt in enumerate(ground_truth):
                if match_discovery_to_gt(disc, gt, level=level):
                    matched_gt_idxs.add(gi)
        n_matched = len(matched_gt_idxs)
        prec[k] = n_matched / k if k > 0 else 0.0
        rec[k] = n_matched / n_gt if n_gt > 0 else 0.0

    return {'precision': prec, 'recall': rec}


def null_fpr_calibrated(
    ranked_results: list[dict],
    null_pairs: list[dict],
    threshold_field: str = 'passes_dual_threshold',
) -> float:
    """
    False positive rate on known-null pairs among selected hypotheses.
    """
    selected = [r for r in ranked_results if r.get(threshold_field, False)]
    if not selected:
        return 0.0
    n_null = sum(
        1 for r in selected
        if any(r.get('source') == np.get('source') and r.get('target') == np.get('target')
               for np in null_pairs)
    )
    return n_null / len(selected)


def first_gt_rank(ranked_results: list[dict], ground_truth: list[dict],
                  level: str = 'strict') -> int:
    """Return 1-indexed rank of first GT match, or -1 if none found."""
    for rank, r in enumerate(ranked_results, start=1):
        if any(match_discovery_to_gt(r, g, level) for g in ground_truth):
            return rank
    return -1

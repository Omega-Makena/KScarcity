"""
Step 6: Final ranking, dual threshold, and evaluation against GT.

Score(H) = Z_H × pi_H

where Z_H is the permutation Z-score (from original data) and pi_H is the
selection frequency from block-bootstrap stability selection.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional


def apply_dual_threshold(
    results: list[dict],
    fdr_q: float = 0.10,
    stability_min: float = 0.60,
    verbose: bool = True,
) -> list[dict]:
    """
    Apply dual threshold: FDR-significant AND stable.

    A hypothesis passes if:
        fdr_adjusted_p < fdr_q  AND  selection_frequency >= stability_min

    Reports at 9 threshold combinations without cherry-picking.
    Returns results sorted by score descending with 'passes_dual_threshold' added.
    """
    fdr_levels = [0.05, 0.10, 0.20]
    pi_levels = [0.50, 0.60, 0.70]
    m = len(results)

    if verbose:
        print(f'\n  Dual-threshold report ({m} hypotheses):')
        print(f"  {'FDR q':>7}  {'pi_min':>7}  {'#passed':>8}  {'% passed':>9}  {'Est. FDP':>9}")
        print(f"  {'-' * 50}")

    for q in fdr_levels:
        for pi_min in pi_levels:
            passed = [
                r for r in results
                if (r.get('fdr_adjusted_p', 1.0) < q
                    and r.get('selection_frequency', 0.0) >= pi_min)
            ]
            n_passed = len(passed)
            pct = 100 * n_passed / max(m, 1)
            est_fdp = q  # BH guarantee: expected FDP <= q by construction
            if verbose:
                print(f"  {q:>7.2f}  {pi_min:>7.2f}  {n_passed:>8d}  "
                      f"{pct:>8.1f}%  {est_fdp:>8.2f}")

    # Tag each result with whether it passes the primary threshold
    for r in results:
        r['passes_dual_threshold'] = (
            r.get('fdr_adjusted_p', 1.0) < fdr_q
            and r.get('selection_frequency', 0.0) >= stability_min
        )

    sorted_results = sorted(results, key=lambda r: r.get('score', 0.0), reverse=True)
    return sorted_results


def evaluate_against_gt(
    ranked_results: list[dict],
    ground_truth: list[dict],
    null_pairs: list[dict],
    k_values: list[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the calibrated ranking against ground truth.

    Rank all hypotheses by Score(H) = z_score × selection_frequency, descending.
    Compute precision@k, recall@k, first GT rank, mean GT rank, null FPR.
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 30, 50]

    # Sort by score descending
    ranked = sorted(ranked_results, key=lambda r: r.get('score', 0.0), reverse=True)
    n_gt = len(ground_truth)

    def _matches_gt(r: dict, gt_entry: dict) -> bool:
        """True if discovery r matches GT entry (edge + type, or edge-only)."""
        src_match = (r['source'] == gt_entry['source'] and r['target'] == gt_entry['target'])
        if not src_match:
            # Undirected for correlational/temporal
            if gt_entry['type'] in ('correlational', 'temporal', 'equilibrium'):
                src_match = (r['source'] == gt_entry['target']
                             and r['target'] == gt_entry['source'])
        if not src_match:
            return False
        return r['test_type'] == gt_entry['type']

    def _matches_any_gt(r: dict) -> bool:
        return any(_matches_gt(r, g) for g in ground_truth)

    def _matches_null(r: dict) -> bool:
        return any(
            r['source'] == np['source'] and r['target'] == np['target']
            for np in null_pairs
        )

    # Precision@k and recall@k
    prec_at_k: dict[int, float] = {}
    rec_at_k: dict[int, float] = {}
    for k in sorted(k_values):
        top_k = ranked[:k]
        matched_gt_idxs: set[int] = set()
        for disc in top_k:
            for gi, gt in enumerate(ground_truth):
                if _matches_gt(disc, gt):
                    matched_gt_idxs.add(gi)
        n_matched = len(matched_gt_idxs)
        prec_at_k[k] = n_matched / k if k > 0 else 0.0
        rec_at_k[k] = n_matched / n_gt if n_gt > 0 else 0.0

    # First GT rank (1-indexed), mean GT rank
    first_gt_rank = -1
    gt_ranks = []
    for rank, r in enumerate(ranked, start=1):
        if _matches_any_gt(r):
            if first_gt_rank < 0:
                first_gt_rank = rank
            gt_ranks.append(rank)

    mean_gt_rank = float(sum(gt_ranks) / len(gt_ranks)) if gt_ranks else float('inf')

    # Null FPR: fraction of selected hypotheses that are known null pairs
    selected = [r for r in ranked if r.get('passes_dual_threshold', False)]
    n_sel = len(selected)
    n_sel_null = sum(1 for r in selected if _matches_null(r))
    null_fpr = n_sel_null / n_sel if n_sel > 0 else 0.0
    n_gt_in_selected = sum(1 for r in selected if _matches_any_gt(r))

    if verbose:
        print(f'\n  Calibrated Ranking Evaluation (n={len(ranked)} hypotheses):')
        print(f"  {'k':>5}  {'P@k':>8}  {'R@k':>8}")
        print(f"  {'-' * 30}")
        for k in sorted(k_values):
            print(f"  {k:>5}  {prec_at_k[k]:>8.4f}  {rec_at_k[k]:>8.4f}")
        print(f'\n  First GT rank: {first_gt_rank if first_gt_rank > 0 else "not found"}')
        print(f'  Mean GT rank:  {mean_gt_rank:.1f}')
        print(f'  Null FPR (selected): {null_fpr:.3f}')
        print(f'  Selected (dual threshold): {n_sel}')
        print(f'  GT matches in selected: {n_gt_in_selected}')
        print(f'\n  vs. Old system:')
        print(f'    Old first GT rank: 123 (all top-100 Bayesian conf = FPs)')
        improvement = f'{123 / first_gt_rank:.1f}x' if first_gt_rank > 0 else 'N/A'
        print(f'    New first GT rank: {first_gt_rank if first_gt_rank > 0 else "not found"}')
        print(f'    Improvement: {improvement}')

    return {
        'precision_at_k': prec_at_k,
        'recall_at_k': rec_at_k,
        'first_gt_rank': first_gt_rank,
        'mean_gt_rank': mean_gt_rank,
        'null_fpr': null_fpr,
        'n_selected': n_sel,
        'n_gt_in_selected': n_gt_in_selected,
        'improvement_over_old': {
            'old_first_gt_rank': 123,
            'new_first_gt_rank': first_gt_rank,
            'improvement': f'{123 / first_gt_rank:.1f}x' if first_gt_rank > 0 else 'N/A',
        },
    }

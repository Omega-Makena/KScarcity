"""
Step 4: Benjamini-Hochberg FDR control on the selected hypotheses.

Standard BH 1995 procedure — NOT Benjamini-Yekutieli (too conservative here).

Ensures that among all hypotheses declared significant, at most q fraction
are expected to be false positives.
"""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(
    pvalues: list[float],
    q: float = 0.10,
) -> list[bool]:
    """
    Standard BH procedure (Benjamini & Hochberg 1995).

    1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
    2. Find largest k such that p_(k) <= k * q / m
    3. Reject all hypotheses with p_(i) <= p_(k)

    Returns:
        List of bool, same order as input pvalues (True = reject null).
    """
    m = len(pvalues)
    if m == 0:
        return []

    pvals = np.asarray(pvalues, dtype=np.float64)
    sorted_idx = np.argsort(pvals, kind='stable')
    sorted_pvals = pvals[sorted_idx]

    # BH threshold for each rank k (1-indexed)
    thresholds = (np.arange(1, m + 1) / m) * q
    significant_mask = sorted_pvals <= thresholds

    # Find largest k that satisfies condition
    if not np.any(significant_mask):
        critical_pval = -1.0
    else:
        last_k = int(np.where(significant_mask)[0][-1])  # 0-indexed
        critical_pval = float(sorted_pvals[last_k])

    # Reject all hypotheses with p <= critical_pval
    reject = pvals <= critical_pval
    return reject.tolist()


def _bh_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """
    Compute BH-adjusted p-values (Storey & Tibshirani step-up procedure).
    Adjusted p for hypothesis i = min(m * p_(k) / k) for k >= rank(i).
    """
    m = len(pvalues)
    if m == 0:
        return []
    pvals = np.asarray(pvalues, dtype=np.float64)
    sorted_idx = np.argsort(pvals, kind='stable')
    sorted_pvals = pvals[sorted_idx]

    # Compute adjusted p-values in sorted order
    adj = np.zeros(m)
    adj[-1] = sorted_pvals[-1]
    for k in range(m - 2, -1, -1):
        adj[k] = min(adj[k + 1], (m / (k + 1)) * sorted_pvals[k])
    adj = np.minimum(adj, 1.0)

    # Put back in original order
    result = np.zeros(m)
    result[sorted_idx] = adj
    return result.tolist()


def apply_fdr(
    results: list[dict],
    q_levels: list[float] = None,
) -> list[dict]:
    """
    Apply BH-FDR at multiple q levels and add results to each hypothesis dict.

    Adds fields:
        'fdr_significant_q{q}': bool  for each q in q_levels
        'fdr_adjusted_p': float

    Also prints a summary for each q.
    """
    if q_levels is None:
        q_levels = [0.05, 0.10, 0.20]

    if not results:
        return results

    pvalues = [r['p_value'] for r in results]
    m = len(pvalues)

    # BH-adjusted p-values
    adj_p = _bh_adjusted_pvalues(pvalues)
    for r, ap in zip(results, adj_p):
        r['fdr_adjusted_p'] = ap

    # Significance at each q
    for q in q_levels:
        reject = benjamini_hochberg(pvalues, q=q)
        key = f'fdr_significant_q{q:.2f}'.replace('0.', 'q').replace('q0', 'q')
        # Simpler key: fdr_significant_005, fdr_significant_010, etc.
        key = 'fdr_sig_' + f'{int(round(q * 100)):03d}'
        n_sig = sum(reject)
        pct = 100 * n_sig / m if m > 0 else 0.0
        print(f'  FDR q={q:.2f}: {n_sig}/{m} significant ({pct:.1f}%)')
        for r, rej in zip(results, reject):
            r[key] = bool(rej)

    # Canonical key for downstream steps (use q=0.10)
    q_main = 0.10
    reject_main = benjamini_hochberg(pvalues, q=q_main)
    for r, rej in zip(results, reject_main):
        r['fdr_significant'] = bool(rej)

    return results

"""
Step 2: Convert permutation p-values to Z-scores for a common scale.

z_i = Phi^{-1}(1 - p_i)

where Phi^{-1} is the standard normal quantile function.

At B=200, the smallest possible p is 1/201 ≈ 0.00498,
giving z_max ≈ Phi^{-1}(0.995) ≈ 2.576.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_Z_CAP = 4.0  # hard cap to prevent inf values


def pvalue_to_zscore(p: float) -> float:
    """
    Convert p-value to one-sided Z-score.

    z = Phi^{-1}(1 - p)

    Edge cases:
        p <= 0 → cap at _Z_CAP (should not occur with permutation +1 correction)
        p >= 1 → 0.0 (no evidence)
    """
    if p <= 0.0:
        return _Z_CAP
    if p >= 1.0:
        return 0.0
    z = norm.ppf(1.0 - p)
    return float(min(z, _Z_CAP))


def add_zscores(results: list[dict]) -> list[dict]:
    """
    Add 'z_score' and 'z_significant' fields to each result from Step 1.

    z_significant: z > 1.645 (one-sided p < 0.05), quick pre-filter.

    Modifies list in place and returns it.
    """
    for r in results:
        r['z_score'] = pvalue_to_zscore(r['p_value'])
        r['z_significant'] = r['z_score'] > 1.645
    return results

import numpy as np
from typing import Dict, Any

def apply_bh_fdr(results: Dict[str, Any], q: float = 0.05) -> None:
    """
    Apply Benjamini-Hochberg FDR step-up procedure to control false discovery rate.
    Updates the results dictionary with a 'significant' boolean flag.
    """
    keys = list(results.keys())
    p_values = np.array([results[k]['p_value'] for k in keys])
    m = len(p_values)
    
    if m == 0:
        return

    sorted_idx = np.argsort(p_values)
    significant = np.zeros(m, dtype=bool)
    max_significant = -1

    for rank, idx in enumerate(sorted_idx):
        threshold = (rank + 1) / m * q
        if p_values[idx] <= threshold:
            max_significant = rank

    if max_significant >= 0:
        for rank in range(max_significant + 1):
            significant[sorted_idx[rank]] = True

    for i, k in enumerate(keys):
        results[k]['significant'] = bool(significant[i])

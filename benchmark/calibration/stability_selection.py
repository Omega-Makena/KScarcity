import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

def block_bootstrap_indices(n_samples: int, block_size: int, n_iterations: int) -> List[np.ndarray]:
    """Generate indices for block bootstrap to preserve autocorrelation."""
    indices = []
    n_blocks = n_samples // block_size + (1 if n_samples % block_size > 0 else 0)
    
    for _ in range(n_iterations):
        boot_idx = []
        # Sample blocks with replacement
        sampled_blocks = np.random.randint(0, n_blocks, size=n_blocks)
        for b in sampled_blocks:
            start = b * block_size
            end = min(start + block_size, n_samples)
            boot_idx.extend(range(start, end))
        indices.append(np.array(boot_idx[:n_samples]))
    return indices

def stability_selection(
    X: np.ndarray, 
    y: np.ndarray, 
    estimator_func, # function that returns selected features/significant hypotheses
    block_size: int = 50, 
    n_bootstrap: int = 100, 
    threshold: float = 0.6
) -> Dict[str, float]:
    """
    Perform stability selection via block bootstrap.
    estimator_func(X_boot, y_boot) -> List[str] of selected hypothesis names
    Returns a dictionary of {hypothesis_name: selection_frequency} for those >= threshold.
    """
    n_samples = len(X)
    boot_indices = block_bootstrap_indices(n_samples, block_size, n_bootstrap)
    
    selection_counts = defaultdict(int)
    
    for idx in boot_indices:
        X_boot = X[idx]
        y_boot = y[idx]
        selected = estimator_func(X_boot, y_boot)
        for item in selected:
            selection_counts[item] += 1
            
    frequencies = {k: v / n_bootstrap for k, v in selection_counts.items()}
    return {k: freq for k, freq in frequencies.items() if freq >= threshold}

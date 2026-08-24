"""
Numeric utilities for Controller ⇆ Evaluator math.

Provides robust normalization, clipping, and rolling statistics.
"""

import numpy as np
from typing import List


def clip(x: float, lo: float, hi: float) -> float:
    """
    Clips a value to strict bounds.

    Args:
        x: The input value.
        lo: The lower bound.
        hi: The upper bound.

    Returns:
        The clipped value, guaranteed to be within [lo, hi].
    """
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """
    Performs safe division, handling zero denominators.

    Args:
        a: Numerator.
        b: Denominator.
        default: The value to return if b is close to zero.

    Returns:
        The result of a/b or default.
    """
    if abs(b) < 1e-12:
        return default
    return a / b



def robust_zscore(x: float, median: float, mad: float) -> float:
    """
    Compute robust z-score using median and Median Absolute Deviation (MAD).

    This provides a normalization that is resistant to outliers compared to standard
    mean/std normalization.
    
    Args:
        x: Value to normalize.
        median: Median of the reference distribution.
        mad: Median Absolute Deviation of the reference distribution.
        
    Returns:
        Z-score approximation. Returns 0.0 if MAD is negligible.
    """
    if mad < 1e-12:
        return 0.0
    return (x - median) / mad


def robust_quantiles(values: List[float], quantiles: List[float]) -> List[float]:
    """
    Compute robust quantiles using rank statistics.

    Determines the values at specified percentile ranks in the input distribution.
    
    Args:
        values: List of numeric observations.
        quantiles: List of quantile levels (e.g., [0.16, 0.84]) to compute.
        
    Returns:
        List of computed quantile values corresponding to the inputs. Returns 0.0s if input is empty.
    """
    if len(values) == 0:
        return [0.0] * len(quantiles)
    
    sorted_vals = np.array(sorted(values))
    n = len(sorted_vals)
    
    result = []
    for q in quantiles:
        idx = int(np.round(q * (n - 1)))
        idx = max(0, min(n - 1, idx))
        result.append(float(sorted_vals[idx]))
    
    return result


def softplus(x: float) -> float:
    """
    Calculates the Softplus function: log(1 + exp(x)).

    This is a smooth approximation of the ReLU function, useful for producing positive values
    with a differentiable gradient. Includes numerical stability checks for large x.

    Args:
        x: Input value.

    Returns:
        log(1 + exp(x))
    """
    # For large x, softplus(x) ≈ x
    # For small x, softplus(x) ≈ log(1 + x)
    if x > 20:
        return x
    return np.log1p(np.exp(x))



def compute_median_mad(values: List[float]) -> tuple[float, float]:
    """
    Computes robust location and scale estimators for a dataset.

    Calculates the Median and the Median Absolute Deviation (MAD), which are robust
    alternatives to Mean and Standard Deviation.
    
    Args:
        values: List of numeric values.
        
    Returns:
        A tuple (median, mad). Returns (0.0, 1.0) if input is empty.
    """
    if len(values) == 0:
        return 0.0, 1.0
    
    vals = np.array(values)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    
    # Prevent division by zero
    if mad < 1e-12:
        mad = 1.0
    
    return median, mad

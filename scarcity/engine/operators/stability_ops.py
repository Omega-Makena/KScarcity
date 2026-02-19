"""
Stability Operators — Concordance and drift detection primitives.

Spearman correlation, sign agreement, and Page-Hinkley tests.
"""

import numpy as np
from typing import List


def spearman_concordance(values_list: List[np.ndarray]):
    """
    Compute Spearman correlation concordance across windows.
    
    Measures rank stability of values across consecutive windows.
    
    Args:
        values_list: List of arrays from different windows
        
    Returns:
        Concordance score in [0, 1]
    """
    if len(values_list) < 2:
        return 1.0
    
    # Compute rank correlations
    correlations = []
    for i in range(len(values_list) - 1):
        v1, v2 = values_list[i], values_list[i + 1]
        
        if len(v1) != len(v2) or len(v1) == 0:
            continue
        
        # Compute ranks
        ranks1 = np.argsort(np.argsort(v1)).astype(np.float32)
        ranks2 = np.argsort(np.argsort(v2)).astype(np.float32)
        
        # Pearson correlation of ranks = Spearman
        v1_mean = np.mean(ranks1)
        v2_mean = np.mean(ranks2)
        
        numerator = np.sum((ranks1 - v1_mean) * (ranks2 - v2_mean))
        denom1 = np.sqrt(np.sum((ranks1 - v1_mean) ** 2))
        denom2 = np.sqrt(np.sum((ranks2 - v2_mean) ** 2))
        
        if denom1 * denom2 > 1e-12:
            corr = numerator / (denom1 * denom2)
            correlations.append(corr)
        else:
            correlations.append(1.0)  # Identical ranks
    
    # Average and normalize to [0, 1]
    if len(correlations) == 0:
        return 0.0
    
    mean_corr = np.mean(correlations)
    # Map [-1, 1] to [0, 1]
    return float((mean_corr + 1.0) / 2.0)


def sign_agreement(values_list: List[np.ndarray]):
    """
    Compute sign agreement across windows.
    
    Measures stability of sign patterns.
    
    Args:
        values_list: List of arrays from different windows
        
    Returns:
        Agreement score in [0, 1]
    """
    if len(values_list) < 2:
        return 1.0
    
    signs_list = [np.sign(v) for v in values_list]
    agreement = []
    
    for i in range(len(signs_list) - 1):
        s1, s2 = signs_list[i], signs_list[i + 1]
        
        if len(s1) != len(s2) or len(s1) == 0:
            continue
        
        # Fraction of matching signs
        match = np.mean(s1 == s2)
        agreement.append(match)
    
    return np.mean(agreement) if agreement else 0.0


def page_hinkley(series: np.ndarray, delta: float = 0.05, lambda_threshold: float = 2.0):
    """
    Page-Hinkley test for drift detection.
    
    Detects changes in the mean of a time series.
    
    Args:
        series: Time series values
        delta: Minimum change to detect (relative to std)
        lambda_threshold: Alert threshold
        
    Returns:
        1 if drift detected, else 0
    """
    if len(series) < 2:
        return 0.0
    
    # Estimate parameters
    mean = np.mean(series)
    std = np.std(series)
    
    if std < 1e-12:
        return 0.0
    
    # Compute cumulative deviation
    threshold = delta * std
    
    # Page-Hinkley statistic
    m = 0
    mi = []
    
    for i, val in enumerate(series):
        m = (val - mean) - threshold
        mi_val = max(0.0, (mi[i - 1] + m) if i > 0 else m)
        mi.append(mi_val)
    
    # Check if threshold crossed
    max_deviation = max(mi)
    
    if max_deviation > lambda_threshold * std:
        return 1.0
    
    return 0.0


def stability_score(gain_list: List[float]) -> float:
    """
    Combined stability score from Spearman and sign agreement.
    
    Args:
        gain_list: List of gain values across windows
        
    Returns:
        Stability score in [0, 1]
    """
    if len(gain_list) < 2:
        return 0.5
    
    # Convert to arrays for consistency
    arrays = [np.array([g]) for g in gain_list]
    
    # Spearman concordance
    spearman = spearman_concordance(arrays)
    
    # Sign agreement
    sign_agree = sign_agreement(arrays)
    
    # Weighted combination
    stability = 0.5 * spearman + 0.5 * sign_agree
    
    return float(np.clip(stability, 0.0, 1.0))

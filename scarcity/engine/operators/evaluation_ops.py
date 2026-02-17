"""
Evaluation Operators — Online scoring primitives.

R² gain, NLL gain, and Granger-like tests for path evaluation.

All functions handle edge cases (constant series, small samples) gracefully
by returning 0.0 rather than NaN.
"""

import numpy as np
from typing import Optional

# Try to import scipy for proper F-distribution; fall back to approximation
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def r2_gain(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray) -> float:
    """
    Compute R² gain vs baseline.
    
    Measures how much better y_pred is compared to baseline_pred
    in terms of R² (coefficient of determination).
    
    Args:
        y_true: Actual target values.
        y_pred: Model predictions.
        baseline_pred: Baseline predictions (e.g., mean).
        
    Returns:
        R² improvement. Positive means y_pred is better.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    baseline_pred = np.asarray(baseline_pred).ravel()
    
    if len(y_true) == 0:
        return 0.0
    
    ss_res_pred = np.sum((y_true - y_pred) ** 2)
    ss_res_base = np.sum((y_true - baseline_pred) ** 2)
    
    if ss_res_base == 0:
        return 0.0
    
    var_y = np.var(y_true)
    if var_y < 1e-10:
        return 0.0
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    
    r2_pred = 1 - ss_res_pred / ss_tot
    r2_base = 1 - ss_res_base / ss_tot
    
    gain = r2_pred - r2_base
    return 0.0 if np.isnan(gain) else gain


def nll_gain(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    baseline_pred: np.ndarray,
    sigma: Optional[float] = None
) -> float:
    """
    Compute Negative Log-Likelihood gain vs baseline.
    
    Assumes Gaussian likelihood. Measures how much better the model's
    predictions are in terms of log-likelihood.
    
    Args:
        y_true: Actual target values.
        y_pred: Model predictions.
        baseline_pred: Baseline predictions.
        sigma: Error standard deviation. If None, estimated from residuals.
        
    Returns:
        NLL improvement. Positive means y_pred has lower NLL (better).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    baseline_pred = np.asarray(baseline_pred).ravel()
    
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # Estimate sigma if not provided
    if sigma is None:
        residuals = y_true - y_pred
        sigma = np.std(residuals)
        if sigma < 1e-10:
            sigma = 1.0  # Prevent division by zero
    
    sigma_sq = sigma ** 2
    
    # NLL for prediction: 0.5 * n * log(2π σ²) + 0.5 * Σ(y - ŷ)² / σ²
    nll_pred = 0.5 * n * np.log(2 * np.pi * sigma_sq) + \
               0.5 * np.sum((y_true - y_pred) ** 2) / sigma_sq
    
    # NLL for baseline
    nll_base = 0.5 * n * np.log(2 * np.pi * sigma_sq) + \
               0.5 * np.sum((y_true - baseline_pred) ** 2) / sigma_sq
    
    # Gain = improvement in NLL (lower NLL is better, so base - pred)
    gain = nll_base - nll_pred
    return 0.0 if np.isnan(gain) else gain


def granger_step(x: np.ndarray, y: np.ndarray, lag: int = 1) -> float:
    """
    Granger causality test using F-statistic.
    
    Tests whether lagged values of x improve prediction of y
    beyond using lagged values of y alone (restricted model).
    
    Returns a score in [0, 1] where higher values indicate
    stronger evidence that x Granger-causes y.
    
    Args:
        x: Potential cause series.
        y: Potential effect series.
        lag: Number of lags to test.
        
    Returns:
        Granger causality score (0 = no evidence, 1 = strong evidence).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Need sufficient data for regression
    min_samples = 2 * lag + 5
    if len(x) < min_samples or len(y) < min_samples:
        return 0.0
    
    # Align series
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    
    # Check for constant series
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    
    # Build regression matrices
    # Restricted model: Y_t ~ Y_{t-1}, ..., Y_{t-lag}
    # Unrestricted model: Y_t ~ Y_{t-1}, ..., Y_{t-lag}, X_{t-1}, ..., X_{t-lag}
    
    effective_n = n - lag
    if effective_n < lag + 3:
        return 0.0
    
    Y_target = y[lag:]
    
    # Build lagged Y matrix (restricted model)
    Y_lags = np.column_stack([y[lag-i-1:n-i-1] for i in range(lag)])
    
    # Build lagged X matrix
    X_lags = np.column_stack([x[lag-i-1:n-i-1] for i in range(lag)])
    
    # Unrestricted model: Y lags + X lags
    X_full = np.hstack([Y_lags, X_lags])
    
    # Add constant term
    ones = np.ones((effective_n, 1))
    Y_lags_c = np.hstack([ones, Y_lags])
    X_full_c = np.hstack([ones, X_full])
    
    try:
        # Fit restricted model
        beta_r, residuals_r, rank_r, _ = np.linalg.lstsq(Y_lags_c, Y_target, rcond=None)
        y_pred_r = Y_lags_c @ beta_r
        ss_res_r = np.sum((Y_target - y_pred_r) ** 2)
        
        # Fit unrestricted model
        beta_u, residuals_u, rank_u, _ = np.linalg.lstsq(X_full_c, Y_target, rcond=None)
        y_pred_u = X_full_c @ beta_u
        ss_res_u = np.sum((Y_target - y_pred_u) ** 2)
        
        # F-statistic
        q = lag  # Number of restrictions (X lag coefficients)
        df_num = q
        df_den = effective_n - X_full_c.shape[1]
        
        if df_den <= 0 or ss_res_u < 1e-10:
            return 0.0
        
        f_stat = ((ss_res_r - ss_res_u) / df_num) / (ss_res_u / df_den)
        
        if f_stat < 0:
            return 0.0
        
        # Convert to p-value and then to score
        if SCIPY_AVAILABLE:
            p_value = 1 - scipy_stats.f.cdf(f_stat, df_num, df_den)
        else:
            # Approximation: use empirical threshold
            # F > 4 with reasonable df is roughly p < 0.05
            p_value = 1.0 / (1.0 + f_stat / 4.0)
        
        # Higher score = stronger Granger causality evidence
        score = max(0.0, min(1.0, 1.0 - p_value))
        return 0.0 if np.isnan(score) else score
        
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Compute partial correlation between x and y controlling for z.
    
    Useful for testing conditional independence in causal discovery.
    
    Args:
        x: First variable.
        y: Second variable.
        z: Control variable(s) - can be 1D or 2D.
        
    Returns:
        Partial correlation coefficient in [-1, 1].
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z)
    
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    
    n = len(x)
    if n < z.shape[1] + 3:
        return 0.0
    
    # Check for constant variables
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    
    try:
        # Regress x on z
        z_c = np.hstack([np.ones((n, 1)), z])
        beta_x = np.linalg.lstsq(z_c, x, rcond=None)[0]
        resid_x = x - z_c @ beta_x
        
        # Regress y on z
        beta_y = np.linalg.lstsq(z_c, y, rcond=None)[0]
        resid_y = y - z_c @ beta_y
        
        # Correlation of residuals = partial correlation
        std_x, std_y = np.std(resid_x), np.std(resid_y)
        if std_x < 1e-10 or std_y < 1e-10:
            return 0.0
        
        corr = np.corrcoef(resid_x, resid_y)[0, 1]
        return 0.0 if np.isnan(corr) else corr
        
    except (np.linalg.LinAlgError, ValueError):
        return 0.0



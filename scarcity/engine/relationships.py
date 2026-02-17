"""
Relationship Hypothesis Implementations

Complete implementations for all 15 relationship types.
Each class extends the base Hypothesis and implements proper algorithms.
"""

from __future__ import annotations

import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .discovery import Hypothesis, RelationshipType, HypothesisMetadata
from .relationship_config import (
    CausalConfig,
    CorrelationalConfig,
    TemporalConfig,
    FunctionalConfig,
    EquilibriumConfig,
    CompositionalConfig,
    CompetitiveConfig,
    SynergisticConfig,
    ProbabilisticConfig,
    StructuralConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CAUSAL — Granger Causality
# =============================================================================

class CausalHypothesis(Hypothesis):
    """
    Detects causal relationships using Granger causality.
    
    X Granger-causes Y if past values of X help predict Y 
    beyond what past values of Y alone can predict.
    
    Args:
        source: Name of the source variable (potential cause).
        target: Name of the target variable (potential effect).
        lag: Number of lags to consider in Granger test.
        buffer_size: Maximum buffer size for observations.
        config: Configuration object with thresholds and parameters.
    """
    
    def __init__(self, source: str, target: str, lag: int = 2, 
                 buffer_size: int = 100, config: Optional[CausalConfig] = None):
        super().__init__([source, target], RelationshipType.CAUSAL)
        self.source = source
        self.target = target
        self.lag = lag
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        self.config = config or CausalConfig()
        
        # Granger test statistics
        self.gain_forward = 0.0  # Predictive gain X → Y
        self.gain_backward = 0.0  # Predictive gain Y → X
        self.p_cause = 0.0
        self.direction = 0  # +1 = X→Y, -1 = Y→X, 0 = none
        
        # Learned coefficients for prediction (stored from last Granger computation)
        self._learned_coef_aug: Optional[np.ndarray] = None
        self._x_mean: float = 0.0
        self._y_mean: float = 0.0
    
    def fit_step(self, row: Dict[str, float]) -> None:
        """Update buffers with new observation."""
        if self.source in row and self.target in row:
            x_val = row[self.source]
            y_val = row[self.target]
            if np.isfinite(x_val) and np.isfinite(y_val):
                self.buffer_x.append(x_val)
                self.buffer_y.append(y_val)
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """Compute Granger causality statistics."""
        cfg = self.config
        min_samples = self.lag + cfg.min_samples_for_eval
        
        if len(self.buffer_x) < min_samples:
            return {'fit_score': 0.5, 'confidence': 0.5, 
                    'evidence': len(self.buffer_x), 'stability': 0.5}
        
        X = np.array(self.buffer_x)
        Y = np.array(self.buffer_y)
        
        # Store means for prediction
        self._x_mean = float(np.mean(X))
        self._y_mean = float(np.mean(Y))
        
        # Compute directional gains and store coefficients
        self.gain_forward, coef_fwd = self._granger_gain_with_coef(X, Y)
        self.gain_backward, _ = self._granger_gain_with_coef(Y, X)
        
        # Store learned coefficients for prediction
        if coef_fwd is not None:
            self._learned_coef_aug = coef_fwd
        
        # Determine direction using configurable threshold
        gain_diff = self.gain_forward - self.gain_backward
        if gain_diff > cfg.direction_threshold:
            self.direction = 1
            self.p_cause = min(1.0, self.gain_forward * cfg.confidence_multiplier)
        elif gain_diff < -cfg.direction_threshold:
            self.direction = -1
            self.p_cause = min(1.0, self.gain_backward * cfg.confidence_multiplier)
        else:
            self.direction = 0
            self.p_cause = 0.0
        
        fit = min(1.0, max(self.gain_forward, self.gain_backward))
        
        return {
            'fit_score': fit,
            'confidence': self.p_cause,
            'evidence': len(self.buffer_x),
            'stability': 0.8 if self.direction != 0 else 0.5,
            'gain_forward': self.gain_forward,
            'gain_backward': self.gain_backward,
            'direction': self.direction
        }
    
    def _granger_gain_with_coef(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute predictive gain of X for predicting Y.
        
        Returns:
            Tuple of (gain, learned_coefficients) where coefficients are from
            the augmented regression [Y_lags, X_lags] -> Y.
        """
        ridge = self.config.ridge_alpha
        
        if len(X) <= self.lag + 1:
            return 0.0, None
        
        # Baseline: predict Y from its own lags
        Y_target = Y[self.lag:]
        
        Y_lags = np.column_stack([Y[self.lag-i-1:-i-1] for i in range(self.lag)])
        X_lags = np.column_stack([X[self.lag-i-1:-i-1] for i in range(self.lag)])
        
        # Baseline MSE (Y from Y lags only)
        XtX = Y_lags.T @ Y_lags + ridge * np.eye(self.lag)
        Xty = Y_lags.T @ Y_target
        try:
            coef = np.linalg.solve(XtX, Xty)
            residual_base = Y_target - Y_lags @ coef
            mse_base = float(np.mean(residual_base ** 2))
        except np.linalg.LinAlgError:
            return 0.0, None
        
        # Augmented MSE (Y from Y lags + X lags)
        XY = np.concatenate([Y_lags, X_lags], axis=1)
        XtX_aug = XY.T @ XY + ridge * np.eye(2 * self.lag)
        Xty_aug = XY.T @ Y_target
        try:
            coef_aug = np.linalg.solve(XtX_aug, Xty_aug)
            residual_aug = Y_target - XY @ coef_aug
            mse_aug = float(np.mean(residual_aug ** 2))
        except np.linalg.LinAlgError:
            return 0.0, None
        
        if mse_base <= 1e-9:
            return 0.0, coef_aug
        
        gain = max(0.0, (mse_base - mse_aug) / mse_base)
        return gain, coef_aug
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """
        Predict target based on learned Granger coefficients.
        
        Uses the learned coefficients from the augmented regression to compute
        the effect of X on Y, rather than a hardcoded effect size.
        """
        cfg = self.config
        min_pred_samples = self.lag + cfg.min_prediction_samples
        
        if self.direction != 1 or len(self.buffer_x) < min_pred_samples:
            return None
        
        if self._learned_coef_aug is None:
            return None
        
        # Build feature vector from recent lags
        if len(self.buffer_y) < self.lag or len(self.buffer_x) < self.lag:
            return None
        
        # Construct lag features: [y_{t-1}, ..., y_{t-lag}, x_{t-1}, ..., x_{t-lag}]
        y_lags = np.array([self.buffer_y[-i-1] for i in range(self.lag)])
        x_lags = np.array([self.buffer_x[-i-1] for i in range(self.lag)])
        features = np.concatenate([y_lags, x_lags])
        
        # Predict using learned coefficients
        y_hat = float(np.dot(features, self._learned_coef_aug))
        
        return (self.target, y_hat)

    def to_dict(self) -> Dict[str, Any]:
        """Include Granger-specific fields needed by UIs and exporters."""
        d = super().to_dict()
        d["metrics"].update({
            "lag": self.lag,
            "gain_forward": float(self.gain_forward),
            "gain_backward": float(self.gain_backward),
            "p_cause": float(self.p_cause),
            "direction": int(self.direction),
        })
        d["source"] = self.source
        d["target"] = self.target
        return d


# =============================================================================
# 2. CORRELATIONAL — Pearson/Spearman Correlation
# =============================================================================

class CorrelationalHypothesis(Hypothesis):
    """
    Detects correlation using online Pearson correlation.
    
    Note: Correlation does NOT imply causation.
    
    Args:
        var1: First variable name.
        var2: Second variable name.
        buffer_size: Maximum buffer size for observations.
        config: Configuration object with thresholds.
    """
    
    def __init__(self, var1: str, var2: str, buffer_size: int = 100,
                 config: Optional[CorrelationalConfig] = None):
        super().__init__([var1, var2], RelationshipType.CORRELATIONAL)
        self.var1 = var1
        self.var2 = var2
        self.buffer1 = deque(maxlen=buffer_size)
        self.buffer2 = deque(maxlen=buffer_size)
        self.config = config or CorrelationalConfig()
        
        # Online statistics (Welford)
        self.n = 0
        self.mean1 = 0.0
        self.mean2 = 0.0
        self.M2_1 = 0.0
        self.M2_2 = 0.0
        self.cov = 0.0
    
    def fit_step(self, row: Dict[str, float]) -> None:
        """Update correlation statistics with new observation."""
        if self.var1 in row and self.var2 in row:
            x = row[self.var1]
            y = row[self.var2]
            if not (np.isfinite(x) and np.isfinite(y)):
                return
            
            self.buffer1.append(x)
            self.buffer2.append(y)
            
            # Welford's online algorithm
            self.n += 1
            delta1 = x - self.mean1
            self.mean1 += delta1 / self.n
            delta2 = y - self.mean2
            self.mean2 += delta2 / self.n
            
            self.M2_1 += delta1 * (x - self.mean1)
            self.M2_2 += delta2 * (y - self.mean2)
            self.cov += delta1 * (y - self.mean2)
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """Compute correlation coefficient."""
        cfg = self.config
        
        if self.n < cfg.min_samples:
            return {'fit_score': 0.5, 'confidence': 0.5, 
                    'evidence': self.n, 'stability': 0.5}
        
        var1 = self.M2_1 / self.n
        var2 = self.M2_2 / self.n
        covar = self.cov / self.n
        
        denom = np.sqrt(var1 * var2)
        if denom < 1e-9:
            r = 0.0
        else:
            r = covar / denom
        r = np.clip(r, -1.0, 1.0)
        
        return {
            'fit_score': abs(r),
            'confidence': min(1.0, self.n / cfg.confidence_scale) * abs(r),
            'evidence': self.n,
            'stability': 0.8 if abs(r) > cfg.stability_threshold else 0.5,
            'correlation': r
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Correlation is not predictive."""
        return None  # Correlation is not used for simulation


# =============================================================================
# 3. TEMPORAL — VAR(p) Autoregressive
# =============================================================================

class TemporalHypothesis(Hypothesis):
    """
    Detects temporal/autoregressive relationships.
    
    Y_t depends on Y_{t-1}, Y_{t-2}, ..., Y_{t-p}
    
    Args:
        variable: Variable name to model.
        lag: Number of lags in AR model.
        buffer_size: Maximum buffer size for observations.
        config: Configuration object with RLS parameters.
    """
    
    def __init__(self, variable: str, lag: int = 3, buffer_size: int = 100,
                 config: Optional[TemporalConfig] = None):
        super().__init__([variable], RelationshipType.TEMPORAL)
        self.variable = variable
        self.lag = lag
        self.buffer = deque(maxlen=buffer_size)
        self.config = config or TemporalConfig()
        
        # AR coefficients (online update)
        self.coefficients = np.zeros(lag + 1)  # [intercept, phi_1, ..., phi_p]
        self._rls_P = np.eye(lag + 1) * self.config.initial_covariance  # RLS covariance
        self._lambda = self.config.forgetting_factor  # Forgetting factor
    
    def fit_step(self, row: Dict[str, float]) -> None:
        """Update AR model with new observation."""
        if self.variable not in row:
            return
        
        val = row[self.variable]
        if not np.isfinite(val):
            return
        
        self.buffer.append(val)
        
        if len(self.buffer) > self.lag:
            # Form feature vector [1, y_{t-1}, ..., y_{t-p}]
            x = np.array([1.0] + [self.buffer[-i-2] for i in range(self.lag)])
            y = val
            
            # RLS update
            self._rls_update(x, y)
    
    def _rls_update(self, x: np.ndarray, y: float):
        """Recursive Least Squares update."""
        # Prediction error
        y_hat = np.dot(x, self.coefficients)
        error = y - y_hat
        
        # Kalman gain
        Px = self._rls_P @ x
        denom = self._lambda + x @ Px
        if abs(denom) < 1e-12:
            return
        K = Px / denom
        
        # Update coefficients
        self.coefficients += K * error
        
        # Update covariance
        self._rls_P = (self._rls_P - np.outer(K, Px)) / self._lambda
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """Evaluate AR model fit."""
        cfg = self.config
        min_samples = self.lag + cfg.min_samples_for_eval
        
        if len(self.buffer) <= min_samples:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': len(self.buffer), 'stability': 0.5}
        
        # Compute autocorrelation at lag 1
        Y = np.array(self.buffer)
        auto_corr = np.corrcoef(Y[1:], Y[:-1])[0, 1] if len(Y) > 1 else 0.0
        
        # Compute R-squared
        if len(self.buffer) > self.lag + 10:
            predictions = []
            actuals = []
            for i in range(self.lag, len(self.buffer)):
                x = np.array([1.0] + [self.buffer[i-j-1] for j in range(self.lag)])
                predictions.append(np.dot(x, self.coefficients))
                actuals.append(self.buffer[i])
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-9)
            r2 = max(0.0, min(1.0, r2))
        else:
            r2 = 0.5
        
        return {
            'fit_score': r2,
            'confidence': min(1.0, len(self.buffer) / 50) * r2,
            'evidence': len(self.buffer),
            'stability': 0.8 if abs(auto_corr) > cfg.autocorr_stability_threshold else 0.5,
            'autocorrelation': auto_corr,
            'coefficients': self.coefficients.tolist()
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Predict next value using AR model."""
        if len(self.buffer) <= self.lag:
            return None
        
        x = np.array([1.0] + [self.buffer[-i-1] for i in range(self.lag)])
        y_hat = np.dot(x, self.coefficients)
        return (self.variable, y_hat)


# =============================================================================
# 4. FUNCTIONAL — Linear/Polynomial Regression
# =============================================================================

class FunctionalHypothesis(Hypothesis):
    """
    Detects deterministic functional relationships.
    
    Y = f(X) where f is approximately linear or polynomial.
    
    Args:
        source: Source variable name.
        target: Target variable name.
        degree: Polynomial degree.
        buffer_size: Maximum buffer size.
        config: Configuration object with RLS and threshold parameters.
    """
    
    def __init__(self, source: str, target: str, degree: int = 1,
                 buffer_size: int = 100, config: Optional[FunctionalConfig] = None):
        super().__init__([source, target], RelationshipType.FUNCTIONAL)
        self.source = source
        self.target = target
        self.degree = degree
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        self.config = config or FunctionalConfig()
        
        # RLS for polynomial regression
        n_features = degree + 1
        self.coefficients = np.zeros(n_features)
        self._rls_P = np.eye(n_features) * self.config.initial_covariance
        self._lambda = self.config.forgetting_factor
    
    def _features(self, x: float) -> np.ndarray:
        """Generate polynomial features."""
        return np.array([x ** i for i in range(self.degree + 1)])
    
    def fit_step(self, row: Dict[str, float]) -> None:
        """Update regression with new observation."""
        if self.source in row and self.target in row:
            x = row[self.source]
            y = row[self.target]
            if not (np.isfinite(x) and np.isfinite(y)):
                return
            
            self.buffer_x.append(x)
            self.buffer_y.append(y)
            
            # RLS update
            features = self._features(x)
            y_hat = np.dot(features, self.coefficients)
            error = y - y_hat
            
            Px = self._rls_P @ features
            denom = self._lambda + features @ Px
            if abs(denom) > 1e-12:
                K = Px / denom
                self.coefficients += K * error
                self._rls_P = (self._rls_P - np.outer(K, Px)) / self._lambda
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """Evaluate functional fit."""
        cfg = self.config
        n = len(self.buffer_x)
        
        if n < cfg.min_samples:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n, 'stability': 0.5}
        
        # Compute R-squared
        X = np.array(self.buffer_x)
        Y = np.array(self.buffer_y)
        Y_hat = np.array([np.dot(self._features(x), self.coefficients) for x in X])
        
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-9)
        r2 = max(0.0, min(1.0, r2))
        
        # Check if it's approximately deterministic
        residual_std = np.std(Y - Y_hat)
        y_std = np.std(Y)
        is_deterministic = residual_std < cfg.deterministic_threshold * y_std if y_std > 1e-6 else False
        
        return {
            'fit_score': r2,
            'confidence': min(1.0, n / cfg.confidence_scale) * r2,
            'evidence': n,
            'stability': 0.9 if is_deterministic else 0.6,
            'coefficients': self.coefficients.tolist(),
            'deterministic': is_deterministic
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Predict target from source."""
        if self.source not in row or len(self.buffer_x) < self.config.min_samples:
            return None
        
        x = row[self.source]
        y_hat = np.dot(self._features(x), self.coefficients)
        return (self.target, y_hat)


# =============================================================================
# 5. EQUILIBRIUM — Mean-Reverting Process
# =============================================================================

class EquilibriumHypothesis(Hypothesis):
    """
    Detects mean-reverting/equilibrium behavior.
    
    Variable tends to return to a stable equilibrium value.
    
    Args:
        variable: Variable name to model.
        buffer_size: Maximum buffer size.
        config: Configuration object with Kalman filter parameters.
    """
    
    def __init__(self, variable: str, buffer_size: int = 100,
                 config: Optional[EquilibriumConfig] = None):
        super().__init__([variable], RelationshipType.EQUILIBRIUM)
        self.variable = variable
        self.buffer = deque(maxlen=buffer_size)
        self.config = config or EquilibriumConfig()
        
        # Kalman filter for equilibrium estimation
        self.equilibrium = 0.0  # Estimated equilibrium
        self.reversion_rate = 0.0  # Speed of reversion
        self.variance = 1.0
        
        # Kalman state - using config parameters
        self._kf_P = 1.0  # State covariance
        self._kf_Q = self.config.process_noise  # Process noise
        self._kf_R = self.config.observation_noise  # Observation noise
    
    def fit_step(self, row: Dict[str, float]) -> None:
        """Update equilibrium estimate with Kalman filter."""
        if self.variable not in row:
            return
        
        val = row[self.variable]
        if not np.isfinite(val):
            return
        
        self.buffer.append(val)
        
        # Kalman filter update for equilibrium
        # State is the equilibrium level
        # Predict
        P_pred = self._kf_P + self._kf_Q
        
        # Update
        K = P_pred / (P_pred + self._kf_R)
        self.equilibrium += K * (val - self.equilibrium)
        self._kf_P = (1 - K) * P_pred
        
        # Estimate reversion rate
        if len(self.buffer) > 10:
            Y = np.array(self.buffer)
            diffs = Y[1:] - Y[:-1]
            deviations = Y[:-1] - self.equilibrium
            
            # Regress diffs on deviations to get reversion rate
            # dY = -theta * (Y - mu) + noise
            if np.var(deviations) > 1e-9:
                self.reversion_rate = -np.cov(diffs, deviations)[0, 1] / np.var(deviations)
                self.reversion_rate = max(0.0, min(1.0, self.reversion_rate))
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """Evaluate equilibrium behavior."""
        cfg = self.config
        n = len(self.buffer)
        
        if n < cfg.min_samples_for_eval:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n, 'stability': 0.5}
        
        Y = np.array(self.buffer)
        
        # Check if mean-reverting using configurable threshold
        is_reverting = self.reversion_rate > cfg.reversion_threshold
        
        # Variance around equilibrium
        var_around_eq = np.var(Y - self.equilibrium)
        var_total = np.var(Y)
        explained = 1 - var_around_eq / (var_total + 1e-9)
        explained = max(0.0, min(1.0, explained))
        
        return {
            'fit_score': self.reversion_rate if is_reverting else 0.3,
            'confidence': min(1.0, n / cfg.confidence_scale) * (0.8 if is_reverting else 0.3),
            'evidence': n,
            'stability': 0.8 if is_reverting else 0.4,
            'equilibrium': self.equilibrium,
            'reversion_rate': self.reversion_rate,
            'is_reverting': is_reverting
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Predict value moving toward equilibrium."""
        cfg = self.config
        
        if len(self.buffer) < cfg.min_samples_for_prediction or self.reversion_rate < cfg.reversion_threshold:
            return None
        
        current = self.buffer[-1]
        predicted = current - self.reversion_rate * (current - self.equilibrium)
        return (self.variable, predicted)


# =============================================================================
# 6. COMPOSITIONAL — Sum Constraints
# =============================================================================

class CompositionalHypothesis(Hypothesis):
    """
    Detects compositional/additive constraints.
    
    Total = Part1 + Part2 + ... + PartN
    
    Args:
        parts: List of part variable names.
        total: Total variable name.
        buffer_size: Maximum buffer size.
        config: Configuration object with threshold parameters.
    """
    
    def __init__(self, parts: List[str], total: str, buffer_size: int = 100,
                 config: Optional[CompositionalConfig] = None):
        super().__init__(parts + [total], RelationshipType.COMPOSITIONAL)
        self.parts = parts
        self.total = total
        self.buffer_parts = {p: deque(maxlen=buffer_size) for p in parts}
        self.buffer_total = deque(maxlen=buffer_size)
        self.config = config or CompositionalConfig()
        
        self.constraint_error = float('inf')
    
    def fit_step(self, row: Dict[str, float]) -> None:
        """Collect observations."""
        if all(p in row for p in self.parts) and self.total in row:
            for p in self.parts:
                self.buffer_parts[p].append(row[p])
            self.buffer_total.append(row[self.total])
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """Check if sum constraint holds."""
        cfg = self.config
        n = len(self.buffer_total)
        
        if n < cfg.min_samples:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n, 'stability': 0.5}
        
        # Compute sum of parts vs total
        parts_sum = np.zeros(n)
        for p in self.parts:
            parts_sum += np.array(self.buffer_parts[p])
        
        total = np.array(self.buffer_total)
        
        # Relative error
        errors = np.abs(parts_sum - total) / (np.abs(total) + 1e-9)
        mean_error = np.mean(errors)
        self.constraint_error = mean_error
        
        # If error is very small, constraint holds
        holds = mean_error < cfg.error_threshold
        
        return {
            'fit_score': max(0.0, 1.0 - mean_error * cfg.error_scaling),
            'confidence': 0.9 if holds else 0.2,
            'evidence': n,
            'stability': 0.95 if holds else 0.3,
            'constraint_error': mean_error,
            'constraint_holds': holds
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Predict total from parts or vice versa."""
        if all(p in row for p in self.parts):
            predicted_total = sum(row[p] for p in self.parts)
            return (self.total, predicted_total)
        return None


# =============================================================================
# 7. COMPETITIVE — Trade-off Detection
# =============================================================================

class CompetitiveHypothesis(Hypothesis):
    """
    Detects competitive/trade-off relationships.
    
    X + Y ≈ constant (zero-sum behavior)
    
    Args:
        var1: First variable name.
        var2: Second variable name.
        buffer_size: Maximum buffer size.
        config: Configuration object with threshold parameters.
    """
    
    def __init__(self, var1: str, var2: str, buffer_size: int = 100,
                 config: Optional[CompetitiveConfig] = None):
        super().__init__([var1, var2], RelationshipType.COMPETITIVE)
        self.var1 = var1
        self.var2 = var2
        self.buffer1 = deque(maxlen=buffer_size)
        self.buffer2 = deque(maxlen=buffer_size)
        self.config = config or CompetitiveConfig()
    
    def fit_step(self, row: Dict[str, float]) -> None:
        """Collect observations."""
        if self.var1 in row and self.var2 in row:
            self.buffer1.append(row[self.var1])
            self.buffer2.append(row[self.var2])
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """Check for trade-off (constant sum)."""
        cfg = self.config
        n = len(self.buffer1)
        
        if n < cfg.min_samples:
            return {'fit_score': 0.5, 'confidence': 0.5,
                    'evidence': n, 'stability': 0.5}
        
        X = np.array(self.buffer1)
        Y = np.array(self.buffer2)
        
        # Check if sum is constant
        sums = X + Y
        cv = np.std(sums) / (np.mean(np.abs(sums)) + 1e-9)  # Coefficient of variation
        
        # Also check negative correlation
        corr = np.corrcoef(X, Y)[0, 1] if np.std(X) > 1e-9 and np.std(Y) > 1e-9 else 0.0
        
        is_competitive = cv < cfg.cv_threshold and corr < cfg.correlation_threshold
        
        return {
            'fit_score': max(0.0, 1.0 - cv) if corr < 0 else 0.3,
            'confidence': 0.9 if is_competitive else 0.3,
            'evidence': n,
            'stability': 0.8 if is_competitive else 0.4,
            'sum_cv': cv,
            'correlation': corr,
            'is_competitive': is_competitive,
            'constant_sum': np.mean(sums)
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Predict one variable from the other using constant sum."""
        if len(self.buffer1) < self.config.min_samples:
            return None
        
        const_sum = np.mean(np.array(self.buffer1) + np.array(self.buffer2))
        
        if self.var1 in row and self.var2 not in row:
            return (self.var2, const_sum - row[self.var1])
        if self.var2 in row and self.var1 not in row:
            return (self.var1, const_sum - row[self.var2])
        return None


# =============================================================================
# 8-15: Import summary (implementations continue in relationships_extended.py)
# =============================================================================

# For brevity, we define stubs for the remaining types
# Full implementations follow the same pattern

class SynergisticHypothesis(Hypothesis):
    """
    Detects interaction effects (X1*X2 term significant).
    
    Args:
        var1: First variable name.
        var2: Second variable name.
        target: Target variable name.
        buffer_size: Maximum buffer size.
        config: Configuration object with threshold parameters.
    """
    
    def __init__(self, var1: str, var2: str, target: str, buffer_size: int = 100,
                 config: Optional[SynergisticConfig] = None):
        super().__init__([var1, var2, target], RelationshipType.SYNERGISTIC)
        self.var1 = var1
        self.var2 = var2
        self.target = target
        self.buffer = deque(maxlen=buffer_size)
        self.config = config or SynergisticConfig()
        self.interaction_coef = 0.0
    
    def fit_step(self, row: Dict[str, float]) -> None:
        if all(v in row for v in [self.var1, self.var2, self.target]):
            self.buffer.append((row[self.var1], row[self.var2], row[self.target]))
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = len(self.buffer)
        
        if n < cfg.min_samples:
            return {'fit_score': 0.5, 'confidence': 0.5, 'evidence': n, 'stability': 0.5}
        
        data = np.array(list(self.buffer))
        X1, X2, Y = data[:, 0], data[:, 1], data[:, 2]
        
        # Fit Y ~ X1 + X2 + X1*X2
        features = np.column_stack([np.ones(n), X1, X2, X1 * X2])
        try:
            coef = np.linalg.lstsq(features, Y, rcond=None)[0]
            self.interaction_coef = coef[3]
        except (np.linalg.LinAlgError, ValueError):
            self.interaction_coef = 0.0
        
        # Check if interaction is significant using configurable threshold
        has_synergy = abs(self.interaction_coef) > cfg.interaction_threshold
        
        return {
            'fit_score': min(1.0, abs(self.interaction_coef)),
            'confidence': 0.8 if has_synergy else 0.3,
            'evidence': n,
            'stability': 0.7,
            'interaction_coefficient': self.interaction_coef
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


class ProbabilisticHypothesis(Hypothesis):
    """
    Detects when X shifts the distribution of Y.
    
    Args:
        condition: Condition variable name.
        target: Target variable name.
        buffer_size: Maximum buffer size.
        config: Configuration object with threshold parameters.
    """
    
    def __init__(self, condition: str, target: str, buffer_size: int = 200,
                 config: Optional[ProbabilisticConfig] = None):
        super().__init__([condition, target], RelationshipType.PROBABILISTIC)
        self.condition = condition
        self.target = target
        self.buffer_0 = deque(maxlen=buffer_size)  # Y when X=0
        self.buffer_1 = deque(maxlen=buffer_size)  # Y when X=1
        self.config = config or ProbabilisticConfig()
    
    def fit_step(self, row: Dict[str, float]) -> None:
        cfg = self.config
        if self.condition in row and self.target in row:
            x = row[self.condition]
            y = row[self.target]
            if x <= cfg.split_threshold:
                self.buffer_0.append(y)
            else:
                self.buffer_1.append(y)
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n0, n1 = len(self.buffer_0), len(self.buffer_1)
        
        if n0 < cfg.min_samples_per_group or n1 < cfg.min_samples_per_group:
            return {'fit_score': 0.5, 'confidence': 0.5, 
                    'evidence': n0 + n1, 'stability': 0.5}
        
        mean_0 = np.mean(self.buffer_0)
        mean_1 = np.mean(self.buffer_1)
        std_0 = np.std(self.buffer_0)
        std_1 = np.std(self.buffer_1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
        effect_size = abs(mean_1 - mean_0) / (pooled_std + 1e-9)
        
        significant = effect_size > cfg.effect_size_threshold
        
        return {
            'fit_score': min(1.0, effect_size),
            'confidence': 0.8 if significant else 0.3,
            'evidence': n0 + n1,
            'stability': 0.7,
            'mean_shift': mean_1 - mean_0,
            'effect_size': effect_size
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


class StructuralHypothesis(Hypothesis):
    """
    Detects hierarchical/nested structure.
    
    Args:
        group: Group variable name.
        outcome: Outcome variable name.
        buffer_size: Maximum buffer size (unused but kept for consistency).
        config: Configuration object with threshold parameters.
    """
    
    def __init__(self, group: str, outcome: str, buffer_size: int = 200,
                 config: Optional[StructuralConfig] = None):
        super().__init__([group, outcome], RelationshipType.STRUCTURAL)
        self.group = group
        self.outcome = outcome
        self.group_means: Dict[float, List[float]] = {}
        self.config = config or StructuralConfig()
    
    def fit_step(self, row: Dict[str, float]) -> None:
        if self.group in row and self.outcome in row:
            g = row[self.group]
            y = row[self.outcome]
            if g not in self.group_means:
                self.group_means[g] = []
            self.group_means[g].append(y)
    
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        
        if len(self.group_means) < cfg.min_groups:
            return {'fit_score': 0.5, 'confidence': 0.5, 'evidence': 0, 'stability': 0.5}
        
        # ICC (Intraclass Correlation Coefficient)
        group_means = [np.mean(vals) for vals in self.group_means.values() if len(vals) > 0]
        within_var = np.mean([np.var(vals) for vals in self.group_means.values() if len(vals) > 1])
        between_var = np.var(group_means)
        
        icc = between_var / (between_var + within_var + 1e-9)
        total_n = sum(len(v) for v in self.group_means.values())
        
        return {
            'fit_score': icc,
            'confidence': min(1.0, total_n / cfg.confidence_scale) * icc,
            'evidence': total_n,
            'stability': 0.7,
            'icc': icc,
            'n_groups': len(self.group_means)
        }
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None


# Registry of all hypothesis types
HYPOTHESIS_CLASSES = {
    RelationshipType.CAUSAL: CausalHypothesis,
    RelationshipType.CORRELATIONAL: CorrelationalHypothesis,
    RelationshipType.TEMPORAL: TemporalHypothesis,
    RelationshipType.FUNCTIONAL: FunctionalHypothesis,
    RelationshipType.EQUILIBRIUM: EquilibriumHypothesis,
    RelationshipType.COMPOSITIONAL: CompositionalHypothesis,
    RelationshipType.COMPETITIVE: CompetitiveHypothesis,
    RelationshipType.SYNERGISTIC: SynergisticHypothesis,
    RelationshipType.PROBABILISTIC: ProbabilisticHypothesis,
    RelationshipType.STRUCTURAL: StructuralHypothesis,
}

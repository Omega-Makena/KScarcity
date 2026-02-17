"""
Online Relationship Discovery — Hardened Hypothesis Implementations.

This module provides **production-hardened** implementations of hypothesis classes
with industrial robustness features:
- Winsorization: Clips extreme values to reduce outlier impact
- Huber Loss: Robust gradient that limits influence of large errors
- MAD Statistics: Median Absolute Deviation for robust variance estimation

These are DISTINCT from the classes in `relationships.py` which are the reference
implementations for the 15 relationship types. Use these hardened versions for
production workloads with potentially noisy or adversarial data.

Usage:
    from scarcity.engine.algorithms_online import FunctionalLinearHypothesis
    # For standard reference implementations, use:
    # from scarcity.engine.relationships import FunctionalHypothesis
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from .discovery import Hypothesis, RelationshipType
from .robustness import OnlineWinsorizer, OnlineMAD, huber_gradient

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class RLSConfig:
    """Configuration for Recursive Least Squares estimator."""
    lambda_forget: float = 0.99  # Forgetting factor (0.95-1.0 typical)
    initial_covariance: float = 10.0  # Initial P matrix scaling


@dataclass
class KalmanConfig:
    """Configuration for 1D Kalman Filter."""
    process_noise: float = 1e-4  # Q: model uncertainty
    observation_noise: float = 1e-2  # R: measurement uncertainty


@dataclass
class HardenedHypothesisConfig:
    """Configuration for all hardened hypothesis classes."""
    rls: RLSConfig = None
    kalman: KalmanConfig = None
    min_samples_correlation: int = 10
    history_max_size: int = 50
    history_trim_size: int = 20
    
    def __post_init__(self):
        if self.rls is None:
            self.rls = RLSConfig()
        if self.kalman is None:
            self.kalman = KalmanConfig()


# Default config instance
DEFAULT_HARDENED_CONFIG = HardenedHypothesisConfig()


# =============================================================================
# Safe Math Utilities
# =============================================================================

def safe_float(x: Any) -> float:
    try:
        if x is None: return float('nan')
        if isinstance(x, str): return float(x)
        return float(x)
    except (ValueError, TypeError):
        return float('nan')

def is_valid(x: float) -> bool:
    return math.isfinite(x)


# =============================================================================
# Core Building Blocks
# =============================================================================

class RecursiveLeastSquares:
    """
    Robust Recursive Least Squares (RLS) estimator for online linear regression.

    Maintains a running estimate of the weight vector `w` and the inverse covariance
    matrix `P`. It incorporates:
    - Forgetting Factor (lambda): To adapt to non-stationary data.
    - Huber Loss: To reduce the impact of outliers on the gradient steps.
    
    Args:
        n_features: Number of features in the model.
        lambda_forget: Forgetting factor (default from config).
        initial_covariance: Initial P matrix scaling (default from config).
        config: Optional RLSConfig for all parameters.
    """
    def __init__(self, n_features: int, lambda_forget: float = None,
                 initial_covariance: float = None, config: Optional[RLSConfig] = None):
        cfg = config or DEFAULT_HARDENED_CONFIG.rls
        self.n_features = n_features
        self.lambda_forget = lambda_forget if lambda_forget is not None else cfg.lambda_forget
        init_cov = initial_covariance if initial_covariance is not None else cfg.initial_covariance
        self.w = np.zeros(n_features)
        self.P = np.eye(n_features) * init_cov

    def predict(self, x: np.ndarray) -> float:
        if not np.all(np.isfinite(x)): return 0.0
        return float(np.dot(self.w, x))

    def update(self, x: np.ndarray, y: float) -> None:
        if not (is_valid(y) and np.all(np.isfinite(x))): return
        
        # traditional rls gain calculation
        Px = np.dot(self.P, x)
        denom = self.lambda_forget + np.dot(x, Px)
        if abs(denom) < 1e-9: return
        
        g = Px / denom
        
        # raw error
        raw_error = y - np.dot(self.w, x)
        
        # robust error (huber) - prevents exploding gradients from outliers
        robust_error = huber_gradient(raw_error)
        
        self.w += g * robust_error
        self.P = (self.P - np.outer(g, Px)) / self.lambda_forget

class KalmanFilter1D:
    """
    Simple 1D Kalman Filter for estimating a scalar state with noise.

    Used for tracking equilibrium values or slowly drifting means.
    
    Args:
        q: Process noise covariance (model uncertainty).
        r: Measurement noise covariance (sensor uncertainty).
        config: Optional KalmanConfig for all parameters.
    """
    def __init__(self, q: float = None, r: float = None, 
                 config: Optional[KalmanConfig] = None):
        cfg = config or DEFAULT_HARDENED_CONFIG.kalman
        self.q = q if q is not None else cfg.process_noise
        self.r = r if r is not None else cfg.observation_noise
        self.x = 0.0
        self.p = 1.0
        
    def update(self, z: float) -> float:
        if not is_valid(z): return self.x
        p_pred = self.p + self.q
        k = p_pred / (p_pred + self.r) if (p_pred + self.r) != 0 else 0
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * p_pred
        return self.x

# --- implementations ---

class FunctionalLinearHypothesis(Hypothesis):
    """
    Proposes a functional linear relationship: Y = aX + b.

    Uses RLS to learn the coefficients (a, b) online. Measures fit based on the
    prediction error relative to the robust standard deviation of the error.
    """
    def __init__(self, input_var: str, target_var: str):
        super().__init__([input_var, target_var], RelationshipType.FUNCTIONAL)
        self.input = input_var; self.target = target_var
        self.rls = RecursiveLeastSquares(2)
        # robust variance tracking
        self.err_stats = OnlineMAD()
        # input clipping
        self.win_x = OnlineWinsorizer()
        self.win_y = OnlineWinsorizer()
        
    def fit_step(self, row: Dict) -> None:
        if self.input not in row or self.target not in row: return
        x, y = safe_float(row[self.input]), safe_float(row[self.target])
        if not (is_valid(x) and is_valid(y)): return
        
        # winsorize inputs before feeding to rls
        x = self.win_x.update(x)
        y = self.win_y.update(y)
        
        self.rls.update(np.array([1.0, x]), y)
        
    def evaluate(self, row: Dict) -> Dict[str, float]:
        res = {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.evidence, "stability": self.stability}
        if self.input not in row or self.target not in row: return res
        x, y = safe_float(row[self.input]), safe_float(row[self.target])
        if not (is_valid(x) and is_valid(y)): return res
        
        # use winsorized values for eval too? strictly yes for stability
        # but for 'true' fit score on raw data? debatable.
        # let's use raw for 'prediction check' but winsorized for 'training stability'
        # actually, to be fair, we should use winsorized x to predict y
        if self.win_x.window:
            x_safe = max(self.win_x.lower_bound, min(x, self.win_x.upper_bound))
        else:
            x_safe = x
        
        pred = self.rls.predict(np.array([1.0, x_safe]))
        err = abs(y - pred)
        self.err_stats.update(err)
        
        # stability via mad
        sigma = self.err_stats.std_proxy
        mu = max(self.err_stats.median, 1e-6) # use median
        cv = sigma / mu
        stability = 1.0 / (1.0 + cv)
        
        fit_score = float(np.exp(-0.5 * (err / sigma)**2))
        return {"fit_score": fit_score, "confidence": self.confidence, "evidence": self.evidence, "stability": stability}

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self.input not in row: return None
        x = safe_float(row[self.input])
        if not is_valid(x): return None
        return (self.target, self.rls.predict(np.array([1.0, x])))

class CorrelationalHypothesis(Hypothesis):
    """
    Proposes a correlational relationship: Pearson(X, Y).

    Tracks sufficient statistics (means, variances, covariance) online using
    Welford's algorithm (or similar incremental updates). Capable of handling
    outliers via winsorization before updates.
    """
    def __init__(self, var_a: str, var_b: str):
        super().__init__([var_a, var_b], RelationshipType.CORRELATIONAL)
        self.a = var_a; self.b = var_b
        # robust clipping for correlation
        self.win_a = OnlineWinsorizer()
        self.win_b = OnlineWinsorizer()
        
        # we still use welford-like logic for correlation calc, but on partial data
        # actually, standard pearson is bad with outliers.
        # we can use winsorized pearson.
        self.mean_a = 0.0; self.mean_b = 0.0
        self.m2_a = 0.0; self.m2_b = 0.0
        self.cov = 0.0
        self.n = 0
        
    def fit_step(self, row: Dict) -> None:
        if self.a not in row or self.b not in row: return
        av, bv = safe_float(row[self.a]), safe_float(row[self.b])
        if not (is_valid(av) and is_valid(bv)): return

        # clip
        av = self.win_a.update(av)
        bv = self.win_b.update(bv)

        self.n += 1
        delta_a = av - self.mean_a
        self.mean_a += delta_a / self.n
        delta_b = bv - self.mean_b
        self.mean_b += delta_b / self.n
        
        self.m2_a += delta_a * (av - self.mean_a)
        self.m2_b += delta_b * (bv - self.mean_b)
        self.cov += delta_a * (bv - self.mean_b)
        
    def evaluate(self, row: Dict) -> Dict[str, float]:
        if self.n < 10: 
             return {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.n, "stability": 0.5}
        denom = np.sqrt(self.m2_a * self.m2_b)
        r = abs(self.cov / denom) if denom > 0 else 0.0
        return {"fit_score": r, "confidence": self.confidence, "evidence": self.n, "stability": 1.0}
    
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return None

class TemporalLagHypothesis(Hypothesis):
    """
    Proposes a lagged causal relationship: Y_t = a * X_{t-k} + b.

    Maintains a sliding history buffer of the input variable X to align it
    temporally with the current value of Y.
    """
    def __init__(self, inp: str, out: str, lag: int=1):
        super().__init__([inp, out], RelationshipType.TEMPORAL)
        self.inp, self.out, self.lag = inp, out, lag
        self.hist: List[float] = []
        self.rls = RecursiveLeastSquares(2)
        self.err_stats = OnlineMAD()
        self.win_in = OnlineWinsorizer()
        self.win_out = OnlineWinsorizer()
        
    def fit_step(self, row: Dict) -> None:
        if self.inp not in row or self.out not in row: return
        val_in = safe_float(row[self.inp])
        val_out = safe_float(row[self.out])
        if not is_valid(val_in) or not is_valid(val_out): return

        # winsorize inputs
        val_in = self.win_in.update(val_in)
        val_out = self.win_out.update(val_out)

        self.hist.append(val_in)
        if len(self.hist) > self.lag:
            prev = self.hist[-(self.lag+1)]
            self.rls.update(np.array([1.0, prev]), val_out)
            if len(self.hist) > 50: self.hist = self.hist[-20:]
            
    def evaluate(self, row: Dict) -> Dict[str, float]:
        if len(self.hist) < self.lag or self.out not in row: 
            return {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.evidence, "stability": 0.5}
            
        val_out = safe_float(row[self.out])
        if not is_valid(val_out): 
            return {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.evidence, "stability": 0.5}
        
        pred = self.rls.predict(np.array([1.0, self.hist[-self.lag]]))
        err = abs(val_out - pred)
        self.err_stats.update(err)
        
        sigma = self.err_stats.std_proxy
        mu = max(self.err_stats.median, 1e-6)
        cv = sigma / mu
        stability = 1.0 / (1.0 + cv)
        fit_score = float(np.exp(-0.5 * (err / sigma)**2))
        return {"fit_score": fit_score, "confidence": self.confidence, "evidence": self.evidence, "stability": stability}

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self.inp not in row: return None
        # Prediction for NEXT step is based on CURRENT value (lag=1)
        # If lag > 1, we need history.
        # For simplicity, if lag=1, use current row.
        if self.lag == 1:
             x = safe_float(row[self.inp])
             if not is_valid(x): return None
             return (self.out, self.rls.predict(np.array([1.0, x])))
        else:
             # Use history if available
             if len(self.hist) < self.lag: return None
             return (self.out, self.rls.predict(np.array([1.0, self.hist[-self.lag]])))

    def observe(self, row: Dict[str, float]) -> None:
        if self.inp in row:
             x = safe_float(row[self.inp])
             if is_valid(x):
                 self.hist.append(x)
                 if len(self.hist) > 50: self.hist = self.hist[-20:]

class EquilibriumHypothesis(Hypothesis):
    """
    Proposes that a variable fluctuates around a stable equilibrium value.

    Uses a Kalman Filter to track the 'true' state of the variable. Large deviations
    from the filtered state are considered errors against this hypothesis.
    """
    def __init__(self, var: str):
        super().__init__([var], RelationshipType.EQUILIBRIUM)
        self.var = var; self.kf = KalmanFilter1D()
        self.err_stats = OnlineMAD()
        self.win = OnlineWinsorizer()
        
    def fit_step(self, row: Dict) -> None:
        if self.var in row: 
            v = safe_float(row[self.var])
            if is_valid(v): 
                v = self.win.update(v)
                self.kf.update(v)
        
    def evaluate(self, row: Dict) -> Dict[str, float]:
        if self.var not in row:
             return {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.evidence, "stability": 0.5}
        v = safe_float(row[self.var])
        if not is_valid(v): 
             return {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.evidence, "stability": 0.5}
        
        err = abs(v - self.kf.x)
        self.err_stats.update(err)
        
        sigma = self.err_stats.std_proxy
        mu = max(self.err_stats.median, 1e-6)
        cv = sigma / mu
        stability = 1.0 / (1.0 + cv)
        
        fit_score = float(np.exp(-0.5 * (err / sigma)**2))
        return {"fit_score": fit_score, "confidence": self.confidence, "evidence": self.evidence, "stability": stability}
        
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        return (self.var, self.kf.x)

class VectorizedFunctionalHypothesis(Hypothesis):
    """
    A lightweight, zero-copy wrapper for hypothesis managed by a vectorized backend.

    This class does not store its own weights. Instead, it holds an index into
    a global tensor (managed by `VectorizedHypothesisPool`) where parameters are
    stored. This allows for massive SIMD updates during the `fit_step` of the pool.
    """
    def __init__(self, input_var: str, target_var: str, idx: int, engine_ref):
        super().__init__([input_var, target_var], RelationshipType.FUNCTIONAL)
        self.input = input_var; self.target = target_var
        self.idx = idx
        self.engine = engine_ref # reference to vectorizedrls
        self.err_stats = OnlineMAD()
        
    def fit_step(self, row: Dict) -> None:
        # no-op! update happens in batch in hypothesispool.update_all()
        # this prevents python loop overhead.
        pass
        
    def evaluate(self, row: Dict) -> Dict[str, float]:
        # read from tensor state directly
        if self.input not in row or self.target not in row: 
            return {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.evidence, "stability": 0.5}
            
        x = safe_float(row[self.input])
        y = safe_float(row[self.target])
        if not (is_valid(x) and is_valid(y)):
            return {"fit_score": 0.5, "confidence": self.confidence, "evidence": self.evidence, "stability": 0.5}
            
        # prediction: w[0] + w[1]*x
        # access numpy array directly
        W = self.engine.W[self.idx]
        pred = W[0]*1.0 + W[1]*x
        
        err = abs(y - pred)
        self.err_stats.update(err)
        
        sigma = self.err_stats.std_proxy
        fit_score = float(np.exp(-0.5 * (err / sigma)**2)) if sigma > 0 else 0.0
        
        # stability
        mu = max(self.err_stats.median, 1e-6)
        cv = sigma / mu
        stability = 1.0 / (1.0 + cv)

        return {"fit_score": fit_score, "confidence": self.confidence, "evidence": self.evidence, "stability": stability}
        
    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self.input not in row: return None
        x = safe_float(row[self.input])
        if not is_valid(x): return None
        W = self.engine.W[self.idx]
        val = W[0] + W[1]*x
        return (self.target, float(val))

    def observe(self, row: Dict[str, float]) -> None:
        """
        Observe state updates. For vectorized/functional hypotheses (markovian), 
        this is largely stateless, but required by the simulation interface.
        """
        pass

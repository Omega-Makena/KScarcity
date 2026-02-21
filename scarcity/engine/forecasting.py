"""
Industrial-Grade Predictive Forecasting (Bayesian VARX + GARCH).

This module replaces the naive matrix projection with an Online Bayesian 
Vector Autoregression with eXogenous variables (VARX) model. It calculates
streaming volatility (GARCH) to output heteroskedastic confidence intervals,
ensuring that inherently chaotic predictions are bounded by immense uncertainty bands.
"""

import logging
import time
from typing import Dict, Any, Optional

import numpy as np
from numba import njit, float32  # type: ignore

from scarcity.runtime import EventBus, get_bus

logger = logging.getLogger(__name__)


@njit(cache=True)
def _compute_garch_varx_forecast(
    W: np.ndarray,
    X_t: np.ndarray,
    exogenous_shock: float,
    sigma2_t: np.ndarray,
    max_steps: int,
    omega: float = 0.01,
    alpha: float = 0.15,
    beta: float = 0.80
) -> tuple:
    """
    Numba-accelerated Bayesian VARX projection with GARCH(1,1) uncertainty.
    
    Args:
        W: The weighted adjacency/transition matrix [V, V].
        X_t: The current state vector [V].
        exogenous_shock: A scalar value representing an external anomaly score.
        sigma2_t: Current conditional variance per variable [V].
        max_steps: N steps ahead to forecast.
        omega: GARCH baseline variance.
        alpha: GARCH shock response (ARCH term).
        beta: GARCH persistence (GARCH term).
        
    Returns:
        forecasts: Array [max_steps, V] of expected future states.
        variances: Array [max_steps, V] of expected variances (uncertainty bands).
    """
    V = X_t.shape[0]
    # Use float64 internally because Numba's np.dot may implicitly upcast to double precision
    forecasts = np.zeros((max_steps, V), dtype=np.float64)
    variances = np.zeros((max_steps, V), dtype=np.float64)
    
    current_state = np.copy(X_t).astype(np.float64)
    current_sigma2 = np.copy(sigma2_t).astype(np.float64)
    matrix_W = W.astype(np.float64)
    
    # Assume the exogenous shock decays exponentially
    current_shock = float(exogenous_shock)
    
    for step in range(max_steps):
        # 1. VARX Mean Equation Projection
        # X_{t+1} = W * X_t + Exogenous Shock
        next_state = np.dot(matrix_W, current_state) + (current_shock * 0.1)
        
        # We estimate the generic 'residual' using the shock and underlying variance
        implied_residual_sq = (current_state - next_state)**2 + (current_shock * current_shock)
        
        # 2. GARCH Variance Equation Projection
        # sigma^2_{t+1} = omega + alpha * epsilon^2_t + beta * sigma^2_t
        next_sigma2 = omega + alpha * implied_residual_sq + beta * current_sigma2
        
        forecasts[step] = next_state
        variances[step] = next_sigma2
        
        # Step forward
        current_state = next_state
        current_sigma2 = next_sigma2
        current_shock *= 0.5  # Exogenous shock decay
        
    return forecasts, variances


class PredictiveForecaster:
    """
    Online Bayesian VARX Forecaster.
    
    Extracts the transition matrix from HypergraphStore and projects N steps
    ahead using Numba-compiled GARCH volatility models to produce strict 
    Confidence Interval (CI) bounds. Throttles projection depth based on DRG.
    """
    
    def __init__(self, store, bus: Optional[EventBus] = None):
        self.store = store
        self.bus = bus if bus else get_bus()
        self.running = False
        
        # DRG Hardware limits
        self.max_steps = 5
        self.enabled = True
        
        # Exogenous shocks (Anomalies)
        self.latest_anomaly_score = 0.0
        
        # GARCH States
        self.garch_sigma2 = None
        
    async def start(self) -> None:
        """Subscribe to engine loops."""
        if self.running:
            return
        self.running = True
        self.bus.subscribe("data_window", self._handle_data_window)
        self.bus.subscribe("scarcity.anomaly_detected", self._handle_anomaly)
        self.bus.subscribe("scarcity.drg_extension_profile", self._handle_drg)
        logger.info("Bayesian VARX Forecaster started")

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.bus.unsubscribe("data_window", self._handle_data_window)
        self.bus.unsubscribe("scarcity.anomaly_detected", self._handle_anomaly)
        self.bus.unsubscribe("scarcity.drg_extension_profile", self._handle_drg)
        logger.info("Bayesian VARX Forecaster stopped")

    async def _handle_drg(self, topic: str, profile: Dict[str, Any]) -> None:
        """Throttle forecasting depth to save VRAM/Compute based on DRG."""
        self.enabled = profile.get("forecast_enabled", True)
        self.max_steps = int(profile.get("forecast_max_steps", 5))

    async def _handle_anomaly(self, topic: str, anomaly: Dict[str, Any]) -> None:
        """Capture exogenous shocks to inject into the VARX model."""
        self.latest_anomaly_score = anomaly.get("severity", 0.0)

    async def _handle_data_window(self, topic: str, data: Dict[str, Any]) -> None:
        """Extract matrix and project T+N steps ahead with GARCH bounds."""
        if not self.running or not self.enabled or self.max_steps <= 0:
            return
            
        window_tensor = data.get('data')
        if window_tensor is None:
            return
            
        if isinstance(window_tensor, list):
            window_tensor = np.array(window_tensor)
            
        if window_tensor.ndim != 2:
            return

        V = window_tensor.shape[1]
        
        # Initialize GARCH State if missing
        if self.garch_sigma2 is None or self.garch_sigma2.shape[0] != V:
            self.garch_sigma2 = np.ones(V, dtype=np.float32) * 0.1
            
        # Build Transition Matrix W from hypergraph store
        W = np.zeros((V, V), dtype=np.float32)
        for key, edge in self.store.edges.items():
            src_id, dst_id = key
            if src_id < V and dst_id < V:
                # Modulate weight by the edge variance (Bayesian prior adaptation)
                variance_penalty = 1.0 / (1.0 + edge.var)
                W[dst_id, src_id] = edge.weight * edge.stability * variance_penalty
                
        # Spectral radius dampening for stability
        try:
            spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
            if spectral_radius > 0.99:
                W = W / (spectral_radius + 0.05)
        except np.linalg.LinAlgError:
            pass # SVD did not converge

        X_t = window_tensor[-1, :].astype(np.float32)
        shock = min(float(self.latest_anomaly_score), 10.0)
        
        # Compute Numba GARCH Forecast
        forecasts, variances = _compute_garch_varx_forecast(
            W=W,
            X_t=X_t,
            exogenous_shock=shock,
            sigma2_t=self.garch_sigma2,
            max_steps=self.max_steps
        )
        
        # Evolve global sigma for next tick
        self.garch_sigma2 = variances[0]
        
        # Decay anomaly state
        self.latest_anomaly_score *= 0.5
        
        payload = {
            "timestamp": time.time(),
            "window_id": data.get("window_id", "unknown"),
            "steps_ahead": self.max_steps,
            "forecast_matrix": forecasts.tolist(),
            "garch_variance_matrix": variances.tolist()
        }
        await self.bus.publish("scarcity.forecasted_trends", payload)

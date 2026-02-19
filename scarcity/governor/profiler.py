"""
Resource profiling utilities for the DRG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np  # type: ignore


@dataclass
class ProfilerConfig:
    ema_alpha: float = 0.3
    kalman_Q: float = 0.01
    kalman_R: float = 0.1


@dataclass
class KalmanState:
    estimate: float = 0.0
    error_cov: float = 1.0


class ResourceProfiler:
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self._ema: Dict[str, float] = {}
        self._kalman: Dict[str, KalmanState] = {}

    def update(self, metrics: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        ema = {}
        forecast = {}
        for key, value in metrics.items():
            ema[key] = self._update_ema(key, value)
            forecast[key] = self._update_kalman(key, value)
        return ema, forecast

    def _update_ema(self, key: str, value: float) -> float:
        alpha = self.config.ema_alpha
        if key not in self._ema:
            self._ema[key] = value
        else:
            self._ema[key] = (1 - alpha) * self._ema[key] + alpha * value
        return self._ema[key]

    def _update_kalman(self, key: str, measurement: float) -> float:
        state = self._kalman.get(key, KalmanState(estimate=measurement, error_cov=1.0))

        # Predict
        predicted_estimate = state.estimate
        predicted_error_cov = state.error_cov + self.config.kalman_Q

        # Update
        kalman_gain = predicted_error_cov / (predicted_error_cov + self.config.kalman_R)
        updated_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        updated_error_cov = (1 - kalman_gain) * predicted_error_cov

        self._kalman[key] = KalmanState(updated_estimate, updated_error_cov)
        return updated_estimate


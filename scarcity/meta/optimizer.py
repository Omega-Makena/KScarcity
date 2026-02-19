"""
Online Reptile optimizer for meta learning.

This module implements an Online Reptile optimizer, which maintains a set of global
priors and updates them based on aggregated updates from multiple tasks (domains).
It includes mechanisms for stability, such as adaptive beta (learning rate) and
rollback capability in case of performance degradation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class MetaOptimizerConfig:
    """Configuration for the OnlineReptileOptimizer."""
    beta_init: float = 0.1
    beta_max: float = 0.3
    beta_decay_rate: float = 0.8  # Multiplier when under resource pressure
    beta_growth_rate: float = 1.1  # Multiplier when bandwidth is free
    beta_min_factor: float = 0.5  # Minimum beta as fraction of beta_init
    ema_alpha: float = 0.3
    rollback_delta: float = 0.1
    backup_versions: int = 10


@dataclass
class MetaOptimizerState:
    """Internal state of the optimizer."""
    prior: Dict[str, float] = field(default_factory=dict)
    beta: float = 0.1
    reward_ema: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)
    last_reward: float = 0.0


class OnlineReptileOptimizer:
    """
    Maintains a global meta prior using a Reptile-style EMA and supports rollback.
    
    This optimizer updates the global prior towards the direction of aggregated
    meta-updates. It adjusts the interpolation factor (`beta`) based on system
    resource constraints (DRG profile) to ensure safe and efficient learning.
    """

    def __init__(self, config: Optional[MetaOptimizerConfig] = None):
        """
        Initialize the optimizer.

        Args:
            config: Configuration object. Defaults to default settings.
        """
        self.config = config or MetaOptimizerConfig()
        self.state = MetaOptimizerState(beta=self.config.beta_init)

    def apply(
        self,
        aggregated_vector: np.ndarray,
        keys: List[str],
        reward: float,
        drg_profile: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply an aggregated update to the global prior.

        Args:
            aggregated_vector: The update vector derived from multiple domains.
            keys: The list of parameter keys corresponding to the vector.
            reward: The current system reward/score.
            drg_profile: Current resource usage profile (e.g., VRAM, latency).

        Returns:
            The updated global prior parameters (Dict[str, float]).
        """
        cfg = self.config
        state = self.state

        self._update_beta(drg_profile)

        if not state.prior:
            state.prior = {key: 0.0 for key in keys}

        self._record_history()

        prior_vector = np.array([state.prior.get(key, 0.0) for key in keys], dtype=np.float32)
        updated_vector = prior_vector + state.beta * aggregated_vector

        state.prior = dict(zip(keys, updated_vector.tolist()))
        self._update_reward(reward)

        return dict(state.prior)

    def rollback(self) -> Dict[str, float]:
        """
        Rollback the prior to the previous version.

        Returns:
            The restored prior parameters.
        """
        if not self.state.history:
            return self.state.prior
        self.state.prior = self.state.history.pop()
        return self.state.prior

    def should_rollback(self, reward: float) -> bool:
        """
        Check if a rollback is necessary based on reward degradation.

        Args:
            reward: The current reward.

        Returns:
            True if the drop in reward exceeds the configured threshold.
        """
        delta = self.state.reward_ema - reward
        return delta > self.config.rollback_delta

    def _update_beta(self, drg_profile: Dict[str, float]) -> None:
        """Adaptive beta adjustment based on resource pressure."""
        cfg = self.config
        vram_high = drg_profile.get("vram_high", 0.0)
        latency_high = drg_profile.get("latency_high", 0.0)
        bandwidth_free = drg_profile.get("bandwidth_free", 0.0)

        beta = self.state.beta
        if vram_high or latency_high:
            beta *= cfg.beta_decay_rate
        if bandwidth_free:
            beta *= cfg.beta_growth_rate
        beta = min(cfg.beta_max, max(cfg.beta_init * cfg.beta_min_factor, beta))
        self.state.beta = beta

    def _update_reward(self, reward: float) -> None:
        """Update the EMA of the reward."""
        cfg = self.config
        state = self.state
        state.reward_ema = (1 - cfg.ema_alpha) * state.reward_ema + cfg.ema_alpha * reward
        state.last_reward = reward

    def _record_history(self) -> None:
        """Save current prior to history for potential rollback."""
        cfg = self.config
        state = self.state
        state.history.append(dict(state.prior))
        if len(state.history) > cfg.backup_versions:
            state.history = state.history[-cfg.backup_versions :]

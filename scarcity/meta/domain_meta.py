"""
Per-domain meta learner for SCARCITY.

This module implements the logic for tracking learning progress within individual
domains. It maintains historical performance metrics and confidence scores to
generate adaptive meta-updates for global aggregation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class DomainMetaConfig:
    """Configuration for the DomainMetaLearner."""
    ema_alpha: float = 0.3
    meta_lr_min: float = 0.05
    meta_lr_max: float = 0.2
    stability_floor: float = 0.1
    confidence_decay: float = 0.9
    confidence_boost: float = 0.05  # Boost when sign agreement and positive delta
    max_history: int = 20


@dataclass
class DomainMetaState:
    """Internal state for a single domain's meta-learning process."""
    last_score: float = 0.0
    ema_score: float = 0.0
    last_timestamp: float = field(default_factory=time.time)
    history: List[float] = field(default_factory=list)
    confidence: float = 0.0
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class DomainMetaUpdate:
    """A computed meta-update from a specific domain."""
    domain_id: str
    vector: np.ndarray
    keys: List[str]
    confidence: float
    timestamp: float
    score_delta: float


class DomainMetaLearner:
    """
    Tracks domain-level learning signals and produces meta updates.
    
    This learner observes performance metrics from a domain, updates its internal
    confidence and EMA score, and computes a gradient-based update vector for
    meta-parameters. It adapts the learning rate based on confidence.
    """

    def __init__(self, config: Optional[DomainMetaConfig] = None):
        """
        Initialize the DomainMetaLearner.

        Args:
            config: Configuration object. Defaults to default settings.
        """
        self.config = config or DomainMetaConfig()
        self._states: Dict[str, DomainMetaState] = {}

    def observe(self, domain_id: str, metrics: Dict[str, float], parameters: Dict[str, float]) -> DomainMetaUpdate:
        """
        Process a new observation from a domain and generate a meta-update.

        Args:
            domain_id: Unique identifier for the domain.
            metrics: Dictionary of performance metrics (e.g., gain, stability).
            parameters: Dictionary of current parameter values in the domain.

        Returns:
            A DomainMetaUpdate object containing the calculated update vector and metadata.
        """
        cfg = self.config
        state = self._states.get(domain_id, DomainMetaState())

        score = float(metrics.get("meta_score", metrics.get("gain_p50", 0.0)))
        stability = float(metrics.get("stability_avg", 0.0))
        now = time.time()

        # Update EMA
        if state.history:
            state.ema_score = (1 - cfg.ema_alpha) * state.ema_score + cfg.ema_alpha * score
        else:
            state.ema_score = score

        score_delta = score - state.last_score
        state.history.append(score_delta)
        if len(state.history) > cfg.max_history:
            state.history = state.history[-cfg.max_history :]

        # Update confidence
        stability_term = max(stability, cfg.stability_floor)
        sign_agreement = np.sign(score_delta) == np.sign(state.confidence)
        state.confidence = cfg.confidence_decay * state.confidence + (1 - cfg.confidence_decay) * stability_term
        if sign_agreement and score_delta > 0:
            state.confidence += cfg.confidence_boost
        state.confidence = float(np.clip(state.confidence, 0.0, 1.0))

        # Compute adaptive meta learning rate
        meta_lr = cfg.meta_lr_min + (cfg.meta_lr_max - cfg.meta_lr_min) * state.confidence

        keys = sorted(parameters.keys())
        param_vector = np.array([parameters[k] for k in keys], dtype=np.float32)

        prev_params = state.parameters or {k: 0.0 for k in keys}
        prev_vector = np.array([prev_params.get(k, 0.0) for k in keys], dtype=np.float32)

        delta_vector = meta_lr * (param_vector - prev_vector)

        state.parameters = dict(zip(keys, param_vector.tolist()))
        state.last_score = score
        state.last_timestamp = now

        self._states[domain_id] = state

        return DomainMetaUpdate(
            domain_id=domain_id,
            vector=delta_vector,
            keys=keys,
            confidence=state.confidence,
            timestamp=now,
            score_delta=score_delta,
        )

    def state(self, domain_id: str) -> DomainMetaState:
        """Retrieve the current state for a domain."""
        return self._states.get(domain_id, DomainMetaState())

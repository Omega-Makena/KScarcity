"""
Trust scoring heuristics for federated participants.

This module implements the `TrustScorer`, which assigns and updates trust scores
for each peer in the federation. Scores are based on agreement with the consensus,
compliance with protocols, and the impact of their contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class TrustConfig:
    """Configuration for TrustScorer."""
    decay: float = 0.98
    min_trust: float = 0.0
    max_trust: float = 1.0
    agreement_weight: float = 0.6
    compliance_weight: float = 0.3
    impact_weight: float = 0.1
    penalty: float = 0.2


class TrustScorer:
    """
    Maintains reputation scores for federation peers.
    """

    def __init__(self, config: Optional[TrustConfig] = None):
        """
        Initialize the trust scorer.

        Args:
            config: Trust configuration object.
        """
        self.config = config or TrustConfig()
        self._scores: Dict[str, float] = {}

    def update(
        self,
        peer_id: str,
        agreement: float,
        compliance: float,
        impact_delta: float,
        violation: bool = False,
    ) -> float:
        """
        Update the trust score for a peer based on recent activity.

        Args:
            peer_id: Unique identifier of the peer.
            agreement: Score representing agreement with the aggregate (0.0 to 1.0).
            compliance: Score representing protocol compliance (0.0 to 1.0).
            impact_delta: Measure of positive/negative impact (-1.0 to 1.0).
            violation: Whether a severe protocol violation occurred.

        Returns:
            The new trust score for the peer.
        """
        cfg = self.config
        trust = self._scores.get(peer_id, 0.5)

        weighted_score = (
            cfg.agreement_weight * np.clip(agreement, 0.0, 1.0)
            + cfg.compliance_weight * np.clip(compliance, 0.0, 1.0)
            + cfg.impact_weight * np.clip(impact_delta, -1.0, 1.0)
        )

        # Apply exponential moving average
        trust = cfg.decay * trust + (1 - cfg.decay) * weighted_score
        
        if violation:
            trust = max(cfg.min_trust, trust - cfg.penalty)

        trust = float(np.clip(trust, cfg.min_trust, cfg.max_trust))
        self._scores[peer_id] = trust
        return trust

    def score(self, peer_id: str) -> float:
        """
        Get the current trust score for a peer.
        Defaults to 0.5 for unknown peers.
        """
        return self._scores.get(peer_id, 0.5)

    def sandbox(self, peer_id: str) -> None:
        """
        Manually downgrade a peer's trust, effectively sandboxing them.
        """
        self._scores[peer_id] = max(self.config.min_trust, 0.1)

    def trusted_peers(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Return a dictionary of peers meeting a minimum trust threshold.

        Args:
            threshold: Minimum trust score required.

        Returns:
            Dictionary mapping peer_id to trust score.
        """
        return {peer: score for peer, score in self._scores.items() if score >= threshold}

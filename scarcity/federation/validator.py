"""
Packet validation and policy enforcement.

This module provides the `PacketValidator` class, which enforces rules on incoming
data packets. It checks for trust thresholds, resource limits (e.g., max edges),
and structural validity before allowing data to be processed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .packets import PathPack, EdgeDelta, PolicyPack, CausalSemanticPack


@dataclass
class ValidatorConfig:
    """Configuration for packet validation."""
    trust_min: float = 0.2
    allow_policy_share: bool = True
    max_edges: int = 2048
    max_concepts: int = 256


class PacketValidator:
    """
    Validates incoming packets against security policies and limits.
    """

    def __init__(self, config: ValidatorConfig):
        """
        Initialize the validator.

        Args:
            config: Validator configuration object.
        """
        self.config = config

    def validate_path_pack(self, pack: PathPack, trust: float) -> bool:
        """
        Validate a PathPack.

        Args:
            pack: The PathPack to validate.
            trust: The sender's trust score.

        Returns:
            True if valid, False otherwise.
        """
        if trust < self.config.trust_min:
            return False
        if len(pack.edges) > self.config.max_edges:
            return False
        if not pack.operator_stats:
            # Must have at least empty stats, none implies malformed
            return False
        return True

    def validate_edge_delta(self, delta: EdgeDelta, trust: float) -> bool:
        """
        Validate an EdgeDelta.

        Args:
            delta: The EdgeDelta to validate.
            trust: The sender's trust score.

        Returns:
            True if valid, False otherwise.
        """
        if trust < self.config.trust_min:
            return False
        if len(delta.upserts) + len(delta.prunes) > self.config.max_edges:
            return False
        return True

    def validate_policy_pack(self, pack: PolicyPack, trust: float) -> bool:
        """
        Validate a PolicyPack.

        Args:
            pack: The PolicyPack to validate.
            trust: The sender's trust score.

        Returns:
            True if valid, False otherwise.
        """
        if not self.config.allow_policy_share:
            return False
        if trust < self.config.trust_min:
            return False
        return True

    def validate_causal_pack(self, pack: CausalSemanticPack, trust: float) -> bool:
        """
        Validate a CausalSemanticPack.

        Args:
            pack: The CausalSemanticPack to validate.
            trust: The sender's trust score.

        Returns:
            True if valid, False otherwise.
        """
        if trust < self.config.trust_min:
            return False
        if len(pack.concepts) > self.config.max_concepts:
            return False
        return True

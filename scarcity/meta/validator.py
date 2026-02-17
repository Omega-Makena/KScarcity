"""
Meta packet validation utilities.

This module validates incoming meta-updates from domains to ensure they meet
quality and safety standards before being aggregated. It checks for sufficient
confidence, reasonable vector sizes, and finite values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Optional

import numpy as np

from .domain_meta import DomainMetaUpdate


@dataclass
class MetaValidatorConfig:
    """Configuration for the MetaPacketValidator."""
    min_confidence: float = 0.1
    max_keys: int = 32
    max_score_delta: float = 1.0


class MetaPacketValidator:
    """
    Validates meta-update packets.
    """

    def __init__(self, config: Optional[MetaValidatorConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Configuration object. Defaults to default settings.
        """
        self.config = config or MetaValidatorConfig()

    def validate_update(self, update: DomainMetaUpdate) -> bool:
        """
        Check if a given update is valid.

        Args:
            update: The DomainMetaUpdate object to validate.

        Returns:
            True if validity checks pass, False otherwise.
        """
        cfg = self.config
        if update.confidence < cfg.min_confidence:
            return False
        if len(update.keys) > cfg.max_keys:
            return False
        if abs(update.score_delta) > cfg.max_score_delta:
            return False
        if not all(np.isfinite(update.vector)):
            return False
        return True

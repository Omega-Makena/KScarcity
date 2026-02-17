"""
Privacy and secure aggregation utilities.

This module implements privacy-preserving mechanisms for federated learning,
including Differential Privacy (DP) noise injection and Secure Aggregation (SA)
masking to protect individual client updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math
import secrets
import numpy as np


@dataclass
class PrivacyConfig:
    """Configuration for PrivacyGuard."""
    secure_aggregation: bool = True
    dp_noise_sigma: float = 0.0
    dp_epsilon: float = 0.0
    dp_delta: float = 0.0
    dp_sensitivity: float = 1.0
    dp_noise_type: str = "gaussian"  # gaussian | laplace
    seed_length: int = 16


class PrivacyGuard:
    """
    Applies differential privacy noise and simple secure aggregation masking.

    Secure aggregation is approximated by generating common masks that can be
    cancelled by the aggregator; a production deployment should integrate with
    a full SA protocol.
    """

    def __init__(self, config: PrivacyConfig):
        """
        Initialize the PrivacyGuard.

        Args:
            config: Privacy configuration object.
        """
        self.config = config

    def apply_noise(self, values: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Apply Differential Privacy noise to a batch of values.

        Args:
            values: Input sequence of value vectors.

        Returns:
            Numpy array with added Gaussian noise (if configured).
        """
        array = np.asarray(values, dtype=np.float32)
        noise_type = self.config.dp_noise_type.lower()
        sigma = self._resolve_sigma(noise_type)
        if sigma <= 0:
            return array
        if noise_type == "laplace":
            noise = np.random.laplace(0.0, sigma, size=array.shape)
        else:
            noise = np.random.normal(0.0, sigma, size=array.shape)
        return array + noise.astype(np.float32)

    def _resolve_sigma(self, noise_type: str) -> float:
        """Resolve Gaussian noise sigma from config or epsilon/delta."""
        if self.config.dp_noise_sigma > 0:
            return self.config.dp_noise_sigma
        if noise_type == "laplace" and self.config.dp_epsilon > 0:
            return self.config.dp_sensitivity / self.config.dp_epsilon
        if self.config.dp_epsilon > 0 and self.config.dp_delta > 0:
            c = math.sqrt(2 * math.log(1.25 / self.config.dp_delta))
            return self.config.dp_sensitivity * c / self.config.dp_epsilon
        return 0.0

    def secure_mask(self, array: np.ndarray) -> Tuple[np.ndarray, bytes]:
        """
        Apply a secure aggregation mask to an array.

        Args:
            array: Input numpy array.

        Returns:
            Tuple containing:
            - Masked array.
            - The random seed used to generate the mask.
        """
        if not self.config.secure_aggregation:
            return array, b""
        mask_seed = secrets.token_bytes(self.config.seed_length)
        rng = np.random.default_rng(int.from_bytes(mask_seed, "big", signed=False))
        mask = rng.normal(0.0, 1.0, size=array.shape).astype(np.float32)
        return array + mask, mask_seed

    def unmask(self, masked_sum: np.ndarray, seeds: Iterable[bytes], count: int) -> np.ndarray:
        """
        Remove the combined mask using provided seeds.
        
        This assumes the 'masked_sum' contains the sum of multiple masked updates.
        By regenerating the masks from the seeds and subtracting them, the true sum
        is revealed.

        Args:
            masked_sum: The sum of masked arrays.
            seeds: Iterable of seeds used for masking.
            count: Number of updates aggregated (used for validation/scaling logic if needed).

        Returns:
            The unmasked sum array.
        """
        if not self.config.secure_aggregation or count == 0:
            return masked_sum
        combined_mask = np.zeros_like(masked_sum, dtype=np.float32)
        for seed in seeds:
            rng = np.random.default_rng(int.from_bytes(seed, "big", signed=False))
            combined_mask += rng.normal(0.0, 1.0, size=masked_sum.shape).astype(np.float32)
        return masked_sum - combined_mask

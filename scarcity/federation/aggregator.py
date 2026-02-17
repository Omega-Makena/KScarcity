"""
Robust aggregation utilities for SCARCITY federation.

This module provides aggregation methods (e.g., Trimmed Mean, Median, Krum, Bulyan)
to combine updates from multiple federated clients while resisting poisoning attacks
and Byzantine failures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class AggregationMethod(str, Enum):
    """Supported aggregation methods."""
    FEDAVG = "fedavg"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    BULYAN = "bulyan"


@dataclass
class AggregationConfig:
    """Configuration for FederatedAggregator."""
    method: AggregationMethod = AggregationMethod.TRIMMED_MEAN
    trim_alpha: float = 0.1
    multi_krum_m: int = 5
    trust_min: float = 0.2
    adaptive_metric_is_loss: bool = True


def _parse_updates(
    updates: Sequence[Sequence[float] | tuple | dict],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    vectors: List[np.ndarray] = []
    weights: List[Optional[float]] = []
    metrics: List[Optional[float]] = []
    for item in updates:
        weight = None
        metric = None
        if isinstance(item, dict):
            vec = item.get("vector") or item.get("update") or item.get("values")
            weight = item.get("weight") or item.get("n_samples") or item.get("count")
            metric = item.get("loss")
            if metric is None:
                metric = item.get("metric") or item.get("score") or item.get("gain")
        elif isinstance(item, tuple):
            vec = item[0]
            if len(item) > 1:
                weight = item[1]
            if len(item) > 2:
                metric = item[2]
        else:
            vec = item
        vectors.append(np.asarray(vec, dtype=np.float32))
        weights.append(float(weight) if weight is not None else None)
        metrics.append(float(metric) if metric is not None else None)

    shapes = {vec.shape for vec in vectors}
    if len(shapes) > 1:
        raise ValueError("Updates must share the same shape")

    array = np.stack(vectors, axis=0)
    weight_array = None
    if any(w is not None for w in weights):
        weight_array = np.array([w if w is not None else 0.0 for w in weights], dtype=np.float32)
    metric_array = None
    if any(m is not None for m in metrics):
        metric_array = np.array([m if m is not None else np.nan for m in metrics], dtype=np.float32)
    return array, weight_array, metric_array


def _weighted_mean(array: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        return np.mean(array, axis=0)
    total = float(np.sum(weights))
    if total <= 0:
        return np.mean(array, axis=0)
    norm = weights / total
    return np.sum(array * norm[:, None], axis=0)


def _adaptive_mean(array: np.ndarray, metrics: Optional[np.ndarray], is_loss: bool) -> np.ndarray:
    if metrics is None:
        return np.mean(array, axis=0)
    eps = 1e-6
    weights = []
    for m in metrics:
        if not np.isfinite(m):
            weights.append(0.0)
            continue
        if is_loss:
            weights.append(1.0 / max(m, eps))
        else:
            weights.append(max(m, 0.0))
    return _weighted_mean(array, np.array(weights, dtype=np.float32))


def _stack_updates(updates: Sequence[Sequence[float]]) -> np.ndarray:
    """Stack a sequence of updates into a numpy array."""
    if not updates:
        raise ValueError("No updates supplied")
    array = np.asarray(updates, dtype=np.float32)
    if len({arr.shape for arr in array}) > 1:
        raise ValueError("Updates must share the same shape")
    return array


def _pairwise_distances(array: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between row vectors."""
    norms = np.sum(array ** 2, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * array @ array.T
    np.maximum(distances, 0.0, out=distances)
    return distances


def _trimmed_mean(array: np.ndarray, alpha: float) -> np.ndarray:
    """Compute coordinate-wise trimmed mean."""
    if array.shape[0] == 1:
        return array[0]

    total = array.shape[0]
    trim_count = int(round(alpha * total))
    if alpha > 0.0 and trim_count == 0:
        trim_count = 1
    trim_count = min(trim_count, total - 2)

    if trim_count <= 0:
        return np.mean(array, axis=0)

    median = np.median(array, axis=0)
    distances = np.linalg.norm(array - median, axis=1)
    keep = np.argsort(distances)[: total - trim_count]
    trimmed = array[keep]
    return np.mean(trimmed, axis=0)


def _krum_select(array: np.ndarray, m: int) -> Tuple[np.ndarray, List[int]]:
    """Select m vectors that minimize the sum of distances to their nearest neighbors (Krum)."""
    distances = _pairwise_distances(array)
    n = array.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    neighbours = max(1, min(n - 2, n - m - 2))
    for i in range(n):
        sorted_indices = np.argsort(distances[i])
        scores[i] = np.sum(distances[i, sorted_indices[1 : neighbours + 1]])
    selected = np.argsort(scores)[:m]
    return array[selected], selected.tolist()


def _bulyan(array: np.ndarray, m: int, alpha: float) -> np.ndarray:
    """Bulyan aggregation: Krum selection followed by Trimmed Mean."""
    if array.shape[0] <= m:
        return np.mean(array, axis=0)
    selected, _ = _krum_select(array, m)
    return _trimmed_mean(selected, alpha)


class FederatedAggregator:
    """
    Entry-point for federated aggregation with fallback logic.
    
    Routes aggregation requests to the configured method (Median, Trimmed Mean, Krum, etc.)
    and returns the aggregated result along with metadata.
    """

    def __init__(self, config: AggregationConfig):
        """
        Initialize the aggregator.

        Args:
            config: Aggregation configuration.
        """
        self.config = config

    def aggregate(self, updates: Sequence[Sequence[float]]) -> Tuple[np.ndarray, dict]:
        """
        Aggregate a list of updates.

        Args:
            updates: A sequence of update vectors.

        Returns:
            A tuple containing:
            - The aggregated vector (np.ndarray).
            - Metadata dictionary (e.g., method used, participant count).
        """
        array, weights, metrics = _parse_updates(updates)
        method = self.config.method
        meta: dict = {"method": method.value, "participants": array.shape[0]}

        if method == AggregationMethod.FEDAVG:
            meta["strategy"] = "mean"
            return np.mean(array, axis=0), meta

        if method == AggregationMethod.WEIGHTED:
            meta["strategy"] = "weighted"
            return _weighted_mean(array, weights), meta

        if method == AggregationMethod.ADAPTIVE:
            meta["strategy"] = "adaptive"
            meta["adaptive_metric_is_loss"] = self.config.adaptive_metric_is_loss
            return _adaptive_mean(array, metrics, self.config.adaptive_metric_is_loss), meta

        if method == AggregationMethod.MEDIAN:
            meta["trim_alpha"] = 0.0
            return np.median(array, axis=0), meta

        if method == AggregationMethod.TRIMMED_MEAN:
            meta["trim_alpha"] = self.config.trim_alpha
            return _trimmed_mean(array, self.config.trim_alpha), meta

        if method == AggregationMethod.KRUM:
            selected, indices = _krum_select(array, 1)
            meta["selected"] = indices
            return selected.mean(axis=0), meta

        if method == AggregationMethod.MULTI_KRUM:
            m = min(self.config.multi_krum_m, array.shape[0])
            selected, indices = _krum_select(array, m)
            meta["selected"] = indices
            return np.mean(selected, axis=0), meta

        if method == AggregationMethod.BULYAN:
            m = min(self.config.multi_krum_m, array.shape[0])
            meta["selected_size"] = m
            meta["trim_alpha"] = self.config.trim_alpha
            return _bulyan(array, m, self.config.trim_alpha), meta

        raise ValueError(f"Unsupported aggregation method: {method}")

    @staticmethod
    def detect_outliers(updates: Sequence[Sequence[float]], reference: Sequence[float], z_thresh: float = 4.0) -> List[int]:
        """
        Return indices of updates that diverge strongly from the aggregate reference.
        
        Args:
            updates: List of update vectors.
            reference: Reference vector (e.g., the aggregate).
            z_thresh: Z-score threshold for outlier detection.
            
        Returns:
            List of indices of detected outliers.
        """
        array = _stack_updates(updates)
        ref = np.asarray(reference, dtype=np.float32)
        diff_norms = np.linalg.norm(array - ref, axis=1)
        mean = float(np.mean(diff_norms))
        std = float(np.std(diff_norms) + 1e-6)
        z_scores = (diff_norms - mean) / std
        return [i for i, z in enumerate(z_scores) if z > z_thresh]

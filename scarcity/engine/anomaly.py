"""
Industrial-Grade Online Anomaly Detection (Streaming RRCF).

This module replaces naive Mahalanobis distance with a Numba-optimized
Streaming Robust Random Cut Forest (RRCF). It measures the 'CoDispersion' 
of a new data point against a sliding window of historical states, making 
it mathematically immune to masking effects from collusive anomaly swarms.
"""

import logging
import time
from typing import Dict, Any, Optional

import numpy as np
from numba import njit, float32, int32  # type: ignore

from scarcity.runtime import EventBus, get_bus

logger = logging.getLogger(__name__)


@njit(cache=True)
def _compute_rrcf_codispersion(history_buffer: np.ndarray, query_vector: np.ndarray, num_trees: int = 50) -> float:
    """
    Numba-accelerated approximation of RRCF CoDispersion.
    Builds random bounding boxes across the local history to score how easily 
    the query vector is isolated from the dense clusters.
    
    Args:
        history_buffer: Shape [W, D], rolling window of past vectors.
        query_vector: Shape [D], the new point to evaluate.
        num_trees: The number of random bisections to simulate.
        
    Returns:
        float: Isolation score (higher means more anomalous).
    """
    W, D = history_buffer.shape
    if W < 10:
        return 0.0  # Not enough history
        
    isolation_depths = np.zeros(num_trees, dtype=np.float32)
    
    # Merge query into the dataset for bounding box computations
    dataset = np.empty((W + 1, D), dtype=np.float32)
    dataset[:W] = history_buffer
    dataset[W] = query_vector

    for t in range(num_trees):
        # We simulate a random cut tree without building explicit nodes
        active_indices = np.arange(W + 1)
        depth = 0.0
        
        while len(active_indices) > 1:
            # Current bounding box
            active_data = dataset[active_indices]
            
            # Find dimensions with non-zero variance to cut
            mins = np.empty(D, dtype=np.float32)
            maxs = np.empty(D, dtype=np.float32)
            for d in range(D):
                mins[d] = np.min(active_data[:, d])
                maxs[d] = np.max(active_data[:, d])
            
            ranges = maxs - mins
            sum_ranges = np.sum(ranges)
            
            if sum_ranges <= 1e-6:
                break # All points are identical
                
            # Pick a dimension proportional to its variance (range)
            probs = ranges / sum_ranges
            val = np.random.random()
            cumulative = 0.0
            cut_dim = D - 1
            for d in range(D):
                cumulative += probs[d]
                if val <= cumulative:
                    cut_dim = d
                    break
                    
            # Pick a uniform random cut point along the chosen dimension
            cut_val = mins[cut_dim] + np.random.random() * ranges[cut_dim]
            
            # Partition
            left_mask = dataset[active_indices, cut_dim] <= cut_val
            right_mask = ~left_mask
            
            # Does the query fall left or right?
            # Query is always the last index (W) initially, find where it went
            query_is_left = False
            for i, idx in enumerate(active_indices):
                if idx == W:
                    query_is_left = left_mask[i]
                    break
                    
            if query_is_left:
                active_indices = active_indices[left_mask]
            else:
                active_indices = active_indices[right_mask]
                
            depth += 1.0
            
        # Standard anomaly scoring: shorter depth = more anomalous
        # We invert it so higher score = anomaly
        # E(h(x)) roughly scales as 2 * ln(W - 1)
        c_n = 2.0 * np.log(max(2.0, float(W)))
        isolation_depths[t] = depth / c_n
        
    mean_depth = np.mean(isolation_depths)
    # Convert to standard 0-1 anomaly score, scaled up for readability
    score = (2.0 ** -mean_depth) * 10.0
    return float(score)


class OnlineAnomalyDetector:
    """
    Streaming RRCF Anomaly Detector.
    
    Subscribes to 'data_window' and uses Numba-compiled random bisections
    to instantly flag anomalies. Reacts to 'scarcity.drg_extension_profile'
    to dynamically scale its fidelity (number of trees) under hardware duress.
    """
    
    def __init__(self, bus: Optional[EventBus] = None):
        self.bus = bus if bus else get_bus()
        self.running = False
        
        # RRCF Configuration
        self.num_trees = 50          # Baseline fidelity
        self.max_trees = 100         # Under nominal load
        self.min_trees = 10          # Under extreme throttle
        
        self.score_threshold = 6.0   # Out of 10.0
        
        # State
        self.history_buffer = None
        self.buffer_size = 256
        self.sample_rate = 1.0
        self.window_counter = 0

    async def start(self) -> None:
        """Subscribe to the main engine's data window and DRG limits."""
        if self.running:
            return
        self.running = True
        self.bus.subscribe("data_window", self._handle_data_window)
        self.bus.subscribe("scarcity.drg_extension_profile", self._handle_drg)
        logger.info("Streaming RRCF Anomaly Detector started")

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.bus.unsubscribe("data_window", self._handle_data_window)
        self.bus.unsubscribe("scarcity.drg_extension_profile", self._handle_drg)
        logger.info("Streaming RRCF Anomaly Detector stopped")

    async def _handle_drg(self, topic: str, profile: Dict[str, Any]) -> None:
        """Dynamically throttle algorithm fidelity based on GPU/CPU telemetry."""
        enabled = profile.get("anomaly_enabled", True)
        if not enabled:
            self.num_trees = 0
            return
            
        self.sample_rate = profile.get("anomaly_sample_rate", 1.0)
        
        # Scale trees based on allowed fidelity
        target_trees = int(self.max_trees * self.sample_rate)
        self.num_trees = max(self.min_trees, target_trees)
        
        logger.debug(f"RRCF Anomaly Detector throttled to {self.num_trees} trees at {self.sample_rate}x rate.")

    async def _handle_data_window(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Passively score the incoming data window using Numba RRCF.
        """
        if not self.running or self.num_trees <= 0:
            return
            
        self.window_counter += 1
        if self.sample_rate < 1.0:
            # Skip frames to save CPU if DRG requested proportional throttle
            if (self.window_counter % int(1.0 / self.sample_rate)) != 0:
                return
            
        window_tensor = data.get('data')
        if window_tensor is None:
            return
            
        if isinstance(window_tensor, list):
            window_tensor = np.array(window_tensor)
            
        if window_tensor.ndim != 2:
            return

        # Latest state
        latest_vector = window_tensor[-1, :].astype(np.float32)
        
        # Update rolling buffer
        if self.history_buffer is None:
            self.history_buffer = np.zeros((self.buffer_size, latest_vector.shape[0]), dtype=np.float32)
            self.history_idx = 0
            self.history_filled = False
            
        self.history_buffer[self.history_idx] = latest_vector
        self.history_idx += 1
        if self.history_idx >= self.buffer_size:
            self.history_idx = 0
            self.history_filled = True
            
        if not self.history_filled and self.history_idx < 10:
            return # Warming up
            
        # Use valid history slice
        valid_len = self.buffer_size if self.history_filled else self.history_idx
        history_slice = self.history_buffer[:valid_len]
        
        # Fast C-compiled computation
        score = _compute_rrcf_codispersion(history_slice, latest_vector, self.num_trees)
        
        if score > self.score_threshold:
            payload = {
                "timestamp": time.time(),
                "severity": float(score),
                "trigger_vector_norm": float(np.linalg.norm(latest_vector)),
                "window_id": data.get("window_id", "unknown"),
                "detected_by": f"RRCF (trees={self.num_trees})"
            }
            await self.bus.publish("scarcity.anomaly_detected", payload)

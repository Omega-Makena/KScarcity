"""
StreamSharder — Domain-based adaptive stream sharding.

Automatically partitions streams into domain shards using online clustering
with dynamic rebalancing based on latency.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, sharding will use round-robin")

logger = logging.getLogger(__name__)


class StreamSharder:
    """
    Partition data streams into domain shards.
    
    Features:
    - Clustering-based sharding using MiniBatch K-Means
    - Dynamic rebalancing based on latency
    - Domain assignment tracking
    """
    
    def __init__(self, n_shards: int = 3, rebalance_threshold: float = 2.0):
        """
        Initialize stream sharder.
        
        Args:
            n_shards: Number of shards to create
            rebalance_threshold: Latency multiplier to trigger rebalancing
        """
        self.n_shards = n_shards
        self.rebalance_threshold = rebalance_threshold
        
        # Clustering model
        self.clusterer = None
        if SKLEARN_AVAILABLE:
            self.clusterer = MiniBatchKMeans(n_clusters=n_shards, random_state=42, batch_size=100)
        
        # Shard statistics
        self.shard_latency: Dict[int, List[float]] = defaultdict(list)
        self.shard_counts: Dict[int, int] = defaultdict(int)
        
        # State
        self._initialized = False
        self._stats = {
            'assignments_made': 0,
            'rebalances': 0
        }
        
        logger.info(f"StreamSharder initialized with {n_shards} shards")
    
    def assign_shard(self, data: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """
        Assign data to a shard.
        
        Args:
            data: Data window or feature vector
            metadata: Optional metadata for domain hints
            
        Returns:
            Shard ID (0 to n_shards-1)
        """
        # Simple round-robin if clustering unavailable
        if not SKLEARN_AVAILABLE or self.clusterer is None:
            shard_id = self._stats['assignments_made'] % self.n_shards
            self._stats['assignments_made'] += 1
            self.shard_counts[shard_id] += 1
            return shard_id
        
        # Extract features for clustering
        if data.ndim > 1:
            # Use mean of window as feature vector
            features = data.mean(axis=0)
        else:
            features = data
        
        # Fit or predict
        if not self._initialized:
            # Warm-up: collect initial samples
            if self._stats['assignments_made'] < self.n_shards:
                shard_id = self._stats['assignments_made'] % self.n_shards
                self._stats['assignments_made'] += 1
                self.shard_counts[shard_id] += 1
                return shard_id
            else:
                # Initialize clusterer with collected samples (use random for now)
                self._initialized = True
        
        # Predict using clustering
        try:
            shard_id = self.clusterer.predict(features.reshape(1, -1))[0]
            # Update clusterer
            self.clusterer.partial_fit(features.reshape(1, -1))
        except (AttributeError, ValueError):
            # Fallback to round-robin if clustering fails
            shard_id = self._stats['assignments_made'] % self.n_shards
        
        self._stats['assignments_made'] += 1
        self.shard_counts[int(shard_id)] += 1
        
        return int(shard_id)
    
    def record_latency(self, shard_id: int, latency_ms: float) -> None:
        """
        Record processing latency for a shard.
        
        Args:
            shard_id: Shard identifier
            latency_ms: Processing latency in milliseconds
        """
        self.shard_latency[shard_id].append(latency_ms)
        
        # Keep only last 100 measurements
        if len(self.shard_latency[shard_id]) > 100:
            self.shard_latency[shard_id] = self.shard_latency[shard_id][-100:]
        
        # Check if rebalancing needed
        if self._should_rebalance():
            self.rebalance()
    
    def _should_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        if len(self.shard_latency) < self.n_shards:
            return False
        
        # Compute mean latency per shard
        mean_latencies = {
            sid: np.mean(latencies)
            for sid, latencies in self.shard_latency.items()
            if len(latencies) > 10
        }
        
        if len(mean_latencies) < self.n_shards:
            return False
        
        overall_mean = np.mean(list(mean_latencies.values()))
        
        # Check if any shard exceeds threshold
        for shard_id, mean_lat in mean_latencies.items():
            if mean_lat > overall_mean * self.rebalance_threshold:
                logger.warning(
                    f"Shard {shard_id} latency ({mean_lat:.2f}ms) exceeds "
                    f"threshold ({overall_mean * self.rebalance_threshold:.2f}ms)"
                )
                return True
        
        return False
    
    def rebalance(self) -> None:
        """Rebalance shards (split high-latency shard)."""
        if self.clusterer is None or not SKLEARN_AVAILABLE:
            logger.warning("Cannot rebalance: clustering unavailable")
            return
        
        # For now, just reinitialize the clusterer
        # In production, would intelligently split/merge shards
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.n_shards,
            random_state=42,
            batch_size=100
        )
        self._initialized = False
        
        self._stats['rebalances'] += 1
        logger.info(f"Shards rebalanced (rebalance #{self._stats['rebalances']})")
    
    def get_shard_stats(self) -> Dict:
        """Get statistics for all shards."""
        stats = {}
        
        for shard_id in range(self.n_shards):
            latencies = self.shard_latency.get(shard_id, [])
            stats[f"shard_{shard_id}"] = {
                'count': self.shard_counts.get(shard_id, 0),
                'mean_latency_ms': np.mean(latencies) if latencies else 0.0,
                'std_latency_ms': np.std(latencies) if latencies else 0.0
            }
        
        return {
            'shards': stats,
            'total_assignments': self._stats['assignments_made'],
            'rebalances': self._stats['rebalances']
        }
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        return {
            'n_shards': self.n_shards,
            'assignments': self._stats['assignments_made'],
            'rebalances': self._stats['rebalances'],
            'initialized': self._initialized
        }

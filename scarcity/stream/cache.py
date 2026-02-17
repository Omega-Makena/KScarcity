"""
CacheManager — Adaptive LRU cache for data windows.

Maintains in-memory cache with temporal decay for recently processed windows.
"""

import logging
import time
import numpy as np
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Adaptive LRU cache with temporal decay.
    
    Features:
    - LRU eviction policy
    - Temporal decay weighting
    - Configurable max size
    - Hit/miss statistics
    """
    
    def __init__(self, max_size: int = 1000, decay_factor: float = 0.01):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of items in cache
            decay_factor: Temporal decay factor (lambda)
        """
        self.max_size = max_size
        self.decay_factor = decay_factor
        self.cache = OrderedDict()  # LRU ordered
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'insertions': 0
        }
        
        logger.info(f"CacheManager initialized with max_size={max_size}")
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self._stats['hits'] += 1
            return self.cache[key]['data']
        
        self._stats['misses'] += 1
        return None
    
    def put(self, key: str, data: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """
        Insert item into cache.
        
        Args:
            key: Cache key
            data: Data to cache
            metadata: Optional metadata
        """
        # Check if key already exists
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict()
        
        # Insert new item
        self.cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self._stats['insertions'] += 1
    
    def _evict(self) -> None:
        """Evict least recently used item."""
        if self.cache:
            self.cache.popitem(last=False)  # Remove oldest
            self._stats['evictions'] += 1
    
    def get_weight(self, key: str) -> float:
        """
        Get temporal weight for cached item.
        
        Args:
            key: Cache key
            
        Returns:
            Weight based on temporal decay
        """
        if key not in self.cache:
            return 0.0
        
        item = self.cache[key]
        age_seconds = time.time() - item['timestamp']
        weight = np.exp(-self.decay_factor * age_seconds)
        return weight
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_ratio = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_ratio': hit_ratio,
            'evictions': self._stats['evictions'],
            'insertions': self._stats['insertions']
        }
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

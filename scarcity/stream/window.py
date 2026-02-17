"""
WindowBuilder — Online windowing and normalization.

Transforms raw sequential data into normalized overlapping windows using
Welford's algorithm for online statistics and EMA smoothing.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

logger = logging.getLogger(__name__)


class WelfordStats:
    """
    Welford's online algorithm for computing mean and variance incrementally.
    
    Updates mean and variance in O(1) per sample without storing all values.
    """
    
    def __init__(self, n_features: int):
        """
        Initialize statistics tracker.
        
        Args:
            n_features: Number of features to track
        """
        self.n_features = n_features
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.var = np.zeros(n_features, dtype=np.float64)
        self.count = 0
    
    def update(self, x: np.ndarray) -> None:
        """
        Update statistics with new sample(s).
        
        Args:
            x: Array of shape (n_samples, n_features) or (n_features,)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        for sample in x:
            self.count += 1
            delta = sample - self.mean
            self.mean += delta / self.count
            self.var += delta * (sample - self.mean)
    
    def get_std(self) -> np.ndarray:
        """
        Get standard deviation.
        
        Returns:
            Standard deviation for each feature
        """
        if self.count < 2:
            return np.ones(self.n_features, dtype=np.float64)
        return np.sqrt(np.maximum(self.var / (self.count - 1), 1e-8))


class EMASmoother:
    """
    Exponential Moving Average for noise reduction.
    """
    
    def __init__(self, alpha: float = 0.3, n_features: int = 1):
        """
        Initialize EMA smoother.
        
        Args:
            alpha: Smoothing factor (0-1), lower = more smoothing
            n_features: Number of features to smooth
        """
        self.alpha = alpha
        self.value = np.zeros(n_features, dtype=np.float64)
        self.initialized = False
    
    def smooth(self, x: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing.
        
        Args:
            x: Input values
            
        Returns:
            Smoothed values
        """
        if not self.initialized:
            self.value = x.copy()
            self.initialized = True
            return self.value
        
        self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class WindowBuilder:
    """
    Build normalized overlapping windows from raw data.
    
    Features:
    - Rolling window slicing
    - Online normalization using Welford's algorithm
    - EMA smoothing for noise
    - LOCF / interpolation for missing data
    - Adaptive window size based on feedback
    """
    
    def __init__(
        self,
        window_size: int = 2048,
        stride: int = 1024,
        normalization: str = "z-score",
        ema_alpha: float = 0.3,
        fill_method: str = "locf"
    ):
        """
        Initialize window builder.
        
        Args:
            window_size: Number of samples per window
            stride: Window overlap (step size)
            normalization: Type of normalization ('z-score', 'min-max', 'none')
            ema_alpha: EMA smoothing factor
            fill_method: Method for handling missing data ('locf', 'linear')
        """
        self.window_size = window_size
        self.stride = stride
        self.normalization = normalization
        self.fill_method = fill_method
        self.ema_alpha = ema_alpha
        
        # Statistics tracking
        self.welford_stats: Optional[WelfordStats] = None
        self.ema_smoother: Optional[EMASmoother] = None
        self.min_values = None
        self.max_values = None
        
        # Buffer for rolling windows
        self.buffer = deque(maxlen=window_size * 2)
        
        # State
        self.window_count = 0
        self.last_window_time = 0.0
        self._stats = {
            'windows_created': 0,
            'normalization_time_ms': 0.0,
            'empty_windows': 0
        }
        
        logger.info(
            f"WindowBuilder initialized: window_size={window_size}, "
            f"stride={stride}, normalization={normalization}"
        )
    
    def process_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Process a data chunk and generate windows.
        
        Args:
            chunk: Raw data array of shape (n_samples, n_features)
            
        Returns:
            List of normalized windows
        """
        if chunk is None or len(chunk) == 0:
            return []
        
        # Initialize statistics on first chunk
        n_features = chunk.shape[1] if chunk.ndim > 1 else 1
        if self.welford_stats is None:
            self.welford_stats = WelfordStats(n_features)
            self.ema_smoother = EMASmoother(alpha=self.ema_alpha, n_features=n_features)
        
        # Add chunk to buffer
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        self.buffer.extend(chunk)
        
        # Extract windows
        windows = self._extract_windows()
        
        # Update statistics with new chunk
        self.welford_stats.update(chunk)
        
        return windows
    
    def _extract_windows(self) -> List[np.ndarray]:
        """
        Extract rolling windows from buffer.
        
        Returns:
             List of numpy arrays representing the windows
        """
        if len(self.buffer) < self.window_size:
            return []
        
        windows = []
        buffer_array = np.array(self.buffer)
        
        # Extract overlapping windows
        for i in range(0, len(buffer_array) - self.window_size + 1, self.stride):
            window = buffer_array[i:i + self.window_size].copy()
            
            # Handle missing data
            window = self._handle_missing_data(window)
            
            # Normalize
            window = self._normalize(window)
            
            windows.append(window)
            self.window_count += 1
            self._stats['windows_created'] += 1
            
        self.last_window_time = time.time()
        return windows
    
    def _handle_missing_data(self, window: np.ndarray) -> np.ndarray:
        """
        Handle missing values using specified method.
        
        Args:
            window: Data window
            
        Returns:
            Window with missing data filled
        """
        if np.isnan(window).any():
            if self.fill_method == "locf":
                # Last Observed Carried Forward
                window = self._apply_locf(window)
            elif self.fill_method == "linear":
                # Linear interpolation
                window = self._apply_interpolation(window)
        
        return window
    
    def _apply_locf(self, window: np.ndarray) -> np.ndarray:
        """Apply Last Observed Carried Forward."""
        for i in range(1, len(window)):
            mask = np.isnan(window[i])
            window[i][mask] = window[i-1][mask]
        
        # Forward fill any remaining NaNs at the start
        for i in range(len(window) - 2, -1, -1):
            mask = np.isnan(window[i])
            window[i][mask] = window[i+1][mask]
        
        return window
    
    def _apply_interpolation(self, window: np.ndarray) -> np.ndarray:
        """Apply linear interpolation."""
        # For each feature
        for feature_idx in range(window.shape[1]):
            values = window[:, feature_idx]
            if np.isnan(values).any():
                # Use numpy's interpolation
                valid_mask = ~np.isnan(values)
                if valid_mask.any():
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = values[valid_indices]
                    interp_values = np.interp(
                        np.arange(len(values)),
                        valid_indices,
                        valid_values
                    )
                    window[:, feature_idx] = interp_values
                else:
                    # All NaN - fill with zeros
                    window[:, feature_idx] = 0.0
        
        return window
    
    def _normalize(self, window: np.ndarray) -> np.ndarray:
        """
        Normalize window using configured method.
        
        Args:
            window: Data window
            
        Returns:
            Normalized window
        """
        if self.normalization == "none":
            return window
        elif self.normalization == "z-score":
            if self.welford_stats and self.welford_stats.count > 1:
                mean = self.welford_stats.mean
                std = self.welford_stats.get_std()
                return (window - mean) / std
            return window
        elif self.normalization == "min-max":
            if self.min_values is not None and self.max_values is not None:
                return (window - self.min_values) / (self.max_values - self.min_values + 1e-8)
            return window
        else:
            logger.warning(f"Unknown normalization method: {self.normalization}")
            return window
    
    def reset_stats(self) -> None:
        """Reset statistics (useful for concept drift scenarios)."""
        if self.welford_stats:
            self.welford_stats = WelfordStats(self.welford_stats.n_features)
        logger.info("WindowBuilder statistics reset")
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'windows_created': self._stats['windows_created'],
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': self.normalization,
            'buffer_size': len(self.buffer),
            'samples_seen': self.welford_stats.count if self.welford_stats else 0
        }
    
    def set_window_size(self, new_size: int) -> None:
        """Adaptively change window size (from governor feedback)."""
        self.window_size = new_size
        self.buffer = deque(maxlen=new_size * 2)
        logger.info(f"Window size adapted to {new_size}")
    
    def set_stride(self, new_stride: int) -> None:
        """Adaptively change stride."""
        self.stride = new_stride
        logger.info(f"Stride adapted to {new_stride}")

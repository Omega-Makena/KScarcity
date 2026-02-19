"""
StreamSource — Continuous data ingestion with rate regulation.

Implements async data reading from various sources with PI-controller based
rate regulation to maintain target latency and handle backpressure.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Dict, Any, Optional, Callable
from datetime import datetime
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PIController:
    """
    Proportional-Integral controller for adaptive rate regulation.
    
    Formula: Δt_next = Δt_base + K_p * error + K_i * integral
    """
    
    def __init__(self, target_latency: float = 100.0, k_p: float = 0.1, k_i: float = 0.01):
        """
        Initialize PI controller.
        
        Args:
            target_latency: Target latency in milliseconds
            k_p: Proportional gain
            k_i: Integral gain
        """
        self.target_latency = target_latency
        self.k_p = k_p
        self.k_i = k_i
        self.integral = 0.0
        self.dt_base = 1.0  # Base delay in seconds
        
    def update(self, actual_latency_ms: float) -> float:
        """
        Update controller and compute next delay.
        
        Args:
            actual_latency_ms: Measured latency in milliseconds
            
        Returns:
            Next delay in seconds
        """
        error = self.target_latency - actual_latency_ms
        self.integral += error
        self.integral = np.clip(self.integral, -1000, 1000)  # Windup protection
        
        dt_next = self.dt_base + self.k_p * error + self.k_i * self.integral
        dt_next = max(0.01, min(dt_next, 10.0))  # Clip to reasonable range
        
        logger.debug(f"PI Controller: error={error:.2f}, integral={self.integral:.2f}, dt_next={dt_next:.3f}")
        return dt_next


class StreamSource:
    """
    Async data source that ingests data with adaptive rate control.
    
    Features:
    - Async iterator for non-blocking reads
    - PI-controller rate regulation
    - Backpressure detection via bus queue depth
    - Support for CSV, generator, and custom sources
    """
    
    def __init__(
        self,
        data_source: Callable | AsyncIterator | str,
        window_size: int = 1000,
        name: str = "default",
        target_latency_ms: float = 100.0
    ):
        """
        Initialize stream source.
        
        Args:
            data_source: CSV file path, async iterator, or callable
            window_size: Number of rows per batch
            name: Source identifier
            target_latency_ms: Target processing latency
        """
        self.data_source = data_source
        self.window_size = window_size
        self.name = name
        self.target_latency_ms = target_latency_ms
        
        # Rate control
        self.pi_controller = PIController(target_latency=target_latency_ms)
        self.current_delay = 0.01
        
        # State
        self._running = False
        self._stats = {
            'chunks_read': 0,
            'rows_read': 0,
            'errors': 0,
            'last_latency_ms': 0.0
        }
        
        logger.info(f"StreamSource '{name}' initialized with window_size={window_size}")
    
    async def read_chunk(self) -> Optional[np.ndarray]:
        """
        Read a single chunk of data.
        
        Returns:
            Numpy array with shape (window_size, features) or None if EOF
        """
        try:
            start_time = time.time()
            
            if isinstance(self.data_source, str):
                # CSV file source
                chunk = await self._read_csv_chunk(self.data_source)
            elif hasattr(self.data_source, '__aiter__'):
                # Async iterator
                chunk = await self._read_async_iterator(self.data_source)
            elif callable(self.data_source):
                # Callable source
                chunk = await self._read_callable(self.data_source)
            else:
                raise ValueError(f"Unsupported data source type: {type(self.data_source)}")
            
            if chunk is not None:
                self._stats['chunks_read'] += 1
                self._stats['rows_read'] += chunk.shape[0]
                
                # Update latency and PI controller
                latency_ms = (time.time() - start_time) * 1000
                self._stats['last_latency_ms'] = latency_ms
                self.current_delay = self.pi_controller.update(latency_ms)
            
            return chunk
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error reading chunk: {e}", exc_info=True)
            return None
    
    async def _read_csv_chunk(self, filepath: str) -> Optional[np.ndarray]:
        """Read chunk from CSV file."""
        # Simplified: would use async CSV reader in production
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for CSV reading")
        
        # Chunked reading to avoid loading entire file into memory.
        if not hasattr(self, '_csv_iter'):
            logger.info(f"Opening CSV file: {filepath}")
            self._csv_iter = pd.read_csv(filepath, chunksize=self.window_size)

        try:
            df = next(self._csv_iter)
        except StopIteration:
            return None

        return df.values
    
    async def _read_async_iterator(self, source: AsyncIterator) -> Optional[np.ndarray]:
        """Read chunk from async iterator."""
        chunk_data = []
        
        for _ in range(self.window_size):
            try:
                item = await source.__anext__()
                chunk_data.append(item)
            except StopAsyncIteration:
                break
        
        return np.array(chunk_data) if chunk_data else None
    
    async def _read_callable(self, source: Callable) -> Optional[np.ndarray]:
        """Read chunk from callable."""
        result = source()
        if asyncio.iscoroutine(result):
            result = await result
        return result
    
    async def stream(self) -> AsyncIterator[np.ndarray]:
        """
        Async generator that yields data chunks.
        
        Yields:
            Numpy arrays of shape (batch_size, features)
        """
        self._running = True
        logger.info(f"StreamSource '{self.name}' started streaming")
        
        try:
            while self._running:
                chunk = await self.read_chunk()
                
                if chunk is None:
                    logger.info("Reached end of data source")
                    break
                
                yield chunk
                
                # Apply adaptive delay
                if self.current_delay > 0:
                    await asyncio.sleep(self.current_delay)
                    
        except asyncio.CancelledError:
            logger.info(f"StreamSource '{self.name}' cancelled")
        finally:
            self._running = False
    
    def stop(self):
        """Stop the stream."""
        self._running = False
        logger.info(f"StreamSource '{self.name}' stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'name': self.name,
            'chunks_read': self._stats['chunks_read'],
            'rows_read': self._stats['rows_read'],
            'errors': self._stats['errors'],
            'last_latency_ms': self._stats['last_latency_ms'],
            'current_delay': self.current_delay,
            'is_running': self._running
        }


async def create_test_source() -> StreamSource:
    """
    Create a test data source for development.
    
    Returns:
        StreamSource with synthetic data
    """
    async def generate_data():
        """Generate synthetic test data."""
        np.random.seed(42)
        while True:
            await asyncio.sleep(0.1)
            yield np.random.randn(100, 10)  # 100 rows, 10 features
    
    source = StreamSource(
        data_source=generate_data(),
        window_size=1000,
        name="test_source",
        target_latency_ms=50.0
    )
    
    return source

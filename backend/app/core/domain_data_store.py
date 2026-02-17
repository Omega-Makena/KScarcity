"""Domain data storage for real-time data visualization."""

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(str, Enum):
    """Data source types."""
    SYNTHETIC = "synthetic"
    MANUAL = "manual"


@dataclass
class DataWindow:
    """Represents a single data window with features and metadata."""
    timestamp: datetime
    domain_id: int
    domain_name: str
    window_id: int
    features: np.ndarray  # Shape: (window_size, num_features)
    scarcity_signal: float
    source: DataSource
    upload_id: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp.isoformat() + "Z",
            "domain_id": self.domain_id,
            "domain_name": self.domain_name,
            "window_id": self.window_id,
            "features_shape": list(self.features.shape),
            "features_sample": self.features[:5].tolist() if len(self.features) > 0 else [],  # First 5 rows
            "scarcity_signal": float(self.scarcity_signal),
            "source": self.source.value,
            "upload_id": self.upload_id
        }


@dataclass
class DomainStatistics:
    """Aggregated statistics for a domain."""
    domain_id: int
    total_windows: int
    synthetic_count: int
    manual_count: int
    avg_scarcity: float
    min_scarcity: float
    max_scarcity: float
    generation_rate: float  # windows per minute
    last_window_at: Optional[datetime]
    first_window_at: Optional[datetime]
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "domain_id": self.domain_id,
            "total_windows": self.total_windows,
            "synthetic_count": self.synthetic_count,
            "manual_count": self.manual_count,
            "avg_scarcity": float(self.avg_scarcity),
            "min_scarcity": float(self.min_scarcity),
            "max_scarcity": float(self.max_scarcity),
            "generation_rate": float(self.generation_rate),
            "last_window_at": self.last_window_at.isoformat() + "Z" if self.last_window_at else None,
            "first_window_at": self.first_window_at.isoformat() + "Z" if self.first_window_at else None
        }


class DomainDataStore:
    """Stores and manages domain data windows in memory."""
    
    def __init__(self, max_windows_per_domain: int = 1000, bus=None):
        """
        Initialize the domain data store.
        
        Args:
            max_windows_per_domain: Maximum windows to store per domain (ring buffer size)
            bus: Event bus for subscribing to data_window events
        """
        self.buffers: Dict[int, deque] = {}  # domain_id -> deque of DataWindow
        self.max_windows = max_windows_per_domain
        self.bus = bus
        self._running = False
        
        logger.info(f"DomainDataStore initialized with max {max_windows_per_domain} windows per domain")
    
    async def start(self):
        """Start the data store and subscribe to event bus."""
        if self._running:
            logger.warning("DomainDataStore already running")
            return
        
        if not self.bus:
            logger.warning("No event bus provided, DomainDataStore will not receive real-time data")
            return
        
        # Subscribe to data_window events
        self.bus.subscribe("data_window", self._on_data_window)
        self._running = True
        
        logger.info("DomainDataStore started and subscribed to data_window events")
    
    async def stop(self):
        """Stop the data store and unsubscribe from event bus."""
        if not self._running:
            return
        
        if self.bus:
            # Note: EventBus doesn't have unsubscribe, so we just stop processing
            pass
        
        self._running = False
        logger.info("DomainDataStore stopped")
    
    async def _on_data_window(self, topic: str, data: dict):
        """
        Handle incoming data window from event bus.
        
        Args:
            topic: Event topic (should be "data_window")
            data: Event payload containing window data
        """
        try:
            # Extract required fields
            domain_id = data.get("domain_id")
            domain_name = data.get("domain_name", f"Domain-{domain_id}")
            window_id = data.get("window_id", 0)
            timestamp_str = data.get("timestamp")
            source = data.get("source", "synthetic")
            
            # Parse timestamp
            if timestamp_str:
                # Remove 'Z' suffix if present and parse
                timestamp_str = timestamp_str.rstrip('Z')
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.utcnow()
            
            # Extract features (numpy array or list)
            features_data = data.get("data")
            if features_data is None:
                features_data = data.get("features", [])
            
            if isinstance(features_data, np.ndarray):
                features = features_data
            elif isinstance(features_data, list):
                features = np.array(features_data)
            else:
                logger.warning(f"Invalid features data type: {type(features_data)}")
                return
            
            # Extract scarcity signal (default to random if not provided)
            scarcity_signal = data.get("scarcity_signal")
            if scarcity_signal is None:
                # Generate random scarcity signal if not provided
                scarcity_signal = float(np.random.uniform(0.0, 1.0))
            else:
                scarcity_signal = float(scarcity_signal)
            
            # Create DataWindow
            window = DataWindow(
                timestamp=timestamp,
                domain_id=domain_id,
                domain_name=domain_name,
                window_id=window_id,
                features=features,
                scarcity_signal=scarcity_signal,
                source=DataSource(source),
                upload_id=data.get("upload_id")
            )
            
            # Store window
            self._store_window(window)
            
            logger.debug(f"Stored window {window_id} for domain {domain_id} ({domain_name})")
            
        except Exception as e:
            logger.error(f"Error processing data window: {e}", exc_info=True)
    
    def _store_window(self, window: DataWindow):
        """
        Store a data window in the appropriate domain buffer.
        
        Args:
            window: DataWindow to store
        """
        domain_id = window.domain_id
        
        # Create buffer for domain if it doesn't exist
        if domain_id not in self.buffers:
            self.buffers[domain_id] = deque(maxlen=self.max_windows)
            logger.info(f"Created buffer for domain {domain_id}")
        
        # Add window to buffer (automatically removes oldest if at max capacity)
        self.buffers[domain_id].append(window)
    
    def get_windows(
        self,
        domain_id: int,
        limit: int = 50,
        offset: int = 0,
        source: Optional[str] = None
    ) -> List[DataWindow]:
        """
        Retrieve windows for domain with pagination and filtering.
        
        Args:
            domain_id: Domain ID to retrieve windows for
            limit: Maximum number of windows to return
            offset: Number of windows to skip (for pagination)
            source: Filter by source type ("synthetic" or "manual")
        
        Returns:
            List of DataWindow objects, sorted by timestamp descending (newest first)
        """
        if domain_id not in self.buffers:
            return []
        
        # Get all windows for domain
        windows = list(self.buffers[domain_id])
        
        # Sort by timestamp descending (newest first)
        windows.sort(key=lambda w: w.timestamp, reverse=True)
        
        # Apply source filter if specified
        if source:
            try:
                source_enum = DataSource(source)
                windows = [w for w in windows if w.source == source_enum]
            except ValueError:
                logger.warning(f"Invalid source filter: {source}")
        
        # Apply pagination
        start = offset
        end = offset + limit
        
        return windows[start:end]
    
    def get_latest_window(self, domain_id: int) -> Optional[DataWindow]:
        """
        Get most recent window for domain.
        
        Args:
            domain_id: Domain ID
        
        Returns:
            Most recent DataWindow or None if no windows exist
        """
        if domain_id not in self.buffers or len(self.buffers[domain_id]) == 0:
            return None
        
        # Get all windows and find the one with latest timestamp
        windows = list(self.buffers[domain_id])
        return max(windows, key=lambda w: w.timestamp)
    
    def get_statistics(self, domain_id: int) -> Optional[DomainStatistics]:
        """
        Compute aggregated statistics for domain.
        
        Args:
            domain_id: Domain ID
        
        Returns:
            DomainStatistics object or None if no windows exist
        """
        if domain_id not in self.buffers or len(self.buffers[domain_id]) == 0:
            return None
        
        windows = list(self.buffers[domain_id])
        
        # Count by source
        synthetic_count = sum(1 for w in windows if w.source == DataSource.SYNTHETIC)
        manual_count = sum(1 for w in windows if w.source == DataSource.MANUAL)
        
        # Scarcity statistics
        scarcity_values = [w.scarcity_signal for w in windows]
        avg_scarcity = float(np.mean(scarcity_values))
        min_scarcity = float(np.min(scarcity_values))
        max_scarcity = float(np.max(scarcity_values))
        
        # Time statistics
        timestamps = [w.timestamp for w in windows]
        first_window_at = min(timestamps)
        last_window_at = max(timestamps)
        
        # Calculate generation rate (windows per minute)
        time_diff = (last_window_at - first_window_at).total_seconds()
        if time_diff > 0:
            generation_rate = (len(windows) - 1) / (time_diff / 60.0)
        else:
            generation_rate = 0.0
        
        return DomainStatistics(
            domain_id=domain_id,
            total_windows=len(windows),
            synthetic_count=synthetic_count,
            manual_count=manual_count,
            avg_scarcity=avg_scarcity,
            min_scarcity=min_scarcity,
            max_scarcity=max_scarcity,
            generation_rate=generation_rate,
            last_window_at=last_window_at,
            first_window_at=first_window_at
        )
    
    def get_total_windows(self) -> int:
        """Get total number of windows across all domains."""
        return sum(len(buffer) for buffer in self.buffers.values())
    
    def get_domain_count(self) -> int:
        """Get number of domains with data."""
        return len(self.buffers)
    
    def clear_domain(self, domain_id: int):
        """Clear all windows for a domain."""
        if domain_id in self.buffers:
            self.buffers[domain_id].clear()
            logger.info(f"Cleared all windows for domain {domain_id}")
    
    def clear_all(self):
        """Clear all windows for all domains."""
        self.buffers.clear()
        logger.info("Cleared all domain data")

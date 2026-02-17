"""
Telemetry & Metrics — real-time monitoring and performance feedback for SCARCITY.

This module continuously collects, aggregates, and publishes system metrics to
enable adaptive control and debugging across all components.

Core algorithms:
- Exponential Moving Average (EMA) for latency tracking
- EWMA for error detection
- Page-Hinkley Test for drift detection
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, Callable
from collections import deque
from datetime import datetime
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, CPU metrics will be disabled")

try:
    import torch
    import pynvml  # NVIDIA Management Library
    TORCH_AVAILABLE = True
    NVML_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    NVML_AVAILABLE = False
    try:
        import torch
        TORCH_AVAILABLE = True
        logging.warning("pynvml not available, basic GPU metrics only")
    except (ImportError, OSError):
        logging.warning("torch not available, GPU metrics will be disabled")

from .bus import EventBus, get_bus

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Track latency using Exponential Moving Average (EMA)."""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize latency tracker.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Lower = more smoothing.
        """
        self.alpha = alpha
        self.latency_ms = 0.0
        self.count = 0
    
    def record(self, duration_ms: float) -> None:
        """
        Record a latency measurement.
        
        Args:
            duration_ms: Latency in milliseconds
        """
        self.count += 1
        if self.count == 1:
            self.latency_ms = duration_ms
        else:
            # EMA update: L_t = α * Δt + (1-α) * L_{t-1}
            self.latency_ms = self.alpha * duration_ms + (1 - self.alpha) * self.latency_ms
    
    def get_latency(self) -> float:
        """Get current EMA latency estimate."""
        return self.latency_ms
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.latency_ms = 0.0
        self.count = 0


class ThroughputCounter:
    """Track throughput using sliding window counting."""
    
    def __init__(self, window_seconds: float = 1.0):
        """
        Initialize throughput counter.
        
        Args:
            window_seconds: Time window for calculating rate
        """
        self.window_seconds = window_seconds
        self.timestamps = deque()
    
    def record_event(self) -> None:
        """Record an event occurrence."""
        now = time.time()
        self.timestamps.append(now)
        # Remove old timestamps outside window
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
    
    def get_rate(self) -> float:
        """
        Get events per second.
        
        Returns:
            Average rate over the window
        """
        return len(self.timestamps) / self.window_seconds


class DriftMonitor:
    """
    Detect data/model drift using simplified Page-Hinkley test.
    
    Monitors a metric for abrupt changes in mean value.
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize drift monitor.
        
        Args:
            threshold: Alert threshold (higher = less sensitive)
        """
        self.threshold = threshold
        self.mean = None
        self.variance = None
        self.count = 0
    
    def update(self, value: float) -> Optional[float]:
        """
        Update monitor with new value and check for drift.
        
        Args:
            value: New measurement
            
        Returns:
            Drift score if detected, None otherwise
        """
        self.count += 1
        
        # Online mean and variance update (Welford's algorithm)
        if self.count == 1:
            self.mean = value
            self.variance = 0.0
            return None
        
        old_mean = self.mean
        self.mean = old_mean + (value - old_mean) / self.count
        self.variance = self.variance + (value - old_mean) * (value - self.mean)
        
        # Calculate z-score
        if self.count > 10 and self.variance > 0:  # Need some data
            std = np.sqrt(self.variance / (self.count - 1))
            z_score = abs(value - self.mean) / std if std > 0 else 0
            
            # Alert if z-score exceeds threshold
            if z_score > self.threshold:
                logger.warning(f"Drift detected: z-score={z_score:.3f}, mean={self.mean:.3f}, value={value:.3f}")
                return z_score
        
        return None


class SystemProbe:
    """Probe system resources (CPU, GPU, memory)."""
    
    def __init__(self):
        """Initialize system probe."""
        self._cpu_util = 0.0
        self._memory_mb = 0.0
        self._gpu_util = 0.0
        self._vram_gb = 0.0
        self._nvml_initialized = False
        
        # Initialize NVML if available
        if 'NVML_AVAILABLE' in globals() and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                logger.debug(f"NVML initialization failed: {e}")
    
    def probe(self) -> Dict[str, float]:
        """
        Probe current system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        metrics = {}
        
        # CPU metrics
        if PSUTIL_AVAILABLE:
            self._cpu_util = psutil.cpu_percent(interval=0.1)
            self._memory_mb = psutil.virtual_memory().used / (1024 ** 2)
            metrics['cpu_percent'] = self._cpu_util
            metrics['memory_mb'] = self._memory_mb
        
        # GPU metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Get VRAM info from PyTorch
                free_vram, total_vram_bytes = torch.cuda.mem_get_info()
                total_vram = total_vram_bytes / (1024 ** 3)
                used_vram = (total_vram_bytes - free_vram) / (1024 ** 3)
                
                self._vram_gb = free_vram / (1024 ** 3)
                metrics['vram_total_gb'] = total_vram
                metrics['vram_used_gb'] = used_vram
                
                # Get GPU utilization from NVML if available
                if self._nvml_initialized:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self._gpu_util = util.gpu / 100.0  # Convert to fraction
                        metrics['gpu_util'] = self._gpu_util * 100  # Report as percentage
                        metrics['gpu_memory_util'] = util.memory / 100.0
                    except Exception as e:
                        logger.debug(f"NVML utilization query failed: {e}")
                        self._gpu_util = 0.0
                        metrics['gpu_util'] = 0.0
                else:
                    # Fallback to PyTorch-only metrics
                    metrics['gpu_util'] = 0.0  # Cannot determine without NVML
                    
                # Additional GPU info
                metrics['gpu_count'] = torch.cuda.device_count()
                metrics['gpu_name'] = torch.cuda.get_device_name(0)
                
            except Exception as e:
                logger.warning(f"GPU metrics collection failed: {e}")
        
        return metrics
    
    def get_last_metrics(self) -> Dict[str, float]:
        """Get last probed metrics."""
        return {
            'cpu_percent': self._cpu_util,
            'memory_mb': self._memory_mb,
            'gpu_util': self._gpu_util,
            'vram_gb': self._vram_gb
        }


class Telemetry:
    """
    Main telemetry orchestrator for SCARCITY runtime.
    
    Collects metrics from:
    - Event bus (latency, throughput)
    - System resources (CPU, GPU, VRAM)
    - Custom module metrics (drift, errors)
    
    Publishes telemetry events to the bus and optionally writes to logs.
    """
    
    def __init__(self, bus: Optional[EventBus] = None, publish_interval: float = 3.0):
        """
        Initialize telemetry system.
        
        Args:
            bus: EventBus instance (defaults to global bus)
            publish_interval: Seconds between telemetry snapshots
        """
        self.bus = bus if bus else get_bus()
        self.publish_interval = publish_interval
        
        # Trackers
        self.latency = LatencyTracker(alpha=0.3)
        self.throughput = ThroughputCounter(window_seconds=1.0)
        self.drift_monitor = DriftMonitor(threshold=3.0)
        self.system_probe = SystemProbe()
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Metrics
        self.errors_last_minute = deque()
        self.custom_metrics: Dict[str, float] = {}
        self._meta_metrics_handler = self._handle_meta_metrics
        self.bus.subscribe("meta_metrics", self._meta_metrics_handler)
        
        logger.info(f"Telemetry initialized with publish_interval={publish_interval}s")
    
    async def start(self) -> None:
        """Start continuous telemetry collection and publishing."""
        if self._running:
            logger.warning("Telemetry already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._telemetry_loop())
        logger.info("Telemetry started")
    
    async def stop(self) -> None:
        """Stop telemetry collection."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self.bus.unsubscribe("meta_metrics", self._meta_metrics_handler)
        
        logger.info("Telemetry stopped")
    
    async def _telemetry_loop(self) -> None:
        """Main telemetry collection loop."""
        try:
            while self._running:
                # Gather metrics
                snapshot = await self._collect_snapshot()
                
                # Publish to bus
                await self.bus.publish("telemetry", snapshot)
                
                # Log to console/file (simplified for now)
                logger.info(
                    f"Telemetry: cpu={snapshot.get('cpu_percent', 0):.1f}%, "
                    f"latency={snapshot.get('bus_latency_ms', 0):.2f}ms, "
                    f"throughput={snapshot.get('bus_throughput', 0):.0f}/s"
                )
                
                # Wait for next interval
                await asyncio.sleep(self.publish_interval)
                
        except asyncio.CancelledError:
            logger.info("Telemetry loop cancelled")
        except Exception as e:
            logger.error(f"Error in telemetry loop: {e}", exc_info=True)
    
    async def _collect_snapshot(self) -> Dict[str, Any]:
        """
        Collect current metrics snapshot.
        
        Returns:
            Dictionary of all current metrics
        """
        # Probe system
        system_metrics = self.system_probe.probe()
        
        # Clean old error timestamps
        now = time.time()
        while self.errors_last_minute and self.errors_last_minute[0] < now - 60:
            self.errors_last_minute.popleft()
        
        # Build snapshot
        snapshot = {
            'timestamp': now,
            'datetime': datetime.utcnow().isoformat(),
            
            # System metrics
            **system_metrics,
            
            # Bus metrics
            'bus_latency_ms': self.latency.get_latency(),
            'bus_throughput': self.throughput.get_rate(),

            # Compatibility aliases for consumers expecting generic keys
            'latency_ms': self.latency.get_latency(),
            'fps': 1000.0 / max(self.latency.get_latency(), 1e-6),
            
            # Error metrics
            'errors_last_minute': len(self.errors_last_minute),
            'drift_score': self.drift_monitor.mean if self.drift_monitor.mean is not None else 0.0,
            
            # Custom metrics
            **self.custom_metrics
        }
        
        return snapshot
    
    def record_latency(self, duration_ms: float) -> None:
        """
        Record a latency measurement.
        
        Args:
            duration_ms: Latency in milliseconds
        """
        self.latency.record(duration_ms)
    
    def record_message(self) -> None:
        """Record a message event (for throughput calculation)."""
        self.throughput.record_event()
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self.errors_last_minute.append(time.time())
    
    def record_metric(self, name: str, value: float) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.custom_metrics[name] = value

    async def _handle_meta_metrics(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Consume meta metrics published on the bus.
        """
        if not isinstance(data, dict):
            return

        recorded: Dict[str, float] = {}
        for key, value in data.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            self.record_metric(key, numeric)
            recorded[key] = numeric

        if recorded:
            logger.debug("Recorded custom meta metrics: %s", recorded)
    
    def check_drift(self, value: float) -> Optional[float]:
        """
        Check for drift in a value.
        
        Args:
            value: Value to check
            
        Returns:
            Drift score if detected, None otherwise
        """
        return self.drift_monitor.update(value)


async def main():
    """Example usage and testing."""
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting SCARCITY Runtime Telemetry Test")
    
    # Initialize
    bus = get_bus()
    telemetry = Telemetry(bus=bus, publish_interval=2.0)
    
    # Start telemetry
    await telemetry.start()
    
    # Simulate some activity
    for i in range(5):
        await asyncio.sleep(2.5)
        telemetry.record_message()
        logger.info(f"Test iteration {i+1}")
    
    # Cleanup
    await telemetry.stop()
    await bus.shutdown()
    
    logger.info("Test complete")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

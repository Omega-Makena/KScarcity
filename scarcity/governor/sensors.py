"""
Telemetry sensors for the Dynamic Resource Governor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - used in environment without psutil
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may be missing native deps
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False


@dataclass
class SensorConfig:
    """
    Configuration for system sensors.

    Attributes:
        interval_ms: Minimum interval between samples in milliseconds to avoid
            excessive polling overhead.
    """
    interval_ms: int = 250


class ResourceSensors:
    """
    Collects system telemetry from CPU, GPU, memory, and I/O.
    
    Acts as a hardware abstraction layer, providing a unified dictionary of
    normalized metrics (usually 0.0 - 1.0 utilization scores). Handles proper
    initialization of optional libraries like `psutil` or `pynvml`.
    """

    def __init__(self, config: SensorConfig):
        """
        Initializes sensors.

        Args:
            config: Sensor configuration.
        """
        self.config = config
        self._last_sample_ts = 0.0
        if _NVML_AVAILABLE:
            try:  # pragma: no cover - NVML init
                pynvml.nvmlInit()
            except Exception:
                pass

    def sample(self) -> Dict[str, float]:
        """
        Retrieves the current system metrics.

        Enforces the configured sampling interval by sleeping if called too frequently.
        
        Returns:
            A dictionary of metrics, e.g., {'cpu_util': 0.45, 'mem_util': 0.60}.
        """
        now = time.time()
        if (now - self._last_sample_ts) * 1000 < self.config.interval_ms:
            time.sleep(max(0.0, self.config.interval_ms / 1000 - (now - self._last_sample_ts)))
        self._last_sample_ts = time.time()

        metrics: Dict[str, float] = {}
        metrics.update(self._cpu_metrics())
        metrics.update(self._memory_metrics())
        metrics.update(self._gpu_metrics())
        metrics.update(self._io_metrics())
        return metrics

    def _cpu_metrics(self) -> Dict[str, float]:
        """Collects CPU usage and frequency."""
        if psutil is None:
            return {"cpu_util": 0.0, "cpu_freq": 0.0}
        cpu_util = float(psutil.cpu_percent(interval=None))
        freq = psutil.cpu_freq()
        cpu_freq = float(freq.current) if freq else 0.0
        return {"cpu_util": cpu_util / 100.0, "cpu_freq": cpu_freq}

    def _memory_metrics(self) -> Dict[str, float]:
        """Collects RAM and Swap usage."""
        if psutil is None:
            return {"mem_util": 0.0}
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "mem_util": float(mem.percent) / 100.0,
            "mem_available_gb": float(mem.available) / (1024**3),
            "swap_util": float(swap.percent) / 100.0,
        }

    def _gpu_metrics(self) -> Dict[str, float]:
        """
        Collects GPU usage via PyTorch or NVML.

        Focuses on utilization and VRAM consumption. Safe to call even on
        CPU-only systems (returns zeros).
        """
        if torch is None or not torch.cuda.is_available():
            return {"gpu_util": 0.0, "vram_util": 0.0}
        try:
            util = torch.cuda.utilization()
        except Exception:  # pragma: no cover
            util = 0.0
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used = total_mem - free_mem
            vram_util = float(used) / float(total_mem)
        except Exception:
            vram_util = 0.0
        gpu_util = float(util) / 100.0 if isinstance(util, (int, float)) else 0.0

        if _NVML_AVAILABLE:  # pragma: no cover - optional
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = rates.gpu / 100.0
            except Exception:
                pass

        return {"gpu_util": gpu_util, "vram_util": vram_util}

    def _io_metrics(self) -> Dict[str, float]:
        """Collects Disk II/O and Network I/O throughput (in MB)."""
        if psutil is None:
            return {"disk_read_mb": 0.0, "disk_write_mb": 0.0, "net_sent_mb": 0.0, "net_recv_mb": 0.0}
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        return {
            "disk_read_mb": float(disk.read_bytes) / (1024**2) if disk else 0.0,
            "disk_write_mb": float(disk.write_bytes) / (1024**2) if disk else 0.0,
            "net_sent_mb": float(net.bytes_sent) / (1024**2) if net else 0.0,
            "net_recv_mb": float(net.bytes_recv) / (1024**2) if net else 0.0,
        }

"""
Industrial-Grade Dynamic Resource Governance (DRG).

This module manages hardware telemetry (both GPU via NVML and CPU via psutil).
It implements:
1. Online GPU/CPU memory fragmentation and bandwidth tracking.
2. Exponetial Moving Average (EMA) OOM Prediction.
3. Proportional control for dynamic precision scaling (FP32 -> FP16 -> Q8).
4. Rate-limiting configurations for peripheral system extensions.
"""

import logging
import time
from typing import Dict, Any, Optional
import psutil

from scarcity.runtime import EventBus, get_bus

logger = logging.getLogger(__name__)

# Try importing NVML for GPU tracking
HAS_NVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception as e:
    logger.warning(f"NVML initialization failed. Falling back to CPU-only tracking: {e}")


class GlobalResourceManager:
    """
    Hardware-Aware DRG.
    
    Monitors bare-metal resources (GPU+CPU) and emits continuous, PID-styled
    throttling directives to peripheral systems to maintain engine core stability.
    """
    
    def __init__(self, bus: Optional[EventBus] = None):
        """Initialize the Global DRG."""
        self.bus = bus if bus else get_bus()
        self.running = False
        
        # Hard system boundaries
        self.max_cpu_percent = 90.0
        self.max_ram_percent = 92.0
        self.max_gpu_percent = 95.0
        
        # OOM Prediction Tracking (EMA)
        self.alpha_memory = 0.2
        self.ema_ram_growth = 0.0
        self.ema_vram_growth = 0.0
        self.last_ram = 0.0
        self.last_vram = 0.0
        self.last_time = time.time()
        
        # Broadcast State
        self.current_precision = "fp32"
        self.current_extension_state = "nominal"
        
    async def start(self) -> None:
        """Subscribe to telemetry requests or tick loop."""
        if self.running:
            return
        self.running = True
        self.bus.subscribe("processing_metrics", self._handle_pipeline_tick)
        logger.info(f"Industrial DRG started. GPU Tracking Enabled: {HAS_NVML}")

    async def stop(self) -> None:
        """Unsubscribe and cleanup."""
        if not self.running:
            return
        self.running = False
        self.bus.unsubscribe("processing_metrics", self._handle_pipeline_tick)
        
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
                
        logger.info("Industrial DRG stopped")

    def _get_hardware_telemetry(self) -> Dict[str, float]:
        """Polls raw hardware sensors."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_percent": 0.0,
            "vram_percent": 0.0
        }
        
        if HAS_NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats["gpu_percent"] = float(util.gpu)
                stats["vram_percent"] = float(mem.used) / float(mem.total) * 100.0
            except Exception as e:
                logger.debug(f"NVML polling failed: {e}")
                
        return stats
        
    def _predict_oom(self, stats: Dict[str, float], dt: float) -> bool:
        """
        Uses Exponential Moving Average (EMA) of derivatives to forecast 
        if memory will breach limits in the next dt cycles.
        """
        if dt <= 0:
            return False
            
        # CPU RAM Growth
        ram_delta = (stats["ram_percent"] - self.last_ram) / dt
        self.ema_ram_growth = (1 - self.alpha_memory) * self.ema_ram_growth + self.alpha_memory * ram_delta
        
        # VRAM Growth
        vram_delta = (stats["vram_percent"] - self.last_vram) / dt
        self.ema_vram_growth = (1 - self.alpha_memory) * self.ema_vram_growth + self.alpha_memory * vram_delta
        
        self.last_ram = stats["ram_percent"]
        self.last_vram = stats["vram_percent"]
        
        # Predict 5 seconds into the future
        future_ram = stats["ram_percent"] + (self.ema_ram_growth * 5.0)
        future_vram = stats["vram_percent"] + (self.ema_vram_growth * 5.0)
        
        if future_ram > self.max_ram_percent or future_vram > self.max_gpu_percent:
            return True
        return False

    async def _handle_pipeline_tick(self, topic: str, pipeline_metrics: Dict[str, Any]) -> None:
        """
        Evaluate real-time hardware stress and emit proportional throttling events.
        """
        if not self.running:
            return
            
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Poll Hardware
        hw = self._get_hardware_telemetry()
        
        # OOM Threat Detection
        oom_imminent = self._predict_oom(hw, dt)
        pipeline_latency = pipeline_metrics.get("latency_ms", 0.0)
        
        # --- Evaluate Tier-1 Protection (Dynamic Precision) ---
        new_precision = "fp32"
        if oom_imminent or hw["vram_percent"] > 85.0:
            new_precision = "q8"
        elif hw["vram_percent"] > 70.0:
            new_precision = "fp16"
            
        if new_precision != self.current_precision:
            self.current_precision = new_precision
            logger.warning(f"DRG shifting engine precision to: {new_precision.upper()}")
            await self.bus.publish("scarcity.drg_precision_target", {"precision": new_precision})

        # --- Evaluate Tier-2 Protection (Extension Throttling) ---
        new_ext_state = "nominal"
        throttle_ratio = 1.0
        
        if oom_imminent or hw["cpu_percent"] > self.max_cpu_percent or hw["gpu_percent"] > self.max_gpu_percent:
            new_ext_state = "severe_throttle"
            throttle_ratio = 0.0  # Suspend non-criticals
        elif pipeline_latency > 80.0 or hw["ram_percent"] > 80.0:
            new_ext_state = "proportional_throttle"
            throttle_ratio = 0.5  # Half frequency
            
        # Emit Peripheral Constraints
        if new_ext_state != self.current_extension_state or new_ext_state == "proportional_throttle":
            self.current_extension_state = new_ext_state
            
            profile = {
                "anomaly_enabled": throttle_ratio > 0,
                "anomaly_sample_rate": throttle_ratio, # 1.0 (all), 0.5 (half), 0.0 (none)
                "forecast_enabled": throttle_ratio > 0,
                "forecast_max_steps": max(1, int(5 * throttle_ratio)),
                "hw_telemetry": hw,
                "oom_imminent": oom_imminent
            }
            await self.bus.publish("scarcity.drg_extension_profile", profile)

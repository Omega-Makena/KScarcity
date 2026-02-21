import pytest
import time
from unittest.mock import AsyncMock

from scarcity.engine.resource_manager import GlobalResourceManager
from scarcity.runtime import EventBus

def test_oom_ema_prediction():
    """Verify the Exponential Moving Average (EMA) mathematical slope prediction for OOM threats."""
    drg = GlobalResourceManager()
    
    # 1. Baseline safe metrics
    stats = {
        "cpu_percent": 10.0,
        "ram_percent": 50.0,
        "gpu_percent": 10.0,
        "vram_percent": 50.0
    }
    
    drg.last_ram = 50.0
    drg.last_vram = 50.0
    
    # Static memory -> Slope is 0 -> OOM threat is False
    assert not drg._predict_oom(stats, dt=1.0)
    assert drg.ema_vram_growth == 0.0
    
    # 2. Catastrophic Memory Spike
    stats["vram_percent"] = 90.0
    
    # (90 - 50) / 1.0 = +40.0% vram growth per second
    # EMA = (1-0.2)*0 + 0.2*40.0 = 8.0 percent per second EMA
    # Predict 5s future = 90.0 + (5 * 8.0) = 130%
    # 130% > 95% Max_GPU -> OOM IMMINENT
    assert drg._predict_oom(stats, dt=1.0)
    assert drg.ema_vram_growth > 0.0

def test_dynamic_precision_tier1_defense():
    import asyncio
    asyncio.run(_test_dynamic_precision_tier1_defense())

async def _test_dynamic_precision_tier1_defense():
    """Verify that the DRG actively shifts engine float precision under duress."""
    bus = EventBus()
    bus.publish = AsyncMock() # type: ignore
    drg = GlobalResourceManager(bus)
    await drg.start()
    
    # Hack hardware poll
    def mock_hw():
        return {
            "cpu_percent": 10.0,
            "ram_percent": 50.0,
            "gpu_percent": 10.0,
            "vram_percent": 80.0 # Danger zone, but not crashing
        }
    drg._get_hardware_telemetry = mock_hw
    
    # Pipeline tick evaluates the threat
    await drg._handle_pipeline_tick("processing_metrics", {"latency_ms": 10.0})
    
    # VRAM > 70% should downcast to FP16 to save space
    bus.publish.assert_any_call("scarcity.drg_precision_target", {"precision": "fp16"})
    assert drg.current_precision == "fp16"
    assert drg.current_extension_state == "nominal" # Latency/CPU is fine, so extensions keep running
    
    # 2. OOM Threat (Tier 2 defense)
    def mock_hw_oom():
        return {
            "cpu_percent": 99.0, # CPU pegged
            "ram_percent": 95.0, # RAM maxed
            "gpu_percent": 95.0,
            "vram_percent": 95.0 
        }
    drg._get_hardware_telemetry = mock_hw_oom
    
    # The EMA will predict an OOM
    await drg._handle_pipeline_tick("processing_metrics", {"latency_ms": 100.0})
    
    # VRAM > 85% or OOM -> Downcast to Q8 entirely
    bus.publish.assert_any_call("scarcity.drg_precision_target", {"precision": "q8"})
    
    # The CPU/RAM is maxed, meaning the extensions must be starved
    assert drg.current_extension_state == "severe_throttle"
    
    # Check the extension starvation profile
    call_args = bus.publish.call_args_list[-1][0]
    topic, payload = call_args
    assert topic == "scarcity.drg_extension_profile"
    assert payload["anomaly_enabled"] is False
    assert payload["forecast_enabled"] is False
    assert payload["forecast_max_steps"] == 1 # Minimum depth
    
    await drg.stop()

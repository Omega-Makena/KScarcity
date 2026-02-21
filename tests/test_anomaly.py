import pytest
import numpy as np
import time
from unittest.mock import AsyncMock

from scarcity.engine.anomaly import _compute_rrcf_codispersion, OnlineAnomalyDetector
from scarcity.runtime import EventBus

def test_rrcf_codispersion_baseline():
    """Verify that normal points score low and extreme outliers score high."""
    np.random.seed(42)
    history = np.random.normal(loc=0.0, scale=1.0, size=(100, 5)).astype(np.float32)
    
    # Normal point from the same distribution
    normal_point = np.array([0.1, -0.2, 0.5, -0.1, 0.0], dtype=np.float32)
    normal_score = _compute_rrcf_codispersion(history, normal_point, num_trees=50)
    
    # Extreme outlier
    outlier_point = np.array([100.0, -100.0, 50.0, 50.0, -50.0], dtype=np.float32)
    outlier_score = _compute_rrcf_codispersion(history, outlier_point, num_trees=50)
    
    assert normal_score < outlier_score, "RRCF failed to distinguish an outlier from a normal point"
    assert outlier_score > 3.0, "Outlier score is too low, indicating poor isolation tree depth."

def test_rrcf_codispersion_masking():
    """Verify RRCF relies on CoDispersion to detect groups of anomalies (swarm) without being masked."""
    np.random.seed(42)
    history = np.random.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
    
    # Anomaly Swarm (multiple points clustered far away from the origin)
    swarm_p1 = np.array([50.0, 50.0], dtype=np.float32)
    swarm_p2 = np.array([51.0, 49.0], dtype=np.float32)
    
    # Inject one into history to try and "mask" the second one
    poisoned_history = np.vstack([history, swarm_p1])
    
    # Mahalanobis would fail here because the covariance matrix gets skewed, 
    # but RRCF should still easily isolate the swarm cluster from the main distribution
    score = _compute_rrcf_codispersion(poisoned_history, swarm_p2, num_trees=50)
    
    assert score > 5.0, f"Swarm masking immunity failed. Score: {score}"

def test_anomaly_detector_throttling():
    import asyncio
    asyncio.run(_test_anomaly_detector_throttling())

async def _test_anomaly_detector_throttling():
    """Verify the detector responds to DRG throttling constraints."""
    bus = EventBus()
    bus.publish = AsyncMock() # type: ignore
    
    detector = OnlineAnomalyDetector(bus)
    await detector.start()
    
    # Simulate severe starvation
    await detector._handle_drg("scarcity.drg_extension_profile", {
        "anomaly_enabled": True,
        "anomaly_sample_rate": 0.5
    })
    
    assert detector.sample_rate == 0.5
    assert detector.num_trees < detector.max_trees
    
    # Fire two data windows. First should skip (counter=1, 1 % 2 != 0)
    await detector._handle_data_window("data_window", {"data": [[0.0]]})
    assert detector.history_buffer is None # Skipped processing
    
    # Second should process (counter=2, 2 % 2 == 0)
    data = np.zeros((10, 1), dtype=np.float32)
    await detector._handle_data_window("data_window", {"data": data})
    assert detector.history_buffer is not None # Processed
    
    await detector.stop()

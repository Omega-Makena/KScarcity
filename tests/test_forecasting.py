import pytest
import numpy as np
from unittest.mock import AsyncMock

from scarcity.engine.forecasting import _compute_garch_varx_forecast, PredictiveForecaster
from scarcity.engine.store import HypergraphStore
from scarcity.runtime import EventBus

def test_varx_garch_projection():
    """Verify the pure mathematical projection of VARX and GARCH confidence bounds."""
    # 2-variable transition matrix
    W = np.array([
        [0.8, 0.1],
        [0.2, 0.9]
    ], dtype=np.float32)
    
    X_t = np.array([1.0, 2.0], dtype=np.float32)
    sigma2_t = np.array([0.1, 0.1], dtype=np.float32)
    
    # 1. Baseline Test (No Anomaly Shock)
    f_base, v_base = _compute_garch_varx_forecast(
        W, X_t, exogenous_shock=0.0, sigma2_t=sigma2_t, max_steps=5
    )
    
    assert f_base.shape == (5, 2)
    assert v_base.shape == (5, 2)
    # Without shocks, the GARCH variance decays towards the omega / (1 - beta) asymptote
    # It started at 0.1, so it should be decreasing towards ~0.05
    assert np.all(v_base[-1] < v_base[0])

    
    # 2. Exogenous Shock Test
    massive_shock = 10.0
    f_shock, v_shock = _compute_garch_varx_forecast(
        W, X_t, exogenous_shock=massive_shock, sigma2_t=sigma2_t, max_steps=5
    )
    
    # The shock should significantly inflate the predicted state magnitude
    assert np.linalg.norm(f_shock[0]) > np.linalg.norm(f_base[0])
    
    # The shock should trigger massive heteroskedastic volatility bursts (GARCH alpha term)
    assert np.all(v_shock[0] > v_base[0] * 5.0), "GARCH failed to inflate uncertainty under exogenous shock"

def test_forecaster_drg_throttling():
    import asyncio
    asyncio.run(_test_forecaster_drg_throttling())

async def _test_forecaster_drg_throttling():
    """Verify the PredictiveForecaster reads from the Hypergraph and obeys DRG max_steps."""
    store = HypergraphStore()
    bus = EventBus()
    bus.publish = AsyncMock() # type: ignore

    
    forecaster = PredictiveForecaster(store, bus)
    await forecaster.start()
    
    # Restrict depth to 2 via DRG
    await forecaster._handle_drg("scarcity.drg_extension_profile", {
        "forecast_enabled": True,
        "forecast_max_steps": 2
    })
    
    assert forecaster.max_steps == 2
    
    data = np.ones((5, 2), dtype=np.float32)
    await forecaster._handle_data_window("data_window", {"data": data})
    
    # Verify the bus received exactly 2 steps ahead
    bus.publish.assert_called_once()
    topic, payload = bus.publish.call_args[0]
    
    assert topic == "scarcity.forecasted_trends"
    assert payload["steps_ahead"] == 2
    assert len(payload["forecast_matrix"]) == 2
    assert len(payload["garch_variance_matrix"]) == 2
    
    await forecaster.stop()

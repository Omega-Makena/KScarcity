import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import sys

# Important: scarcity_bridge is inside kshiked/ui/institution/backend, so 5 levels up to root
project_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from scarcity.runtime import EventBus
from scarcity.engine.store import HypergraphStore
from scarcity.engine.anomaly import OnlineAnomalyDetector
from scarcity.engine.forecasting import PredictiveForecaster

class ScarcityBridge:
    """
    A secure wrapper to spin up the actual Numba-compiled Scarcity Engine inside
    the Streamlit process. This strips out all toyish mockups, mathematically
    proving the core extensions on the Spoke's local data.
    """
    def __init__(self):
        self.bus = EventBus()
        self.store = HypergraphStore()
        
        # Initialize the heavy industrial extensions
        self.anomaly_detector = OnlineAnomalyDetector(self.bus)
        self.forecaster = PredictiveForecaster(self.store, self.bus)

        
    def process_dataframe(self, df: pd.DataFrame, basket_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs the Numba-compiled Scarcity anomaly detector and Bayesian VARX bounds
        over the verified DataFrame.
        """
        # Extract numeric data, aligning with the schema rules if necessary
        if basket_schema and "required_columns" in basket_schema:
            cols = [c for c in basket_schema["required_columns"] if c in df.columns]
            data_matrix = df[cols].values.astype(np.float32)
        else:
            data_matrix = df.select_dtypes(include=[np.number]).values.astype(np.float32)
            
        if data_matrix.shape[0] == 0 or data_matrix.shape[1] == 0:
            return {"anomalies": [], "forecasts": [], "drg_profiles": []}
            
        V = data_matrix.shape[1]
        
        collected_anomalies = []
        collected_forecasts = []
        collected_drg = []
        
        # Wire up the EventBus Listeners
        async def anomaly_handler(topic: str, data: Any):
            collected_anomalies.append(data.get("severity", 0.0))
            
        async def forecast_handler(topic: str, data: Any):
            collected_forecasts.append({
                "forecasts": data.get("forecast_matrix"),
                "variances": data.get("garch_variance_matrix")
            })
            
        async def drg_handler(topic: str, data: Any):
            # Capture if the DRG had to downcast precision due to load
            collected_drg.append(data)
            
        self.bus.subscribe("scarcity.anomaly_detected", anomaly_handler)
        self.bus.subscribe("scarcity.forecasted_trends", forecast_handler)
        self.bus.subscribe("scarcity.drg_extension_profile", drg_handler)
        
        async def run_pipeline():
            await self.anomaly_detector.start()
            await self.forecaster.start()
            
            for i in range(len(data_matrix)):
                # Publish the row as a [1, V] window for the streaming engines
                row_window = data_matrix[i:i+1]
                payload = {"data": row_window, "window_id": f"spoke_frame_{i}"}
                await self.bus.publish("data_window", payload)
                
            # Allow EventBus to flush final mathematical frames
            await self.bus.wait_for_idle()
            
            await self.anomaly_detector.stop()
            await self.forecaster.stop()
            await self.bus.shutdown()
            
        # Execute the highly optimized asynchronous mathematical pipeline
        asyncio.run(run_pipeline())
        
        return {
            "anomalies": collected_anomalies,
            "forecasts": collected_forecasts,
            "drg": collected_drg
        }

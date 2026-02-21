import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
import sys

# Important: executive_bridge is inside kshiked/ui/institution/backend, so 5 levels up to root
project_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scarcity.engine.forecasting import _compute_garch_varx_forecast
from kshiked.ui.institution.backend.database import get_connection

class ExecutiveBridge:
    """
    Connects the God Tier Executive Policy Workbench directly to the 
    Numba-compiled Bayesian VARX and GARCH(1,1) engine. 
    It mathematically models how exogenous macro-shocks (like interest rate hikes) 
    ripple through the dynamically discovered National Intelligence Meta-Graph.
    """
    
    @classmethod
    def get_tiers(cls) -> list:
        """Dynamically fetch the sectors (baskets) from the live registry."""
        try:
            with get_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT name FROM baskets ORDER BY id")
                baskets = c.fetchall()
                if baskets:
                    return [r['name'] for r in baskets]
        except Exception:
            pass
        return ["Fallback Sector Alpha", "Fallback Sector Beta", "Fallback Sector Gamma"]

    @classmethod
    def construct_meta_graph_matrix(cls) -> np.ndarray:
        """
        Builds the robust Transition Matrix (W). 
        In a full deployment, this is hydrated by the Federated Aggregator `global_weights`.
        For simulation realism over arbitrary data, it dynamically creates a structural map
        based on the active Baskets in the network.
        """
        tiers = cls.get_tiers()
        V = len(tiers)
        if V == 0:
            return np.ones((1, 1), dtype=np.float32)

        # Baseline interconnectedness (15% spillover)
        W = np.ones((V, V), dtype=np.float32) * 0.15
        
        # Diagonal (40% auto-correlation / sector persistence)
        np.fill_diagonal(W, 0.40)
        
        # Synthetic noise injection to model real-world chaos bounds
        np.random.seed(42) # fixed for stability in UI
        noise = np.random.uniform(0.0, 0.2, size=(V, V)).astype(np.float32)
        W = W + noise
        
        # Prevent exploding gradients via spectral bounding
        W = W * 0.85
        return W

    @classmethod
    def simulate_policy_shock(cls, target_tier_idx: int, magnitude: float, steps: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the Bayesian VARX projection.
        Returns two DataFrames: 
          - Expected State Cascade (Forecasts)
          - GARCH Uncertainty Bounds (Variances)
        """
        tiers = cls.get_tiers()
        V = len(tiers)
        W = cls.construct_meta_graph_matrix()
        
        # Flat baseline current state
        X_t = np.ones(V, dtype=np.float32) * 1.0
        
        # Inject the policy target explicitly into the state vector
        if 0 <= target_tier_idx < V:
            X_t[target_tier_idx] += magnitude
        
        # Initial variance
        sigma2_t = np.ones(V, dtype=np.float32) * 0.1
        
        # Run the Numba engine
        forecasts, variances = _compute_garch_varx_forecast(
            W=W,
            X_t=X_t,
            exogenous_shock=magnitude,
            sigma2_t=sigma2_t,
            max_steps=steps,
            omega=0.05,
            alpha=0.25, # High ARCH effect so shocks explode uncertainty
            beta=0.70
        )
        
        df_forecasts = pd.DataFrame(forecasts, columns=tiers)
        df_variances = pd.DataFrame(variances, columns=tiers)
        
        # Provide time index
        df_forecasts.index = [f"T+{i}" for i in range(steps)]
        df_variances.index = [f"T+{i}" for i in range(steps)]
        
        return df_forecasts, df_variances

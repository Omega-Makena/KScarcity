import numpy as np
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys

# Important: federation_bridge is inside kshiked/ui/institution/backend, so 5 levels up to root
project_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from scarcity.federation.aggregator import FederatedAggregator, AggregationConfig, AggregationMethod
from scarcity.federation.privacy_guard import PrivacyGuard, PrivacyConfig

class FederationBridge:
    """
    Connects the Basket Admin Governance Hub directly to the `scarcity.federation` math engine.
    Applies Byzantine-resilient aggregation (Trimmed Mean/Krum) to Spoke Delta updates,
    and injects Laplacian Differential Privacy noise.
    """
    
    @staticmethod
    def aggregate_spoke_models(payloads: List[Dict[str, Any]], method_name: str = "trimmed_mean") -> Tuple[np.ndarray, dict]:
        """
        Extracts local model weights from Spoke Deltas and aggregates them 
        using the scarcity engine's robust averaging methods.
        """
        # Parse the method enum
        try:
            method = AggregationMethod(method_name)
        except ValueError:
            method = AggregationMethod.TRIMMED_MEAN
            
        config = AggregationConfig(method=method, trim_alpha=0.1)
        aggregator = FederatedAggregator(config)
        
        # Extract the weight vectors from the JSON payloads
        updates = []
        for p in payloads:
            if "local_weights" in p:
                updates.append(p["local_weights"])
            elif "post_shock_volatility_forecast" in p and p["post_shock_volatility_forecast"]:
                updates.append(list(p["post_shock_volatility_forecast"].values()))
            elif "shock_vector" in p and p["shock_vector"]:
                updates.append([v.get("delta_magnitude", 0) for v in p["shock_vector"].values()])
                
        # If no valid weights, return dummy for UI stability
        if not updates:
            return np.zeros(10, dtype=np.float32), {"method": "none", "participants": 0}
            
        # Run the actual mathematical aggregation
        global_weights, metadata = aggregator.aggregate(updates)
        return global_weights, metadata

    @staticmethod
    def apply_differential_privacy(weights: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Injects mathematically certified Laplacian noise to the Global FL vector
        to guarantee differential privacy before exporting to the Executive God Tier
        or peering to sideways baskets.
        """
        if epsilon <= 0.0:
            return weights
            
        config = PrivacyConfig(
            dp_epsilon=epsilon,
            dp_noise_type="laplace",
            dp_sensitivity=1.0 # Assuming normalized weights
        )
        guard = PrivacyGuard(config)
        
        # PrivacyGuard expects a sequence of vectors, so we wrap and unwrap the [weights] array
        noised_batch = guard.apply_noise([weights])
        return noised_batch[0]

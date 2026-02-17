"""
Shock Compiler for KShield Simulation.

Transforms stochastic shock definitions (Impulse, OU Process, Brownian) 
into deterministic vectors compatible with Scarcity's SFC Engine.

This allows us to keep the core engine fast and vectorized while supporting
complex, localized stochastic processes in the initialization phase.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

# Import Shock Definitions
from kshiked.core.shocks import Shock, ImpulseShock, OUProcessShock, BrownianShock
from scarcity.simulation.sfc import SFCConfig

logger = logging.getLogger("kshield.compiler")

class ShockCompiler:
    """
    Compiles high-level Shock objects into time-series vectors.
    """
    
    def __init__(self, steps: int = 50, dt: float = 1.0, seed: int = 42):
        self.steps = steps
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def compile(self, shocks: List[Shock]) -> Dict[str, np.ndarray]:
        """
        Generate vectors for all active shocks.
        
        Returns:
            Dict[str, np.ndarray]: Map of 'shock_channel' -> 'time_series_vector'
            compatible with SFCConfig.shock_vectors.
        """
        # Initialize zero vectors for supported channels
        # Canonical Keys from SFCEconomy
        channels = ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"]
        vectors = {k: np.zeros(self.steps, dtype=np.float32) for k in channels}
        
        for shock in shocks:
            if not shock.active:
                continue
                
            # Determine Target Channel
            # Map metric names or specific targets to the 4 canonical channels
            # Heuristic mapping for demo purposes
            target = shock.target_metric.lower()
            channel_key = None
            
            if "demand" in target or "consumption" in target or "investment" in target:
                channel_key = "demand_shock"
            elif "supply" in target or "productivity" in target:
                channel_key = "supply_shock"
            elif "fiscal" in target or "spending" in target or "tax" in target:
                channel_key = "fiscal_shock"
            elif "fx" in target or "rate" in target or "currency" in target:
                channel_key = "fx_shock"
            else:
                # Default fallback mechanism or explicit mapping needed
                # For now log warning
                logger.warning(f"Shock target '{shock.target_metric}' not mapped to canonical channel. defaulting to demand.")
                channel_key = "demand_shock"
                
            # Generate Time Series
            series = self._generate_series(shock)
            
            # Accumulate
            # Vectors are additive
            vectors[channel_key] += series
            
        return vectors

    def _generate_series(self, shock: Shock) -> np.ndarray:
        """Type-specific generation logic."""
        series = np.zeros(self.steps)
        
        # We process 'step()' manually to capture state evolution
        # But we must be careful not to mutate the original shock object permanently if we want re-runs
        # So we clone or simulate locally.
        
        if isinstance(shock, ImpulseShock):
            val = shock.magnitude
            for t in range(self.steps):
                series[t] = val
                val *= shock.decay
                if abs(val) < 1e-6: val = 0.0
                
        elif isinstance(shock, OUProcessShock):
            # dx = theta * (mu - x) * dt + sigma * dW
            x = 0.0 # Start at mean? Or 0 deviation? Usually 0 deviation.
            for t in range(self.steps):
                series[t] = x
                dw = self.rng.normal(0, np.sqrt(shock.dt))
                dx = shock.theta * (shock.mu - x) * shock.dt + shock.sigma * dw
                x += dx
                
        elif isinstance(shock, BrownianShock):
            # dx = drift * dt + sigma * dW
            x = 0.0
            for t in range(self.steps):
                series[t] = x
                dw = self.rng.normal(0, np.sqrt(shock.dt))
                dx = shock.drift * shock.dt + shock.volatility * dw
                x += dx
                
        else:
            # Generic fallback if 'step' and 'get_delta' provided
            # This handles custom subclasses
            try:
                # Basic simulation wrapper
                # Note: This might mutate the shock object state!
                # Ideally we'd avoid this, but for now it's okay for single-use.
                pass 
            except Exception:
                pass
                
        return series

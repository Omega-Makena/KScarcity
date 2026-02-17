"""
Exogenous Shock System for Kshield.
Phase 4: Adversarial Stress Test (Stochastic Processes)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
import random


class ShockType(Enum):
    """Types of economic shocks."""
    IMPULSE = "impulse"
    OU_PROCESS = "ou_process"
    BROWNIAN = "brownian"

@dataclass
class Shock:
    """Base Shock Class"""
    name: str
    target_metric: str
    active: bool = True

    def get_delta(self, current_val: float) -> float:
        return 0.0

    def step(self):
        pass

@dataclass
class ImpulseShock(Shock):
    """Classic 'Hit' shock"""
    magnitude: float = 0.0
    decay: float = 0.9
    _current_val: float = 0.0
    
    def __post_init__(self):
        self._current_val = self.magnitude

    def get_delta(self, current_val: float) -> float:
        if not self.active: return 0.0
        return self._current_val

    def step(self):
        self._current_val *= self.decay
        if abs(self._current_val) < 1e-4:
            self.active = False

@dataclass
class OUProcessShock(Shock):
    """
    Ornstein-Uhlenbeck Process.
    Mean-reverting stochastic process. Use for realistic volatility.
    dx = theta * (mu - x) * dt + sigma * dW
    """
    theta: float = 0.15 # Speed of mean reversion
    mu: float = 0.0     # Long-Run Mean (offset, usually 0)
    sigma: float = 0.2  # Volatility
    dt: float = 1.0     # Time step
    _x: float = 0.0
    
    def get_delta(self, current_val: float) -> float:
        if not self.active: return 0.0
        # Return the NOISE offset to add to the value
        # We assume this shock adds specific noise to the metric
        return self._x

    def step(self):
        # Euler-Maruyama method
        dw = np.random.normal(0, np.sqrt(self.dt))
        dx = self.theta * (self.mu - self._x) * self.dt + self.sigma * dw
        self._x += dx

@dataclass
class BrownianShock(Shock):
    """
    Geometric Brownian Motion (Random Walk).
    Good for asset prices (Drift + Volatility).
    """
    drift: float = 0.0
    volatility: float = 0.1
    dt: float = 1.0
    _accumulated: float = 0.0

    def get_delta(self, current_val: float) -> float:
        if not self.active: return 0.0
        return self._accumulated

    def step(self):
        epsilon = np.random.normal(0, 1)
        # Delta S = S * (mu dt + sigma epsilon sqrt(dt))
        # Simplified additive walk for generic metric:
        change = self.drift * self.dt + self.volatility * epsilon * np.sqrt(self.dt)
        self._accumulated += change

class ShockManager:
    """
    Manages active stochastic processes.
    """
    def __init__(self):
        self.active_shocks: List[Shock] = []
        
    def add_shock(self, shock: Shock):
        self.active_shocks.append(shock)
        
    def apply_shocks(self, env_state) -> List[str]:
        """
        Applies shocks to environment state (Additively).
        """
        applied = []
        node_map = {name: i for i, name in enumerate(env_state.node_ids)}
        
        for shock in self.active_shocks:
            if not shock.active: continue
            
            if shock.target_metric in node_map:
                idx = node_map[shock.target_metric]
                val = env_state.values[idx]
                
                delta = shock.get_delta(val)
                
                if abs(delta) > 1e-6:
                    # Apply
                    # Note: We add the delta. 
                    # For GBM, we might want multiplicative, but additive logic is safer for generic variables.
                    env_state.values[idx] += delta
                    # applied.append(f"{shock.name}: {delta:+.4f}")
            
            shock.step()
            
        return applied

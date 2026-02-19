"""
Scenario Management Core

Defines the primitives for the Professional Scenario Platform:
1. Shocks as Processes (Time + Shape + Distribution)
2. Policies as Instruments (Rules + Constraints + Lags)
3. Scenarios as Persistable Objects
"""

from __future__ import annotations

import uuid
import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Directory for saving scenarios
SCENARIO_DIR = Path(__file__).parent.parent.parent / "scenarios"
SCENARIO_DIR.mkdir(exist_ok=True)


class ShockShape(str, Enum):
    STEP = "step"       # Jump and stay
    PULSE = "pulse"     # Jump and return immediately
    RAMP = "ramp"       # Linear increase
    DECAY = "decay"     # Jump and exponential decay
    CYCLICAL = "cyclical" # Sine wave


@dataclass
class ShockProcess:
    """
    Defines a shock as a process over time, not just a point value.
    """
    target: str  # e.g., "demand_shock", "supply_shock"
    magnitude: float
    start_time: int
    duration: int
    shape: ShockShape = ShockShape.STEP
    
    # Advanced parameters
    decay_rate: float = 0.5 # For DECAY shape
    frequency: float = 0.1  # For CYCLICAL shape
    
    # Stochastic parameters (optional)
    distribution: Optional[str] = None # "normal", "uniform"
    std_dev: float = 0.0
    
    def generate_vector(self, total_steps: int) -> np.ndarray:
        """Generate the time-series vector for this shock."""
        vec = np.zeros(total_steps)
        
        start = max(0, min(self.start_time, total_steps))
        end = max(0, min(self.start_time + self.duration, total_steps))
        
        if start >= total_steps:
            return vec
            
        t_range = np.arange(end - start)
        
        if self.shape == ShockShape.STEP:
            vec[start:end] = self.magnitude
            
        elif self.shape == ShockShape.PULSE:
            if start < total_steps:
                vec[start] = self.magnitude
                
        elif self.shape == ShockShape.RAMP:
            # Linear ramp from 0 to magnitude
            slope = self.magnitude / max(1, self.duration)
            vec[start:end] = slope * (t_range + 1)
            
        elif self.shape == ShockShape.DECAY:
            # Mag * (1 - decay)^t
            vec[start:end] = self.magnitude * (1.0 - self.decay_rate) ** t_range
            
        elif self.shape == ShockShape.CYCLICAL:
            # Mag * sin(2pi * freq * t)
            vec[start:end] = self.magnitude * np.sin(2 * np.pi * self.frequency * t_range)
            
        return vec


@dataclass
class PolicyInstrument:
    """
    Defines a policy lever with rules and constraints.
    """
    name: str # e.g. "Central Bank Rate"
    key: str  # e.g. "policy_rate"
    
    # Mode: "fixed" or "rule"
    mode: str = "fixed"
    
    # For "fixed" mode
    fixed_value: float = 0.0
    
    # For "rule" mode (e.g. Taylor Rule parameters)
    rule_params: Dict[str, float] = field(default_factory=dict)
    # Example: {"phi_infl": 1.5, "phi_gap": 0.5, "target_infl": 0.02}
    
    # Constraints (Hard bounds)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Implementation Lags
    decision_lag: int = 0
    implementation_lag: int = 0
    
    def apply_constraints(self, value: float) -> float:
        """Clip value to defined bounds."""
        if self.min_value is not None:
            value = max(self.min_value, value)
        if self.max_value is not None:
            value = min(self.max_value, value)
        return value


@dataclass
class Scenario:
    """
    A unified container for a simulation configuration.
    This is the "First-Class Object".
    """
    name: str
    description: str = ""
    owner: str = "User"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Components
    shocks: List[ShockProcess] = field(default_factory=list)
    policies: List[PolicyInstrument] = field(default_factory=list)
    
    # Base configuration overrides (Simulation settings)
    base_settings: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"steps": 100, "dt": 0.5}

    @classmethod
    def from_dict(cls, data: Dict) -> Scenario:
        """Hydrate from JSON dict."""
        shocks = [ShockProcess(**s) for s in data.get("shocks", [])]
        policies = [PolicyInstrument(**p) for p in data.get("policies", [])]
        
        # Filter out keys that might not exist in older versions or are handled above
        meta = {k: v for k, v in data.items() if k not in ["shocks", "policies"]}
        
        return cls(shocks=shocks, policies=policies, **meta)
        
    def to_dict(self) -> Dict:
        """Serialize to JSON dict."""
        return asdict(self)

    def compile_to_config(self) -> Any:
        """
        Compile this high-level scenario into an executable SFCConfig.
        Returns SFCConfig object (typed Any to avoid circular import at module level).
        """
        # Local import to avoid circular dependency with sfc.py
        from scarcity.simulation.sfc import SFCConfig
        
        # 1. Base Settings
        steps = self.base_settings.get("steps", 50)
        dt = self.base_settings.get("dt", 1.0)
        
        # 2. Compile Shocks to Vectors
        shock_vectors = {}
        # Union all keys found in shocks
        all_keys = set([s.target for s in self.shocks])
        
        for key in all_keys:
            # Sum all processes targeting this key
            total_vec = np.zeros(steps)
            for s in self.shocks:
                if s.target == key:
                    total_vec += s.generate_vector(steps)
            shock_vectors[key] = total_vec
            
        # 3. Extract Constraints & Rules
        constraints = {}
        rule_overrides = {}
        
        for p in self.policies:
            # Constraints
            if p.min_value is not None or p.max_value is not None:
                constraints[p.key] = (p.min_value, p.max_value)
            
            # Rule Parameters (e.g. Taylor Rule coefficients)
            if p.mode == "rule" and p.rule_params:
                # Map standardized keys to SFCConfig keys if needed
                # For now assume direct mapping or simple translation
                for k, v in p.rule_params.items():
                    rule_overrides[k] = v
                
        # 4. Construct Config
        return SFCConfig(
            steps=steps,
            dt=dt,
            shock_vectors=shock_vectors,
            constraints=constraints,
            # Pass through other base settings and rule overrides
            **{**{k:v for k,v in self.base_settings.items() if k not in ["steps", "dt"]}, **rule_overrides}
        )

class ScenarioManager:
    """
    Persistence layer for Scenarios.
    """
    
    @staticmethod
    def save_scenario(scenario: Scenario) -> Path:
        """Save scenario to disk."""
        path = SCENARIO_DIR / f"{scenario.id}.json"
        with open(path, "w") as f:
            json.dump(scenario.to_dict(), f, indent=2)
        logger.info(f"Saved scenario {scenario.name} to {path}")
        return path
    
    @staticmethod
    def load_scenario(scenario_id: str) -> Optional[Scenario]:
        """Load scenario from disk."""
        path = SCENARIO_DIR / f"{scenario_id}.json"
        if not path.exists():
            return None
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return Scenario.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load scenario {scenario_id}: {e}")
            return None

    @staticmethod
    def list_scenarios() -> List[Dict[str, str]]:
        """List available scenarios (lightweight)."""
        results = []
        for p in SCENARIO_DIR.glob("*.json"):
            try:
                with open(p, "r") as f:
                    # Just read header info if possible, but JSON loads all. 
                    # For MVP small files, full load is fine.
                    data = json.load(f)
                    results.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "description": data.get("description"),
                        "created_at": data.get("created_at")
                    })
            except Exception:
                continue
        # Sort by date desc
        return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)
    
    @staticmethod
    def delete_scenario(scenario_id: str) -> bool:
        """Delete a scenario."""
        path = SCENARIO_DIR / f"{scenario_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False



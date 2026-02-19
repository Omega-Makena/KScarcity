"""
Policy Controller for KShield (The "Brain").

Wraps the Scarcity SFCEconomy and injects policy actions dynamically
using the Tensor Policy Engine (PID Control).

Architecture:
[Controller]
   |--> [PolicyTensorEngine] (Decides Action)
   |--> [SFCEconomy] (Simulates World)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from scarcity.simulation.sfc import SFCEconomy, SFCConfig
from kshiked.core.tensor_policies import PolicyTensorEngine, TensorEngineConfig
from kshiked.core.policies import EconomicPolicy

logger = logging.getLogger("kshield.controller")

class PolicyController:
    """
    Orchestrates the feedback loop between Economy and Policy.
    """
    
    def __init__(self, economy: SFCEconomy, policies: Dict[str, List[EconomicPolicy]]):
        self.economy = economy
        self.engine = PolicyTensorEngine(TensorEngineConfig())
        
        # 1. Define Observable Metrics
        # These keys must match what we extract from economy state
        self.metric_names = [
            "gdp_growth", 
            "inflation", 
            "unemployment", 
            "interest_rate", 
            "output_gap",
            "inflation_gap",
            "credit_spread"
        ]
        
        # 2. Compile Policy Engine
        self.engine.compile(policies, self.metric_names)
        
    def run(self, steps: int) -> List[Dict[str, Any]]:
        """
        Run the simulation with active control loop.
        """
        trajectory = []
        
        for t in range(steps):
            # 1. Observe State (from PREVIOUS step, or initial)
            # We use 'current_outcomes' and 'current_channels'
            # For t=0, these are initialized defaults
            state_vector = self._extract_state_vector()
            
            # 2. Compute Action
            # Returns dict: {"tighten_policy": 0.5, "stimulus": 0.2}
            actions = self.engine.evaluate(state_vector, dt=self.economy.config.dt)
            
            # 3. Apply Actions (Inject Overrides)
            # Map abstract actions to concrete model inputs
            self._apply_actions(actions)
            
            # 4. Step Economy
            frame = self.economy.step()
            trajectory.append(frame)
            
        return trajectory

    def _extract_state_vector(self) -> np.ndarray:
        """Convert economy state dicts to strict numpy vector."""
        # Combine all sources
        source = {
            **self.economy.current_outcomes,
            **self.economy.current_channels,
            "interest_rate": float(self.economy.interest_rate) # Core attribute
        }
        
        vec = np.zeros(len(self.metric_names))
        for i, name in enumerate(self.metric_names):
            # Partial match or exact?
            # Metric names in policies might be "Inflation..." verbose.
            # But here we defined metric_names strictly.
            # Policies must be written to use THESE names or mapped.
            # For now assuming strict match.
            vec[i] = float(source.get(name, 0.0))
        return vec

    def _apply_actions(self, actions: Dict[str, float]):
        """
        Map generic actions (tighten/stimulus) to Model Inputs (rate/fiscal).
        """
        # Base settings
        # We are modifying the config for the CURRENT step??
        # No, SFCEconomy reads 'policy_schedule' or inner specific attributes.
        # But 'step()' recalculates target_rate unless we override.
        # 'step()' has a section: "2. POLICY". it calculates target_rate.
        # It DOES NOT expose a public override variable easily unless we trick it.
        #
        # TRICK: We can inject into `economy.config.policy_schedule` for the CURRENT time `t`.
        # Or better: We subclassed/wrapped it.
        #
        # Let's use `policy_schedule`.
        current_t = self.economy.time + 1 # Target time is next step
        
        rate_change = 0.0
        fiscal_change = 0.0
        
        # Mapping Logic
        for action, mag in actions.items():
            if action == "tighten_policy":
                rate_change += mag * 0.01 # Mag=1 -> +100bps
            elif action == "ease_policy":
                rate_change -= mag * 0.01
            elif action == "stimulus_package":
                fiscal_change += mag * 0.05 # Mag=1 -> +5% G/Y
            elif action == "austerity":
                fiscal_change -= mag * 0.05
        
        # We need to construct an item for the schedule
        # But SFCEconomy reads schedule at top of step.
        # Simpler: We create a single-step entry.
        override = {
            "t": current_t,
            "policy_rate": self.economy.interest_rate + rate_change,
            "fiscal_impulse": fiscal_change # Impulse is delta
        }
        
        # Helper on Economy must exist, or we append to list
        if self.economy.config.policy_schedule is None:
            self.economy.config.policy_schedule = []
        
        # Remove old entry for t if exists (inefficient list scan, but fine for prototype)
        self.economy.config.policy_schedule = [x for x in self.economy.config.policy_schedule if x["t"] != current_t]
        self.economy.config.policy_schedule.append(override)

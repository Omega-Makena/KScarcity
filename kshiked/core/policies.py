from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scarcity.governor.policies import PolicyRule

@dataclass
class EconomicPolicy(PolicyRule):
    """
    Enhanced policy definition for Economic Governance (V3).
    Now supports PID Control and Crisis Management.
    """
    authority: str = "Central Bank"
    cooldown: int = 5
    temporal_lag: int = 0
    uncertainty_tolerance: float = 0.2
    
    # PID Control Parameters
    kp: float = 0.0          # Proportional Gain (replaces 'factor')
    ki: float = 0.0          # Integral Gain
    kd: float = 0.0          # Derivative Gain
    
    # Crisis Management
    crisis_threshold: float = 999.0 # Value which triggers Crisis Mode
    crisis_multiplier: float = 5.0  # Weight multiplier during crisis

def default_economic_policies() -> Dict[str, List[EconomicPolicy]]:
    """
    Returns a set of default policies for economic governance (V3).
    """
    return {
        "monetary": [
            # Target: Inflation <= 5.0%
            EconomicPolicy(
                metric="Inflation, consumer prices (annual %)",
                threshold=5.0,
                action="tighten_policy",
                direction=">",
                factor=0.0, # Deprecated, using Kp
                priority=1,
                authority="Central Bank",
                cooldown=0, # PID needs continuous control
                uncertainty_tolerance=0.15,
                
                # PID Settings
                kp=0.1,  # Response to current error
                ki=0.01, # Accumulate error (eliminate steady-state)
                kd=0.05, # Dampen rapid changes
                
                # Crisis
                crisis_threshold=15.0 # Hyperinflation trigger
            ),
             # Target: GDP Growth >= 1.0% (Stimulus)
            EconomicPolicy(
                metric="GDP (current US$)", # Proxy for growth level
                threshold=-1.0, # Not used directly for PID setpoint usually, but trigger
                action="stimulus_package",
                direction="<", 
                factor=0.0,
                priority=2,
                authority="Treasury",
                cooldown=0,
                
                kp=0.2,
                ki=0.0,
                kd=0.0,
                
                crisis_threshold=-5.0 # Depression
            ),
        ],
        "fiscal": [
            EconomicPolicy(
                metric="External debt stocks, total (DOD, current US$)",
                threshold=70.0, # Using % as value here for demo simplicity
                action="austerity",
                direction=">",
                factor=0.0,
                priority=3,
                authority="IMF",
                cooldown=10, 
                uncertainty_tolerance=0.05,
                
                kp=0.05,
                ki=0.001,
                kd=0.0
            )
        ]
    }

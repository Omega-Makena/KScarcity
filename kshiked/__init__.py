"""
KShield - National Threat Detection & Economic Governance System

Exposes core components for the unified dashboard.
"""

# Core Governance Modules
from .core.governance import (
    EconomicGovernor, 
    EconomicGovernorConfig
)
from .core.policies import (
    default_economic_policies,
    EconomicPolicy
)
from .core.shocks import (
    Shock, 
    ShockManager, 
    ShockType
)

# Unified Hub (safe import — may fail if pulse dependencies missing)
try:
    from .hub import KShieldHub, get_hub
except ImportError:
    KShieldHub = None
    get_hub = None

# Pulse Components (safe import)
try:
    from .pulse import PulseSensor, compute_threat_report
except ImportError:
    PulseSensor = None
    compute_threat_report = None

__all__ = [
    "EconomicGovernor",
    "EconomicGovernorConfig",
    "default_economic_policies",
    "EconomicPolicy",
    "Shock",
    "ShockManager",
    "ShockType",
    "KShieldHub",
    "get_hub",
    "PulseSensor",
    "compute_threat_report"
]


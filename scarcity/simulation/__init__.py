"""
Scarcity Simulation Package.

Public API surface for the simulation sub-system.
"""

from scarcity.simulation.engine import SimulationEngine, SimulationConfig
from scarcity.simulation.agents import AgentRegistry
from scarcity.simulation.sfc import SFCEconomy, SFCConfig, SectorType, Sector
from scarcity.simulation.learned_sfc import LearnedSFCEconomy, LearnedSFCConfig

__all__ = [
    # Core engine
    "SimulationEngine",
    "SimulationConfig",
    # Agent registry
    "AgentRegistry",
    # SFC model
    "SFCEconomy",
    "SFCConfig",
    "SectorType",
    "Sector",
    # Learned SFC
    "LearnedSFCEconomy",
    "LearnedSFCConfig",
]

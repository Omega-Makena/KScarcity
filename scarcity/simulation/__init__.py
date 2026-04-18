"""
Scarcity Simulation Package.

Public API surface for the simulation sub-system.
"""

from scarcity.simulation.engine import SimulationEngine, SimulationConfig
from scarcity.simulation.agents import AgentRegistry
from scarcity.simulation.sfc import SFCEconomy, SFCConfig, SectorType, Sector
from scarcity.simulation.learned_sfc import LearnedSFCEconomy, LearnedSFCConfig
from scarcity.simulation.types import (
    EconomyState,
    PolicyState,
    Sector as ProductionSector,
    SectorFeedback,
    ShockVector,
    StepResult,
)
from scarcity.simulation.parameters import AllParams
from scarcity.simulation.coupling_interface import (
    AggregatedFeedback,
    MacroExposure,
    SectorModelProtocol,
    aggregate_feedback,
)
from scarcity.simulation.accounting import accounting_warnings, run_accounting_checks
from scarcity.simulation.sfc_engine import (
    EngineConfig,
    MultiSectorSFCEngine,
    default_initial_state,
    find_steady_state,
    step,
)

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
    # Step 1 core typed interfaces
    "EconomyState",
    "PolicyState",
    "ProductionSector",
    "SectorFeedback",
    "ShockVector",
    "StepResult",
    # Step 1 parameters and coupling
    "AllParams",
    "AggregatedFeedback",
    "MacroExposure",
    "SectorModelProtocol",
    "aggregate_feedback",
    # Step 1 accounting
    "run_accounting_checks",
    "accounting_warnings",
    # Step 2 engine
    "EngineConfig",
    "MultiSectorSFCEngine",
    "default_initial_state",
    "step",
    "find_steady_state",
]

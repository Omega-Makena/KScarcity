"""FastAPI dependency wiring."""

from fastapi import Depends, Request

from app.core.datasets import DatasetRegistry
from app.core.scarcity_manager import ScarcityCoreManager
from app.engine import EngineRunner
from app.simulation.manager import SimulationManager


def get_simulation_manager(request: Request) -> SimulationManager:
    """Resolve the simulation manager from the application state."""

    return request.app.state.simulation  # type: ignore[attr-defined]


def get_engine_runner(request: Request) -> EngineRunner:
    """Resolve the MPIE engine runner from application state."""

    runner = getattr(request.app.state, "engine_runner", None)
    if runner is None:
        raise RuntimeError("Engine runner not initialised")
    return runner


def get_dataset_registry(request: Request) -> DatasetRegistry:
    """Resolve dataset registry from application state."""

    registry = getattr(request.app.state, "dataset_registry", None)
    if registry is None:
        raise RuntimeError("Dataset registry not initialised")
    return registry


def get_scarcity_manager(request: Request) -> ScarcityCoreManager:
    """Resolve the scarcity core manager from application state."""
    
    manager = getattr(request.app.state, "scarcity_manager", None)
    if manager is None:
        raise RuntimeError("Scarcity core manager not initialised")
    return manager


SimulationManagerDep = Depends(get_simulation_manager)
EngineRunnerDep = Depends(get_engine_runner)
DatasetRegistryDep = Depends(get_dataset_registry)
ScarcityManagerDep = Depends(get_scarcity_manager)

__all__ = [
    "SimulationManagerDep",
    "EngineRunnerDep",
    "DatasetRegistryDep",
    "ScarcityManagerDep",
    "get_simulation_manager",
    "get_engine_runner",
    "get_dataset_registry",
    "get_scarcity_manager",
]


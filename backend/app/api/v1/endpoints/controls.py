"""Simulation control endpoints."""

from fastapi import APIRouter

from app.core.dependencies import SimulationManagerDep
from app.schemas.controls import DriftRequest, PauseRequest
from app.simulation.manager import SimulationManager

router = APIRouter()


@router.post("/drift")
def inject_drift(payload: DriftRequest, simulation: SimulationManager = SimulationManagerDep) -> dict[str, str]:
    """Trigger a drift scenario."""

    return simulation.inject_drift(payload)


@router.post("/pause")
def pause_simulation(payload: PauseRequest, simulation: SimulationManager = SimulationManagerDep) -> dict[str, str]:
    """Pause or resume parts of the simulation."""

    return simulation.apply_pause(payload)


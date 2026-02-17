"""Risk and assurance endpoints."""

from fastapi import APIRouter

from app.core.dependencies import SimulationManagerDep
from app.schemas.risk import RiskStatusResponse
from app.simulation.manager import SimulationManager


router = APIRouter()


@router.get("/", response_model=RiskStatusResponse)
def read_risk(simulation: SimulationManager = SimulationManagerDep) -> RiskStatusResponse:
    """Return risk card data."""

    return simulation.get_risk_status()


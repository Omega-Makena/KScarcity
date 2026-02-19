"""Status and heartbeat endpoints."""

import warnings
from fastapi import APIRouter, Response

from app.core.config import get_settings
from app.core.dependencies import SimulationManagerDep
from app.schemas.status import HealthResponse, SimulationStatus
from app.simulation.manager import SimulationManager

router = APIRouter()


@router.get("/", response_model=HealthResponse, deprecated=True)
def read_status(
    response: Response,
    simulation: SimulationManager = SimulationManagerDep
) -> HealthResponse:
    """
    Return service heartbeat.
    
    **DEPRECATED**: This endpoint is deprecated. Please use /api/v2/runtime/status instead.
    """
    # Add deprecation warning header
    response.headers["X-API-Warn"] = "Deprecated: Use /api/v2/runtime/status instead"
    response.headers["Deprecation"] = "true"
    
    settings = get_settings()
    sim_status = SimulationStatus(
        mode="paused" if simulation.is_paused else "live",
        tick_seconds=simulation.tick_seconds,
    )
    return HealthResponse(
        service=settings.project_name,
        status="ok",
        simulation=sim_status,
    )


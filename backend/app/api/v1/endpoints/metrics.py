"""Metrics endpoints for KPI summaries."""

from fastapi import APIRouter

from app.core.dependencies import SimulationManagerDep
from app.schemas.metrics import SummaryMetricsResponse
from app.simulation.manager import SimulationManager

router = APIRouter()


@router.get("/summary", response_model=SummaryMetricsResponse)
def read_summary(mode: str = "stakeholder", simulation: SimulationManager = SimulationManagerDep) -> SummaryMetricsResponse:
    """Return KPI summary metrics for the requested mode."""

    mode_literal = "client" if mode == "client" else "stakeholder"
    return simulation.get_summary(mode_literal)


"""Endpoints for ingesting external data windows."""

from fastapi import APIRouter

from app.core.dependencies import SimulationManagerDep
from app.schemas.data import DataEntry, DataEntryRecord
from app.simulation.manager import SimulationManager

router = APIRouter()


@router.post("/", response_model=dict[str, str])
def ingest_data(payload: DataEntry, simulation: SimulationManager = SimulationManagerDep) -> dict[str, str]:
    """Ingest a new data entry."""

    return simulation.ingest_data(payload)


@router.get("/recent", response_model=list[DataEntryRecord])
def recent_data(simulation: SimulationManager = SimulationManagerDep) -> list[DataEntryRecord]:
    """Return recently ingested data windows."""

    return simulation.recent_ingest()



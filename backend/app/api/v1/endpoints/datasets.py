"""Dataset hub endpoints exposing locally ingested datasets."""

from fastapi import APIRouter, HTTPException

from app.core.datasets import DatasetRegistry
from app.core.dependencies import DatasetRegistryDep, SimulationManagerDep
from app.schemas.datasets import DatasetDetail, DatasetListResponse, DatasetSummary
from app.simulation.manager import SimulationManager

router = APIRouter()


@router.get("/", response_model=DatasetListResponse)
def list_datasets(
    registry: DatasetRegistry = DatasetRegistryDep,
) -> DatasetListResponse:
    """Return summaries plus aggregated domain stats."""

    summaries = [DatasetSummary(**summary) for summary in registry.summaries()]
    domains = registry.domain_breakdown()
    totals = registry.totals()
    return DatasetListResponse(
        datasets=summaries,
        domains=domains,
        totals=totals,
    )


@router.get("/{dataset_id}", response_model=DatasetDetail)
def dataset_detail(dataset_id: str, registry: DatasetRegistry = DatasetRegistryDep) -> DatasetDetail:
    """Return full metadata for a dataset."""

    record = registry.detail(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return DatasetDetail(**record)


@router.post("/refresh", response_model=DatasetListResponse)
def refresh_datasets(
    registry: DatasetRegistry = DatasetRegistryDep,
    simulation: SimulationManager = SimulationManagerDep,
) -> DatasetListResponse:
    """Force reload from artifacts/local_dataset_report.json."""

    registry.refresh()
    simulation.refresh_dataset_overlay()
    return list_datasets(registry)

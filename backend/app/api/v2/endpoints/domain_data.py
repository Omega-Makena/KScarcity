"""Domain data visualization API endpoints - v2."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class DataWindowResponse(BaseModel):
    """Single data window response."""

    timestamp: str
    domain_id: int
    domain_name: str
    window_id: int
    features_shape: List[int]
    features_sample: List[List[float]]
    scarcity_signal: float
    source: str
    upload_id: Optional[int]


class DomainDataResponse(BaseModel):
    """Paginated domain data response."""

    domain_id: int
    domain_name: str
    windows: List[DataWindowResponse]
    total_count: int
    limit: int
    offset: int
    has_more: bool


class DomainStatisticsResponse(BaseModel):
    """Domain statistics response."""

    domain_id: int
    total_windows: int
    synthetic_count: int
    manual_count: int
    avg_scarcity: float
    min_scarcity: float
    max_scarcity: float
    generation_rate: float
    last_window_at: Optional[str]
    first_window_at: Optional[str]


@router.get("/{domain_id}/data", response_model=DomainDataResponse)
async def get_domain_data(
    domain_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    source: Optional[str] = Query(default=None, regex="^(synthetic|manual)$"),
    scarcity: ScarcityCoreManager = ScarcityManagerDep,
) -> DomainDataResponse:
    """
    Get data windows for specific domain.

    Query Parameters:
    - limit: Max windows to return (default 50, max 200)
    - offset: Pagination offset (default 0)
    - source: Filter by source ("synthetic" or "manual")
    """
    if not scarcity.domain_data_store:
        raise HTTPException(
            status_code=503, detail="Domain data store not initialized"
        )

    # Verify domain exists
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")

    domain = scarcity.domain_manager.get_domain(domain_id)
    if domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")

    # Get windows
    windows = scarcity.domain_data_store.get_windows(
        domain_id=domain_id, limit=limit, offset=offset, source=source
    )

    # Get total count (without pagination)
    all_windows = scarcity.domain_data_store.get_windows(
        domain_id=domain_id, limit=10000, offset=0, source=source
    )
    total_count = len(all_windows)

    # Convert to response format
    window_responses = [
        DataWindowResponse(**window.to_dict()) for window in windows
    ]

    has_more = (offset + limit) < total_count

    return DomainDataResponse(
        domain_id=domain_id,
        domain_name=domain.name,
        windows=window_responses,
        total_count=total_count,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


@router.get("/{domain_id}/statistics", response_model=DomainStatisticsResponse)
async def get_domain_statistics(
    domain_id: int, scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DomainStatisticsResponse:
    """
    Get aggregated statistics for domain.

    Returns:
    - total_windows: Total windows processed
    - synthetic_count: Number of synthetic windows
    - manual_count: Number of manual windows
    - avg_scarcity: Average scarcity signal
    - min_scarcity: Minimum scarcity signal
    - max_scarcity: Maximum scarcity signal
    - generation_rate: Windows per minute
    - last_window_at: Timestamp of last window
    - first_window_at: Timestamp of first window
    """
    if not scarcity.domain_data_store:
        raise HTTPException(
            status_code=503, detail="Domain data store not initialized"
        )

    # Verify domain exists
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")

    domain = scarcity.domain_manager.get_domain(domain_id)
    if domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")

    # Get statistics
    stats = scarcity.domain_data_store.get_statistics(domain_id)

    if stats is None:
        # Return empty statistics if no data
        return DomainStatisticsResponse(
            domain_id=domain_id,
            total_windows=0,
            synthetic_count=0,
            manual_count=0,
            avg_scarcity=0.0,
            min_scarcity=0.0,
            max_scarcity=0.0,
            generation_rate=0.0,
            last_window_at=None,
            first_window_at=None,
        )

    return DomainStatisticsResponse(**stats.to_dict())


@router.get("/{domain_id}/latest", response_model=DataWindowResponse)
async def get_latest_window(
    domain_id: int, scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DataWindowResponse:
    """Get most recent data window for domain."""
    if not scarcity.domain_data_store:
        raise HTTPException(
            status_code=503, detail="Domain data store not initialized"
        )

    # Verify domain exists
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")

    domain = scarcity.domain_manager.get_domain(domain_id)
    if domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")

    # Get latest window
    window = scarcity.domain_data_store.get_latest_window(domain_id)

    if window is None:
        raise HTTPException(
            status_code=404, detail=f"No data windows found for domain {domain_id}"
        )

    return DataWindowResponse(**window.to_dict())

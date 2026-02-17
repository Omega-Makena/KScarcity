"""Aggregate FastAPI routers."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    controls,
    data,
    datasets,
    domains,
    metrics,
    mpie,
    onboarding,
    risk,
    status,
    streams,
)

api_router = APIRouter()
api_router.include_router(status.router, prefix="/status", tags=["status"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(domains.router, prefix="/domains", tags=["domains"])
api_router.include_router(risk.router, prefix="/risk", tags=["risk"])
api_router.include_router(controls.router, prefix="/controls", tags=["controls"])
api_router.include_router(streams.router, prefix="/ws", tags=["streams"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
api_router.include_router(mpie.router, prefix="/mpie", tags=["mpie"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])

__all__ = ["api_router"]


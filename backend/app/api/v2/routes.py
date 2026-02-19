"""Aggregate v2 API routers."""

from fastapi import APIRouter

from app.api.v2.endpoints import health, runtime, mpie, drg, federation, meta, simulation, metrics, domains, federation_v2, domain_data, demo

api_v2_router = APIRouter()

# Health endpoint
api_v2_router.include_router(health.router, tags=["health-v2"])

# Metrics endpoints
api_v2_router.include_router(metrics.router, prefix="/metrics", tags=["metrics-v2"])

# Domain management endpoints
api_v2_router.include_router(domains.router, prefix="/domains", tags=["domains-v2"])

# Domain data visualization endpoints
api_v2_router.include_router(domain_data.router, prefix="/domains", tags=["domain-data-v2"])

# Demo mode endpoints
api_v2_router.include_router(demo.router, prefix="/demo", tags=["demo"])

# Runtime Bus endpoints
api_v2_router.include_router(runtime.router, prefix="/runtime", tags=["runtime-v2"])

# MPIE endpoints
api_v2_router.include_router(mpie.router, prefix="/mpie", tags=["mpie-v2"])

# DRG endpoints
api_v2_router.include_router(drg.router, prefix="/drg", tags=["drg-v2"])

# Federation endpoints (stub - not yet implemented)
api_v2_router.include_router(federation.router, prefix="/federation", tags=["federation-v2"])

# Federation v2 endpoints (multi-domain)
api_v2_router.include_router(federation_v2.router, prefix="/federation-v2", tags=["federation-multi-domain"])

# Meta Learning endpoints
api_v2_router.include_router(meta.router, prefix="/meta", tags=["meta-v2"])

# Simulation endpoints (stub - not yet implemented)
api_v2_router.include_router(simulation.router, prefix="/simulation", tags=["simulation-v2"])

__all__ = ["api_v2_router"]

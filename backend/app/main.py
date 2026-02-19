"""FastAPI application entry-point."""


import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.api.routes import api_router
from app.api.v2.routes import api_v2_router
from app.core.config import get_settings
from app.core.datasets import DatasetRegistry
from app.core.scarcity_manager import ScarcityCoreManager
from app.core.logging_config import setup_logging
from app.core.error_handlers import (
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from app.engine import EngineRunner
from app.simulation.manager import SimulationManager
from scarcity.dashboard.server import attach_simulation_manager, create_app as create_scic_app
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Setup logging
setup_logging()


def create_app() -> FastAPI:
    """Application factory."""

    settings = get_settings()
    app = FastAPI(
        title=settings.project_name,
        description="""
# Scarcity Backend API

This API provides access to the SCARCITY framework - an online-first framework for 
scarcity-aware deep learning with adaptive, resource-efficient machine learning.

## API Versions

### v1 (Deprecated)
- **Base Path**: `/api/v1`
- **Status**: Deprecated - uses mock data
- **Migration**: Please migrate to v2 endpoints

### v2 (Current)
- **Base Path**: `/api/v2`
- **Status**: Active - backed by scarcity core components
- **Features**: Real-time data from Runtime Bus, MPIE, DRG, Meta Learning

## Components

- **Runtime Bus**: Event-driven communication fabric
- **MPIE**: Multi-Path Inference Engine for causal discovery
- **DRG**: Dynamic Resource Governor for adaptive resource management
- **Federation**: Decentralized learning (coming soon)
- **Meta Learning**: Cross-domain optimization
- **Simulation**: 3D hypergraph visualization (coming soon)

## Getting Started

1. Check system health: `GET /api/v2/health`
2. View Runtime Bus status: `GET /api/v2/runtime/status`
3. Ingest data: `POST /api/v2/mpie/ingest`
4. Monitor resources: `GET /api/v2/drg/status`
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    scic_app = create_scic_app()
    scic_app.state.parent_app = app  # type: ignore[attr-defined]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register error handlers
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    @app.on_event("startup")
    async def startup_event() -> None:
        dataset_registry = DatasetRegistry(Path(BASE_DIR) / "artifacts" / "local_dataset_report.json")
        app.state.dataset_registry = dataset_registry

        simulation = SimulationManager(
            seed=settings.simulation_seed,
            tick_seconds=settings.simulation_tick_seconds,
            dataset_registry=dataset_registry,
        )
        app.state.simulation = simulation
        attach_simulation_manager(scic_app, simulation)

        engine_runner = EngineRunner()
        await engine_runner.start()
        app.state.engine_runner = engine_runner
        
        # Initialize and start Scarcity Core components
        if settings.scarcity_enabled:
            scarcity_manager = ScarcityCoreManager(settings=settings)
            await scarcity_manager.initialize()
            await scarcity_manager.start()
            app.state.scarcity_manager = scarcity_manager

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        # Shutdown Scarcity Core components first
        scarcity_manager = getattr(app.state, "scarcity_manager", None)
        if scarcity_manager is not None:
            await scarcity_manager.stop()
        
        engine_runner = getattr(app.state, "engine_runner", None)
        if engine_runner is not None:
            await engine_runner.stop()

        if hasattr(app.state, "simulation"):
            del app.state.simulation

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    app.include_router(api_v2_router, prefix=settings.api_v2_prefix)
    app.mount("/scic", scic_app)
    return app


app = create_app()

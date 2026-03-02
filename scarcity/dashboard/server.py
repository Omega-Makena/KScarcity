"""
Scarcity Dashboard Server Module.

Provides a minimal ASGI application stub for the dashboard.
This module is the integration surface between the FastAPI backend and
the Scarcity simulation dashboard.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI

    _fastapi_available = True
except ImportError:  # pragma: no cover
    _fastapi_available = False


def create_app() -> Any:  # type: ignore[return]
    """
    Create and return the Scarcity dashboard ASGI application.

    Returns a FastAPI instance if FastAPI is available, otherwise raises
    ImportError with actionable guidance.
    """
    if not _fastapi_available:
        raise ImportError(
            "FastAPI is required to run the Scarcity dashboard server. "
            "Install it with: pip install scarcity[dashboard]"
        )
    app = FastAPI(
        title="Scarcity Dashboard",
        description="Scarcity simulation and causal discovery dashboard.",
        version="1.0.0",
        docs_url="/docs",
    )
    logger.info("Scarcity dashboard app created")
    return app


def attach_simulation_manager(app: Any, simulation_manager: Any) -> None:
    """
    Attach a SimulationManager to the dashboard app's state.

    Args:
        app: The FastAPI application instance returned by ``create_app()``.
        simulation_manager: A SimulationManager (or compatible) instance whose
            tick loop should be accessible by dashboard endpoints.
    """
    if app is None:
        logger.warning("attach_simulation_manager: app is None; skipping.")
        return
    app.state.simulation = simulation_manager
    logger.info("SimulationManager attached to Scarcity dashboard app.")

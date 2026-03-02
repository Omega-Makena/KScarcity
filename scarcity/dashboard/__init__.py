"""
Scarcity Dashboard Package.

Provides the dashboard ASGI application entry point.
"""

from scarcity.dashboard.server import attach_simulation_manager, create_app

__all__ = [
    "attach_simulation_manager",
    "create_app",
]

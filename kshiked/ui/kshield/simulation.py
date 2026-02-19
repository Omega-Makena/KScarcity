"""
K-SHIELD: Simulations -- Economic Scenario Engine

Thin wrapper -- all logic lives in the sim/ subpackage.
"""

from __future__ import annotations

from .sim import render_simulation

__all__ = ["render_simulation"]

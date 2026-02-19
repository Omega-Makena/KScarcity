"""Schema definitions related to service status."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SimulationStatus(BaseModel):
    """Metadata for the running simulation loop."""

    mode: Literal["live", "paused"] = Field(default="live")
    tick_seconds: float = Field(default=1.0)


class HealthResponse(BaseModel):
    """Response payload for the heartbeat endpoint."""

    service: str = Field(default="scarce-backend")
    status: Literal["ok", "degraded"] = Field(default="ok")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    modes: list[str] = Field(default_factory=lambda: ["stakeholder", "client"])
    simulation: SimulationStatus = Field(default_factory=SimulationStatus)


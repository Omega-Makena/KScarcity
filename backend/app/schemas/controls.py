"""Payloads for simulation control endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class DriftRequest(BaseModel):
    """Request payload to inject concept drift."""

    domain_id: str | None = Field(default=None, description="Target domain or None for global.")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)


class PauseRequest(BaseModel):
    """Request payload to pause or resume simulation scopes."""

    scope: Literal["global", "domain", "client"]
    target_id: str | None = None
    action: Literal["pause", "resume"]


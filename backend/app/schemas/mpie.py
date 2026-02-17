"""Schemas for MPIE engine surface."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MetricSnapshot(BaseModel):
    timestamp: datetime
    values: Dict[str, Any]


class EngineStatus(BaseModel):
    running: bool
    windows_ingested: int
    latest_metric: Optional[MetricSnapshot] = None
    history: List[MetricSnapshot] = Field(default_factory=list)
    resource_profile: Optional[MetricSnapshot] = None
    meta_policy: Optional[MetricSnapshot] = None


class EngineSimulationFrame(BaseModel):
    frame_id: int
    positions: List[List[float]]
    colors: List[List[float]]
    edges: List[List[float]]
    timestamp: datetime

"""Schema definitions for KPI and timeline metrics."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class KpiMetric(BaseModel):
    """Summary KPI metric displayed in hero strip."""

    id: str
    label: str
    value: float
    unit: str | None = None
    delta: float = Field(default=0.0)
    trend: Literal["up", "down", "flat"] = Field(default="flat")


class SummaryMetricsResponse(BaseModel):
    """Wrapper for KPI metrics list."""

    mode: Literal["stakeholder", "client"]
    metrics: list[KpiMetric]
    last_updated: datetime = Field(default_factory=datetime.utcnow)


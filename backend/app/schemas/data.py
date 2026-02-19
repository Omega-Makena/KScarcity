"""Schema definitions for external data ingestion."""

from datetime import datetime
import math
from typing import Literal

from pydantic import BaseModel, Field, conlist, confloat


class MetricValue(BaseModel):
    """Single named metric supplied with the data entry."""

    name: str = Field(..., min_length=1, max_length=64)
    value: float | str = Field(..., description="Raw value for the metric; numeric or categorical.")
    direction: Literal["positive", "negative"] = Field(
        default="positive", description="Whether higher values increase (positive) or decrease (negative) confidence."
    )
    weight: confloat(ge=0.0, le=10.0) = Field(default=1.0, description="Relative weight to apply to the metric.")

    def numeric_value(self) -> float | None:
        """Return the metric as a float if possible; otherwise None."""
        if isinstance(self.value, (int, float)):
            return float(self.value)
        if isinstance(self.value, str):
            try:
                return float(self.value)
            except ValueError:
                return None
        return None


class DataEntry(BaseModel):
    """Payload representing an external data window."""

    domain_id: str = Field(..., description="Target domain identifier, e.g. econ-ke")
    source: str = Field(default="user", description="Descriptor for the data source.")
    metrics: conlist(MetricValue, min_length=1) = Field(
        ..., description="Collection of metrics describing the incoming data window."
    )
    narrative: str | None = Field(default=None, description="Optional note about the update.")

    def derived_delta(self) -> float:
        """Compute a synthetic delta to nudge domain accuracy."""

        total = 0.0
        for metric in self.metrics:
            metric_value = metric.numeric_value()
            if metric_value is None:
                continue
            normalized = math.tanh(metric_value * metric.weight / 100.0)
            contribution = normalized if metric.direction == "positive" else -normalized
            total += contribution

        # Scale to simulation range and clamp.
        return max(-3.0, min(3.0, round(total * 2.5, 2)))


class DataEntryRecord(BaseModel):
    """Internal representation of an ingested entry stored by the simulation."""

    domain_id: str
    source: str
    metrics: list[MetricValue]
    narrative: str | None = None
    impact: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")



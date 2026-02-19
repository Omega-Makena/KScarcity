"""Dataset hub response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetField(BaseModel):
    name: str
    dtype: str = Field(default="unknown", description="Inferred field datatype.")
    missing: int = Field(default=0, description="Number of missing entries.")


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    path: str
    domain: str
    domain_id: int
    rows: int
    columns: int
    windows_emitted: int
    schema_version: str
    last_ingested: str | None = Field(default=None, description="Filesystem modification timestamp if available.")


class DatasetDetail(DatasetSummary):
    fields: list[DatasetField] = Field(default_factory=list)
    missing_total: int = 0


class DomainDatasetAggregate(BaseModel):
    domain: str
    domain_id: int
    datasets: int
    rows: int
    windows: int


class DatasetListResponse(BaseModel):
    datasets: list[DatasetSummary]
    domains: list[DomainDatasetAggregate]
    totals: dict


__all__ = [
    "DatasetField",
    "DatasetSummary",
    "DatasetDetail",
    "DomainDatasetAggregate",
    "DatasetListResponse",
]


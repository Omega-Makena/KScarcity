"""Schema definitions for domain-oriented federated learning views."""

from datetime import datetime

from pydantic import BaseModel, Field


class DomainCompliance(BaseModel):
    """Compliance metadata for a domain."""

    frameworks: list[str] = Field(default_factory=list)
    last_audit: datetime | None = None
    notes: str | None = None


class ClientSummary(BaseModel):
    """Minimal client information for the map/graph."""

    client_id: str
    region: str
    status: str


class DomainSummary(BaseModel):
    """Aggregated domain information."""

    domain_id: str
    name: str
    sector: str
    phase: str
    accuracy: float
    delta: float
    clients: list[ClientSummary] = Field(default_factory=list)
    compliance: DomainCompliance = Field(default_factory=DomainCompliance)
    origin: str = Field(default="scic", description="Source of the domain record (scic|dataset).")
    dataset_count: int = 0
    dataset_rows: int = 0
    dataset_windows: int = 0
    dataset_last_ingested: str | None = None


class DomainListResponse(BaseModel):
    """Envelope for domain list responses."""

    domains: list[DomainSummary]
    selected_domain: DomainSummary | None = None


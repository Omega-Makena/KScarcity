"""Schema definitions for risk and assurance panels."""

from datetime import datetime

from pydantic import BaseModel, Field


class RiskCard(BaseModel):
    """Card displayed in the risk & assurance panel."""

    id: str
    title: str
    status: str
    summary: str
    badges: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    """Individual audit log entry."""

    event_id: str
    domain_id: str | None = None
    description: str
    level: str = "info"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskStatusResponse(BaseModel):
    """Composite response for risk endpoint."""

    cards: list[RiskCard]
    audit_events: list[AuditEvent]


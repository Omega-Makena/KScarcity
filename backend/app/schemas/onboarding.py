"""Schemas for onboarding and federated client registration flows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DomainInfo(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    schema_hash_current: Optional[str] = None


class DomainCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class DomainList(BaseModel):
    domains: List[DomainInfo]
    selected_domain: Optional[DomainInfo] = None


class ClientRegistrationRequest(BaseModel):
    display_name: str
    domain_id: str
    profile_class: str = Field(default="cpu")
    vram_gb: float = Field(default=0.0, ge=0.0)
    email: Optional[str] = None


class ClientRegistrationResponse(BaseModel):
    client_id: str
    api_key: Optional[str]
    state: str
    domain_id: str


class HeartbeatRequest(BaseModel):
    metrics: Dict[str, float] | None = None


class UploadColumn(BaseModel):
    name: str
    suggested_dtype: str
    role: str
    null_pct: float


class UploadPreview(BaseModel):
    upload_id: str
    rows: int
    schema_hash: str
    columns: List[UploadColumn]


class UploadCommitRequest(BaseModel):
    upload_id: str
    mapping: List[Dict[str, str]]


class BasketAssignRequest(BaseModel):
    client_id: str


class BasketAssignment(BaseModel):
    basket_id: str
    label: str
    capacity: int
    policy: Dict[str, Any]
    peers: List[Dict[str, Any]]


class GossipRelayRequest(BaseModel):
    basket_id: str
    payload: Dict[str, Any]


class GossipHistoryEntry(BaseModel):
    basket_id: str
    from_client: str
    payload: Dict[str, Any]
    timestamp: datetime



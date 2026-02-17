"""Wrapper endpoints for onboarding the federated dashboard clients."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import logging

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from dataclasses import asdict, is_dataclass

from app.schemas.onboarding import (
    BasketAssignRequest,
    BasketAssignment,
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    DomainCreateRequest,
    DomainInfo,
    DomainList,
    GossipHistoryEntry,
    GossipRelayRequest,
    HeartbeatRequest,
    UploadColumn,
    UploadCommitRequest,
    UploadPreview,
)
from scarcity.dashboard.onboarding import baskets, clients, domains, gossip, ingestion

router = APIRouter()
logger = logging.getLogger(__name__)


def _to_domain_info(domain) -> DomainInfo:
    if is_dataclass(domain):
        payload = asdict(domain)
    elif isinstance(domain, dict):
        payload = domain
    else:
        payload = domain.__dict__
    return DomainInfo.model_validate(payload)


def _require_api_client(api_key: str | None) -> clients.Client:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    try:
        return clients.authenticate_api_key(api_key)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/domains", response_model=DomainList)
def list_domains(domain_id: str | None = None) -> DomainList:
    domain_objs = domains.list_domains(domain_id=domain_id)
    payload = [_to_domain_info(domain) for domain in domain_objs]
    selected = payload[0] if domain_id and payload else None
    return DomainList(domains=payload, selected_domain=selected)


@router.post("/domains", response_model=DomainInfo, status_code=201)
def create_domain(payload: DomainCreateRequest) -> DomainInfo:
    try:
        domain = domains.create_domain(payload.name, payload.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_domain_info(domain)


@router.get("/domains/{domain_id}", response_model=DomainInfo)
def get_domain(domain_id: str) -> DomainInfo:
    domain = domains.get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found.")
    return _to_domain_info(domain)


@router.post("/clients/register", response_model=ClientRegistrationResponse, status_code=201)
def register_client(payload: ClientRegistrationRequest) -> ClientRegistrationResponse:
    try:
        client, api_key = clients.register_client(
            display_name=payload.display_name,
            domain_id=payload.domain_id,
            profile_class=payload.profile_class,
            vram_gb=payload.vram_gb,
            email=payload.email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ClientRegistrationResponse(
        client_id=client.id,
        api_key=api_key,
        state=client.state.value,
        domain_id=client.domain_id,
    )


@router.post("/clients/heartbeat", response_model=Dict[str, Any])
def heartbeat(
    payload: HeartbeatRequest,
    api_key: str | None = Header(default=None, alias="X-Client-Key"),
) -> Dict[str, Any]:
    client = _require_api_client(api_key)
    updated = clients.record_heartbeat(client.id, payload.metrics)
    return {
        "client_id": updated.id,
        "state": updated.state.value,
        "last_seen": updated.last_seen.isoformat() + "Z" if updated.last_seen else None,
    }


@router.get("/clients", response_model=List[ClientRegistrationResponse])
def list_clients(domain_id: str | None = None) -> List[ClientRegistrationResponse]:
    client_objs = clients.list_clients(domain_id=domain_id)
    return [
        ClientRegistrationResponse(
            client_id=item.id,
            api_key=None,
            state=item.state.value,
            domain_id=item.domain_id,
        )
        for item in client_objs
    ]


@router.post("/ingestion/upload", response_model=UploadPreview, status_code=201)
async def upload_csv(
    domain_id: str,
    file: UploadFile = File(...),
    api_key: str | None = Header(default=None, alias="X-Client-Key"),
) -> UploadPreview:
    client = _require_api_client(api_key)
    try:
        record = ingestion.create_upload(client.id, domain_id, file.filename, file.file)
    except ValueError as exc:
        logger.warning(
            "Upload failed for client=%s domain=%s filename=%s: %s",
            client.id,
            domain_id,
            file.filename,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    columns = [
        UploadColumn(
            name=col.name,
            suggested_dtype=col.suggested_dtype,
            role=col.role,
            null_pct=col.null_pct,
        )
        for col in record.columns
    ]
    return UploadPreview(upload_id=record.id, rows=record.rows, schema_hash=record.schema_hash, columns=columns)


@router.get("/ingestion/schema/{upload_id}", response_model=UploadPreview)
def preview_schema(upload_id: str) -> UploadPreview:
    try:
        record = ingestion.preview_schema(upload_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    columns = [
        UploadColumn(
            name=col.name,
            suggested_dtype=col.suggested_dtype,
            role=col.role,
            null_pct=col.null_pct,
        )
        for col in record.columns
    ]
    return UploadPreview(upload_id=record.id, rows=record.rows, schema_hash=record.schema_hash, columns=columns)


@router.post("/ingestion/commit", response_model=UploadPreview)
def commit_upload(payload: UploadCommitRequest) -> UploadPreview:
    try:
        record = ingestion.commit_upload(payload.upload_id, payload.mapping)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    columns = [
        UploadColumn(
            name=col.name,
            suggested_dtype=col.suggested_dtype,
            role=col.role,
            null_pct=col.null_pct,
        )
        for col in record.columns
    ]
    return UploadPreview(upload_id=record.id, rows=record.rows, schema_hash=record.schema_hash, columns=columns)


@router.post("/baskets/assign", response_model=BasketAssignment)
def assign_basket(
    payload: BasketAssignRequest,
    api_key: str | None = Header(default=None, alias="X-Client-Key"),
) -> BasketAssignment:
    client = _require_api_client(api_key)
    if client.id != payload.client_id:
        raise HTTPException(status_code=403, detail="Client mismatch.")
    try:
        basket_obj, peers = baskets.assign_client(payload.client_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BasketAssignment(
        basket_id=basket_obj.id,
        label=basket_obj.label,
        capacity=basket_obj.capacity,
        policy=basket_obj.policy,
        peers=peers,
    )


@router.get("/baskets/{basket_id}/peers", response_model=List[Dict[str, Any]])
def basket_peers(basket_id: str) -> List[Dict[str, Any]]:
    try:
        return baskets.list_peers(basket_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/gossip/send", response_model=Dict[str, Any])
async def send_gossip(
    payload: GossipRelayRequest,
    api_key: str | None = Header(default=None, alias="X-Client-Key"),
) -> Dict[str, Any]:
    client = _require_api_client(api_key)
    if client.basket_id != payload.basket_id:
        raise HTTPException(status_code=403, detail="Client not part of basket.")
    await gossip.relay_message(payload.basket_id, client.id, payload.payload)
    return {"status": "ok"}


@router.get("/gossip/history", response_model=List[GossipHistoryEntry])
def gossip_history(limit: int = 50) -> List[GossipHistoryEntry]:
    history = gossip.gossip_history(limit=limit)
    return [
        GossipHistoryEntry(
            basket_id=item["basket_id"],
            from_client=item["from_client"],
            payload=item["payload"],
            timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
        )
        for item in history
    ]



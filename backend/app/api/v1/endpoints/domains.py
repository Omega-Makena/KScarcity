"""Domain-related endpoints."""

from fastapi import APIRouter, HTTPException

from app.core.dependencies import SimulationManagerDep
from app.schemas.domains import DomainListResponse
from app.schemas.risk import AuditEvent
from app.simulation.manager import SimulationManager

router = APIRouter()


@router.get("/", response_model=DomainListResponse)
def list_domains(
    domain_id: str | None = None,
    simulation: SimulationManager = SimulationManagerDep,
) -> DomainListResponse:
    """Return summary information about each domain."""

    return simulation.get_domains(domain_id=domain_id)


@router.get("/{domain_id}/audit", response_model=list[AuditEvent])
def domain_audit(domain_id: str, simulation: SimulationManager = SimulationManagerDep) -> list[AuditEvent]:
    """Return audit events for a given domain."""

    domains = {domain.domain_id for domain in simulation.get_domains().domains}
    if domain_id not in domains:
        raise HTTPException(status_code=404, detail="Domain not found.")
    return simulation.get_audit_for_domain(domain_id)


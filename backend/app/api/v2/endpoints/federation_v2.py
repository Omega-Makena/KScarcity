"""Federation API endpoints - v2."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class FederationStatus(BaseModel):
    """Federation status response."""
    active: bool
    total_connections: int
    total_rounds: int
    total_updates_shared: int
    strategy: str
    privacy_enabled: bool
    pending_updates: Dict[int, int]


class ConnectionCreate(BaseModel):
    """Connection creation request."""
    from_domain: int
    to_domain: int


class TopologyCreate(BaseModel):
    """Topology creation request."""
    domain_ids: List[int]
    topology_type: str  # "mesh" or "ring"


@router.post("/enable", status_code=200)
async def enable_federation(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, str]:
    """
    Enable federated learning.
    
    Activates federation across all domains.
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    scarcity.federation_coordinator_v2.enable_federation()
    
    return {"status": "enabled", "message": "Federation activated"}


@router.post("/disable", status_code=200)
async def disable_federation(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, str]:
    """
    Disable federated learning.
    
    Deactivates federation across all domains.
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    scarcity.federation_coordinator_v2.disable_federation()
    
    return {"status": "disabled", "message": "Federation deactivated"}


@router.get("/status", response_model=FederationStatus)
async def get_federation_status(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> FederationStatus:
    """
    Get federation status.
    
    Returns current federation state and metrics.
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    metrics = scarcity.federation_coordinator_v2.get_metrics()
    
    return FederationStatus(
        active=metrics["active"],
        total_connections=metrics["total_connections"],
        total_rounds=metrics["total_rounds"],
        total_updates_shared=metrics["total_updates_shared"],
        strategy=metrics["strategy"],
        privacy_enabled=metrics["privacy_enabled"],
        pending_updates=metrics["pending_updates"]
    )


@router.get("/connections")
async def get_connections(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> List[Dict[str, Any]]:
    """
    Get active P2P connections.
    
    Returns list of all domain-to-domain connections.
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    connections = scarcity.federation_coordinator_v2.get_connections()
    
    return [
        {
            "from_domain": conn.from_domain,
            "to_domain": conn.to_domain,
            "established_at": conn.established_at,
            "updates_shared": conn.updates_shared,
            "last_update_at": conn.last_update_at
        }
        for conn in connections
    ]


@router.post("/connections", status_code=201)
async def create_connection(
    request: ConnectionCreate,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, str]:
    """
    Create P2P connection between domains.
    
    Establishes a connection for model sharing.
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    # Verify domains exist
    from_domain = scarcity.domain_manager.get_domain(request.from_domain)
    to_domain = scarcity.domain_manager.get_domain(request.to_domain)
    
    if from_domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {request.from_domain} not found")
    
    if to_domain is None:
        raise HTTPException(status_code=404, detail=f"Domain {request.to_domain} not found")
    
    scarcity.federation_coordinator_v2.create_connection(
        request.from_domain,
        request.to_domain
    )
    
    return {
        "status": "created",
        "message": f"Connection created from domain {request.from_domain} to {request.to_domain}"
    }


@router.delete("/connections/{from_domain}/{to_domain}", status_code=204)
async def remove_connection(
    from_domain: int,
    to_domain: int,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
):
    """
    Remove P2P connection.
    
    Removes connection between two domains.
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    scarcity.federation_coordinator_v2.remove_connection(from_domain, to_domain)


@router.post("/topology", status_code=201)
async def create_topology(
    request: TopologyCreate,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, Any]:
    """
    Create network topology.
    
    Creates connections between domains in specified topology.
    Supported topologies: "mesh" (all-to-all), "ring" (circular).
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    if not scarcity.domain_manager:
        raise HTTPException(status_code=503, detail="Domain manager not initialized")
    
    # Verify all domains exist
    for domain_id in request.domain_ids:
        domain = scarcity.domain_manager.get_domain(domain_id)
        if domain is None:
            raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
    
    if request.topology_type == "mesh":
        scarcity.federation_coordinator_v2.create_full_mesh(request.domain_ids)
        connections_created = len(request.domain_ids) * (len(request.domain_ids) - 1)
    elif request.topology_type == "ring":
        scarcity.federation_coordinator_v2.create_ring_topology(request.domain_ids)
        connections_created = len(request.domain_ids)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown topology type: {request.topology_type}. Use 'mesh' or 'ring'"
        )
    
    return {
        "status": "created",
        "topology_type": request.topology_type,
        "domains": request.domain_ids,
        "connections_created": connections_created
    }


@router.get("/metrics")
async def get_federation_metrics(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, Any]:
    """
    Get detailed federation metrics.
    
    Returns comprehensive metrics about federation activity.
    """
    if not scarcity.federation_coordinator_v2:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    return scarcity.federation_coordinator_v2.get_metrics()

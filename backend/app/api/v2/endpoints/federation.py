"""Federation Layer API endpoints - v2."""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class PeerInfo(BaseModel):
    """Federation peer information."""
    id: str
    trust_score: float
    updates: int
    latency_ms: float
    status: str


class AggregationStrategy(BaseModel):
    """Aggregation strategy configuration."""
    strategy: str
    trim_percent: float = 0.0
    min_confidence: float = 0.5


class PrivacyMetrics(BaseModel):
    """Privacy metrics."""
    epsilon: float
    delta: float
    noise_level: float
    privacy_budget: float


class FederationUpdate(BaseModel):
    """Federation update submission."""
    domain_id: str
    update_data: Dict[str, Any]
    confidence: float = 1.0


@router.get("/peers", response_model=List[PeerInfo])
async def get_federation_peers(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> List[PeerInfo]:
    """
    Get peer network status.
    
    Returns list of connected peers with trust scores.
    """
    if not scarcity.federation_coordinator:
        raise HTTPException(status_code=503, detail="Federation coordinator not initialized")
    
    peers = scarcity.federation_coordinator.peers()
    
    peer_list = []
    for peer_id, peer_info in peers.items():
        peer_list.append(PeerInfo(
            id=peer_id,
            trust_score=peer_info.trust,
            updates=0,  # TODO: Track update count
            latency_ms=0.0,  # TODO: Track latency
            status="active"
        ))
    
    return peer_list


@router.get("/aggregation/strategy", response_model=AggregationStrategy)
async def get_aggregation_strategy(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> AggregationStrategy:
    """
    Get current aggregation strategy.
    
    Returns aggregation method and configuration.
    """
    if not scarcity.federation_client:
        raise HTTPException(status_code=503, detail="Federation client not initialized")
    
    # Get config from aggregator
    config = scarcity.federation_client.aggregator.config
    
    return AggregationStrategy(
        strategy=config.method.value,
        trim_percent=config.trim_percent,
        min_confidence=config.min_confidence
    )


@router.get("/privacy/metrics", response_model=PrivacyMetrics)
async def get_privacy_metrics(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> PrivacyMetrics:
    """
    Get privacy metrics.
    
    Returns differential privacy parameters and noise levels.
    """
    if not scarcity.federation_client:
        raise HTTPException(status_code=503, detail="Federation client not initialized")
    
    # Get config from privacy guard
    config = scarcity.federation_client.privacy_guard.config
    
    return PrivacyMetrics(
        epsilon=config.epsilon,
        delta=config.delta,
        noise_level=config.noise_scale,
        privacy_budget=config.budget
    )


@router.post("/update", response_model=Dict[str, str])
async def submit_federation_update(
    update: FederationUpdate,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, str]:
    """
    Submit local update for aggregation.
    
    Submits a local model update to the federation network.
    """
    if not scarcity.federation_client:
        raise HTTPException(status_code=503, detail="Federation client not initialized")
    
    try:
        # Create a simple packet with the update data
        packet = {
            "domain_id": update.domain_id,
            "data": update.update_data,
            "confidence": update.confidence,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Queue for export
        await scarcity.federation_client._outbound_queue.put(("federation.update", packet))
        
        return {
            "status": "queued",
            "message": "Update queued for federation export"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit update: {str(e)}")

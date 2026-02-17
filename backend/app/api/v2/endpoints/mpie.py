"""MPIE (Multi-Path Inference Engine) API endpoints - v2."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class MPIEStatus(BaseModel):
    """MPIE Orchestrator status."""
    windows_processed: int
    avg_latency_ms: float
    avg_accept_rate: float
    oom_backoff: bool
    running: bool


class ControllerStats(BaseModel):
    """Controller bandit statistics."""
    proposal_entropy: float
    diversity_index: float
    drift_detections: int
    thompson_mode: bool


class EncoderMetrics(BaseModel):
    """Encoder performance metrics."""
    avg_latency_ms: float
    cache_hit_rate: float


class EvaluatorResults(BaseModel):
    """Evaluator results."""
    accept_rate: float
    gain_p50: float
    gain_p90: float
    ci_width_avg: float
    stability_avg: float
    total_evaluated: int


class HypergraphNode(BaseModel):
    """Hypergraph node."""
    id: int
    name: str
    domain: int


class HypergraphEdge(BaseModel):
    """Hypergraph edge."""
    source: str
    target: str
    weight: float
    stability: float
    domain: int


class DataIngestionRequest(BaseModel):
    """Data ingestion request."""
    data: List[List[float]]
    schema: Optional[Dict[str, Any]] = None
    window_id: Optional[int] = None


@router.get("/status", response_model=MPIEStatus)
async def get_mpie_status(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> MPIEStatus:
    """
    Get overall MPIE status.
    
    Returns processing statistics and current state.
    """
    if not scarcity.mpie:
        raise HTTPException(status_code=503, detail="MPIE not initialized")
    
    stats = scarcity.mpie.get_stats()
    
    return MPIEStatus(
        windows_processed=stats["windows_processed"],
        avg_latency_ms=stats["avg_latency_ms"],
        avg_accept_rate=stats["avg_accept_rate"],
        oom_backoff=stats["oom_backoff"],
        running=stats["running"]
    )


@router.get("/controller/stats", response_model=ControllerStats)
async def get_controller_stats(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> ControllerStats:
    """
    Get bandit controller statistics.
    
    Returns exploration parameters, drift detection, and sampling mode.
    """
    if not scarcity.mpie or not scarcity.mpie.controller:
        raise HTTPException(status_code=503, detail="MPIE Controller not initialized")
    
    stats = scarcity.mpie.controller.get_stats()
    
    return ControllerStats(
        proposal_entropy=stats.get("proposal_entropy", 0.0),
        diversity_index=stats.get("diversity_index", 0.0),
        drift_detections=stats.get("drift_detections", 0),
        thompson_mode=stats.get("thompson_mode", False)
    )


@router.get("/encoder/metrics", response_model=EncoderMetrics)
async def get_encoder_metrics(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> EncoderMetrics:
    """
    Get encoder performance metrics.
    
    Returns encoding latency and cache statistics.
    """
    if not scarcity.mpie or not scarcity.mpie.encoder:
        raise HTTPException(status_code=503, detail="MPIE Encoder not initialized")
    
    # TODO: Implement encoder stats collection
    return EncoderMetrics(
        avg_latency_ms=scarcity.mpie.latency_ema,
        cache_hit_rate=0.0  # TODO: Add cache hit tracking
    )


@router.get("/evaluator/results", response_model=EvaluatorResults)
async def get_evaluator_results(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> EvaluatorResults:
    """
    Get recent evaluation results.
    
    Returns acceptance rates, gain statistics, and stability metrics.
    """
    if not scarcity.mpie or not scarcity.mpie.evaluator:
        raise HTTPException(status_code=503, detail="MPIE Evaluator not initialized")
    
    stats = scarcity.mpie.evaluator.get_stats()
    
    return EvaluatorResults(
        accept_rate=stats.get("accept_rate", 0.0),
        gain_p50=stats.get("gain_p50", 0.0),
        gain_p90=stats.get("gain_p90", 0.0),
        ci_width_avg=stats.get("ci_width_avg", 0.0),
        stability_avg=stats.get("stability_avg", 0.0),
        total_evaluated=stats.get("total_evaluated", 0)
    )


@router.get("/store/nodes", response_model=List[HypergraphNode])
async def get_hypergraph_nodes(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> List[HypergraphNode]:
    """
    Get hypergraph nodes.
    
    Returns all nodes in the hypergraph store.
    """
    if not scarcity.mpie or not scarcity.mpie.store:
        raise HTTPException(status_code=503, detail="MPIE Hypergraph Store not initialized")
    
    snapshot = scarcity.mpie.store.snapshot()
    nodes_data = snapshot.get("nodes", {})
    
    nodes = []
    for node_id, data in nodes_data.items():
        try:
            nodes.append(HypergraphNode(
                id=int(node_id),
                name=data.get("name", f"node_{node_id}"),
                domain=int(data.get("domain", 0))
            ))
        except (ValueError, TypeError):
            continue
    
    return nodes


@router.get("/store/edges", response_model=List[HypergraphEdge])
async def get_hypergraph_edges(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> List[HypergraphEdge]:
    """
    Get hypergraph edges with weights and stability.
    
    Returns all edges in the hypergraph store.
    """
    if not scarcity.mpie or not scarcity.mpie.store:
        raise HTTPException(status_code=503, detail="MPIE Hypergraph Store not initialized")
    
    snapshot = scarcity.mpie.store.snapshot()
    edges_data = snapshot.get("edges", {})
    nodes_data = snapshot.get("nodes", {})
    
    edges = []
    for edge_key, data in edges_data.items():
        try:
            # Parse edge key "(src, dst)"
            inner = edge_key.strip()[1:-1]
            src_s, dst_s = inner.split(",")
            src_id = int(src_s.strip())
            dst_id = int(dst_s.strip())
            
            src_name = nodes_data.get(src_id, {}).get("name", f"node_{src_id}")
            dst_name = nodes_data.get(dst_id, {}).get("name", f"node_{dst_id}")
            
            edges.append(HypergraphEdge(
                source=src_name,
                target=dst_name,
                weight=float(data.get("weight", 0.0)),
                stability=float(data.get("stability", 0.0)),
                domain=int(data.get("domain", 0))
            ))
        except (ValueError, TypeError, KeyError):
            continue
    
    return edges


@router.get("/store/regimes", response_model=Dict[str, Any])
async def get_hypergraph_regimes(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, Any]:
    """
    Get discovered regimes.
    
    Returns regime information from the hypergraph store.
    """
    if not scarcity.mpie or not scarcity.mpie.store:
        raise HTTPException(status_code=503, detail="MPIE Hypergraph Store not initialized")
    
    snapshot = scarcity.mpie.store.snapshot()
    
    # Count edges by domain (proxy for regimes)
    edges_data = snapshot.get("edges", {})
    regime_counts: Dict[int, int] = {}
    
    for data in edges_data.values():
        domain = int(data.get("domain", 0))
        regime_counts[domain] = regime_counts.get(domain, 0) + 1
    
    return {
        "regimes": [
            {"domain": domain, "edge_count": count}
            for domain, count in sorted(regime_counts.items())
        ],
        "total_regimes": len(regime_counts)
    }


@router.post("/ingest", response_model=Dict[str, Any])
async def ingest_data(
    request: DataIngestionRequest,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, Any]:
    """
    Ingest data for processing.
    
    Publishes data to the Runtime Bus for MPIE processing.
    """
    if not scarcity.mpie or not scarcity.bus:
        raise HTTPException(status_code=503, detail="MPIE or Runtime Bus not initialized")
    
    try:
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float32)
        
        # Publish to bus
        await scarcity.bus.publish("data_window", {
            "data": data_array,
            "schema": request.schema or {},
            "window_id": request.window_id or 0,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        return {
            "status": "ingested",
            "rows": data_array.shape[0],
            "features": data_array.shape[1],
            "window_id": request.window_id or 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest data: {str(e)}")

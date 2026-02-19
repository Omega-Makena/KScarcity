"""Metrics and observability endpoints - v2."""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class SystemMetrics(BaseModel):
    """System-wide metrics."""
    uptime_seconds: float
    requests_total: int
    errors_total: int
    timestamp: str


class ComponentMetrics(BaseModel):
    """Component-specific metrics."""
    runtime_bus: Dict[str, Any]
    mpie: Dict[str, Any]
    drg: Dict[str, Any]
    meta: Dict[str, Any]


@router.get("/prometheus")
async def get_prometheus_metrics(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> str:
    """
    Get metrics in Prometheus format.
    
    Returns metrics compatible with Prometheus scraping.
    """
    metrics = []
    
    # Runtime Bus metrics
    if scarcity.bus:
        bus_stats = scarcity.bus.get_stats()
        metrics.append(f"# HELP scarcity_bus_messages_published Total messages published to bus")
        metrics.append(f"# TYPE scarcity_bus_messages_published counter")
        metrics.append(f"scarcity_bus_messages_published {bus_stats['messages_published']}")
        
        metrics.append(f"# HELP scarcity_bus_messages_delivered Total messages delivered")
        metrics.append(f"# TYPE scarcity_bus_messages_delivered counter")
        metrics.append(f"scarcity_bus_messages_delivered {bus_stats['messages_delivered']}")
        
        metrics.append(f"# HELP scarcity_bus_delivery_errors Total delivery errors")
        metrics.append(f"# TYPE scarcity_bus_delivery_errors counter")
        metrics.append(f"scarcity_bus_delivery_errors {bus_stats['delivery_errors']}")
        
        metrics.append(f"# HELP scarcity_bus_topics_active Active topics")
        metrics.append(f"# TYPE scarcity_bus_topics_active gauge")
        metrics.append(f"scarcity_bus_topics_active {bus_stats['topics_active']}")
    
    # MPIE metrics
    if scarcity.mpie:
        mpie_stats = scarcity.mpie.get_stats()
        metrics.append(f"# HELP scarcity_mpie_windows_processed Total windows processed")
        metrics.append(f"# TYPE scarcity_mpie_windows_processed counter")
        metrics.append(f"scarcity_mpie_windows_processed {mpie_stats['windows_processed']}")
        
        metrics.append(f"# HELP scarcity_mpie_latency_ms Average latency in milliseconds")
        metrics.append(f"# TYPE scarcity_mpie_latency_ms gauge")
        metrics.append(f"scarcity_mpie_latency_ms {mpie_stats['avg_latency_ms']}")
        
        metrics.append(f"# HELP scarcity_mpie_accept_rate Average acceptance rate")
        metrics.append(f"# TYPE scarcity_mpie_accept_rate gauge")
        metrics.append(f"scarcity_mpie_accept_rate {mpie_stats['avg_accept_rate']}")
    
    # Component status
    status = scarcity.get_status()
    for component, state in status["components"].items():
        value = 1 if state == "online" else 0
        metrics.append(f"# HELP scarcity_component_status Component status (1=online, 0=offline)")
        metrics.append(f"# TYPE scarcity_component_status gauge")
        metrics.append(f'scarcity_component_status{{component="{component}"}} {value}')
    
    return "\n".join(metrics)


@router.get("/system")
async def get_system_metrics(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, Any]:
    """
    Get system-wide metrics.
    
    Returns aggregated metrics across all components.
    """
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {}
    }
    
    # Runtime Bus
    if scarcity.bus:
        metrics["components"]["runtime_bus"] = scarcity.bus.get_stats()
    
    # MPIE
    if scarcity.mpie:
        metrics["components"]["mpie"] = scarcity.mpie.get_stats()
    
    # DRG
    if scarcity.drg:
        metrics["components"]["drg"] = {
            "running": scarcity.drg._running,
            "control_interval": scarcity.drg.config.control_interval
        }
    
    # Meta
    if scarcity.meta:
        metrics["components"]["meta"] = {
            "running": scarcity.meta._running,
            "rollback_count": scarcity.meta.optimizer.state.rollback_count
        }
    
    return metrics

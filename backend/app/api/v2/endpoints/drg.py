"""Dynamic Resource Governor API endpoints - v2."""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class DRGStatus(BaseModel):
    """DRG current status."""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    vram_percent: float


class DRGForecast(BaseModel):
    """DRG Kalman forecast."""
    cpu_forecast: float
    memory_forecast: float
    gpu_forecast: float
    vram_forecast: float
    timestamp: str


class PolicyTrigger(BaseModel):
    """Policy trigger event."""
    policy_id: str
    subsystem: str
    action: str
    metric: str
    threshold: float
    triggered_at: str


class SubsystemStatus(BaseModel):
    """Subsystem registration status."""
    name: str
    status: str
    load: float
    throttle: float


class PolicyRule(BaseModel):
    """Policy rule definition."""
    metric: str
    threshold: float
    direction: str  # "above" or "below"
    action: str
    factor: float = 1.0


@router.get("/status", response_model=DRGStatus)
async def get_drg_status(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DRGStatus:
    """
    Get current resource utilization.
    
    Returns CPU, memory, GPU, and VRAM usage percentages.
    """
    if not scarcity.drg:
        raise HTTPException(status_code=503, detail="DRG not initialized")
    
    # Sample current metrics
    metrics = scarcity.drg.sensors.sample()
    
    return DRGStatus(
        cpu_percent=metrics.get("cpu", 0.0),
        memory_percent=metrics.get("memory", 0.0),
        gpu_percent=metrics.get("gpu", 0.0),
        vram_percent=metrics.get("vram", 0.0)
    )


@router.get("/forecast", response_model=DRGForecast)
async def get_drg_forecast(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DRGForecast:
    """
    Get Kalman forecasts for next intervals.
    
    Returns predicted resource utilization.
    """
    if not scarcity.drg:
        raise HTTPException(status_code=503, detail="DRG not initialized")
    
    # Get forecast from profiler
    forecast = scarcity.drg.profiler._kalman
    
    return DRGForecast(
        cpu_forecast=forecast.get("cpu", 0.0),
        memory_forecast=forecast.get("memory", 0.0),
        gpu_forecast=forecast.get("gpu", 0.0),
        vram_forecast=forecast.get("vram", 0.0),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@router.get("/policies", response_model=Dict[str, Any])
async def get_drg_policies(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, Any]:
    """
    Get policy status and trigger history.
    
    Returns active policies and their trigger counts.
    """
    if not scarcity.drg:
        raise HTTPException(status_code=503, detail="DRG not initialized")
    
    # Get policies from config
    policies_dict = scarcity.drg.config.policies
    
    policies_list = []
    for subsystem, rules in policies_dict.items():
        for rule in rules:
            policies_list.append({
                "subsystem": subsystem,
                "metric": rule.metric,
                "threshold": rule.threshold,
                "direction": rule.direction,
                "action": rule.action,
                "factor": rule.factor,
                "status": "active"  # TODO: Track actual status
            })
    
    return {
        "policies": policies_list,
        "total_policies": len(policies_list),
        "triggers": []  # TODO: Implement trigger history tracking
    }


@router.get("/subsystems", response_model=List[SubsystemStatus])
async def get_drg_subsystems(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> List[SubsystemStatus]:
    """
    Get registered subsystems and throttle status.
    
    Returns all subsystems registered with DRG.
    """
    if not scarcity.drg:
        raise HTTPException(status_code=503, detail="DRG not initialized")
    
    # Get registered subsystems
    subsystems = []
    try:
        if hasattr(scarcity.drg, 'registry') and hasattr(scarcity.drg.registry, '_subsystems'):
            for name in scarcity.drg.registry._subsystems.keys():
                subsystems.append(SubsystemStatus(
                    name=name,
                    status="running",
                    load=0.0,  # TODO: Track actual load
                    throttle=1.0  # TODO: Track actual throttle
                ))
    except Exception as e:
        # Return empty list if registry not available
        pass
    
    return subsystems


@router.post("/policy", response_model=Dict[str, str])
async def register_policy(
    policy: PolicyRule,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, str]:
    """
    Register or update a policy.
    
    Adds a new policy rule to the DRG.
    """
    if not scarcity.drg:
        raise HTTPException(status_code=503, detail="DRG not initialized")
    
    # TODO: Implement dynamic policy registration
    # For now, return success
    return {
        "status": "registered",
        "message": f"Policy for {policy.metric} registered (TODO: implement dynamic registration)"
    }

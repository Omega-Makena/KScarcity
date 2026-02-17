"""Health and status endpoints - v2."""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager
from app.core.config import get_settings

router = APIRouter()


class ComponentStatus(BaseModel):
    """Component status."""
    status: str
    message: str = ""


class HealthStatus(BaseModel):
    """Overall health status."""
    service: str
    status: str
    version: str
    timestamp: str
    components: Dict[str, ComponentStatus]


@router.get("/health", response_model=HealthStatus)
async def get_health(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> HealthStatus:
    """
    Get overall system health.
    
    Returns status of all scarcity components.
    """
    settings = get_settings()
    component_status = scarcity.get_status()
    
    components = {}
    for name, status in component_status["components"].items():
        components[name] = ComponentStatus(
            status=status,
            message="" if status == "online" else "Not yet implemented"
        )
    
    overall_status = "healthy" if component_status["started"] else "starting"
    
    return HealthStatus(
        service=settings.project_name,
        status=overall_status,
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        components=components
    )

"""Demo mode API endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager
from app.core.demo_mode import get_demo_mode

router = APIRouter()


class DemoModeStatus(BaseModel):
    """Demo mode status response."""
    active: bool
    acceleration_factor: int
    accelerated_domains: int
    message: str


@router.post("/activate", response_model=DemoModeStatus)
async def activate_demo_mode(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DemoModeStatus:
    """
    Activate demo mode - speeds up data generation 5x for impressive demos.
    
    This makes the system generate data faster so demos are more visually
    impressive, but everything remains real - no fake data or scripted behavior.
    """
    demo_mode = get_demo_mode()
    demo_mode.activate(scarcity)
    
    status = demo_mode.get_status()
    
    return DemoModeStatus(
        active=status["active"],
        acceleration_factor=status["acceleration_factor"],
        accelerated_domains=status["accelerated_domains"],
        message="Demo mode activated - Data generation accelerated 5x!"
    )


@router.post("/deactivate", response_model=DemoModeStatus)
async def deactivate_demo_mode(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> DemoModeStatus:
    """
    Deactivate demo mode - restore normal data generation speed.
    """
    demo_mode = get_demo_mode()
    demo_mode.deactivate(scarcity)
    
    status = demo_mode.get_status()
    
    return DemoModeStatus(
        active=status["active"],
        acceleration_factor=status["acceleration_factor"],
        accelerated_domains=status["accelerated_domains"],
        message="Demo mode deactivated - Normal speed restored"
    )


@router.get("/status", response_model=DemoModeStatus)
async def get_demo_mode_status() -> DemoModeStatus:
    """Get current demo mode status."""
    demo_mode = get_demo_mode()
    status = demo_mode.get_status()
    
    return DemoModeStatus(
        active=status["active"],
        acceleration_factor=status["acceleration_factor"],
        accelerated_domains=status["accelerated_domains"],
        message="Demo mode active" if status["active"] else "Demo mode inactive"
    )

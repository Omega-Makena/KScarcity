"""Simulation Engine API endpoints - v2."""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class NodeState(BaseModel):
    """Simulation node state."""
    id: int
    name: str
    position: List[float]
    velocity: List[float]
    domain: int


class EdgeState(BaseModel):
    """Simulation edge state."""
    source: int
    target: int
    weight: float
    stability: float


class SimulationState(BaseModel):
    """Current simulation state."""
    nodes: List[NodeState]
    edges: List[EdgeState]
    time_step: int
    is_playing: bool


class WhatIfScenario(BaseModel):
    """What-if scenario definition."""
    node_id: int
    intervention: str
    magnitude: float


class WhatIfResult(BaseModel):
    """What-if scenario result."""
    scenario_id: str
    affected_nodes: List[int]
    impact_score: float
    timestamp: str


class SimulationControl(BaseModel):
    """Simulation control command."""
    action: str  # "play", "pause", "reset"


@router.get("/state", response_model=SimulationState)
async def get_simulation_state(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> SimulationState:
    """
    Get current simulation state.
    
    Returns node positions, velocities, and edge states.
    """
    if not scarcity.simulation:
        raise HTTPException(status_code=503, detail="Simulation engine not initialized")
    
    # Get visualizer snapshot
    snapshot = scarcity.simulation.visualizer_snapshot()
    
    # Convert to API format
    nodes = []
    positions = snapshot.get("positions", [])
    values = snapshot.get("values", [])
    
    for i, pos in enumerate(positions):
        nodes.append(NodeState(
            id=i,
            name=f"node_{i}",
            position=pos.tolist() if hasattr(pos, 'tolist') else list(pos),
            velocity=[0.0, 0.0, 0.0],  # TODO: Track velocities
            domain=0  # TODO: Get domain from registry
        ))
    
    # Get edges from adjacency matrix
    edges = []
    adjacency = snapshot.get("adjacency", [])
    stability = snapshot.get("stability", [])
    
    if len(adjacency) > 0:
        for i in range(len(adjacency)):
            for j in range(len(adjacency[i])):
                if adjacency[i][j] > 0:
                    edges.append(EdgeState(
                        source=i,
                        target=j,
                        weight=float(adjacency[i][j]),
                        stability=float(stability[i][j]) if len(stability) > i and len(stability[i]) > j else 0.0
                    ))
    
    return SimulationState(
        nodes=nodes,
        edges=edges,
        time_step=snapshot.get("frame_id", 0),
        is_playing=scarcity.simulation._running
    )


@router.get("/trajectory", response_model=Dict[str, Any])
async def get_simulation_trajectory(
    scarcity: ScarcityCoreManager = ScarcityManagerDep,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get historical trajectory data.
    
    Returns time series of node positions and states.
    
    Args:
        limit: Maximum number of trajectory points to return
    """
    if not scarcity.simulation:
        raise HTTPException(status_code=503, detail="Simulation engine not initialized")
    
    # Get stored trajectories
    trajectories = scarcity.simulation.storage.list_trajectories(limit=limit)
    
    return {
        "trajectories": trajectories,
        "time_steps": len(trajectories),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.post("/whatif", response_model=Dict[str, str])
async def execute_whatif_scenario(
    scenario: WhatIfScenario,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, str]:
    """
    Execute what-if scenario.
    
    Runs a counterfactual simulation with specified intervention.
    """
    if not scarcity.simulation:
        raise HTTPException(status_code=503, detail="Simulation engine not initialized")
    
    try:
        # Create scenario ID
        scenario_id = f"whatif_{scenario.node_id}_{scenario.intervention}_{datetime.utcnow().timestamp()}"
        
        # Create node shocks based on intervention
        node_shocks = {str(scenario.node_id): scenario.magnitude}
        
        # Run what-if scenario
        result = scarcity.simulation.run_whatif(
            scenario_id=scenario_id,
            node_shocks=node_shocks,
            edge_shocks=None,
            horizon=None  # Use default horizon
        )
        
        return {
            "status": "completed",
            "scenario_id": scenario_id,
            "message": f"What-if scenario executed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute scenario: {str(e)}")


@router.get("/whatif/{scenario_id}", response_model=WhatIfResult)
async def get_whatif_result(
    scenario_id: str,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> WhatIfResult:
    """
    Get what-if scenario results.
    
    Returns the results of a previously executed scenario.
    """
    if not scarcity.simulation:
        raise HTTPException(status_code=503, detail="Simulation engine not initialized")
    
    # Get stored what-if result
    result = scarcity.simulation.storage.load_whatif(scenario_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
    
    return WhatIfResult(
        scenario_id=scenario_id,
        affected_nodes=result.get("affected_nodes", []),
        impact_score=result.get("impact_score", 0.0),
        timestamp=result.get("timestamp", datetime.utcnow().isoformat() + "Z")
    )


@router.post("/control", response_model=Dict[str, str])
async def control_simulation(
    control: SimulationControl,
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> Dict[str, str]:
    """
    Control simulation playback.
    
    Play, pause, or reset the simulation.
    """
    if not scarcity.simulation:
        raise HTTPException(status_code=503, detail="Simulation engine not initialized")
    
    try:
        if control.action == "play":
            if not scarcity.simulation._running:
                await scarcity.simulation.start()
            return {"status": "playing", "action": control.action}
        
        elif control.action == "pause":
            if scarcity.simulation._running:
                await scarcity.simulation.stop()
            return {"status": "paused", "action": control.action}
        
        elif control.action == "reset":
            # Stop if running
            if scarcity.simulation._running:
                await scarcity.simulation.stop()
            
            # Reset environment
            scarcity.simulation.environment.reset()
            
            return {"status": "reset", "action": control.action}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {control.action}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to control simulation: {str(e)}")

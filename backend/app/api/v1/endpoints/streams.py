"""WebSocket endpoints for live data streaming."""

from typing import Literal

from fastapi import APIRouter, Depends, WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.core.dependencies import get_simulation_manager
from app.simulation.manager import SimulationManager

router = APIRouter()


async def _stream_to_websocket(websocket: WebSocket, stream):
    await websocket.accept()
    try:
        async for payload in stream:
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        return


@router.websocket("/metrics")
async def metrics_stream(
    websocket: WebSocket,
    simulation: SimulationManager = Depends(get_simulation_manager),
) -> None:
    """Stream KPI metrics updates."""

    mode = websocket.query_params.get("mode", "stakeholder")
    mode_literal: Literal["stakeholder", "client"]
    mode_literal = "client" if mode == "client" else "stakeholder"
    await _stream_to_websocket(websocket, simulation.metrics_stream(mode_literal))


@router.websocket("/gossip")
async def gossip_stream(
    websocket: WebSocket,
    simulation: SimulationManager = Depends(get_simulation_manager),
) -> None:
    """Stream gossip events between clients."""

    await _stream_to_websocket(websocket, simulation.gossip_stream())


@router.websocket("/timeline")
async def timeline_stream(
    websocket: WebSocket,
    simulation: SimulationManager = Depends(get_simulation_manager),
) -> None:
    """Stream online learning timeline updates."""

    await _stream_to_websocket(websocket, simulation.timeline_stream())


@router.websocket("/meta")
async def meta_stream(
    websocket: WebSocket,
    simulation: SimulationManager = Depends(get_simulation_manager),
) -> None:
    """Stream meta-learning sandbox data."""

    await _stream_to_websocket(websocket, simulation.meta_stream())


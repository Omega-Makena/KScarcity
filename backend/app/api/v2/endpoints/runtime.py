"""Runtime Bus API endpoints - v2."""

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.dependencies import ScarcityManagerDep
from app.core.scarcity_manager import ScarcityCoreManager

router = APIRouter()


class RuntimeTelemetry(BaseModel):
    """Runtime Bus telemetry data."""
    messages_published: int
    messages_delivered: int
    delivery_errors: int
    topics_active: int
    total_subscribers: int
    timestamp: str


class RuntimeStatus(BaseModel):
    """Runtime Bus status."""
    status: str
    telemetry: RuntimeTelemetry


@router.get("/status", response_model=RuntimeStatus)
async def get_runtime_status(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> RuntimeStatus:
    """
    Get Runtime Bus statistics.
    
    Returns message counts, active topics, and subscriber information.
    """
    if not scarcity.bus:
        raise HTTPException(status_code=503, detail="Runtime Bus not initialized")
    
    stats = scarcity.bus.get_stats()
    
    telemetry = RuntimeTelemetry(
        messages_published=stats["messages_published"],
        messages_delivered=stats["messages_delivered"],
        delivery_errors=stats["delivery_errors"],
        topics_active=stats["topics_active"],
        total_subscribers=stats["total_subscribers"],
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    
    return RuntimeStatus(
        status="online" if scarcity.bus._running else "offline",
        telemetry=telemetry
    )


@router.get("/topics", response_model=List[str])
async def get_runtime_topics(
    scarcity: ScarcityCoreManager = ScarcityManagerDep
) -> List[str]:
    """
    Get list of active topics.
    
    Returns all topics that have at least one subscriber.
    """
    if not scarcity.bus:
        raise HTTPException(status_code=503, detail="Runtime Bus not initialized")
    
    return scarcity.bus.topics()


@router.get("/metrics", response_model=Dict[str, Any])
async def get_runtime_metrics(
    scarcity: ScarcityCoreManager = ScarcityManagerDep,
    limit: int = 100,
    topic: str = None
) -> Dict[str, Any]:
    """
    Get telemetry metrics history.
    
    Returns historical telemetry data for monitoring and analysis.
    
    Args:
        limit: Maximum number of historical events to return (default: 100)
        topic: Filter by specific topic (optional)
    """
    if not scarcity.bus:
        raise HTTPException(status_code=503, detail="Runtime Bus not initialized")
    
    stats = scarcity.get_bus_statistics()
    history = scarcity.get_telemetry_history(limit=limit, topic=topic)
    
    return {
        "current": stats,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "history": history,
        "history_count": len(history)
    }


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time event streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be closed
                pass


manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_runtime_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    
    Streams Runtime Bus events to connected clients.
    
    The client can send JSON messages to control the stream:
    - {"action": "subscribe", "topics": ["topic1", "topic2"]} - Subscribe to specific topics
    - {"action": "unsubscribe", "topics": ["topic1"]} - Unsubscribe from topics
    - {"action": "ping"} - Keep-alive ping
    """
    await manager.connect(websocket)
    
    # Get scarcity manager from app state
    from app.main import app
    scarcity: ScarcityCoreManager = app.state.scarcity_manager
    
    if not scarcity or not scarcity.bus:
        await websocket.close(code=1011, reason="Runtime Bus not available")
        return
    
    subscribed_topics = set()
    
    async def stream_callback(topic: str, data: Any):
        """Forward bus events to WebSocket client."""
        if topic in subscribed_topics:
            try:
                await websocket.send_json({
                    "type": "event",
                    "topic": topic,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            except Exception as e:
                # Connection might be closed
                pass
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "available_topics": scarcity.get_active_topics()
        })
        
        # Handle client messages
        while True:
            message = await websocket.receive_json()
            action = message.get("action")
            
            if action == "subscribe":
                topics = message.get("topics", [])
                for topic in topics:
                    if topic not in subscribed_topics:
                        scarcity.bus.subscribe(topic, stream_callback)
                        subscribed_topics.add(topic)
                
                await websocket.send_json({
                    "type": "subscribed",
                    "topics": list(subscribed_topics),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            
            elif action == "unsubscribe":
                topics = message.get("topics", [])
                for topic in topics:
                    if topic in subscribed_topics:
                        scarcity.bus.unsubscribe(topic, stream_callback)
                        subscribed_topics.remove(topic)
                
                await websocket.send_json({
                    "type": "unsubscribed",
                    "topics": list(subscribed_topics),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            
            elif action == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            
    except WebSocketDisconnect:
        # Clean up subscriptions
        for topic in subscribed_topics:
            scarcity.bus.unsubscribe(topic, stream_callback)
        manager.disconnect(websocket)
    except Exception as e:
        # Clean up subscriptions
        for topic in subscribed_topics:
            scarcity.bus.unsubscribe(topic, stream_callback)
        manager.disconnect(websocket)

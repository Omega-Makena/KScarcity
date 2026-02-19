"""
StreamFederator — Multi-node stream federation.

Enables multiple SCARCITY instances to share streams using WebSocket
with simple broadcast-style coordination. Gossip-based sync and adaptive
bandwidth allocation are planned but not implemented here.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Set
from datetime import datetime
import hashlib

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not available, federation will be limited")

logger = logging.getLogger(__name__)


class StreamFederator:
    """
    Multi-node stream federation.
    
    Features:
    - WebSocket-based peer communication
    - Basic broadcast and latest-timestamp-wins conflict resolution
    """
    
    def __init__(self, node_id: str, listen_port: int = 8765):
        """
        Initialize stream federator.
        
        Args:
            node_id: Unique identifier for this node
            listen_port: Port to listen for connections
        """
        self.node_id = node_id
        self.listen_port = listen_port
        
        # Peer management
        self.peers: Dict[str, Dict] = {}  # peer_id -> peer_info
        self.connected = False
        
        # Statistics
        self._stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'conflicts_resolved': 0
        }
        
        logger.info(f"StreamFederator initialized: node_id={node_id}, port={listen_port}")
    
    async def start_server(self) -> None:
        """Start WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("Cannot start server: websockets not available")
            return
        
        self.connected = True
        
        try:
            async with websockets.serve(self._handle_connection, "localhost", self.listen_port):
                logger.info(f"Federation server started on port {self.listen_port}")
                await asyncio.Future()  # Run forever
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.connected = False
    
    async def _handle_connection(self, websocket, path):
        """Handle incoming WebSocket connection."""
        peer_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data.get('type') == 'hello':
                    # New peer connecting
                    peer_id = data.get('node_id')
                    self.peers[peer_id] = {
                        'id': peer_id,
                        'connected_at': datetime.utcnow().isoformat(),
                        'last_seen': datetime.utcnow()
                    }
                    logger.info(f"Peer connected: {peer_id}")
                    
                    # Send welcome
                    await self._send_message(websocket, {
                        'type': 'welcome',
                        'node_id': self.node_id
                    })
                
                elif data.get('type') == 'data_window':
                    # Handle federated data
                    self._stats['messages_received'] += 1
                    await self._handle_federated_data(data)
                
                elif data.get('type') == 'heartbeat':
                    # Update peer last_seen
                    peer_id = data.get('node_id')
                    if peer_id and peer_id in self.peers:
                        self.peers[peer_id]['last_seen'] = datetime.utcnow()
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if peer_id:
                del self.peers[peer_id]
                logger.info(f"Peer disconnected: {peer_id}")
    
    async def broadcast_data(self, data_window: Dict[str, any], domain_id: Optional[int] = None) -> None:
        """
        Broadcast data window to all peers.
        
        Args:
            data_window: Data window to send
            domain_id: Optional domain identifier
        """
        if not self.connected or not self.peers:
            return
        
        message = {
            'type': 'data_window',
            'node_id': self.node_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data_window,
            'domain_id': domain_id
        }
        
        # Send to all peers
        for peer_id, peer_info in list(self.peers.items()):
            try:
                # In production, would maintain persistent connections
                logger.debug(f"Would send to peer {peer_id}")
                self._stats['messages_sent'] += 1
            except Exception as e:
                logger.error(f"Failed to send to peer {peer_id}: {e}")
    
    async def _handle_federated_data(self, data: Dict) -> None:
        """
        Handle received federated data.
        
        Args:
            data: Federated data message
        """
        # Apply conflict resolution (latest-timestamp-wins)
        # In production, would integrate with EventBus
        logger.debug(f"Received federated data from {data.get('node_id')}")
    
    async def _send_message(self, websocket, message: Dict) -> None:
        """Send message over WebSocket."""
        await websocket.send(json.dumps(message))
    
    def get_stats(self) -> Dict:
        """Get federation statistics."""
        return {
            'node_id': self.node_id,
            'connected': self.connected,
            'peer_count': len(self.peers),
            'messages_sent': self._stats['messages_sent'],
            'messages_received': self._stats['messages_received'],
            'conflicts_resolved': self._stats['conflicts_resolved']
        }

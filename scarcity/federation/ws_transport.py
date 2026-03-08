"""
WebSocket Transport for SCARCITY Federation.

Enables real distributed federation over WebSocket connections.
Each node runs a WebSocket server and connects to peers/coordinator
for bidirectional, real-time packet exchange.

Uses the existing ``websockets`` dependency (>=11.0).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from .transport import BaseTransport, TransportConfig

logger = logging.getLogger("scarcity.federation.ws_transport")


@dataclass
class WSTransportConfig(TransportConfig):
    """Extended configuration for WebSocket transport."""

    protocol: str = "websocket"
    host: str = "0.0.0.0"
    port: int = 8765
    peer_endpoints: Optional[list] = None  # ["ws://10.0.0.2:8765", ...]
    ping_interval: float = 20.0
    ping_timeout: float = 10.0
    max_message_size: int = 10 * 1024 * 1024  # 10 MB
    auth_token: Optional[str] = None


class WebSocketTransport(BaseTransport):
    """
    Production WebSocket transport for distributed federation.

    Runs a local WebSocket server to receive packets from peers, and
    maintains outbound connections to send packets to peer endpoints.

    Usage::

        config = WSTransportConfig(
            host="0.0.0.0",
            port=8765,
            peer_endpoints=["ws://peer1:8765", "ws://peer2:8765"],
        )
        transport = WebSocketTransport(config)
        transport.register_handler(my_handler)
        await transport.start()
        await transport.send("fl.weights_ready", {"weights": [...]})
    """

    def __init__(self, config: WSTransportConfig):
        super().__init__(config)
        self.ws_config: WSTransportConfig = config
        self._server = None
        self._peer_connections: Dict[str, Any] = {}
        self._connection_locks: Dict[str, asyncio.Lock] = {}
        self._active_clients: Set[Any] = set()
        self._server_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Server (receive side)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket server and connect to known peers."""
        if self._running:
            logger.warning("WebSocketTransport already running")
            return

        try:
            import websockets  # type: ignore
        except ImportError:
            raise ImportError(
                "websockets package required for WebSocket transport. "
                "Install with: pip install 'websockets>=11.0'"
            )

        self._running = True

        # Start server
        self._server = await websockets.serve(
            self._handle_connection,
            self.ws_config.host,
            self.ws_config.port,
            ping_interval=self.ws_config.ping_interval,
            ping_timeout=self.ws_config.ping_timeout,
            max_size=self.ws_config.max_message_size,
        )
        logger.info(
            f"WebSocket federation server started on "
            f"ws://{self.ws_config.host}:{self.ws_config.port}"
        )

        # Pre-connect to known peers
        if self.ws_config.peer_endpoints:
            for endpoint in self.ws_config.peer_endpoints:
                asyncio.create_task(self._ensure_connection(endpoint))

    async def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False

        # Close peer connections
        for endpoint, ws in list(self._peer_connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._peer_connections.clear()
        self._connection_locks.clear()

        # Close active server-side connections
        for ws in list(self._active_clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._active_clients.clear()

        # Stop server
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("WebSocket federation transport stopped")

    async def _handle_connection(self, websocket, path=None) -> None:
        """Handle an incoming WebSocket connection (server side)."""
        self._active_clients.add(websocket)
        remote = getattr(websocket, "remote_address", ("unknown", 0))
        logger.info(f"Federation peer connected from {remote}")

        try:
            async for raw_message in websocket:
                try:
                    message = json.loads(raw_message)
                    topic = message.get("topic", "unknown")
                    payload = message.get("payload", {})

                    # Auth check
                    if self.ws_config.auth_token:
                        msg_token = message.get("auth_token")
                        if msg_token != self.ws_config.auth_token:
                            logger.warning(
                                f"Rejected unauthenticated packet from {remote}"
                            )
                            continue

                    await self._dispatch(topic, payload)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {remote}")
                except Exception as e:
                    logger.error(f"Error processing message from {remote}: {e}")

        except Exception as e:
            logger.debug(f"Connection from {remote} closed: {e}")
        finally:
            self._active_clients.discard(websocket)

    # ------------------------------------------------------------------
    # Client (send side)
    # ------------------------------------------------------------------

    async def send(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Send a packet to all connected peers.

        Broadcasts to all known peer endpoints. Failed sends are logged
        but do not raise (fire-and-forget with reconnection).
        """
        message = json.dumps(
            {
                "topic": topic,
                "payload": payload,
                **(
                    {"auth_token": self.ws_config.auth_token}
                    if self.ws_config.auth_token
                    else {}
                ),
            }
        )

        endpoints = list(self.ws_config.peer_endpoints or [])
        if not endpoints:
            # If no explicit peers, broadcast to connected clients
            await self._broadcast_to_clients(message)
            return

        tasks = [self._send_to_peer(ep, message) for ep in endpoints]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to(
        self, endpoint: str, topic: str, payload: Dict[str, Any]
    ) -> None:
        """Send a packet to a specific peer endpoint."""
        message = json.dumps(
            {
                "topic": topic,
                "payload": payload,
                **(
                    {"auth_token": self.ws_config.auth_token}
                    if self.ws_config.auth_token
                    else {}
                ),
            }
        )
        await self._send_to_peer(endpoint, message)

    async def _broadcast_to_clients(self, message: str) -> None:
        """Broadcast to all server-side connected clients."""
        dead = set()
        for ws in self._active_clients:
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self._active_clients -= dead

    async def _send_to_peer(self, endpoint: str, message: str) -> None:
        """Send a message to a specific peer, with auto-reconnect."""
        ws = await self._ensure_connection(endpoint)
        if ws is None:
            logger.warning(f"Cannot reach peer {endpoint}, message dropped")
            return

        try:
            await ws.send(message)
        except Exception as e:
            logger.warning(f"Send to {endpoint} failed: {e}, reconnecting...")
            self._peer_connections.pop(endpoint, None)
            # Try once more
            ws = await self._ensure_connection(endpoint)
            if ws:
                try:
                    await ws.send(message)
                except Exception:
                    logger.error(f"Retry send to {endpoint} also failed")

    async def _ensure_connection(self, endpoint: str):
        """Get or create a WebSocket connection to a peer."""
        # Check existing
        ws = self._peer_connections.get(endpoint)
        if ws is not None:
            try:
                # Quick liveness check
                if ws.open:
                    return ws
            except Exception:
                pass
            self._peer_connections.pop(endpoint, None)

        # Acquire lock for this endpoint to avoid duplicate connections
        if endpoint not in self._connection_locks:
            self._connection_locks[endpoint] = asyncio.Lock()

        async with self._connection_locks[endpoint]:
            # Double-check after acquiring lock
            ws = self._peer_connections.get(endpoint)
            if ws is not None and getattr(ws, "open", False):
                return ws

            try:
                import websockets  # type: ignore

                ws = await asyncio.wait_for(
                    websockets.connect(
                        endpoint,
                        ping_interval=self.ws_config.ping_interval,
                        ping_timeout=self.ws_config.ping_timeout,
                        max_size=self.ws_config.max_message_size,
                    ),
                    timeout=self.ws_config.reconnect_backoff,
                )
                self._peer_connections[endpoint] = ws
                logger.info(f"Connected to federation peer: {endpoint}")

                # Start listener for this peer (bidirectional)
                asyncio.create_task(self._listen_peer(endpoint, ws))

                return ws
            except Exception as e:
                logger.debug(f"Could not connect to {endpoint}: {e}")
                return None

    async def _listen_peer(self, endpoint: str, ws) -> None:
        """Listen for incoming messages on an outbound peer connection."""
        try:
            async for raw_message in ws:
                try:
                    message = json.loads(raw_message)
                    topic = message.get("topic", "unknown")
                    payload = message.get("payload", {})
                    await self._dispatch(topic, payload)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error processing peer message from {endpoint}: {e}")
        except Exception:
            pass
        finally:
            self._peer_connections.pop(endpoint, None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def connected_peers(self) -> int:
        """Number of currently connected outbound peers."""
        return len(
            [ws for ws in self._peer_connections.values() if getattr(ws, "open", False)]
        )

    @property
    def connected_clients(self) -> int:
        """Number of currently connected inbound clients."""
        return len(self._active_clients)

    @property
    def server_address(self) -> str:
        """Address the server is listening on."""
        return f"ws://{self.ws_config.host}:{self.ws_config.port}"

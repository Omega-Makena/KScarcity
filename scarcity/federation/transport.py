"""
Abstract transport interfaces for federation networking.

This module defines the `BaseTransport` abstract base class and concrete implementations
like `LoopbackTransport` (for local testing) and `SimulatedNetworkTransport` (for
simulation). It establishes the contract for sending and receiving federated packets.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


PacketHandler = Callable[[str, Dict[str, Any]], Awaitable[None]]


@dataclass
class TransportConfig:
    """Configuration for transport layers."""
    protocol: str = "loopback"
    endpoint: Optional[str] = None
    reconnect_backoff: float = 5.0


class BaseTransport:
    """
    Abstract base class for transport implementations.
    """

    def __init__(self, config: TransportConfig):
        """
        Initialize the transport.

        Args:
            config: Transport configuration.
        """
        self.config = config
        self._handler: Optional[PacketHandler] = None
        self._running = False

    def register_handler(self, handler: PacketHandler) -> None:
        """
        Register a callback to handle incoming packets.

        Args:
            handler: Async function taking (topic, payload) and returning None.
        """
        self._handler = handler

    async def start(self) -> None:
        """Start the transport layer."""
        self._running = True

    async def stop(self) -> None:
        """Stop the transport layer."""
        self._running = False

    async def send(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Send a packet to the configured destination.

        Args:
            topic: The message topic (e.g., "federation.path_pack").
            payload: The dictionary payload to send.
        """
        raise NotImplementedError

    async def _dispatch(self, topic: str, payload: Dict[str, Any]) -> None:
        """Internal method to dispatch received packets to the handler."""
        if self._handler is not None:
            await self._handler(topic, payload)


class LoopbackTransport(BaseTransport):
    """
    In-process transport that simply routes messages back to the registered handler.
    Useful for unit tests or single-node development where no actual network is needed.
    """

    async def send(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Immediately dispatch the message to the local handler.
        """
        await self._dispatch(topic, payload)


class SimulatedNetworkTransport(BaseTransport):
    """
    Simulates latency and network delays without external dependencies.
    Useful for testing timeouts and async behavior.
    """

    def __init__(self, config: TransportConfig, latency_ms: float = 20.0):
        """
        Initialize with simulated latency.

        Args:
            config: Transport configuration.
            latency_ms: Simulated network latency in milliseconds.
        """
        super().__init__(config)
        self.latency_ms = latency_ms

    async def send(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Send a packet with simulated delay.
        """
        await asyncio.sleep(self.latency_ms / 1000.0)
        await self._dispatch(topic, payload)


def build_transport(config: TransportConfig) -> BaseTransport:
    """
    Build a transport instance from configuration.

    Falls back to LoopbackTransport for unsupported protocols.
    """
    protocol = (config.protocol or "").lower()
    if protocol in {"loopback", "local"}:
        return LoopbackTransport(config)
    if protocol in {"simulated", "simulated_network", "sim"}:
        return SimulatedNetworkTransport(config)
    return LoopbackTransport(config)

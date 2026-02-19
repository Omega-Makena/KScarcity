"""
Federation coordinator responsible for membership and routing.

This module provides the logic for managing peer membership in the federation.
It tracks active peers, handles heartbeats, updates trust scores, and selects
suitable peers for task assignment or aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class PeerInfo:
    """Information about a peer in the federation."""
    peer_id: str
    endpoint: str
    capabilities: Dict[str, float]
    last_heartbeat: float = field(default_factory=time.time)
    trust: float = 0.5


@dataclass
class CoordinatorConfig:
    """Configuration for the FederationCoordinator."""
    heartbeat_timeout: float = 60.0
    fairness_quota_kb_min: int = 512
    mode: str = "mesh"


class FederationCoordinator:
    """
    Manages peer lifecycle and selection.
    """

    def __init__(self, config: CoordinatorConfig):
        """
        Initialize the coordinator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self._peers: Dict[str, PeerInfo] = {}

    def register_peer(self, peer_id: str, endpoint: str, capabilities: Dict[str, float]) -> PeerInfo:
        """
        Register a new peer or update an existing one.

        Args:
            peer_id: Unique identifier for the peer.
            endpoint: Network endpoint (e.g., URL or IP).
            capabilities: Dictionary of peer capabilities.

        Returns:
            The PeerInfo object for the registered peer.
        """
        info = PeerInfo(peer_id=peer_id, endpoint=endpoint, capabilities=capabilities)
        self._peers[peer_id] = info
        return info

    def update_trust(self, peer_id: str, trust: float) -> None:
        """
        Update the trust score for a peer.

        Args:
            peer_id: Peer identifier.
            trust: New trust score.
        """
        if peer_id in self._peers:
            self._peers[peer_id].trust = trust
            self._peers[peer_id].last_heartbeat = time.time()

    def heartbeat(self, peer_id: str, capabilities: Optional[Dict[str, float]] = None) -> None:
        """
        Process a heartbeat from a peer.

        Args:
            peer_id: Peer identifier.
            capabilities: Optional updated capabilities.
        """
        if peer_id not in self._peers:
            return
        self._peers[peer_id].last_heartbeat = time.time()
        if capabilities:
            self._peers[peer_id].capabilities.update(capabilities)

    def prune_inactive(self) -> List[str]:
        """
        Remove peers that have timed out.

        Returns:
            List of removed peer IDs.
        """
        now = time.time()
        inactive = [
            peer_id
            for peer_id, info in self._peers.items()
            if now - info.last_heartbeat > self.config.heartbeat_timeout
        ]
        for peer_id in inactive:
            del self._peers[peer_id]
        return inactive

    def select_peers(self, count: int, min_trust: float = 0.3) -> List[PeerInfo]:
        """
        Select the most trusted peers.

        Args:
            count: Number of peers to select.
            min_trust: Minimum trust score required.

        Returns:
            List of selected PeerInfo objects.
        """
        candidates = [info for info in self._peers.values() if info.trust >= min_trust]
        candidates.sort(key=lambda info: info.trust, reverse=True)
        return candidates[:count]

    def peers(self) -> Dict[str, PeerInfo]:
        """Return a dictionary of all known peers."""
        return dict(self._peers)

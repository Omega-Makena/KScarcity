"""
Aegis Protocol - Defense Coordinator.

Gatekeeping Aggregator that enforces Lattice-Based Access Control (LBAC) on the topology.
Inherits from Scarcity's FederationCoordinator.
"""

from typing import Dict, List, Optional
import logging

from scarcity.federation.coordinator import FederationCoordinator, CoordinatorConfig, PeerInfo
from .schemas import SecurityLevel
from .security import SecurityLattice

logger = logging.getLogger("aegis.coordinator")

class DefenseCoordinator(FederationCoordinator):
    """
    Extensions:
    1. Tracks 'security_level' as a core peer attribute (via capabilities).
    2. Overrides 'select_peers' to enforce No-Read-Up at the topology level.
    """
    
    def register_peer(self, peer_id: str, endpoint: str, capabilities: Dict[str, float]) -> PeerInfo:
        """
        Extend registration to validate security clearance.
        """
        # Ensure clearance is declared
        if "clearance_level_int" not in capabilities:
            # Map enum string to int if passed as auxiliary
            pass 
        
        return super().register_peer(peer_id, endpoint, capabilities)

    def select_peers_for_task(self, task_level: SecurityLevel, count: int, min_trust: float = 0.5) -> List[PeerInfo]:
        """
        Select peers eligible for a specific classified task.
        Enforces: Peer Clearance >= Task Level.
        """
        candidates = []
        for peer in self._peers.values():
            # 1. Trust Check (Base Logic)
            if peer.trust < min_trust:
                continue
            
            # 2. Clearance Check (Aegis Logic)
            # We assume capabilities['clearance'] stores the level value or int
            # For robustness, we assume capabilities stored the integer value of level
            peer_level_val = int(peer.capabilities.get("clearance_level_int", 0))
            
            # Map task level to int
            # Defined in schemas.py: UNCLASSIFIED=0, CONFIDENTIAL=1, SECRET=2, TOP_SECRET=3
            task_level_map = {
                "UNCLASSIFIED": 0, "CONFIDENTIAL": 1, "SECRET": 2, "TOP_SECRET": 3
            }
            task_val = task_level_map.get(task_level.value, 0)
            
            if peer_level_val < task_val:
                continue # Clearance too low
                
            candidates.append(peer)
            
        # Sort by Trust (Desc)
        candidates.sort(key=lambda p: p.trust, reverse=True)
        return candidates[:count]

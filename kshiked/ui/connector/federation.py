"""
Federation Layer Connector.
"""
from __future__ import annotations
import logging
from typing import List, Dict
from datetime import datetime

from .models import AgencyStatus

logger = logging.getLogger("sentinel.connector.federation")

class FederationConnector:
    """Connect to Federation Layer (Aegis Protocol)."""
    
    def __init__(self):
        self._simulator = None
        self._connected = False
    
    def connect(self) -> bool:
        try:
            from kshiked.federation.integration import DefenseFederationSimulator
            self._simulator = DefenseFederationSimulator()
            self._connected = True
            logger.info("Connected to Aegis Defense Federation")
            return True
        except ImportError as e:
            logger.warning(f"Aegis Federation not available: {e}")
            return False
    
    def get_agency_status(self) -> List[AgencyStatus]:
        if not self._connected or not self._simulator:
             return self._get_demo_agencies()
        
        # Pull live stats from Aegis
        state = self._simulator.tick()
        agencies = state["agencies"]
        
        results = []
        for a in agencies:
            # Map Aegis node data to dashboard DTO
            status_str = "active" if a["links_count"] > 0 else "pending"
            role = "Top Secret" if a.get("clearance") == "TOP_SECRET" else "Secret"
            
            results.append(AgencyStatus(
                id=a["id"],
                name=a["id"].replace("-Nexus", ""),
                full_name=f"{role} Clearance Node",
                status=status_str,
                contribution_score=min(1.0, a["links_count"] / 10.0), # Heuristic
                rounds_participated=state["round"],
                last_update=datetime.now()
            ))
            
        return results
    
    def get_rounds(self, limit: int = 20) -> List[Dict]:
        if not self._connected or not self._simulator:
            return []
            
        # In a real app we'd query history. Here we return the latest tick info.
        # This is a bit of a hack since tick() advances state.
        # Ideally, detailed logs would be fetched separately.
        return [
            {
                "round": self._simulator.round_id,
                "participants": 3,
                "convergence": 0.95, # Mock
                "delta_norm": 0.05,
                "timestamp": datetime.now().timestamp()
            }
        ]

    def _get_demo_agencies(self):
        # Fallback
        import random
        return [
             AgencyStatus("NIS", "NIS", "National Intelligence", "active", 0.9, 10, datetime.now()),
             AgencyStatus("KDF", "KDF", "Defense Forces", "active", 0.8, 10, datetime.now())
        ]

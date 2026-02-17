"""
Aegis Protocol - Integration Layer.

Simulates the Defense Federation by running multiple Agency Nodes in a process.
Connects the 'kshiked' Dashboard to the 'Aegis' Logic.
"""

from typing import List, Dict, Any
import random
from datetime import datetime
from .schemas import SecurityLevel, SignalGraph, CausalNode, CausalLink
from .node import AegisNode

class DefenseFederationSimulator:
    """
    Orchestrates the KDF/NIS/DCI nodes for the dashboard.
    Now includes a DefenseCoordinator to demonstrate Gatekeeping.
    """
    def __init__(self):
        from .coordinator import DefenseCoordinator, CoordinatorConfig
        self.coordinator = DefenseCoordinator(CoordinatorConfig(mode="star"))

        # 1. Init Agencies
        # We need to pass loose args for now because ClientAgentConfig is strict.
        
        self.kdf = AegisNode(
            node_id="KDF-Nexus",
            security_level=SecurityLevel.SECRET, # KDF is Secret
            private_key="kdf-secret-key",
            trusted_peers={"NIS-Nexus": "PARTNER_TS", "DCI-Nexus": "PARTNER"}
        )
        
        self.nis = AegisNode(
            node_id="NIS-Nexus",
            security_level=SecurityLevel.TOP_SECRET, # NIS is TS
            private_key="nis-top-secret-key",
            trusted_peers={"KDF-Nexus": "PARTNER", "DCI-Nexus": "PARTNER"}
        )
        
        self.dci = AegisNode(
            node_id="DCI-Nexus",
            security_level=SecurityLevel.SECRET,
            private_key="dci-secret-key",
            trusted_peers={"KDF-Nexus": "PARTNER", "NIS-Nexus": "PARTNER_TS"}
        )
        
        self.nodes = [self.kdf, self.nis, self.dci]
        
        # Register nodes with Coordinator (Gatekeeper check)
        level_map = {"UNCLASSIFIED": 0, "CONFIDENTIAL": 1, "SECRET": 2, "TOP_SECRET": 3}
        for n in self.nodes:
            self.coordinator.register_peer(
                n.node_id, 
                "memory://", 
                {"clearance_level_int": level_map[n.security_level.value]}
            )
        
        self.round_id = 0
        
        # Seed initial data
        self._seed_data()

    def _seed_data(self):
        """Inject some starting classified intel."""
        # NIS knows about "Election Interference" (TS)
        g1 = SignalGraph(
            id="g-nis-001", topic="Election", security_level=SecurityLevel.TOP_SECRET,
            source_agency="NIS", source_enclave="DeepOps",
            nodes=[CausalNode(id="Election_Rigging", label="Rigging", ontology_mapping="HOSTILE_ACT")],
            links=[CausalLink(source="Election_Rigging", target="Unrest", strength=0.9, confidence=0.95)]
        )
        self.nis.ingest_local_update(g1)
        
        # KDF knows about "Borderskirmish" (Secret)
        g2 = SignalGraph(
            id="g-kdf-001", topic="Border", security_level=SecurityLevel.SECRET,
            source_agency="KDF", source_enclave="NorthernCmd",
            nodes=[CausalNode(id="Border_Skirmish", label="Skirmish", ontology_mapping="HOSTILE_ACT")],
            links=[CausalLink(source="Border_Skirmish", target="Food_prices", strength=0.4, confidence=0.7)]
        )
        self.kdf.ingest_local_update(g2)

    async def tick_async(self) -> Dict[str, Any]:
        """
        Run one simulation step:
        1. Nodes Generate Packets (Push)
        2. Router delivers packets
        3. Nodes Ingest (Merge)
        """
        self.round_id += 1
        
        # 1. PUSH: NIS -> KDF/DCI
        # NIS sanitizes TS -> Secret for KDF
        nis_packets = self.nis.create_push_packets("Election")
        
        # 2. DELIVER
        packet_log = []
        for pkt in nis_packets:
            target = next((n for n in self.nodes if n.node_id == pkt.receiver_id), None)
            if target:
                # Call async ingestion
                await target.process_external_packet(pkt)
                packet_log.append({
                    "src": pkt.sender_id, 
                    "dst": pkt.receiver_id, 
                    "topic": pkt.graph.topic,
                    "sanitized": pkt.graph.security_level.value
                })
        
        # KDF Pushes
        kdf_packets = self.kdf.create_push_packets("Border")
        for pkt in kdf_packets:
             target = next((n for n in self.nodes if n.node_id == pkt.receiver_id), None)
             if target:
                 await target.process_external_packet(pkt)
                 packet_log.append({
                    "src": pkt.sender_id, 
                    "dst": pkt.receiver_id, 
                    "topic": pkt.graph.topic,
                    "sanitized": pkt.graph.security_level.value
                })

        return {
            "round": self.round_id,
            "packets": packet_log,
            "agencies": [
                {
                    "id": n.node_id,
                    "nodes_count": len(n.nodes_registry),
                    "links_count": len(n.knowledge_base),
                    "clearance": n.security_level.value
                }
                for n in self.nodes
            ]
        }

    def tick(self) -> Dict[str, Any]:
        """Sync wrapper for async tick."""
        import asyncio
        loop = asyncio.get_event_loop()
        
        if loop.is_running():
            # Apply nest_asyncio to allow re-entrant loop (common in Streamlit/Jupyter)
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                # Fallback: if nest_asyncio missing, we can't run async in sync loop
                return {
                    "round": self.round_id, 
                    "packets": [], 
                    "agencies": [], 
                    "note": "Error: Async loop conflict and nest_asyncio missing"
                }

        return asyncio.run(self.tick_async())

"""
Aegis Protocol - Node Implementation.

Now uses Scarcity's FederationClientAgent as the base.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json

from scarcity.federation.client_agent import FederationClientAgent, ClientAgentConfig
from scarcity.federation.trust_scorer import TrustConfig
from scarcity.federation.gossip import GossipConfig

from .schemas import SignalGraph, InsightPacket, SecurityLevel, CausalLink, CausalNode
from .security import SecurityLattice, CryptoSigner, Sanitizer
from .gossip import DefenseGossip

# Professional Integration: Use Scarcity's Basket Logic or a concrete implementation
class InMemoryBasketManager:
    """Simple in-memory basket manager for federation nodes."""
    def get_client(self, cid): 
        # Minimal shim to satisfy GossipProtocol
        return type('obj', (object,), {'basket_id': 'defense-basket'})
    def get_basket_for_client(self, cid): return 'defense-basket'
    def get_basket_peers(self, bid): return [] 

logger = logging.getLogger("aegis.node")

class AegisNode(FederationClientAgent):
    """
    The Agency Nexus (Level 1 Node) - Extended Client Agent.
    """
    def __init__(
        self, 
        node_id: str, 
        security_level: SecurityLevel,
        private_key: str,
        trusted_peers: Dict[str, str], # Format: {"PeerID": "CLEARANCE_LEVEL"}
        config: Optional[ClientAgentConfig] = None
    ):
        # Default config if none provided
        if not config:
            config = ClientAgentConfig()
            
        super().__init__(node_id, reconciler=None, config=config, transport=None)
        
        self.security_level = security_level
        self.peers_config = trusted_peers
        
        # Ensure gossip config exists
        gossip_cfg = self.config.gossip if hasattr(self.config, 'gossip') and self.config.gossip else GossipConfig()
        
        # Aegis Components
        self.signer = CryptoSigner(private_key)
        # Custom Gossip Protocol (Subclass)
        self.gossip = DefenseGossip(
            gossip_cfg,
            InMemoryBasketManager(), 
            self.security_level
        )
        
        # State: The Worldview (Merged Graph)
        self.knowledge_base: Dict[str, CausalLink] = {} 
        self.nodes_registry: Dict[str, CausalNode] = {}
        
    async def process_external_packet(self, packet: InsightPacket):
        # ... existing logic ...
        # 1. Trust Scoring (Core Logic)
        trust_score = self.trust.score(packet.sender_id)
        if trust_score < 0.5:
            logger.warning(f"[{self.node_id}] Rejected low trust sender {packet.sender_id}")
            return

        # 2. Security Lattice (Aegis Logic)
        if not SecurityLattice.can_read(self.security_level, packet.graph.security_level):
            logger.error(f"[{self.node_id}] SECURITY VIOLATION: Packet level {packet.graph.security_level} > Node level {self.security_level}")
            self.trust.update(packet.sender_id, agreement=0.0, compliance=0.0, impact_delta=-0.5, violation=True)
            return

        # 3. Merge
        self._merge_graph(packet.graph)
        self.trust.update(packet.sender_id, agreement=1.0, compliance=1.0, impact_delta=0.1)
        logger.info(f"[{self.node_id}] Merged external graph from {packet.sender_id}")

    def create_push_packets(self, topic: str = "General") -> List[InsightPacket]:
        """
        Generate packets. Uses peer filtering from config.
        """
        packets = []
        target_peers = [p for p in self.peers_config if self.peers_config[p] != "BLOCKED"]
        
        current_graph = self._export_state_as_graph(topic)
        
        for peer_id in target_peers:
            # DYNAMIC LEVEL LOOKUP
            # Format in config: "PeerID": "CLEARANCE_LEVEL_STR"
            # e.g. "NIS-Nexus": "TOP_SECRET"
            clearance_str = self.peers_config.get(peer_id, "UNCLASSIFIED")
            
            # Map string to SecurityLevel enum
            try:
                # Handle cases where config might have "PARTNER_TS" suffix (legacy)
                # We strip prefixes/suffixes if needed or rely on strict config.
                # Assuming strict config for professional mode.
                if "TOP_SECRET" in clearance_str: peer_level = SecurityLevel.TOP_SECRET
                elif "SECRET" in clearance_str: peer_level = SecurityLevel.SECRET
                elif "CONFIDENTIAL" in clearance_str: peer_level = SecurityLevel.CONFIDENTIAL
                else: peer_level = SecurityLevel.UNCLASSIFIED
            except Exception:
                peer_level = SecurityLevel.UNCLASSIFIED
            
            # Use Gossip Sanitizer logic
            clean_graph = self.gossip.sanitizer.sanitize(current_graph, peer_level)
            clean_graph.signature = self.signer.sign_graph(clean_graph)
            
            packet = InsightPacket(
                id=f"PKT-{datetime.now().timestamp()}",
                sender_id=self.node_id,
                receiver_id=peer_id,
                graph=clean_graph
            )
            packets.append(packet)
            
        return packets

    def ingest_local_update(self, graph: SignalGraph):
        self._merge_graph(graph)

    def _merge_graph(self, graph: SignalGraph):
        """Merge logic: Update weights, add new nodes."""
        for node in graph.nodes:
            if node.id not in self.nodes_registry:
                self.nodes_registry[node.id] = node
        
        for link in graph.links:
            key = f"{link.source}->{link.target}"
            if key in self.knowledge_base:
                existing = self.knowledge_base[key]
                total_conf = existing.confidence + link.confidence
                new_strength = (existing.strength * existing.confidence + link.strength * link.confidence) / total_conf
                
                existing.strength = new_strength
                existing.confidence = min(1.0, total_conf)
                existing.evidence_hash = link.evidence_hash or existing.evidence_hash
            else:
                self.knowledge_base[key] = link

    def _export_state_as_graph(self, topic: str) -> SignalGraph:
        return SignalGraph(
            id=f"G-{datetime.now().timestamp()}",
            topic=topic,
            nodes=list(self.nodes_registry.values()),
            links=list(self.knowledge_base.values()),
            source_agency=self.node_id,
            source_enclave="Main",
            security_level=self.security_level
        )

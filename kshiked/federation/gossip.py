"""
Defense Gossip Protocol.

Extends Scarcity's GossipProtocol to support Causal Graph payloads.
INHERITANCE: Builds ON TOP OF scarcity.federation.gossip.
"""

from typing import List, Optional, cast
import numpy as np
import logging
import json

from scarcity.federation.gossip import GossipProtocol, GossipMessage, GossipConfig
from scarcity.federation.basket import BasketManager

from .schemas import SignalGraph, SecurityLevel
from .security import Sanitizer

logger = logging.getLogger("aegis.gossip")

class DefenseGossip(GossipProtocol):
    """
    Defense-specific Gossip Protocol.
    
    Inherits from Scarcity's GossipProtocol.
    - REUSES: PeerSampler, MaterialityDetector, MessageBudgetTracker, LocalDPMechanism.
    - EXTENDS: Payload serialization (Graph <-> Vector bytes).
    """
    
    def __init__(
        self, 
        config: GossipConfig, 
        basket_manager: BasketManager, 
        security_level: SecurityLevel
    ):
        super().__init__(config, basket_manager)
        self.security_level = security_level
        # We use a defense-specific sanitizer that uses the core DP mechanism internally
        self.sanitizer = Sanitizer(epsilon=config.local_dp_epsilon)

    def create_message(
        self, 
        client_id: str, 
        graph: SignalGraph  # <--- TYPE CHANGE: We accept Graph, not np.ndarray
    ) -> Optional[GossipMessage]:
        """
        Create a gossip message with a Graph payload.
        Overrides parent method to handle serialization.
        """
        # 1. Sanitize graph (Redaction + DP on weights)
        # In a real system, we'd target per-peer, but for gossip broadcast 
        # we sanitize to a baseline safe level (e.g. SECRET or UNCLASSIFIED)
        # depending on the basket's classification.
        safe_graph = self.sanitizer.sanitize(graph, graph.security_level)
        
        # 2. Serialize Graph -> JSON -> Bytes -> Fake Float Vector
        # Scarcity expects np.ndarray (float32). We pack our bytes into it.
        try:
            json_str = safe_graph.model_dump_json()
            data_bytes = json_str.encode('utf-8')
            
            # Pad to multiple of 4 for float32 view
            padding = 4 - (len(data_bytes) % 4)
            if padding < 4:
                data_bytes += b' ' * padding
                
            # View as float32 array
            vector_payload = np.frombuffer(data_bytes, dtype=np.float32)
            
            # Call parent to handle metadata, sequencing, and budget
            # Note: Parent will try to add DP noise to this "vector". 
            # We must be careful! We already applied DP in `sanitize`.
            # If we let parent add noise to serialized bytes, we corrupt the JSON.
            # HACK: We bypass `dp_mechanism.clip_and_noise` by calling internal helper if possible,
            # or we rely on the fact that we passed a "vector".
            
            # Actually, Scarcity's `create_message` calls `clip_and_noise`.
            # Noise on JSON bytes = Corruption.
            # SOLUTION: We must override `create_message` logic entirely but call parent for state.
            
            # Re-implementing just the flow control to avoid double-noising:
            if not self.budget_tracker.can_send(client_id):
                return None
            
            client_info = self.basket_manager.get_client(client_id)
            if client_info is None:
                return None

            seq = self._sequence_numbers[client_id]
            self._sequence_numbers[client_id] = seq + 1
            
            self.budget_tracker.record_message(client_id)
            
            return GossipMessage(
                sender_id=client_id,
                basket_id=client_info.basket_id,
                summary_vector=vector_payload, # Contains the JSON
                sequence_number=seq,
                round_id=self._round_id
            )
            
        except Exception as e:
            logger.error(f"Failed to serialize graph for gossip: {e}")
            return None

    def merge_messages(
        self, 
        messages: List[GossipMessage], 
        decay_half_life: float = 60.0
    ) -> List[SignalGraph]: # <--- RETURNS GRAPHS
        """
        Decode messages and perform Semantic Merge.
        """
        graphs = []
        for msg in messages:
            try:
                # Recover bytes from float array
                data_bytes = msg.summary_vector.tobytes()
                # Trim parsing/padding errors by finding last '}'
                # Simple decode
                json_str = data_bytes.decode('utf-8').strip()
                # Find the actual JSON end (hack for padding)
                end_idx = json_str.rfind('}')
                if end_idx != -1:
                    json_str = json_str[:end_idx+1]
                
                g = SignalGraph.model_validate_json(json_str)
                graphs.append(g)
            except Exception as e:
                logger.warning(f"Failed to decode gossip message from {msg.sender_id}: {e}")
                
        # We don't average them into one graph here because they might be different topics.
        # We return the list for the Node to integrate.
        return graphs

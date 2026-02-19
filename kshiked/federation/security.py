import hashlib
import json
import logging
import numpy as np
from typing import Dict, List, Optional
from .schemas import SecurityLevel, SignalGraph, CausalNode, CausalLink

# Import Core Logic from Scarcity
try:
    from scarcity.federation.gossip import LocalDPMechanism
except ImportError:
    # Fallback if scarcity is not in path (though it should be)
    LocalDPMechanism = None

logger = logging.getLogger("aegis.security")

class SecurityLattice:
    """
    Enforces the No-Read-Up, No-Write-Down policy.
    """
    
    @staticmethod
    def can_read(reader_level: SecurityLevel, data_level: SecurityLevel) -> bool:
        """
        Can a reader at `reader_level` access data at `data_level`?
        Rule: NO READ UP. Reader must be >= Data.
        """
        return reader_level >= data_level

    @staticmethod
    def can_write(writer_level: SecurityLevel, target_level: SecurityLevel) -> bool:
        """
        Can a writer at `writer_level` write to `target_level`?
        Rule: NO WRITE DOWN (prevents accidental leaks). Writer must be <= Target.
        
        Exception: Declassification/Sanitization (handled separately).
        """
        return writer_level <= target_level


class CryptoSigner:
    """
    Handles digital signatures for provenance.
    Simulated using HMAC-SHA256 for this prototype.
    """
    
    def __init__(self, private_key: str):
        self._key = private_key.encode('utf-8')
    
    def sign_graph(self, graph: SignalGraph) -> str:
        """Generate a signature for the graph content."""
        # Create a canonical representation
        payload = f"{graph.id}|{graph.timestamp.isoformat()}|{len(graph.nodes)}|{len(graph.links)}"
        return hashlib.sha256(self._key + payload.encode('utf-8')).hexdigest()
    
    def verify(self, graph: SignalGraph, public_key: str) -> bool:
        """Verify the signature using the sender's public key (simulated)."""
        # In this sim, public_keys match private keys for simplicity
        # Real impl would use RSA/ECC
        payload = f"{graph.id}|{graph.timestamp.isoformat()}|{len(graph.nodes)}|{len(graph.links)}"
        expected = hashlib.sha256(public_key.encode('utf-8') + payload.encode('utf-8')).hexdigest()
        return graph.signature == expected


from .governance import PolicyEngine, DEFAULT_GOVERNANCE_CONFIG

class Sanitizer:
    """
    The Redaction Engine.
    Uses 'scarcity.federation' primitives for rigorous Differential Privacy
    and 'governance.PolicyEngine' for rule-based redaction.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, rules: Optional[Dict] = None):
        self.epsilon = epsilon
        self.delta = delta
        
        # Init Governance
        self.policy_engine = PolicyEngine(rules or DEFAULT_GOVERNANCE_CONFIG)
        
        if LocalDPMechanism:
            # Rigorous DP from core library
            self._dp = LocalDPMechanism(epsilon, delta, sensitivity=1.0)
        else:
            self._dp = None
            logger.warning("Scarcity LocalDPMechanism not found. Using fallback noise.")
    
    def sanitize(self, graph: SignalGraph, target_level: SecurityLevel) -> SignalGraph:
        """
        Create a sanitized copy of the graph for a lower security level.
        """
        if graph.security_level <= target_level:
            return graph # No sanitization needed
        
        # We need to downgrade.
        # 1. Clone
        new_graph = graph.model_copy(deep=True)
        new_graph.security_level = target_level
        new_graph.signature = None # Signature is broken by modification
        
        # 2. Redact Nodes (Policy Driven)
        for node in new_graph.nodes:
            self.policy_engine.apply_node_policy(node, target_level)
            
        # 3. Redact Links (DP Noise)
        # Note: DP Epsilon might be overridden by policy, but for now we use global or instance
        self._batch_sanitize_links(new_graph.links, target_level)
            
        return new_graph

    def _batch_sanitize_links(self, links: List[CausalLink], level: SecurityLevel):
        """
        Apply rigorous DP to the link weights as a vector.
        """
        if not links: return
        
        # Vectorize weights
        weights = np.array([link.strength for link in links])
        
        # Apply DP
        if self._dp:
            # Use Scarcity's verified mechanism
            args = {"vector": weights} if self._dp else {}
            noisy_weights = self._dp.clip_and_noise(weights)
        else:
            # Fallback
            noise = np.random.normal(0, 0.1, size=weights.shape)
            noisy_weights = weights + noise
            
        # Clip back to valid range [-1, 1]
        noisy_weights = np.clip(noisy_weights, -1.0, 1.0)
        
        # Update links
        for i, link in enumerate(links):
            link.strength = float(noisy_weights[i])
            link.evidence_hash = None # Remove evidence trail

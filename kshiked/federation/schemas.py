"""
Aegis Protocol - Data Contracts.

Defines the core structures for the Distributed Causal Knowledge System.
Strict typing ensures defense-grade reliability.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# --- 1. Security Primitive ---

class SecurityLevel(str, Enum):
    """Lattice-Based Access Control Levels."""
    TOP_SECRET = "TOP_SECRET"   # Highest. Can read all. Can only write to TS.
    SECRET = "SECRET"           # Mid. Can read Secret/Unclass.
    CONFIDENTIAL = "CONFIDENTIAL" # Low. Restricted internal.
    UNCLASSIFIED = "UNCLASSIFIED" # Public / Inter-Agency baseline.

    def __lt__(self, other):
        """Allow comparison: UNCLASSIFIED < TOP_SECRET"""
        levels = {
            "UNCLASSIFIED": 0,
            "CONFIDENTIAL": 1,
            "SECRET": 2,
            "TOP_SECRET": 3
        }
        return levels[self.value] < levels[other.value]
    
    def __ge__(self, other):
         levels = {
            "UNCLASSIFIED": 0,
            "CONFIDENTIAL": 1,
            "SECRET": 2,
            "TOP_SECRET": 3
        }
         return levels[self.value] >= levels[other.value]


# --- 2. Causal Graph Primitives ---

class CausalNode(BaseModel):
    """A node in the Knowledge Graph (e.g., 'Inflation')."""
    id: str
    label: str
    type: str = "variable" # variable, actor, event
    ontology_mapping: Optional[str] = None # e.g. "HOSTILE_ACTOR"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CausalLink(BaseModel):
    """A directional edge representing causality."""
    source: str
    target: str
    strength: float = Field(..., ge=-1.0, le=1.0) # -1 to 1 correlation/causation
    confidence: float = Field(..., ge=0.0, le=1.0) # How sure are we?
    lag_hours: int = 0
    
    # Provenance
    evidence_hash: Optional[str] = None # SHA256 of the source document/intel
    is_discovered: bool = True # False if manually set by analyst

# --- 3. The Atomic Unit of Exchange ---

class SignalGraph(BaseModel):
    """
    A Self-Contained Knowledge Subgraph.
    This is what gets exchanged between Agencies.
    """
    id: str
    topic: str # e.g. "Election_Risk_North"
    timestamp: datetime = Field(default_factory=datetime.now)
    
    nodes: List[CausalNode]
    links: List[CausalLink]
    
    # Who created this?
    source_agency: str
    source_enclave: str
    security_level: SecurityLevel
    
    # Digital Signature (filled by CryptoSigner)
    signature: Optional[str] = None

    @field_validator('nodes')
    def validate_nodes(cls, v):
        if not v:
            raise ValueError("Graph must have at least one node")
        return v


# --- 4. The Diplomat's Packet ---

class InsightPacket(BaseModel):
    """
    The envelope for transmission over the wire.
    """
    id: str
    sender_id: str
    receiver_id: str
    
    timestamp: datetime = Field(default_factory=datetime.now)
    ttl_seconds: int = 3600
    
    # The Payload (May be Encrypted)
    graph: SignalGraph
    
    # Negotiation Metadata
    offer_id: Optional[str] = None # If responding to an offer
    protocol_version: str = "1.0.0-AEGIS"

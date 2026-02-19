"""
Verify the Aegis Protocol Refactor.
Ensures DefenseGossip and AegisNode (Inheritance Version) work end-to-end.
"""

import sys
import os
import asyncio
import logging

# Ensure path
sys.path.append(os.getcwd())

from kshiked.federation.integration import DefenseFederationSimulator
from kshiked.federation.schemas import SignalGraph, CausalNode, CausalLink, SecurityLevel
from kshiked.federation.gossip import DefenseGossip
from kshiked.federation.node import AegisNode

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis.verify")

async def test_flow():
    print("--- 1. Initializing Simulator (Spinning up Agencies) ---")
    sim = DefenseFederationSimulator()
    print("Agencies and Coordinator Online:")
    
    # Verify Coordinator
    from kshiked.federation.coordinator import DefenseCoordinator
    assert isinstance(sim.coordinator, DefenseCoordinator), "Must use DefenseCoordinator"
    print(" [OK] DefenseCoordinator Active")
    
    # Verify Policy Engine
    from kshiked.federation.governance import PolicyEngine
    print(" [OK] Policy Engine Active in Sanitizers")

    for node in sim.nodes:
        print(f" - {node.node_id} (Level: {node.security_level.value})")
        # Check if sanitizer has policy engine
        assert node.gossip.sanitizer.policy_engine is not None
        
    print("\n--- 2. Ticking the Ecosystem (Round 1) ---")
    # This triggers creating push packets (Serialization) and receiving (Deserialization + Merge)
    state_r1 = await sim.tick_async()
    
    # Check Packets
    packets = state_r1["packets"]
    print(f"Packets exchanged: {len(packets)}")
    for p in packets:
        print(f" > Packet: {p['src']} -> {p['dst']} [Topic: {p['topic']}]")
        
    if not packets:
        print("[WARN] No packets exchanged on Round 1")
    
    # Force state change to trigger gossip
    print("\n--- 3. Injecting New Intel (Triggering Drift) ---")
    new_graph = SignalGraph(
        id="G-TEST-NEW",
        topic="Election", 
        security_level=SecurityLevel.TOP_SECRET,
        source_agency="NIS",
        source_enclave="Test",
        nodes=[CausalNode(id="TestNode", label="Test", metadata={"Location": "Nairobi_HQ"})], # Location should be masked!
        links=[CausalLink(source="TestNode", target="Unrest", strength=0.99, confidence=1.0)]
    )
    sim.nis.ingest_local_update(new_graph)
    
    print("--- 4. Ticking Round 2 (Should Gossip & Redact) ---")
    state_r2 = await sim.tick_async()
    
    # Verify KDF received it AND it was redacted
    kdf_nodes = sim.kdf.nodes_registry
    if "TestNode" in kdf_nodes:
        node = kdf_nodes["TestNode"]
        loc = node.metadata.get("Location")
        print(f" [SUCCESS] KDF received 'TestNode'. Location = '{loc}'")
        
        # Verify Redaction Policy (UNCLASSIFIED rule says MASK/GENERALIZE)
        if loc != "Nairobi_HQ":
             print(" [SUCCESS] Redaction Policy Enforced (Location hidden/changed)")
        else:
             print(" [FAIL] Redaction Failed! Location is still raw.")
    else:
        print(" [FAIL] KDF did not receive 'TestNode'")
        
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_flow())

"""
Debug script for growth dynamics.

Usage: python -m kshiked.tests.debug_growth
(requires: pip install -e .)
"""
import numpy as np

from scarcity.simulation.agents import AgentRegistry, NodeAgent, EdgeLink
from scarcity.simulation.environment import SimulationEnvironment, EnvironmentConfig
from scarcity.simulation.dynamics import DynamicsEngine, DynamicsConfig

def test_growth():
    registry = AgentRegistry()
    
    # 1. Create Nodes
    registry._nodes["GDP"] = NodeAgent("GDP", "variable", 0, -1, np.zeros(3), 1.0, 100.0)
    registry._nodes["Source"] = NodeAgent("Source", "source", 0, -1, np.zeros(3), 1.0, 1000.0)
    
    # 2. Create Edge
    link = EdgeLink(
        edge_id="Src->GDP", source="Source", target="GDP",
        weight=0.1, stability=1.0, confidence_interval=1.0, regime=0
    )
    registry._edges[link.edge_id] = link
    
    # 3. Setup Env
    env = SimulationEnvironment(registry, EnvironmentConfig(damping=1.0, seed=42))
    dynamics = DynamicsEngine(env, DynamicsConfig(global_damping=1.0, delta_t=1.0))
    
    print("Initial State:")
    state = env.state()
    print(dict(zip(state.node_ids, state.values)))
    print("Adjacency Sum:", state.adjacency.sum())
    
    # 4. Step
    for t in range(3):
        # Force Source
        src_idx = state.node_ids.index("Source")
        state.values[src_idx] = 1000.0
        
        dynamics.step()
        state = env.state()
        print(f"T={t+1}:", dict(zip(state.node_ids, state.values)))

if __name__ == "__main__":
    test_growth()

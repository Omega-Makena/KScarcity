
import logging
import asyncio
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kshield.demo")

try:
    from scarcity.simulation.agents import AgentRegistry, NodeAgent, EdgeLink
    from scarcity.simulation.environment import SimulationEnvironment, EnvironmentConfig
    from scarcity.simulation.dynamics import DynamicsConfig, DynamicsEngine
    # Optional import for the engine
    try:
        from scarcity.engine.engine import MPIEOrchestrator
    except ImportError:
        logger.warning("MPIEOrchestrator not found or broken. Using Mock.")
        MPIEOrchestrator = None
        
except ImportError as e:
    logger.error(f"Failed to import scarcity simulation components: {e}")
    sys.exit(1)

from kshiked.core.governance import EconomicGovernor, EconomicGovernorConfig
from kshiked.core.shocks import ShockManager, OUProcessShock

async def load_and_train_graph(csv_path: str):
    """
    Loads data and runs MPIE (or mock) to discover the causal graph.
    Returns (AgentRegistry, variable_names, df_pivot)
    """
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        return None, None, None

    # --- Data Loading ---
    logger.info("Loading Kenya Economic Dataset...")
    try:
        raw_df = pd.read_csv(csv_path, skiprows=4)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return None, None, None

    target_indicators = [
        "GDP (current US$)",
        "Inflation, consumer prices (annual %)",
        "External debt stocks, total (DOD, current US$)",
        "Exports of goods and services (BoP, current US$)",
        "Imports of goods and services (BoP, current US$)"
    ]
    
    df_filtered = raw_df[raw_df['Indicator Name'].isin(target_indicators)].copy()
    if df_filtered.empty: 
         raw_df['count'] = raw_df.count(axis=1)
         df_filtered = raw_df.sort_values('count', ascending=False).head(5)

    id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    value_vars = [c for c in df_filtered.columns if c.isdigit()]
    df_long = df_filtered.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')
    df_pivot = df_long.pivot(index='Year', columns='Indicator Name', values='Value')
    
    df_pivot.index = df_pivot.index.astype(int)
    df_pivot = df_pivot.sort_index().interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    variable_names = df_pivot.columns.tolist()

    # --- Learning Phase ---
    registry = AgentRegistry()
    
    if MPIEOrchestrator:
        try:
            logger.info("Initializing MPIE Orchestrator...")
            orchestrator = MPIEOrchestrator()
            await orchestrator.start()
            await orchestrator.stop()
        except Exception as e:
             logger.error(f"MPIE Engine failed: {e}. Falling back to manual seed.")
    
    # Always manually seed for this demo to ensure stability of the Governance test
    logger.info("Seeding registry from dataset variables...")
    for name in variable_names:
        val = df_pivot.iloc[-1][name]
        registry._nodes[name] = NodeAgent(
            node_id=str(name), agent_type="variable", domain=0, regime=-1,
            embedding=np.zeros(3, dtype=np.float32), stability=0.8, value=float(val)
        )
        
    return registry, variable_names, df_pivot

async def main_async():
    # Data path relative to project root
    csv_path = Path(__file__).parent.parent.parent / "API_KEN_DS2_en_csv_v2_14659.csv"
    
    # Load data
    registry, variables, df_pivot = await load_and_train_graph(csv_path)
    
    if not registry:
        return

    # --- Simulation Setup ---
    logger.info("Setting up Simulation...")
    env_config = EnvironmentConfig(damping=0.9, energy_cap=100.0, seed=42)
    env = SimulationEnvironment(registry, env_config)
    
    # Initialize values
    latest_vals = df_pivot.iloc[-1].values
    current_ids = env.state().node_ids
    
    initial_values = np.zeros(len(current_ids), dtype=np.float32)
    for i, node_id in enumerate(current_ids):
        if node_id in variables:
            idx = variables.index(node_id)
            initial_values[i] = latest_vals[idx]
        else:
             initial_values[i] = registry.nodes()[node_id].value
             
    env.update_values(initial_values)

    # --- Governance Setup ---
    gov_config = EconomicGovernorConfig(control_interval=2) 
    governor = EconomicGovernor(gov_config, env)
    
    # --- Shock Setup ---
    shock_manager = ShockManager()
    inflation_node = "Inflation, consumer prices (annual %)"
    
    # Add Stochastic Noise
    ou_shock = OUProcessShock(
        name="InflationVolatility", 
        target_metric=inflation_node,
        mu=0.0, 
        sigma=0.5,
        theta=0.2 
    )
    shock_manager.add_shock(ou_shock)

    dynamics = DynamicsEngine(env, DynamicsConfig(global_damping=0.9, delta_t=0.5))
    
    logger.info("Starting V4 Enterprise Simulation Loop (Tensor + Events)...")
    
    steps = 50
    
    for t in range(steps):
        # 1. Apply Shocks
        shock_manager.apply_shocks(env.state())

        # 2. Log State
        state = env.state()
        if inflation_node in state.node_ids:
            idx = state.node_ids.index(inflation_node)
            val = state.values[idx]
            gdp_val = state.values[state.node_ids.index("GDP (current US$)")]
            # logger.info(f"T={t} | Inflation={val:.2f}% | GDP={gdp_val:.2e}")
        
        # 3. Evolve Dynamics
        dynamics.step()
        
        # 4. Evolve Governance (Async in V4)
        await governor.step(t)

    logger.info("Simulation Complete.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

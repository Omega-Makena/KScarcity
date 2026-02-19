
import logging
import asyncio
import pandas as pd
import numpy as np
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demo_economy")

# Ensure we can import scarcity (if installed in editable mode usually works, but just in case)
try:
    import scarcity
    from scarcity.engine.engine import MPIEOrchestrator
    from scarcity.simulation.whatif import WhatIfManager, WhatIfConfig
    from scarcity.simulation.agents import AgentRegistry
    from scarcity.simulation.environment import SimulationEnvironment, EnvironmentConfig
    from scarcity.simulation.dynamics import DynamicsConfig
    logger.info(f"Successfully imported scarcity version {scarcity.__file__}")
except ImportError as e:
    logger.error(f"Failed to import scarcity: {e}")
    sys.exit(1)

async def main():
    # 1. Data Ingestion
    logger.info("Loading Kenya Economic Dataset...")
    csv_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv"
    
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        return

    # Preprocessing: Wide to Long
    # Skip first 4 rows which are metadata
    try:
        raw_df = pd.read_csv(csv_path, skiprows=4)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return

    # Filter for key indicators to keep the graph manageable for this demo
    # We want: GDP, Inflation, Interest Rate (if avail), External Debt, Imports, Exports
    # Looking at the csv file content viewed previously:
    # "GDP (current US$)" - usually available
    # "Inflation, consumer prices (annual %)"
    # "External debt stocks, total (DOD, current US$)"
    # "Exports of goods and services (% of GDP)"
    # "Imports of goods and services (% of GDP)"
    
    target_indicators = [
        "GDP (current US$)",
        "Inflation, consumer prices (annual %)",
        "External debt stocks, total (DOD, current US$)",
        "Exports of goods and services (BoP, current US$)",
        "Imports of goods and services (BoP, current US$)",
        "Net official development assistance received (current US$)" 
    ]
    
    # We need to find the exact codes or names. Let's do a loose match if exact names vary
    df_filtered = raw_df[raw_df['Indicator Name'].isin(target_indicators)].copy()
    
    if df_filtered.empty:
        logger.error("No target indicators found. Checking available names...")
        # Fallback: Just take top 10 indicators with most data
        raw_df['count'] = raw_df.count(axis=1)
        df_filtered = raw_df.sort_values('count', ascending=False).head(10)
        logger.info(f"Selected top 10 populated indicators: {df_filtered['Indicator Name'].tolist()}")

    # Melt to long format
    id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    value_vars = [c for c in df_filtered.columns if c.isdigit()] # Years
    
    df_long = df_filtered.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')
    
    # Pivot to Time Series (Index=Year, Columns=Indicator)
    df_pivot = df_long.pivot(index='Year', columns='Indicator Name', values='Value')
    
    # Clean up
    df_pivot.index = df_pivot.index.astype(int)
    df_pivot = df_pivot.sort_index()
    # Interpolate missing values
    df_pivot = df_pivot.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    logger.info(f"Data Shape: {df_pivot.shape}")
    logger.info(f"Years: {df_pivot.index.min()} - {df_pivot.index.max()}")
    
    # Normalize data for engine (z-score usually good, but engine might handle it. Let's keep it simple)
    # The MPIE engine expects a list of lists or numpy array for the window
    data_matrix = df_pivot.values # shape (T, N_vars)
    variable_names = df_pivot.columns.tolist()
    
    # 2. Learning Phase (Offline/Batch simulation using Online Engine)
    logger.info("Initializing MPIE Orchestrator...")
    orchestrator = MPIEOrchestrator()
    await orchestrator.start()
    
    # Feed data window by window
    window_size = 15 # Increased context
    training_epochs = 20 # Iterate multiple times to allow Bandit convergence
    
    logger.info(f"Starting causal discovery (Epochs: {training_epochs}, Window: {window_size})...")
    
    for epoch in range(training_epochs):
        # Shuffle order? No, time series must represent sequence. 
        # But for active learning, repeated sequential presentation is standard "replay".
        
        for i in range(len(data_matrix) - window_size):
            window = data_matrix[i : i+window_size]
            
            # Construct payload
            payload = {
                "data": window.tolist(),
                "schema": {
                    "fields": [{"name": name} for name in variable_names],
                    "version": 1
                },
                "window_id": epoch * 1000 + i # Unique monotonicity for window_id
            }
            
            await orchestrator._handle_data_window("data_window", payload)
            
        logger.info(f"Epoch {epoch+1}/{training_epochs} complete. Edges: {orchestrator.store.get_edge_count()}")

    # Allow some time for async processing to settle
    await asyncio.sleep(1)
    
    stats = orchestrator.store.get_stats()
    logger.info(f"Learning complete. Store stats: {stats}")
    
    # 3. Simulation Setup
    logger.info("Setting up Simulation Environment...")
    snapshot = orchestrator.store.snapshot()
    registry = AgentRegistry()
    registry.load_from_store_snapshot(snapshot)
    
    from scarcity.simulation.agents import NodeAgent, EdgeLink # Ensure these are imported

    if len(registry.nodes()) == 0:
        logger.warning("No nodes learned! Manually seeding graph with dataset variables.")
        # Create nodes for all variables
        for name in variable_names:
            # Get latest value
            val = df_pivot.iloc[-1][name]
            agent = NodeAgent(
                node_id=str(name),
                agent_type="variable",
                domain=0,
                regime=-1,
                embedding=np.zeros(3, dtype=np.float32),
                stability=0.8,
                value=float(val)
            )
            registry._nodes[name] = agent
            
        # Create a naive causal chain or random edges to demonstrate propagation
        # For demo purposes, we'll link them in a ring or fully connect them weakly
        logger.info("Seeding naive connections for demonstration...")
        nodes = list(registry._nodes.values())
        for i, agent in enumerate(nodes):
            # Connect to next node (ring)
            target = nodes[(i + 1) % len(nodes)]
            edge_id = f"{agent.node_id}->{target.node_id}"
            link = EdgeLink(
                edge_id=edge_id,
                source=agent.node_id,
                target=target.node_id,
                weight=0.5, # moderate positive influence
                stability=0.9,
                confidence_interval=0.1,
                regime=-1
            )
            registry._edges[edge_id] = link
    latest_values = data_matrix[-1]
    
    # The registry nodes might use names from 'Indicator Name'. 
    # We map current values to them.
    mapped_values = {}
    for idx, name in enumerate(variable_names):
        mapped_values[name] = float(latest_values[idx])
        
    env_config = EnvironmentConfig(
        damping=0.8,
        energy_cap=100.0,
        seed=42
    )
    
    env = SimulationEnvironment(registry, env_config)
    
    # Set initial state close to real latest values if possible, 
    # but the simulation agents primarily use the *learned* graph structure.
    # The `SimulationEnvironment` initializes agent values from registry `value` (avg weight).
    # We can perform a "reset" or "set_state" if we had a mapping method. 
    # For now, we rely on the learned structure.
    
    # 4. Shock Simulation
    logger.info("Running Policy Shock Simulation...")
    whatif_cfg = WhatIfConfig(
        horizon_steps=5,
        bootstrap_runs=20
    )
    
    dynamics_cfg = DynamicsConfig(
        global_damping=0.8, 
        delta_t=0.5
    )
    
    manager = WhatIfManager(env, dynamics_cfg, whatif_cfg)
    
    # Define a shock
    # Find a variable to shock. Let's try to shock something related to "Debt" or "Aid".
    shock_node = None
    for name in variable_names:
        if "debt" in name.lower() or "assistance" in name.lower():
            shock_node = name
            break
            
    if not shock_node and variable_names:
        shock_node = variable_names[0]
        
    if shock_node:
        logger.info(f"Applying shock to: {shock_node} (+10% impulse)")
        
        # Shock magnitude: 10% of standard deviation of that series? 
        # Or just a unit shock. Let's do 1.0 unit (z-score space implied if normalized, but here raw data).
        # Since we passed raw data, 1.0 might be small for GDP. 
        # Let's calculate a relative shock.
        shock_mag = 0.1 * np.nanmean(df_pivot[shock_node].values)
        
        scenario_result = manager.run_scenario(
            scenario_id="policy_shock_1",
            node_shocks={shock_node: shock_mag}
        )
        
        # 5. Reporting
        output_file = os.path.join("kshiked", "results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Simulation Results ===\n")
            f.write(f"Scenario: Shocking {shock_node}\n")
            f.write(f"Top Affected Nodes (Horizon {whatif_cfg.horizon_steps}):\n")
            
            logger.info("\n=== Simulation Results ===")
            logger.info(f"Scenario: Shocking {shock_node}")
            logger.info(f"Top Affected Nodes (Horizon {whatif_cfg.horizon_steps}):")
            
            for impact in scenario_result["top_impacts"]:
                line = f"  - {impact['id']}: {impact['delta']:.4f}"
                print(line)
                f.write(line + "\n")
        
        logger.info(f"Results written to {output_file}")
            
    else:
        logger.warning("No suitable node found to shock.")

    await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())

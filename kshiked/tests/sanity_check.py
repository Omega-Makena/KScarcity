
import os
# Prevent numpy/blas threading deadlocks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import asyncio
import numpy as np
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sanity_check")

try:
    import scarcity
    from scarcity.engine.engine import MPIEOrchestrator
    from scarcity.simulation.whatif import WhatIfManager, WhatIfConfig
    from scarcity.simulation.agents import AgentRegistry
    from scarcity.simulation.environment import SimulationEnvironment, EnvironmentConfig
    from scarcity.simulation.dynamics import DynamicsConfig
    logger.info(f"Imported scarcity library from: {scarcity.__file__}")
except ImportError as e:
    logger.error(f"Failed to import scarcity: {e}")
    sys.exit(1)

async def main():
    logger.info("--- Starting Scarcity Library Sanity Check ---")
    
    # 1. Generate Synthetic Data
    # Perfect relationship: Y follows X with lag 1
    # X is random walk
    n_samples = 200
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    rng = np.random.default_rng(42)
    for t in range(1, n_samples):
        x[t] = 0.9 * x[t-1] + rng.normal(0, 0.1)
        y[t] = 0.7 * y[t-1] + 0.5 * x[t-1] + rng.normal(0, 0.05) # Strong causal link X->Y
        
    data = np.stack([x, y], axis=1) # Shape (200, 2)
    var_names = ["X", "Y"]
    
    # 2. Learn
    orchestrator = MPIEOrchestrator()
    await orchestrator.start()
    
    window_size = 20
    logger.info("Feeding synthetic data (X -> Y)...")
    
    # Epochs to ensure learning
    for epoch in range(2):
        for i in range(len(data) - window_size):
            window = data[i:i+window_size]
            payload = {
                "data": window.tolist(),
                "schema": {"fields": [{"name": n} for n in var_names], "version": 1},
                "window_id": epoch * 1000 + i
            }
            if i % 20 == 0:
                print(f"Epoch {epoch} Step {i} processing...", flush=True)
            
            try:
                await asyncio.wait_for(orchestrator._handle_data_window("dw", payload), timeout=2.0)
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing window {i}")
            except Exception as e:
                logger.error(f"Error processing window {i}: {e}")
            
    stats = orchestrator.store.get_stats()
    logger.info(f"Learning complete. Store stats: {stats}")
    
    # 3. Simulate Shock to X
    snapshot = orchestrator.store.snapshot()
    registry = AgentRegistry()
    registry.load_from_store_snapshot(snapshot)
    
    if len(registry.nodes()) == 0:
        logger.error("FAILED: No nodes learned from synthetic data. Library or Config is broken.")
        return

    env_config = EnvironmentConfig(damping=0.9, energy_cap=10.0, seed=42)
    env = SimulationEnvironment(registry, env_config)
    
    manager = WhatIfManager(env, DynamicsConfig(), WhatIfConfig(horizon_steps=5))
    
    logger.info("Shocking X by +1.0...")
    result = manager.run_scenario("test", node_shocks={"X": 1.0})
    
    # 4. Verify Y response
    logger.info("Impact Results:")
    with open("kshiked/sanity_results.txt", "w", encoding="utf-8") as f:
        found_link = False
        for res in result["top_impacts"]:
            line = f"  {res['id']}: {res['delta']:.4f}"
            print(line)
            f.write(line + "\n")
            if res['id'] == "Y" and abs(res['delta']) > 0.01:
                found_link = True
                
        if found_link:
            msg = "SUCCESS: The library correctly simulated the impact of X on Y."
            logger.info(msg)
            f.write(msg + "\n")
        else:
            msg = "FAILURE: X shock did not significantly affect Y."
            logger.error(msg)
            f.write(msg + "\n")

    await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())

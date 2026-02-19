
import logging
import numpy as np
import sys
from scarcity.engine.evaluator import Evaluator
from scarcity.engine.types import Candidate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sanity_check_sync")

def main():
    logger.info("--- Starting Synchronous Scarcity Math Check ---")
    
    # 1. Generate Synthetic Data
    n_samples = 200
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    rng = np.random.default_rng(42)
    for t in range(1, n_samples):
        x[t] = 0.9 * x[t-1] + rng.normal(0, 0.1)
        y[t] = 0.7 * y[t-1] + 2.0 * x[t-1] + rng.normal(0, 0.05) # Strong causal link X->Y
        
    data = np.stack([x, y], axis=1) # Shape (200, 2)
    # Col 0 = X, Col 1 = Y
    
    # 2. Setup Evaluator
    evaluator = Evaluator(rng=rng)
    
    # 3. Create a candidate representing X(t-1) -> Y(t)
    # X is index 0, Y is index 1.
    # We predict Y (1) using X (0) with lag 1.
    cand_xy = Candidate(
        path_id="X->Y",
        vars=(0, 1),   # Input X, Target Y
        lags=(1, 0),   # X lag 1, Y lag 0 (target)
        ops=(0, 0),    # No-op/Identity
        root=0,
        depth=1,
        domain=0,
        gen_reason="manual"
    )
    
    # Create a candidate representing Reverse causality Y(t-1) -> X(t) (Should be weaker/zero if X is exogenous)
    cand_yx = Candidate(
        path_id="Y->X",
        vars=(1, 0),
        lags=(1, 0),
        ops=(0, 0),
        root=1,
        depth=1,
        domain=0,
        gen_reason="manual"
    )
    
    candidates = [cand_xy, cand_yx]
    
    logger.info("Evaluating candidates on synthetic data...")
    
    # The evaluator expects a window. We'll pass the whole dataset as one big window for the check.
    results = evaluator.score(data, candidates)
    
    for res in results:
        logger.info(f"Path {res.path_id}: Gain={res.gain:.4f}, Stability={res.stability:.4f}, Accepted={res.accepted}")
        
    # Check
    res_xy = next((r for r in results if r.path_id == "X->Y"), None)
    if res_xy:
        logger.info(f"X->Y Result: Gain={res_xy.gain}, Accepted={res_xy.accepted}")
        
    if res_xy and res_xy.gain > 0.1:
        logger.info("SUCCESS: Evaluator correctly identified X->Y causality (Gain > 0.1).")
    else:
        logger.error("FAILURE: Evaluator failed to identify X->Y.")
        sys.exit(1)

if __name__ == "__main__":
    main()

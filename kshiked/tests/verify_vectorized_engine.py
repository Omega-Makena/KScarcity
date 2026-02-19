"""
Verification Script for Vectorized Engine.
"""

import logging
import numpy as np
from scarcity.engine.discovery import HypothesisPool
from scarcity.engine.algorithms_online import VectorizedFunctionalHypothesis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def verify():
    pool = HypothesisPool(capacity=100)
    
    # Create manual hypothesis: y = 2x
    # We need to manually register it with the vectorized pool to get an index
    var_a = "X"
    var_b = "Y"
    idx = pool.vec_pool.get_or_create(var_a, var_b)
    
    hyp = VectorizedFunctionalHypothesis(var_a, var_b, idx, pool.vec_pool.engine)
    pool.add(hyp)
    
    logger.info(f"Created Vectorized Hypothesis idx={idx}")
    
    # Train
    logger.info("Training on y = 2x...")
    for i in range(20):
        x = float(i)
        y = 2.0 * x 
        row = {"X": x, "Y": y}
        
        pool.update_all(row)
        
        if i % 5 == 0:
            logger.info(f"Step {i}: Fit Score = {hyp.fit_score:.4f}, Conf={hyp.confidence:.4f}")
            # Check internal weights
            w = pool.vec_pool.engine.W[idx]
            logger.info(f"   Internal Weights: {w}")

    # Verify
    w_final = pool.vec_pool.engine.W[idx]
    logger.info(f"Final Weights: {w_final}")
    
    # Expect w[0] (bias) ~ 0, w[1] (slope) ~ 2
    slope_err = abs(w_final[1] - 2.0)
    if slope_err < 0.1:
        logger.info("SUCCESS: Vectorized Engine learned the relationship.")
    else:
        logger.error(f"FAILURE: Slope {w_final[1]} != 2.0")

if __name__ == "__main__":
    verify()

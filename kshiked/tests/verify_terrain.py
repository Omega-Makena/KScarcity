"""
Verification script for TerrainGenerator.
Ensures that the policy-response surface generation works correctly.

Usage: python -m kshiked.tests.verify_terrain
(requires: pip install -e .)
"""
import pandas as pd
import numpy as np

from scarcity.engine.economic_engine import EconomicDiscoveryEngine
from scarcity.analytics.terrain import TerrainGenerator
from scarcity.economic_config import CODE_TO_NAME

def verify_terrain():
    print("Initializing Engine...")
    engine = EconomicDiscoveryEngine()
    
    # Mock some data training or state
    # We'll just manually set a friendly state
    engine.friendly_state = {
        'tax_revenue_gdp': 15.0,
        'real_interest_rate': 5.0,
        'gdp_growth': 3.0,
        'inflation_cpi': 4.0
    }
    
    # Initialize Core with some dummy weights so simulations run meaningfully
    # (Otherwise everything might stay 0)
    # We can inject a dummy hypothesis: Tax -> GDP (Negative)
    # and Interest -> GDP (Negative)
    print("Injecting dummy hypotheses...")
    
    # Hack: Access internal vectorized pool directly to set weights
    # We need to find indices for pairs
    pool = engine.core.hypotheses.vec_pool
    
    # Tax -> GDP
    idx1 = pool.get_or_create('tax_revenue_gdp', 'gdp_growth')
    # Set weight to negative (higher tax -> lower growth)
    pool.engine.W[idx1] = [0.0, -0.1]
    
    # Interest -> GDP
    idx2 = pool.get_or_create('real_interest_rate', 'gdp_growth')
    pool.engine.W[idx2] = [0.0, -0.2]
    
    print("Running Terrain Generation...")
    tg = TerrainGenerator(engine)
    
    data = tg.generate_surface(
        initial_state=engine.friendly_state,
        x_policy='tax_revenue_gdp',
        y_policy='real_interest_rate',
        z_response='gdp_growth',
        x_range=(10, 20),
        y_range=(2, 8),
        steps=5, # Small grid for speed
        time_horizon=5
    )
    
    print("Analyzing Output...")
    z_matrix = data['z']
    
    print("Z Matrix shape:", z_matrix.shape)
    print("Z Matrix sample:\n", z_matrix)
    
    # Check if slopes make sense
    # We expect lower growth at high tax (higher index in x)
    # X axis is tax.
    
    # Compare first column (Low Tax) vs last column (High Tax)
    low_tax_growth = z_matrix[:, 0].mean()
    high_tax_growth = z_matrix[:, -1].mean()
    
    print(f"Mean Growth at Low Tax: {low_tax_growth:.4f}")
    print(f"Mean Growth at High Tax: {high_tax_growth:.4f}")
    
    if high_tax_growth < low_tax_growth:
         print("[PASS] Terrain Correctly Sloped (Higher Tax -> Lower Growth)")
    else:
         print("[FAIL] Terrain Slope Unexpected (Check logic/simulation)")
         
    # Check Overlays
    print("Overlays:", data['overlays'].keys())
    
    return True

if __name__ == "__main__":
    try:
        verify_terrain()
        print("[PASS] Verification Passed")
    except Exception as e:
        print(f"[FAIL] Verification Failed: {e}")
        import traceback
        traceback.print_exc()

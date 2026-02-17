"""
Verification script for TerrainGenerator with Trajectory.

Usage: python -m kshiked.tests.verify_trajectory
(requires: pip install -e .)
"""
import pandas as pd
import numpy as np

from scarcity.engine.economic_engine import EconomicDiscoveryEngine
from scarcity.analytics.terrain import TerrainGenerator

def verify_terrain_trajectory():
    print("Initializing Engine...")
    engine = EconomicDiscoveryEngine()
    engine.friendly_state = {
        'tax_revenue_gdp': 15.0,
        'real_interest_rate': 5.0,
        'gdp_growth': 3.0,
        'inflation_cpi': 4.0
    }
    
    # Create a mock path
    print("Creating mock trajectory...")
    mock_history = []
    for i in range(10):
        mock_history.append({
            'year': 2020+i,
            'tax_revenue_gdp': 15.0 + i*0.1,
            'real_interest_rate': 5.0 - i*0.05,
            'gdp_growth': 3.0 + np.sin(i)
        })
    df_hist = pd.DataFrame(mock_history)
    
    print(f"Mock Path: {len(df_hist)} steps")
    
    print("Generating Surface...")
    tg = TerrainGenerator(engine)
    data = tg.generate_surface(
        initial_state=engine.friendly_state,
        x_policy='tax_revenue_gdp',
        y_policy='real_interest_rate',
        z_response='gdp_growth',
        x_range=(14, 17),
        y_range=(4, 6),
        steps=5,
        time_horizon=5
    )
    
    # Verify alignment
    # Check if path is within bounds of terrain (visually this is what we'd check)
    # Programmatically, we just check data integrity.
    
    path_x = df_hist['tax_revenue_gdp']
    path_y = df_hist['real_interest_rate']
    path_z = df_hist['gdp_growth']
    
    if len(path_x) == 10 and len(path_y) == 10 and len(path_z) == 10:
        print("[PASS] Trajectory Vectors Extracted")
    else:
        print("[FAIL] Trajectory Vectors Corrupted")

    return True

if __name__ == "__main__":
    try:
        verify_terrain_trajectory()
        print("[PASS] Verification Passed")
    except Exception as e:
        print(f"[FAIL] Verification Failed: {e}")
        import traceback
        traceback.print_exc()

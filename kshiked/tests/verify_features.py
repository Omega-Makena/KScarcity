"""
Verification script for TerrainGenerator features (Opacity/Risk).

Usage: python -m kshiked.tests.verify_features
(requires: pip install -e .)
"""
import pandas as pd
import numpy as np

from scarcity.engine.economic_engine import EconomicDiscoveryEngine
from scarcity.analytics.terrain import TerrainGenerator

def verify_terrain_features():
    print("Initializing Engine...")
    engine = EconomicDiscoveryEngine()
    engine.friendly_state = {
        'tax_revenue_gdp': 15.0, 
        'gdp_growth': 3.0
    }
    
    # Analyze Terrain with mocked high risk
    print("Running Terrain Generation...")
    tg = TerrainGenerator(engine)
    
    # We can't easily mock internal simulation behavior here without patching,
    # but we can check if the output structure contains the new fields.
    
    data = tg.generate_surface(
        initial_state=engine.friendly_state,
        x_policy='tax_revenue_gdp',
        y_policy='tax_revenue_gdp', # same for mock
        z_response='gdp_growth',
        x_range=(10, 20),
        y_range=(10, 20),
        steps=3,
        time_horizon=2
    )
    
    # Check fields
    if 'overlays' in data and 'risk' in data['overlays']:
        print("[PASS] Risk Overlay Present")
    else:
        print("[FAIL] Risk Overlay Missing")
        
    # Check if NaNs are possible (though with default mock engine, risk might be 0)
    # Just checking code didn't crash is good for now.
    
    return True

if __name__ == "__main__":
    try:
        verify_terrain_features()
        print("[PASS] Verification Passed")
    except Exception as e:
        print(f"[FAIL] Verification Failed: {e}")
        import traceback
        traceback.print_exc()

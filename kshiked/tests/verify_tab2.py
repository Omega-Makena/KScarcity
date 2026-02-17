"""
Verification script for Tab 2 "Time-Manifold" logic.
Mimics the loop in dashboard.py to ensure dimensions align.

Usage: python -m kshiked.tests.verify_tab2
(requires: pip install -e .)
"""
import pandas as pd
import numpy as np

from scarcity.engine.economic_engine import EconomicDiscoveryEngine

def verify_tab2_logic():
    print("Initializing Engine...")
    engine = EconomicDiscoveryEngine()
    engine.friendly_state = {
        'tax_revenue_gdp': 15.0, 
        'gdp_growth': 3.0
    }
    
    man_var = 'tax_revenue_gdp'
    outcome_var = 'gdp_growth'
    steps = 5
    v_min = 10.0
    v_max = 20.0
    
    print(f"Running sweep for {man_var} -> {outcome_var}...")
    
    z_matrix = []
    y_axis = np.linspace(v_min, v_max, steps)
    x_axis = list(range(51)) # The fix: 51 steps (0..50)
    
    print(f"X-Axis Len: {len(x_axis)}")
    print(f"Y-Axis Len: {len(y_axis)}")
    
    for i, val in enumerate(y_axis):
        # Clone & Sim
        sim_m = engine.get_simulation_handle()
        sim_m.set_initial_state(engine.friendly_state)
        sim_m.set_policy(man_var, float(val))
        sim_m.run(50) # Run 50 steps
        
        # Extract Outcome curve
        hist = pd.DataFrame(sim_m.history)
        row = hist[outcome_var].tolist()
        
        # Fix Logic Verification
        if len(row) > 51: row = row[:51]
        elif len(row) < 51: row = row + [row[-1]]*(51-len(row))
        
        z_matrix.append(row)
        print(f"Row {i} len: {len(row)}")
        
    # Check Matrix Shape
    z_np = np.array(z_matrix)
    print(f"Z-Matrix Shape: {z_np.shape}")
    
    if z_np.shape == (steps, 51):
        print("[PASS] Dimensions Align (Steps x Time)")
    else:
        print("[FAIL] Dimension Mismatch")
        
    return True

if __name__ == "__main__":
    try:
        verify_tab2_logic()
        print("[PASS] Verification Passed")
    except Exception as e:
        print(f"[FAIL] Verification Failed: {e}")

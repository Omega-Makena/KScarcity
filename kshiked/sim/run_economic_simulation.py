"""
run economic simulation: fiscal multiplier.

scenario:
does increasing 'military_exp_gdp' affect 'gdp_growth'?
we compare a baseline simulation vs. a stimulus simulation.
"""

import pandas as pd
import logging
from pathlib import Path
from scarcity.engine.economic_engine import EconomicDiscoveryEngine
from scarcity.economic_config import ECONOMIC_VARIABLES, CODE_TO_NAME

# setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Data path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "API_KEN_DS2_en_csv_v2_14659.csv"

def run_fiscal_multiplier_test():
    # 1. load data
    logger.info("loading kenya dataset...")
    df = pd.read_csv(DATA_PATH, skiprows=4)
    
    # pivot logic
    pivoted = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                      var_name='Year', value_name='Value')
    pivoted['Year'] = pd.to_numeric(pivoted['Year'], errors='coerce')
    pivoted = pivoted[pivoted['Year'] >= 1960].sort_values('Year')
    pivoted = pivoted.pivot(index='Year', columns='Indicator Code', values='Value')
    
    # Forward fill but keep NaNs (do NOT fillna(0.0) as it corrupts learning)
    pivoted = pivoted.ffill()
    
    stream_data = pivoted.to_dict('records')
    logger.info(f"stream ready: {len(stream_data)} years of data.")
    
    # 2. Check Data Quality
    logger.info("\n--- Data Quality Check ---")
    for friendly, code in ECONOMIC_VARIABLES.items():
        if code in pivoted.columns:
            non_zeros = (pivoted[code] != 0.0).sum()
            mean_val = pivoted[code].mean()
            logger.info(f"{friendly:<20} ({code}): {non_zeros} non-zeros, Mean={mean_val:.2f}")
    
    # 3. Train Engine
    engine = EconomicDiscoveryEngine()
    # RELAX THRESHOLDS
    engine.core.meta_controller.conf_thresh = 0.3
    engine.core.meta_controller.stab_thresh = 0.3 
    engine.core.meta_controller.min_evidence = 10 
    engine.core.explore_interval = 1 
    
    logger.info("\ntraining economic engine...")
    for i, row in enumerate(stream_data):
        engine.process_row_raw(row)
        if i % 20 == 0:
             stats = engine.core.meta_controller.get_summary(engine.core.hypotheses)
             logger.info(f"Step {i}: Active={stats['active']}, Tentative={stats['tentative']}")
        
    engine.print_top_relationships(k=10)
    
    # 4. Setup Simulation
    sim_baseline = engine.get_simulation_handle()
    sim_stimulus = engine.get_simulation_handle()
    
    last_state = engine.core._sanitize_row(stream_data[-1])
    friendly_state = {}
    for k, v in last_state.items():
        if k in CODE_TO_NAME:
            friendly_state[CODE_TO_NAME[k]] = v
            
    sim_baseline.set_initial_state(friendly_state)
    sim_stimulus.set_initial_state(friendly_state)
    
    # 5. Define Shock: Military Spending
    target_var = "military_exp_gdp"
    response_var = "gdp_growth"
    
    current = friendly_state.get(target_var, 0.0)
    stimulus = current + 1.0 # Increase by 1% of GDP
    
    logger.info(f"\n--- simulation scenario: military stimulus ---")
    logger.info(f"baseline {target_var}: {current:.2f}")
    logger.info(f"stimulus {target_var}: {stimulus:.2f}")
    
    sim_stimulus.perturb(target_var, stimulus)
    
    # 6. Run
    steps = 5
    base_res = sim_baseline.run(steps)
    stim_res = sim_stimulus.run(steps)
    
    # 7. Analysis
    print(f"\n{'year':<5} | {'baseline ' + response_var:<20} | {'stimulus ' + response_var:<20} | {'diff':<10}")
    print("-" * 65)
    
    for i in range(steps):
        b_val = base_res[i].get(response_var, 0.0)
        s_val = stim_res[i].get(response_var, 0.0)
        diff = s_val - b_val
        print(f"{i+1:<5} | {b_val:.4f}{' ':14} | {s_val:.4f}{' ':14} | {diff:+.4f}")

if __name__ == "__main__":
    run_fiscal_multiplier_test()

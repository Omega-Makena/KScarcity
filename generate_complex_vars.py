import sqlite3
import json
import os
import numpy as np
import pandas as pd

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "kshiked", "ui", "institution", "backend", "federated_registry.sqlite"))

def update_ontology():
    print(f"Connecting to database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Expand the Economic Basket schema to allow a vast multi-dimensional state space
    # representing a full macroeconomic meta-graph.
    rich_schema = {
        "required_columns": [
            "inflation_index", 
            "unemployment_rate", 
            "debt_ratio",
            "gdp_growth",
            "policy_rate",
            "fx_reserves",
            "energy_cost",
            "social_stress_index"
        ],
        "allow_extra": True
    }
    
    # We assume basket_id 1 is the Economic Basket based on the seed order.
    # Let's get the ID dynamically.
    cursor.execute("SELECT id FROM baskets WHERE name='Economic Basket'")
    row = cursor.fetchone()
    if row:
        basket_id = row[0]
        cursor.execute(
            "UPDATE ontology_schemas SET schema_definition=? WHERE basket_id=?", 
            (json.dumps(rich_schema), basket_id)
        )
        conn.commit()
        print("Successfully expanded the Economic Basket Ontology to 8 core dimensions.")
    else:
        print("Economic Basket not found inside DB.")
    conn.close()

def generate_complex_csv():
    print("Generating mathematically complex dataset...")
    np.random.seed(42)
    n_steps = 2000
    
    # Create empty dataframe
    df = pd.DataFrame(index=range(n_steps))
    
    # Core macroeconomic dynamics
    # 1. Base autoregressive processes (structural inertia)
    df['inflation_index'] = 2.0 + np.cumsum(np.random.normal(0, 0.1, n_steps))
    df['unemployment_rate'] = 5.0 + np.cumsum(np.random.normal(0, 0.05, n_steps))
    df['debt_ratio'] = 60.0 + np.cumsum(np.random.normal(0, 0.2, n_steps))
    
    # Limit base drifts
    df['unemployment_rate'] = np.clip(df['unemployment_rate'], 3.0, 15.0)
    df['debt_ratio'] = np.clip(df['debt_ratio'], 30.0, 150.0)
    
    # 2. Add structural correlations
    # Phillips curve effect (simplified inverse correlation)
    df['unemployment_rate'] -= df['inflation_index'] * 0.2
    
    # Central bank reaction function (policy rate lags inflation)
    df['policy_rate'] = 1.0 + df['inflation_index'].shift(10).fillna(method='bfill') * 0.8 + np.random.normal(0, 0.2, n_steps)
    
    # GDP growth reacts to policy rate and inflation
    df['gdp_growth'] = 3.0 - (df['policy_rate'] * 0.3) - (df['inflation_index'] * 0.1) + np.random.normal(0, 0.5, n_steps)
    
    # Energy costs follow an Ornstein-Uhlenbeck process (mean reverting) with jumps
    energy = np.zeros(n_steps)
    energy[0] = 100.0
    for t in range(1, n_steps):
        energy[t] = energy[t-1] + 0.1 * (100.0 - energy[t-1]) + np.random.normal(0, 2.0)
    df['energy_cost'] = energy
    
    # 3. Inject Volatility Clustering (GARCH-like behavior)
    # The Scarcity Bayesian Forecaster has a GARCH(1,1) uncertainty bound tracker.
    # We need bursts of volatility to trigger it.
    volatility = np.ones(n_steps)
    for t in range(500, 700): volatility[t] = 5.0  # Regime 1 High Vol
    for t in range(1200, 1300): volatility[t] = 8.0 # Regime 2 Extreme Vol
    
    df['fx_reserves'] = 50.0 + np.cumsum(np.random.normal(0, volatility, n_steps))
    
    # 4. Inject Massive Structural Shocks (For the RRCF Anomaly Detector)
    # Shock 1: Energy Crisis
    df.loc[600:650, 'energy_cost'] += np.linspace(0, 80, 51)
    df.loc[600:700, 'inflation_index'] += np.linspace(0, 5, 101)
    
    # Shock 2: Debt Default Flash Crash
    df.loc[1400:1410, 'debt_ratio'] += np.random.normal(20, 5, 11)
    df.loc[1405:1450, 'fx_reserves'] -= np.linspace(0, 30, 46) # Massive capital flight
    
    # 5. Social Stress (Proxy for cascading impact into human services)
    df['social_stress_index'] = (df['unemployment_rate'] * 10) + (df['inflation_index'] * 15) - df['gdp_growth'] + np.random.normal(0, 5, n_steps)
    
    # Final smoothing step
    df = df.rolling(window=3, min_periods=1).mean()
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "advanced_spoke_data.csv"))
    df.to_csv(output_path, index=False)
    print(f"Generated a {n_steps}-row 8-dimensional dataset at: {output_path}")

if __name__ == "__main__":
    update_ontology()
    generate_complex_csv()

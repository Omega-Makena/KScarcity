import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define sector-specific schemas
SECTOR_FEATURES = {
    "Public Health": ["patient_intake", "essential_drug_stock", "bed_occupancy", "staff_availability"],
    "Water & Sanitation": ["reservoir_level", "water_quality_index", "pipe_pressure", "pump_operational_hrs"],
    "Transport & Logistics": ["fleet_availability", "fuel_reserves", "route_disruption_index", "maintenance_backlog"],
    "Security & Border": ["patrol_frequency", "incident_reports", "border_wait_time_hrs", "equipment_readiness"]
}

def generate_spoke_data(spoke_name, sector_name, num_rows=500):
    np.random.seed(hash(spoke_name) % (2**32))
    
    end_time = datetime.now()
    timestamps = [end_time - timedelta(hours=i) for i in range(num_rows)]
    timestamps.reverse()
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'location': [spoke_name] * num_rows,
        'sector': [sector_name] * num_rows,
    })
    
    features = SECTOR_FEATURES.get(sector_name, ["generic_metric_1", "generic_metric_2", "generic_metric_3", "generic_metric_4"])
    
    # Generate realistic-looking values
    f1, f2, f3, f4 = features
    
    # Feature 1 (e.g. Patient Intake, Reservoir Level - volatile/random walk)
    df[f1] = np.clip(np.cumsum(np.random.normal(0, 5, num_rows)) + 150, 0, 500).round(1)
    
    # Feature 2 (e.g. Drug Stock, Fuel Reserves - downwards trend with sudden jumps representing resupply)
    stock = [100.0]
    for _ in range(num_rows - 1):
        if np.random.random() < 0.05: # Resupply
            stock.append(np.clip(stock[-1] + np.random.uniform(20, 50), 0, 100))
        else: # Depletion
            stock.append(np.clip(stock[-1] - np.random.uniform(0.5, 2.0), 0, 100))
    df[f2] = np.array(stock).round(1)
    
    # Feature 3 (e.g. Bed Occupancy, Pipe Pressure - stable with noise)
    df[f3] = np.clip(np.random.normal(75, 10, num_rows), 0, 100).round(1)
    
    # Feature 4 (e.g. Staff Availability - highly stable)
    df[f4] = np.clip(np.random.normal(90, 5, num_rows), 0, 100).round(1)
    
    # Target variable (threat_score)
    # We create a causal link: high f1 (demand), low f2/f4 (supply), high f3 (load) = high threat
    # Normalize features to 0-1 for the threat equation
    f1_norm = df[f1] / df[f1].max()
    f2_norm = 1.0 - (df[f2] / 100.0) # inverted: lower stock = higher threat
    f3_norm = df[f3] / 100.0
    f4_norm = 1.0 - (df[f4] / 100.0) # inverted: lower staff = higher threat
    
    noise = np.random.normal(0, 0.05, num_rows)
    threat = (f1_norm * 0.3) + (f2_norm * 0.3) + (f3_norm * 0.2) + (f4_norm * 0.2) + noise
    df['threat_score'] = np.clip(threat, 0, 1).round(3)

    return df

def main():
    out_dir = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\data\demo_spokes"
    os.makedirs(out_dir, exist_ok=True)
    
    spokes = [
        {"spoke_name": "Marsabit-Health", "sector": "Public Health"},
        {"spoke_name": "Turkana-Water", "sector": "Water & Sanitation"},
        {"spoke_name": "Nairobi-Transport", "sector": "Transport & Logistics"},
        {"spoke_name": "Garissa-Security", "sector": "Security & Border"}
    ]
        
    for spoke in spokes:
        name = spoke['spoke_name']
        sector = spoke['sector']
        print(f"Generating synthetic {sector} dataset for: {name}")
        
        df = generate_spoke_data(name, sector, num_rows=500)
        
        safe_name = "".join([c if c.isalnum() else "_" for c in name]).strip("_")
        out_file = os.path.join(out_dir, f"{safe_name}_data.csv")
        df.to_csv(out_file, index=False)
        print(f" -> Saved {len(df)} rows to {out_file} (Columns: {', '.join(df.columns)})")
        
    print(f"\nAll 4 domain-specific datasets exported successfully to: {out_dir}")

if __name__ == '__main__':
    main()

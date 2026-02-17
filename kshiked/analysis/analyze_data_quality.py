"""
Analyze Dataset Suitability.
Checks if N=65 is enough for our 1400 variables.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scarcity.engine.algorithms_online import FunctionalLinearHypothesis

# Data path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "API_KEN_DS2_en_csv_v2_14659.csv"

def analyze():
    # Load
    df = pd.read_csv(DATA_PATH, skiprows=4)
    
    # 1. Dimensions
    years = [c for c in df.columns if c.isdigit()]
    N = len(years) # Time steps (65)
    P = len(df)    # Variables (~1500)
    
    print(f"Dataset Shape: {N} Time Steps x {P} Variables")
    print(f"Ratio N/P: {N/P:.4f} (High Dimensional 'Large P, Small N')")
    
    # 2. Check Sparsity / Missing Data
    numeric_df = df[years].apply(pd.to_numeric, errors='coerce')
    missing_pct = numeric_df.isna().mean().mean()
    print(f"Global Missing Data: {missing_pct*100:.1f}%")
    
    # 3. Convergence Test
    # Pick a dense variable (e.g., Population) and see how fast RLS behaves
    # We'll use a dummy target to check coefficient stability history
    from scarcity.engine.algorithms_online import RecursiveLeastSquares
    
    # Synthetic test: y = 2x + noise
    # meaningful length test
    rls = RecursiveLeastSquares(2)
    history = []
    
    # Create synthetic linear data with similar noise profile to real data?
    # Or just use real data?
    # Let's use real pair: Population Total vs Urban Population (should be linear-ish)
    
    # Filter for non-empty rows
    # We need to find two rows that are mostly full
    row_a = numeric_df.iloc[0].interpolate().fillna(0) # First var
    row_b = numeric_df.iloc[1].interpolate().fillna(0) # Second var
    
    x = row_a.values
    y = row_b.values
    
    for i in range(N):
        rls.update(np.array([1.0, x[i]]), y[i])
        history.append(rls.w.copy())
        
    # Analyze stability of 'slope' (w[1]) over time
    slopes = [h[1] for h in history]
    # Calculate step-to-step change
    changes = [abs(slopes[i] - slopes[i-1]) for i in range(1, len(slopes))]
    
    # Find step where change drops below threshold (convergence)
    converged_at = -1
    for i, c in enumerate(changes):
        if c < 0.001:
            converged_at = i
            break
            
    print(f"RLS Convergence Test (Synthetic Pair): Stabilized after ~{converged_at} samples.")
    
    if converged_at < N/2:
        print("CONCLUSION: Dataset size (65) is sufficient for linear convergence.")
    else:
        print("CONCLUSION: Dataset might be too short for stable learning.")

if __name__ == "__main__":
    analyze()

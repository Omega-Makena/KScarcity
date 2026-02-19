"""
Demo: Prove Relationship Types Work on Real Kenya Data

This script loads the Kenya economic dataset and tests the new
hypothesis classes to prove they're not skeletons.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scarcity.engine.relationships import (
    CausalHypothesis,
    CorrelationalHypothesis,
    TemporalHypothesis,
    FunctionalHypothesis,
    EquilibriumHypothesis,
)


def load_kenya_data():
    """Load and process Kenya economic data."""
    # Look in project root
    csv_path = Path(__file__).parent.parent.parent / "API_KEN_DS2_en_csv_v2_14659.csv"
    
    # Read with correct header row
    df = pd.read_csv(csv_path, skiprows=4)
    
    # Filter to key indicators
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP',
        'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
        'FP.CPI.TOTL.ZG': 'Inflation',
        'SL.UEM.TOTL.ZS': 'Unemployment',
        'BX.KLT.DINV.CD.WD': 'FDI',
        'NE.EXP.GNFS.CD': 'Exports',
        'NE.IMP.GNFS.CD': 'Imports',
    }
    
    df_filtered = df[df['Indicator Code'].isin(indicators.keys())]
    
    # Pivot to wide format
    year_cols = [c for c in df.columns if c.isdigit() and 1990 <= int(c) <= 2023]
    
    result = {}
    for _, row in df_filtered.iterrows():
        indicator = indicators.get(row['Indicator Code'], row['Indicator Name'])
        values = []
        for year in year_cols:
            val = row[year]
            if pd.notna(val):
                values.append(float(val))
            else:
                values.append(np.nan)
        result[indicator] = np.array(values)
    
    return result, year_cols


def demo_causal():
    """Test CausalHypothesis on GDP components."""
    print("\n" + "="*60)
    print("1. CAUSAL: Does GDP Growth cause Inflation change?")
    print("="*60)
    
    data, years = load_kenya_data()
    
    if 'GDP_Growth' not in data or 'Inflation' not in data:
        print("Missing required data")
        return
    
    hyp = CausalHypothesis('GDP_Growth', 'Inflation', lag=1)
    
    gdp_growth = data['GDP_Growth']
    inflation = data['Inflation']
    
    # Feed rows
    for i in range(len(gdp_growth)):
        if not (np.isnan(gdp_growth[i]) or np.isnan(inflation[i])):
            row = {'GDP_Growth': gdp_growth[i], 'Inflation': inflation[i]}
            hyp.fit_step(row)
    
    result = hyp.evaluate({})
    
    print(f"  Buffer size: {len(hyp.buffer_x)} years of data")
    print(f"  Forward gain (GDP → Inflation): {result.get('gain_forward', 0):.4f}")
    print(f"  Backward gain (Inflation → GDP): {result.get('gain_backward', 0):.4f}")
    print(f"  Direction: {result.get('direction', 0)} (1=forward, -1=backward)")
    print(f"  Confidence: {result.get('confidence', 0):.4f}")
    
    return result


def demo_correlational():
    """Test CorrelationalHypothesis on Exports vs Imports."""
    print("\n" + "="*60)
    print("2. CORRELATIONAL: Are Exports and Imports correlated?")
    print("="*60)
    
    data, years = load_kenya_data()
    
    if 'Exports' not in data or 'Imports' not in data:
        print("Missing required data")
        return
    
    hyp = CorrelationalHypothesis('Exports', 'Imports')
    
    exports = data['Exports']
    imports = data['Imports']
    
    for i in range(len(exports)):
        if not (np.isnan(exports[i]) or np.isnan(imports[i])):
            row = {'Exports': exports[i], 'Imports': imports[i]}
            hyp.fit_step(row)
    
    result = hyp.evaluate({})
    
    print(f"  Observations: {result.get('evidence', 0)}")
    print(f"  Correlation coefficient: {result.get('correlation', 0):.4f}")
    print(f"  Fit score: {result.get('fit_score', 0):.4f}")
    
    return result


def demo_temporal():
    """Test TemporalHypothesis on GDP time series."""
    print("\n" + "="*60)
    print("3. TEMPORAL: Is GDP autocorrelated?")
    print("="*60)
    
    data, years = load_kenya_data()
    
    if 'GDP' not in data:
        print("Missing GDP data")
        return
    
    hyp = TemporalHypothesis('GDP', lag=2)
    
    gdp = data['GDP']
    
    for i in range(len(gdp)):
        if not np.isnan(gdp[i]):
            row = {'GDP': gdp[i]}
            hyp.fit_step(row)
    
    result = hyp.evaluate({})
    
    print(f"  Observations: {result.get('evidence', 0)}")
    print(f"  Autocorrelation: {result.get('autocorrelation', 0):.4f}")
    print(f"  AR coefficients: {result.get('coefficients', [])[:3]}")
    print(f"  Fit score (R²): {result.get('fit_score', 0):.4f}")
    
    return result


def demo_equilibrium():
    """Test EquilibriumHypothesis on Inflation."""
    print("\n" + "="*60)
    print("4. EQUILIBRIUM: Does Inflation revert to a mean?")
    print("="*60)
    
    data, years = load_kenya_data()
    
    if 'Inflation' not in data:
        print("Missing Inflation data")
        return
    
    hyp = EquilibriumHypothesis('Inflation')
    
    inflation = data['Inflation']
    
    for i in range(len(inflation)):
        if not np.isnan(inflation[i]):
            row = {'Inflation': inflation[i]}
            hyp.fit_step(row)
    
    result = hyp.evaluate({})
    
    print(f"  Observations: {result.get('evidence', 0)}")
    print(f"  Estimated equilibrium: {result.get('equilibrium', 0):.2f}%")
    print(f"  Reversion rate: {result.get('reversion_rate', 0):.4f}")
    print(f"  Is mean-reverting: {result.get('is_reverting', False)}")
    
    return result


def demo_functional():
    """Test FunctionalHypothesis on GDP vs Exports."""
    print("\n" + "="*60)
    print("5. FUNCTIONAL: Is there a functional GDP → Exports relationship?")
    print("="*60)
    
    data, years = load_kenya_data()
    
    if 'GDP' not in data or 'Exports' not in data:
        print("Missing required data")
        return
    
    hyp = FunctionalHypothesis('GDP', 'Exports', degree=1)
    
    gdp = data['GDP']
    exports = data['Exports']
    
    for i in range(len(gdp)):
        if not (np.isnan(gdp[i]) or np.isnan(exports[i])):
            row = {'GDP': gdp[i], 'Exports': exports[i]}
            hyp.fit_step(row)
    
    result = hyp.evaluate({})
    
    print(f"  Observations: {result.get('evidence', 0)}")
    print(f"  Fit score (R²): {result.get('fit_score', 0):.4f}")
    print(f"  Coefficients: {result.get('coefficients', [])}")
    print(f"  Is deterministic: {result.get('deterministic', False)}")
    
    # Test prediction
    if len(gdp) > 0 and not np.isnan(gdp[-1]):
        pred = hyp.predict_value({'GDP': gdp[-1]})
        if pred:
            print(f"  Predicted Exports for GDP={gdp[-1]:.0f}: {pred[1]:.0f}")
    
    return result


if __name__ == "__main__":
    print("="*60)
    print("PROOF: Relationship Types Work on Real Kenya Data")
    print("="*60)
    
    results = {}
    
    try:
        results['causal'] = demo_causal()
        results['correlational'] = demo_correlational()
        results['temporal'] = demo_temporal()
        results['equilibrium'] = demo_equilibrium()
        results['functional'] = demo_functional()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        passed = 0
        total = 5
        
        if results.get('correlational', {}).get('correlation', 0) > 0.8:
            print("✓ CORRELATIONAL: Exports/Imports correlation detected")
            passed += 1
        else:
            print("? CORRELATIONAL: Weak correlation")
        
        if results.get('temporal', {}).get('autocorrelation', 0) > 0.5:
            print("✓ TEMPORAL: GDP autocorrelation detected")
            passed += 1
        else:
            print("? TEMPORAL: Weak autocorrelation")
        
        if results.get('functional', {}).get('fit_score', 0) > 0.7:
            print("✓ FUNCTIONAL: GDP→Exports relationship detected")
            passed += 1
        else:
            print("? FUNCTIONAL: Weak fit")
        
        if results.get('equilibrium', {}).get('evidence', 0) > 10:
            print("✓ EQUILIBRIUM: Processed inflation data")
            passed += 1
        else:
            print("? EQUILIBRIUM: Insufficient data")
        
        if results.get('causal', {}).get('evidence', 0) > 10:
            print("✓ CAUSAL: Processed GDP/Inflation data")
            passed += 1
        else:
            print("? CAUSAL: Insufficient data")
        
        print(f"\n{passed}/{total} hypothesis types demonstrated on real data")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

import json
import numpy as np
import pandas as pd
from scarcity.synthetic.benchmark_generator import create_generator

def verify_generator(schema_path: str, output_path: str = "verification_report.json"):
    print(f"Loading schema from {schema_path}...")
    generator = create_generator(schema_path, seed=42)
    
    N = 5000
    print(f"Generating {N} samples...")
    data_dicts = generator.generate(N)
    df = pd.DataFrame(data_dicts)
    
    # 1. Check NaNs and Infs
    assert not df.isnull().values.any(), "Generated data contains NaNs"
    assert not np.isinf(df.values).any(), "Generated data contains Infs"
    print("Data sanity check passed: No NaNs or Infs.")
    
    report = {
        "metrics": {},
        "null_pairs": {},
        "status": "success"
    }
    
    # 2. Check Null Pairs
    print("\nChecking null pairs...")
    for pair in generator.schema.get("null_pairs", []):
        v1, v2 = pair
        corr = np.corrcoef(df[v1], df[v2])[0, 1]
        report["null_pairs"][f"{v1}-{v2}"] = corr
        print(f"Null pair {v1}-{v2} correlation: {corr:.4f}")
        assert abs(corr) < 0.05, f"Null pair {v1}-{v2} has significant correlation: {corr:.4f}"
        
    # 3. Check Relationships
    print("\nChecking relationships...")
    for rel in generator.relationships:
        r_type = rel['type']
        
        if r_type == 'temporal':
            var = rel['variable']
            # Simplistic check for first lag
            lag = rel['lags'][0]
            corr = np.corrcoef(df[var].iloc[lag:].values, df[var].iloc[:-lag].values)[0, 1]
            report["metrics"][f"temporal_{var}"] = corr
            print(f"Temporal {var} AR({lag}) correlation: {corr:.4f}")
            assert corr > 0.05, f"Temporal relation weak: {corr}"
            
        elif r_type == 'causal':
            source = rel['source']
            target = rel['target']
            lag = rel['lags'][0]
            # Source should predict target
            corr = np.corrcoef(df[target].iloc[lag:].values, df[source].iloc[:-lag].values)[0, 1]
            report["metrics"][f"causal_{source}_{target}"] = corr
            print(f"Causal {source}->{target} (lag {lag}) correlation: {corr:.4f}")
            assert corr > 0.05, f"Causal relation weak: {corr}"
            
        elif r_type == 'correlational':
            v1, v2 = rel['pair']
            expected_corr = rel['correlation']
            corr = np.corrcoef(df[v1], df[v2])[0, 1]
            report["metrics"][f"correlational_{v1}_{v2}"] = corr
            print(f"Correlational {v1}-{v2} correlation: {corr:.4f} (expected ~{expected_corr})")
            assert abs(corr - expected_corr) < 0.15, f"Correlational relation mismatch: {corr} vs {expected_corr}"
            
        elif r_type == 'compositional':
            total = rel['total']
            comps = rel['components']
            sum_comps = df[comps].sum(axis=1)
            # The total is determined completely by sum_comps + noise
            corr = np.corrcoef(df[total], sum_comps)[0, 1]
            report["metrics"][f"compositional_{total}"] = corr
            print(f"Compositional {total} correlation with sum: {corr:.4f}")
            assert corr > 0.7, f"Compositional relation weak: {corr}"
            
        elif r_type == 'equilibrium':
            var = rel['variable']
            mean_val = df[var].mean()
            report["metrics"][f"equilibrium_{var}_mean"] = mean_val
            print(f"Equilibrium {var} mean: {mean_val:.4f} (expected ~{rel['mean']})")
            assert abs(mean_val - rel['mean']) < 0.5, f"Equilibrium mean mismatch: {mean_val}"
            
        elif r_type == 'competitive':
            v1, v2 = rel['pair']
            sum_vals = df[v1] + df[v2]
            mean_sum = sum_vals.mean()
            report["metrics"][f"competitive_{v1}_{v2}_sum"] = mean_sum
            print(f"Competitive {v1}+{v2} mean sum: {mean_sum:.4f} (expected ~{rel['total']})")
            assert abs(mean_sum - rel['total']) < 5.0, f"Competitive sum mismatch: {mean_sum}"

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nVerification successful! Report saved to {output_path}")

if __name__ == "__main__":
    import os
    # Use absolute path to the benchmark_schema.json we created
    base_dir = os.path.dirname(__file__)
    schema_path = os.path.join(base_dir, "benchmark_schema.json")
    report_path = os.path.join(base_dir, "verification_report.json")
    verify_generator(schema_path, report_path)

"""
Full system benchmark orchestrator.

Usage:
    python benchmark/scripts/benchmark_full_system.py --phase all
    python benchmark/scripts/benchmark_full_system.py --phase synthetic
    python benchmark/scripts/benchmark_full_system.py --phase real
    python benchmark/scripts/benchmark_full_system.py --phase federation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.synthetic.pipeline import SyntheticBenchmark
from benchmark.synthetic.reporting import generate_report
from benchmark.real_data.world_bank_loader import prepare_multi_country_data
from benchmark.real_data.rolling_backtest import RollingOriginBacktest
from benchmark.evaluation.federation_metrics import FederationEvaluator


def run_synthetic_benchmark(schema_path: str, n_samples: int = 3000, seed: int = 42) -> dict:
    print("\n--- PHASE: Synthetic Generation & Calibration ---")
    bench = SyntheticBenchmark(schema_path=schema_path, seed=seed, B_perm=100)
    results = bench.run(n_samples=n_samples)
    print(f"  Strict F1={results['metrics']['strict']['f1']:.4f}  "
          f"Null FPR={results['metrics']['null_fpr']:.4f}")
    return results


def run_real_world_backtest(countries=('KEN',), targets=('gdp_growth', 'inflation_cpi'),
                             initial_train_years=15) -> list:
    print("\n--- PHASE: Real-World Rolling Backtest ---")
    try:
        data_dict = prepare_multi_country_data(list(countries))
        rows = []
        for country, df in data_dict.items():
            print(f"  {country}: {df.shape[0]} years × {df.shape[1]} variables", flush=True)
            valid_targets = [t for t in targets if t in df.columns]
            if not valid_targets:
                print(f"  Warning: none of {targets} found in {country} data; skipping.")
                continue
            backtest = RollingOriginBacktest(df, target_variables=valid_targets,
                                             initial_train_years=initial_train_years)
            res_df = backtest.run_backtest({})
            for _, row in res_df.iterrows():
                for method in ('persistence', 'arima', 'var', 'prophet'):
                    rows.append({
                        'country': country,
                        'target': row['target'],
                        'method': method,
                        'mae': row.get(f'{method}_mae'),
                        'dir_acc': row.get(f'{method}_dir'),
                    })
        print(f"  Backtest complete: {len(rows)} evaluation rows", flush=True)
        return rows
    except Exception as e:
        print(f"  Real-world backtest failed: {e}", flush=True)
        return []


def run_federation(data_dict: dict, target: str = 'gdp_growth',
                   train_split: int = 20, num_nodes: int = 3) -> dict:
    print("\n--- PHASE: Federation Evaluation ---")
    try:
        import pandas as pd
        import numpy as np
        # Use first available country
        df = next(iter(data_dict.values()))
        evaluator = FederationEvaluator(df, num_nodes=num_nodes)
        target_col = target if target in df.columns else df.columns[0]
        mem_res = evaluator.evaluate_in_memory(target_col, train_split)
        phys_res = evaluator.evaluate_physical()
        print(f"  In-memory FedAvg MSE={mem_res.get('mse', float('nan')):.4f}, "
              f"comm={mem_res.get('communication_bytes', 0)/1024:.1f} KB", flush=True)
        return {'in_memory': mem_res, 'physical': phys_res}
    except Exception as e:
        print(f"  Federation evaluation failed: {e}", flush=True)
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["synthetic", "real", "federation", "all"])
    parser.add_argument("--schema", default="benchmark/synthetic/benchmark_schema.json")
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="benchmark/reports/outputs")
    args = parser.parse_args()

    synthetic_result = None
    backtest_rows = []
    federation_result = {}
    data_dict = {}

    if args.phase in ("synthetic", "all"):
        synthetic_result = run_synthetic_benchmark(args.schema, args.n_samples, args.seed)

    if args.phase in ("real", "all"):
        backtest_rows = run_real_world_backtest()
        if backtest_rows:
            data_dict = prepare_multi_country_data(['KEN'])

    if args.phase in ("federation", "all"):
        if not data_dict:
            try:
                data_dict = prepare_multi_country_data(['KEN'])
            except Exception:
                pass
        if data_dict:
            federation_result = run_federation(data_dict)

    # Build unified result for reporting
    if synthetic_result is None:
        # Load existing data if available
        data_path = Path("scarcity/synthetic/benchmark_results/benchmark_data.json")
        if data_path.exists():
            with open(data_path) as f:
                loaded = json.load(f)
            synthetic_result = loaded[-1] if isinstance(loaded, list) else loaded
            print(f"\nLoaded existing synthetic results from {data_path}")
        else:
            print("\nNo synthetic results available. Run with --phase synthetic first.")
            return

    if backtest_rows:
        synthetic_result['real_world_backtest'] = backtest_rows
    if federation_result:
        synthetic_result['federation'] = federation_result

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rpt, dat = generate_report(synthetic_result, out_dir=out_dir)
    print(f"\n  Full report: {rpt}")
    print(f"  Full data:   {dat}")


if __name__ == "__main__":
    main()

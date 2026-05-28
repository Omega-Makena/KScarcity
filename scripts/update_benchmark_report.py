"""
Update the benchmark report without re-running the full 675s synthetic benchmark.

Strategy:
1. Load existing benchmark_data.json (has calibration + recovery results).
2. Run anomaly detection on a small fresh synthetic dataset (fast, ~10s).
3. Run Kenya rolling-origin backtest (fast, ~60s).
4. Run in-memory federation evaluation (fast, ~5s).
5. Merge everything and regenerate the benchmark report.

Output: scarcity/synthetic/benchmark_results/benchmark_report.md
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ── 1. Load existing synthetic results ──────────────────────────────────────
DATA_JSON = _ROOT / "scarcity" / "synthetic" / "benchmark_results" / "benchmark_data.json"
SCHEMA_PATH = str(_ROOT / "benchmark" / "synthetic" / "benchmark_schema.json")
OUT_DIR = str(_ROOT / "scarcity" / "synthetic" / "benchmark_results")

print("=" * 60)
print("Scarcity Benchmark Report Updater")
print("=" * 60)

print(f"\n[1/4] Loading existing synthetic results from {DATA_JSON} ...")
with open(DATA_JSON) as f:
    loaded = json.load(f)
base_result = loaded[-1] if isinstance(loaded, list) else loaded
print(f"      Loaded: N={base_result['n_samples']}, "
      f"Strict F1={base_result['metrics']['strict']['f1']:.4f}, "
      f"Null FPR={base_result['metrics']['null_fpr']:.4f}")


# ── 2. Anomaly detection on small synthetic dataset ──────────────────────────
print("\n[2/4] Running anomaly detection on fresh synthetic data ...")
t0 = time.time()
try:
    from benchmark.synthetic.benchmark_generator import create_benchmark_generator
    from benchmark.evaluation.anomaly_detection import AnomalyDetectionEvaluator

    gen = create_benchmark_generator(SCHEMA_PATH, seed=42)
    df_small = gen.generate(500)

    # Inject 5-sigma anomalies at 2% rate
    rng = np.random.RandomState(42 + 1337)
    anomaly_rate = 0.02
    n_per_col = max(1, int(anomaly_rate * len(df_small)))
    mask = pd.DataFrame(False, index=df_small.index, columns=df_small.columns)
    df_anom = df_small.copy()
    for col in df_small.columns:
        idx = rng.choice(len(df_small), n_per_col, replace=False)
        std = float(df_small[col].std()) or 1.0
        signs = rng.choice([-1, 1], n_per_col)
        col_pos = df_small.columns.get_loc(col)
        df_anom.iloc[idx, col_pos] += signs * 5.0 * std
        mask.iloc[idx, col_pos] = True

    ev = AnomalyDetectionEvaluator(df_anom, mask)
    zscore_res = ev.evaluate_zscore(threshold=3.0)
    iso_res = ev.evaluate_isolation_forest()

    anomaly_results = {
        'anomaly_rate': anomaly_rate,
        'n_injected_per_col': n_per_col,
        'zscore': zscore_res,
        'isolation_forest': iso_res,
    }
    print(f"      Z-Score:   P={zscore_res['precision']:.4f}  R={zscore_res['recall']:.4f}  F1={zscore_res['f1']:.4f}")
    print(f"      Iso-Forest: P={iso_res['precision']:.4f}  R={iso_res['recall']:.4f}  F1={iso_res['f1']:.4f}")
    print(f"      Done in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"      Warning: anomaly detection failed: {e}")
    anomaly_results = {'error': str(e)}


# ── 3. Kenya rolling-origin backtest ─────────────────────────────────────────
print("\n[3/4] Running Kenya rolling-origin backtest ...")
t0 = time.time()
backtest_rows = []
data_dict = {}
try:
    from benchmark.real_data.world_bank_loader import prepare_multi_country_data
    from benchmark.real_data.rolling_backtest import RollingOriginBacktest

    data_dict = prepare_multi_country_data(['KEN'])
    kenya_df = data_dict['KEN']
    print(f"      Kenya data: {kenya_df.shape[0]} years × {kenya_df.shape[1]} variables")

    # Pick available targets
    preferred_targets = ['gdp_growth', 'inflation_cpi']
    targets = [t for t in preferred_targets if t in kenya_df.columns]
    if not targets:
        targets = [kenya_df.columns[0]]
    print(f"      Targets: {targets}")

    backtest = RollingOriginBacktest(kenya_df, target_variables=targets, initial_train_years=15)
    res_df = backtest.run_backtest({})

    for _, row in res_df.iterrows():
        for method in ('persistence', 'arima', 'var', 'prophet'):
            mae_key = f'{method}_mae'
            dir_key = f'{method}_dir'
            backtest_rows.append({
                'country': 'KEN',
                'target': row['target'],
                'method': method.upper(),
                'mae': round(float(row[mae_key]), 4) if mae_key in row and pd.notna(row[mae_key]) else None,
                'dir_acc': round(float(row[dir_key]), 3) if dir_key in row and pd.notna(row[dir_key]) else None,
            })

    print(f"      {len(res_df)} test-year evaluations across {len(targets)} targets")
    # Show mean MAE per method
    summary = res_df.groupby('target')[
        [c for c in ['persistence_mae', 'arima_mae', 'var_mae', 'prophet_mae'] if c in res_df.columns]
    ].mean()
    print(summary.to_string())
    print(f"      Done in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"      Warning: rolling backtest failed: {e}")


# ── 4. Federation evaluation ─────────────────────────────────────────────────
print("\n[4/4] Running federation evaluation ...")
t0 = time.time()
federation_result = {}
try:
    from benchmark.evaluation.federation_metrics import FederationEvaluator

    df_fed = next(iter(data_dict.values())) if data_dict else None
    if df_fed is None:
        # fallback: use synthetic data
        df_fed = df_small if 'df_small' in dir() else None

    if df_fed is not None:
        target_col = 'gdp_growth' if 'gdp_growth' in df_fed.columns else df_fed.columns[0]
        evaluator = FederationEvaluator(df_fed, num_nodes=3)
        mem_res = evaluator.evaluate_in_memory(target_col, train_years=20)
        phys_res = evaluator.evaluate_physical()
        federation_result = {'in_memory': mem_res, 'physical': phys_res}
        print(f"      In-memory FedAvg MSE={mem_res.get('mse', float('nan')):.4f}, "
              f"comm={mem_res.get('communication_bytes', 0)/1024:.1f} KB")
        print(f"      Physical: {phys_res.get('error', 'OK')}")
    print(f"      Done in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"      Warning: federation evaluation failed: {e}")


# ── Merge and regenerate report ──────────────────────────────────────────────
print("\n[Report] Merging results and generating report ...")

base_result['anomaly_detection'] = anomaly_results
if backtest_rows:
    base_result['real_world_backtest'] = backtest_rows
if federation_result:
    base_result['federation'] = federation_result

from benchmark.synthetic.reporting import generate_report
rpt_path, data_path = generate_report(base_result, out_dir=OUT_DIR)

print(f"\n{'=' * 60}")
print(f"Report updated: {rpt_path}")
print(f"Data updated:   {data_path}")
print(f"{'=' * 60}")

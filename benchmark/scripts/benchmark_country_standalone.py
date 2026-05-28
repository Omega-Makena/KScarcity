"""
Per-country standalone + federated rolling-origin backtest.

Runs the same full benchmark as benchmark_forecasting_horizons.py but with
an arbitrary primary country instead of hard-coded KEN.  Used for experiment #2:
run TZA and UGA as independent primary evaluation countries.

Protocol (mirrors benchmark_forecasting_horizons.py exactly):
  - Rolling origin: initial_train=10 years, cutoffs until last-h year
  - 9 methods: persistence, arima, prophet, prophet+graph,
               xgb blind/graph, lgbm blind/graph, tft-lite
  - 2 conditions: single (primary only) and federated (primary + pool countries)
  - 4 horizons: h=1,3,5,10

Usage:
    python benchmark/scripts/benchmark_country_standalone.py --primary TZA
    python benchmark/scripts/benchmark_country_standalone.py --primary UGA
    python benchmark/scripts/benchmark_country_standalone.py --primary TZA --pool KEN UGA RWA
    python benchmark/scripts/benchmark_country_standalone.py --primary TZA --horizons 1 3 --no-fed
"""

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
import logging
for _n in ('prophet', 'cmdstanpy'):
    logging.getLogger(_n).setLevel(logging.ERROR)
    logging.getLogger(_n).propagate = False

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse all heavy lifting from benchmark_forecasting_horizons
from benchmark.scripts.benchmark_forecasting_horizons import (
    TARGETS, HORIZONS, INITIAL_TRAIN, CONF_THRESHOLD, MIN_EVIDENCE,
    MAX_PARENTS, MIN_PAIRS, METHODS, METHOD_LABELS,
    load_countries, rolling_backtest,
    _mae, _mean_across_targets, _fmt, _delta_str,
)
from benchmark.real_data.world_bank_loader import prepare_multi_country_data

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_aggregate(all_records, conditions, targets, horizons):
    """Print TABLE 1: aggregate MAE across all targets."""
    W = 112
    cond_labels = [lbl for lbl, _ in conditions]
    single_lbl  = cond_labels[0]
    fed_lbl     = cond_labels[1] if len(cond_labels) > 1 else None

    print('\n' + '=' * W)
    print(f'TABLE 1 -- AGGREGATE MAE  (mean across {len(targets)} targets; lower=better)')
    cond_str = f'{single_lbl}' + (f' | {fed_lbl}' if fed_lbl else '')
    print(f'  Conditions: {cond_str}')
    print('=' * W)

    h_cols = [f'  h={h:2d}' for h in horizons]
    header = f"  {'Method':<22}" + ''.join(f'{c:>12}' for c in h_cols)
    if fed_lbl:
        header += ''.join(f'  fed-sgl h={h}' for h in horizons)
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for method in METHODS:
        row = f"  {METHOD_LABELS[method]:<22}"
        single_maes = {}
        for h in horizons:
            v = _mean_across_targets(all_records, single_lbl, h, method, targets)
            single_maes[h] = v
            row += f'{_fmt(v):>12}'
        if fed_lbl:
            for h in horizons:
                fv = _mean_across_targets(all_records, fed_lbl, h, method, targets)
                row += f'  {_delta_str(fv, single_maes[h]):>10}'
        print(row)


def print_target_table(all_records, primary, conditions, targets, horizons):
    """Print TABLE 2: per-target best method and h=1 XGBoost+Scarcity."""
    W = 112
    single_lbl = conditions[0][0]
    fed_lbl    = conditions[1][0] if len(conditions) > 1 else None

    print('\n' + '=' * W)
    print(f'TABLE 2 -- XGBoost+Scarcity single vs federated h=1  ({primary})')
    print('=' * W)
    print(f"  {'Target':<24} {'single MAE':>10} {'fed MAE':>10} {'delta':>8}  {'Fed helps?':>12}")
    print('  ' + '-' * 70)

    for t in targets:
        ms = _mae(all_records, single_lbl, t, 1, 'xgb_graph')
        mf = _mae(all_records, fed_lbl, t, 1, 'xgb_graph') if fed_lbl else np.nan
        if not np.isnan(ms) and not np.isnan(mf):
            delta  = ms - mf   # positive = fed helps
            helps  = 'YES' if delta > 0 else 'NO'
        else:
            delta, helps = np.nan, 'N/A'
        ds = f'{delta:+.4f}' if not np.isnan(delta) else '     N/A'
        fs = f'{mf:.4f}'     if not np.isnan(mf)     else '   N/A'
        ss = f'{ms:.4f}'     if not np.isnan(ms)      else '   N/A'
        print(f"  {t:<24} {ss:>10} {fs:>10} {ds:>8}  {helps:>12}")


def print_horizon_table(all_records, primary, conditions, targets, horizons):
    """Print TABLE 3: per-target best method per horizon (single condition)."""
    W = 112
    single_lbl = conditions[0][0]

    print('\n' + '=' * W)
    print(f'TABLE 3 -- BEST METHOD PER TARGET × HORIZON  ({primary} single-country)')
    print('=' * W)
    col_str = ''.join(f'  h={h}:best(MAE)' for h in horizons)
    print(f"  {'Target':<24}" + col_str)
    print('  ' + '-' * (24 + 22 * len(horizons)))

    for t in targets:
        row = f"  {t:<24}"
        for h in horizons:
            best_mae, best_m = np.inf, '--'
            for method in METHODS:
                v = _mae(all_records, single_lbl, t, h, method)
                if not np.isnan(v) and v < best_mae:
                    best_mae, best_m = v, method
            if best_m != '--':
                short = METHOD_LABELS[best_m].replace('Scarcity', 'Sc').replace('blind', 'bl')[:12]
                row += f'  {short:<12}({best_mae:.3f})'
            else:
                row += f'  {"N/A":<12}(  N/A )'
        print(row)


def save_results(all_records, out_path):
    import csv
    if not all_records:
        return
    keys = list(all_records[0].keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_records)
    print(f"\n  Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Per-country standalone + federated rolling-origin benchmark')
    parser.add_argument('--primary', default='TZA',
                        help='Primary evaluation country ISO code (default: TZA)')
    parser.add_argument('--pool', nargs='+', default=None,
                        help='Federation pool countries (default: remaining of KEN/TZA/UGA)')
    parser.add_argument('--horizons', nargs='+', type=int, default=HORIZONS)
    parser.add_argument('--targets', nargs='+', default=TARGETS)
    parser.add_argument('--no-fed', action='store_true', help='Skip federated condition')
    args = parser.parse_args()

    primary  = args.primary.upper()
    horizons = args.horizons
    targets  = args.targets

    # Default federation pool: KEN/TZA/UGA minus primary
    if args.pool is None:
        default_pool = ['KEN', 'TZA', 'UGA']
        pool = [c for c in default_pool if c != primary]
    else:
        pool = [c.upper() for c in args.pool if c.upper() != primary]

    print('=' * 80)
    print(f'STANDALONE BACKTEST: {primary}')
    print(f'  Primary:    {primary} (N=34 rolling-origin evaluation)')
    print(f'  Fed pool:   {pool}')
    print(f'  Horizons:   {horizons}')
    print(f'  Targets:    {len(targets)} variables')
    print(f'  Methods:    {len(METHODS)} ({", ".join(METHODS)})')
    print('=' * 80)

    # Load data
    all_countries = list(dict.fromkeys([primary] + pool))
    print(f'\nLoading data for: {all_countries} ...', flush=True)
    data = load_countries(all_countries)

    primary_df = data[primary]
    print(f'  {primary}: {len(primary_df)} years, {primary_df.shape[1]} indicators')
    for cc in pool:
        df = data.get(cc)
        if df is not None:
            pct = df.isnull().mean().mean() * 100
            print(f'  {cc}: {len(df)} years, {pct:.1f}% missing')
        else:
            print(f'  {cc}: FAILED TO LOAD (skipping)')
            pool.remove(cc)

    # Build conditions
    conditions = [(f'{primary}-single', {})]
    if not args.no_fed and pool:
        fed_aux = {cc: data[cc] for cc in pool if cc in data}
        conditions.append((f'{primary}-fed({"+".join(pool)})', fed_aux))

    # Run backtests
    all_records = []
    for label, aux_dfs in conditions:
        recs, _ = rolling_backtest(
            primary_df, aux_dfs, label, horizons, targets,
            INITIAL_TRAIN, CONF_THRESHOLD, MIN_EVIDENCE,
        )
        all_records.extend(recs)
        print(f'  [{label}] {len(recs)} records', flush=True)

    # Report
    print_aggregate(all_records, conditions, targets, horizons)
    print_target_table(all_records, primary, conditions, targets, horizons)
    print_horizon_table(all_records, primary, conditions, targets, horizons)

    # Save
    out_dir = _ROOT / 'artifacts' / 'benchmark_extended'
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_tag = '+'.join(pool) if pool else 'none'
    out_path = out_dir / f'standalone_{primary}_pool_{pool_tag}.csv'
    save_results(all_records, out_path)

    # Summary JSON for documentation
    summary = {}
    for label, _ in conditions:
        summary[label] = {}
        for t in targets:
            summary[label][t] = {}
            for h in horizons:
                best_mae, best_m = np.inf, None
                for method in METHODS:
                    v = _mae(all_records, label, t, h, method)
                    if not np.isnan(v) and v < best_mae:
                        best_mae, best_m = v, method
                summary[label][t][f'h{h}'] = {
                    'best_method': best_m,
                    'best_mae': round(best_mae, 4) if best_m else None,
                    'xgb_graph': _mae(all_records, label, t, h, 'xgb_graph'),
                    'arima': _mae(all_records, label, t, h, 'arima'),
                    'prophet': _mae(all_records, label, t, h, 'prophet'),
                }
    json_path = out_dir / f'standalone_{primary}_pool_{pool_tag}_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f'  Summary saved to {json_path}')


if __name__ == '__main__':
    main()

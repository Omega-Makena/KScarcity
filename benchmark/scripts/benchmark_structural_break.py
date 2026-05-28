"""
Structural-break robustness test: pre-2008 training, post-2008 evaluation.

Compares four conditions for each country (KEN, TZA, UGA):
  1. ARIMA            — rolling train, no graph
  2. XGBoost-blind    — rolling train, all features, no graph
  3. XGBoost-frozen   — graph discovered on pre-2008 data, then FROZEN for
                        all post-2008 predictions (tests whether pre-2008
                        structure survives the GFC regime change)
  4. XGBoost-rolling  — graph re-discovered at each cutoff (standard engine)

The GFC / post-2008 regime change is a natural stress test: capital flows,
inflation dynamics, and fiscal patterns all shifted.  If frozen graph MAE ≈
rolling graph MAE, structure is robust.  If frozen >> rolling, regime change
invalidated the pre-2008 edges.

Evaluation: direct h=1 forecast, cutoffs 2008–2022 (N=15 post-break years).
Graph discovery: full Scarcity engine (same as rolling backtest).

Usage:
    python benchmark/scripts/benchmark_structural_break.py
    python benchmark/scripts/benchmark_structural_break.py --countries KEN
"""

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.scripts.benchmark_forecasting_horizons import (
    load_countries,
    _mae,
    _top_k_graph,
)
from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COUNTRIES     = ['KEN', 'TZA', 'UGA']
TARGETS       = [
    'gdp_growth', 'inflation_cpi', 'unemployment',
    'exports_gdp', 'imports_gdp', 'current_account',
    'real_interest_rate', 'broad_money', 'private_credit',
    'govt_consumption',
]
BREAK_YEAR    = 2008        # first post-break test year
INITIAL_TRAIN = 10          # minimum years to start engine
MIN_TRAIN_XGB = 5
CONF_THRESHOLD = 0.35
MIN_EVIDENCE   = 5
MAX_PARENTS    = 5

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _build_engine_graph(primary_df: pd.DataFrame, cutoff: int) -> dict:
    """
    Stream data up to `cutoff` through the Scarcity engine and return
    graph dict: {target: [parent1, parent2, ...]}.
    """
    var_names = [c for c in TARGETS if c in primary_df.columns
                 and not primary_df[c].isna().all()]
    if len(var_names) < 2:
        return {}

    train = primary_df[primary_df.index <= cutoff]
    if len(train) < INITIAL_TRAIN:
        return {}

    try:
        engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
        schema = {'fields': [{'name': v} for v in var_names]}
        engine.initialize_v2(schema, use_causal=True)

        for yr in sorted(train.index):
            row = {k: float(v) for k, v in train.loc[yr].items()
                   if k in var_names and pd.notna(v)}
            if row:
                engine.process_row(row)

        graph, edges = extract_graph(engine,
                                     conf_threshold=CONF_THRESHOLD,
                                     min_evidence=MIN_EVIDENCE)
        return _top_k_graph(graph, edges, max_parents=MAX_PARENTS)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Forecasting helpers
# ---------------------------------------------------------------------------

def _arima_predict_one(series: pd.Series, cutoff_year: int) -> float:
    """ARIMA(1,1,0) trained up to cutoff, predict 1 step ahead."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        train = series[series.index <= cutoff_year].dropna()
        if len(train) < 4:
            return float('nan')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = ARIMA(train, order=(1, 1, 0)).fit()
            fc = m.forecast(steps=1)
            return float(fc.iloc[0])
    except Exception:
        return float('nan')


def _xgb_predict_one(df: pd.DataFrame, target: str, feature_cols: list,
                     cutoff_year: int) -> float:
    """XGBoost trained up to cutoff, predict target at cutoff+1 using features at cutoff."""
    try:
        import xgboost as xgb

        train = df[df.index <= cutoff_year].copy()
        X_cols = [c for c in feature_cols if c in train.columns]
        if not X_cols or len(train) < MIN_TRAIN_XGB:
            return float('nan')

        # Direct h=1: predict target[t+1] from features[t]
        X = train[X_cols].values[:-1]
        y = train[target].values[1:]

        # Drop rows where target is NaN
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if valid.sum() < MIN_TRAIN_XGB:
            return float('nan')

        m = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
        )
        m.fit(X[valid], y[valid])

        # Feature vector: last available row in train
        last_row = train[X_cols].iloc[-1].values.reshape(1, -1)
        if np.any(np.isnan(last_row)):
            return float(np.mean(y[valid]))
        return float(m.predict(last_row)[0])
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_country(country: str, country_data: dict) -> pd.DataFrame:
    primary_df = country_data[country]
    all_years   = sorted(primary_df.index.tolist())
    post_years  = [y for y in all_years if y >= BREAK_YEAR]

    if not post_years:
        print(f"  {country}: no post-{BREAK_YEAR} data available")
        return pd.DataFrame()

    # Step 1: discover graph on pre-2008 data once → frozen graph
    print(f"  [{country}] Discovering pre-{BREAK_YEAR} graph (data up to {BREAK_YEAR-1})...")
    t0 = time.time()
    frozen_graph = _build_engine_graph(primary_df, cutoff=BREAK_YEAR - 1)
    n_frozen_edges = sum(len(v) for v in frozen_graph.values())
    print(f"  [{country}] Frozen graph: {n_frozen_edges} edges  ({time.time()-t0:.1f}s)")

    rows = []
    for cutoff in post_years[:-1]:   # last year has no next-year truth
        actual_year = cutoff + 1
        if actual_year not in all_years:
            continue

        # XGBoost rolling graph (re-discover at each cutoff)
        rolling_graph = _build_engine_graph(primary_df, cutoff=cutoff)
        n_rolling_edges = sum(len(v) for v in rolling_graph.values())

        for target in TARGETS:
            if target not in primary_df.columns:
                continue
            actual = primary_df.loc[actual_year, target] if actual_year in primary_df.index else float('nan')
            if np.isnan(actual):
                continue

            all_feat = [c for c in TARGETS if c in primary_df.columns and c != target]

            # ARIMA
            pred_arima = _arima_predict_one(primary_df[target], cutoff)
            mae_arima  = abs(actual - pred_arima) if not np.isnan(pred_arima) else float('nan')

            # XGBoost blind (all features)
            pred_blind = _xgb_predict_one(primary_df, target, all_feat, cutoff)
            mae_blind  = abs(actual - pred_blind) if not np.isnan(pred_blind) else float('nan')

            # XGBoost frozen graph (pre-2008 structure)
            frozen_parents = frozen_graph.get(target, [])
            frozen_feat    = [c for c in frozen_parents if c in primary_df.columns] or all_feat
            pred_frozen    = _xgb_predict_one(primary_df, target, frozen_feat, cutoff)
            mae_frozen     = abs(actual - pred_frozen) if not np.isnan(pred_frozen) else float('nan')

            # XGBoost rolling graph (re-discovered each cutoff)
            rolling_parents = rolling_graph.get(target, [])
            rolling_feat    = [c for c in rolling_parents if c in primary_df.columns] or all_feat
            pred_rolling    = _xgb_predict_one(primary_df, target, rolling_feat, cutoff)
            mae_rolling     = abs(actual - pred_rolling) if not np.isnan(pred_rolling) else float('nan')

            rows.append({
                'country': country,
                'cutoff': cutoff,
                'target': target,
                'actual': actual,
                'n_frozen_edges': n_frozen_edges,
                'n_frozen_parents': len(frozen_parents),
                'n_rolling_edges': n_rolling_edges,
                'n_rolling_parents': len(rolling_parents),
                'mae_arima': mae_arima,
                'mae_blind': mae_blind,
                'mae_frozen': mae_frozen,
                'mae_rolling': mae_rolling,
            })

        print(f"  [{country}] cutoff={cutoff}  rolling_edges={n_rolling_edges}", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(df: pd.DataFrame):
    if df.empty:
        print("  No results to display.")
        return

    print()
    print('=' * 90)
    print('STRUCTURAL BREAK TEST: pre-2008 frozen graph vs rolling graph')
    print(f'  Training period: 1990–{BREAK_YEAR-1} (frozen graph)')
    print(f'  Test period:     {BREAK_YEAR}–2022  (post-GFC regime)')
    print('  lower MAE = better; delta_frozen = arima − frozen; delta_rolling = arima − rolling')
    print('=' * 90)

    metrics = ['mae_arima', 'mae_blind', 'mae_frozen', 'mae_rolling']

    for country in df['country'].unique():
        sub = df[df['country'] == country]
        print(f"\n  --- {country} ---")
        print(f"  {'Target':<22}  {'ARIMA':>7}  {'Blind':>7}  {'Frozen':>7}  {'Rolling':>7}  "
              f"{'Frz-ARM':>8}  {'Rol-ARM':>8}  {'Frz≈Rol?':>9}")
        print('  ' + '-' * 85)

        for target in TARGETS:
            t_sub = sub[sub['target'] == target]
            if t_sub.empty:
                continue
            row = {m: float(t_sub[m].mean()) for m in metrics}
            d_frozen  = row['mae_arima'] - row['mae_frozen']
            d_rolling = row['mae_arima'] - row['mae_rolling']
            robust    = 'YES' if abs(d_frozen - d_rolling) < 0.1 * max(abs(d_rolling), 0.001) else 'NO '
            print(f"  {target:<22}  {row['mae_arima']:7.4f}  {row['mae_blind']:7.4f}  "
                  f"{row['mae_frozen']:7.4f}  {row['mae_rolling']:7.4f}  "
                  f"{d_frozen:+8.4f}  {d_rolling:+8.4f}  {robust:>9}")

        # Aggregate row
        agg = {m: float(sub[m].mean()) for m in metrics}
        d_frozen  = agg['mae_arima'] - agg['mae_frozen']
        d_rolling = agg['mae_arima'] - agg['mae_rolling']
        print('  ' + '-' * 85)
        print(f"  {'MEAN (all targets)':<22}  {agg['mae_arima']:7.4f}  {agg['mae_blind']:7.4f}  "
              f"{agg['mae_frozen']:7.4f}  {agg['mae_rolling']:7.4f}  "
              f"{d_frozen:+8.4f}  {d_rolling:+8.4f}")

    # Cross-country summary
    print()
    print('=' * 90)
    print('CROSS-COUNTRY SUMMARY')
    print('=' * 90)
    agg_all = df.groupby('country')[metrics].mean()
    print(f"  {'Country':<8}  {'ARIMA':>7}  {'Blind':>7}  {'Frozen':>7}  {'Rolling':>7}  "
          f"{'FrozenΔ':>8}  {'RollingΔ':>9}  {'Robust?':>8}")
    print('  ' + '-' * 75)
    for c, row in agg_all.iterrows():
        df_val = row['mae_arima'] - row['mae_frozen']
        dr_val = row['mae_arima'] - row['mae_rolling']
        robust = 'YES' if abs(df_val - dr_val) < 0.10 * max(abs(dr_val), 0.001) else 'NO'
        print(f"  {c:<8}  {row['mae_arima']:7.4f}  {row['mae_blind']:7.4f}  "
              f"{row['mae_frozen']:7.4f}  {row['mae_rolling']:7.4f}  "
              f"{df_val:+8.4f}  {dr_val:+9.4f}  {robust:>8}")

    # Regime-change finding
    print()
    print('Key question: does frozen_MAE ≈ rolling_MAE?')
    print('  If YES: pre-2008 causal structure is robust to the GFC regime change.')
    print('  If NO:  regime change invalidated discovered edges — rolling is essential.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Structural break test: pre/post 2008')
    parser.add_argument('--countries', nargs='+', default=COUNTRIES)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    print('=' * 90)
    print('STRUCTURAL BREAK TEST')
    print(f'  Countries:   {args.countries}')
    print(f'  Break year:  {BREAK_YEAR} (GFC)')
    print(f'  Pre-break:   1990–{BREAK_YEAR-1}  (graph frozen from this window)')
    print(f'  Post-break:  {BREAK_YEAR}–2022  (test period)')
    print(f'  Conditions:  ARIMA | XGBoost-blind | XGBoost-frozen | XGBoost-rolling')
    print('=' * 90)

    all_countries = list(set(args.countries + COUNTRIES))
    print(f"\nLoading data: {all_countries} ...")
    country_data = load_countries(all_countries)

    all_rows = []
    t0 = time.time()
    for country in args.countries:
        if country not in country_data:
            print(f"  {country}: data not available, skipping")
            continue
        print(f"\n[{country}] Starting structural break evaluation...")
        df_c = evaluate_country(country, country_data)
        if not df_c.empty:
            all_rows.append(df_c)

    if not all_rows:
        print("No results produced.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    elapsed = time.time() - t0
    print(f'\n  Total: {len(results)} records, {elapsed:.0f}s')

    print_results(results)

    if not args.no_save:
        out_dir = _ROOT / 'artifacts' / 'benchmark_extended'
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / 'structural_break.csv'
        results.to_csv(path, index=False, float_format='%.4f')
        print(f'\n  Saved to {path.relative_to(_ROOT)}')


if __name__ == '__main__':
    main()

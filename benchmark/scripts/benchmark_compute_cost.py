"""
Computational cost comparison: Scarcity engine vs baselines.

Measures wall-clock time for each forecasting method across:
  - Graph discovery phase (once per cutoff)
  - Per-forecast prediction (h=1 direct)

Methods timed:
  1. ARIMA(1,1,0)      — statsmodels ARIMA, per cutoff
  2. Prophet           — fbprophet, per cutoff
  3. XGBoost blind     — XGBRegressor, per cutoff, all features
  4. XGBoost+Scarcity  — engine discovery + XGBRegressor, per cutoff
  5. LightGBM+Scarcity — engine discovery + LGBMRegressor, per cutoff
  6. TFT-lite          — Darts TFT (if available), per cutoff

Reported as:
  - Discovery time (s) per cutoff
  - Prediction time (ms) per forecast
  - Total time (s) for full rolling backtest (24 cutoffs × 10 targets)
  - Relative cost vs ARIMA

Country: KEN (best coverage, representative)
Cutoffs: subset of 10 (to keep runtime manageable)

Usage:
    python benchmark/scripts/benchmark_compute_cost.py
    python benchmark/scripts/benchmark_compute_cost.py --cutoffs 5
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

from benchmark.scripts.benchmark_forecasting_horizons import load_countries

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COUNTRY  = 'KEN'
TARGETS  = [
    'gdp_growth', 'inflation_cpi', 'unemployment',
    'exports_gdp', 'imports_gdp', 'current_account',
    'real_interest_rate', 'broad_money', 'private_credit',
    'govt_consumption',
]
CUTOFF_YEARS = list(range(2004, 2022, 2))   # 9 cutoffs (sparse subset)
MIN_TRAIN    = 5


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Method timing functions
# ---------------------------------------------------------------------------

def time_arima(df: pd.DataFrame, target: str, cutoff: int) -> tuple:
    """Returns (discovery_s, predict_s, prediction)."""
    from statsmodels.tsa.arima.model import ARIMA

    train = df[df.index <= cutoff][target].dropna()
    if len(train) < 4:
        return 0.0, 0.0, float('nan')

    disc_t = 0.0   # no discovery phase
    with Timer() as t:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = ARIMA(train, order=(1, 1, 0)).fit()
                pred = float(m.forecast(steps=1).iloc[0])
        except Exception:
            pred = float(train.mean())
    return disc_t, t.elapsed, pred


def time_prophet(df: pd.DataFrame, target: str, cutoff: int) -> tuple:
    try:
        from prophet import Prophet
        from prophet.diagnostics import cross_validation
    except ImportError:
        return 0.0, 0.0, float('nan')

    train = df[df.index <= cutoff][[target]].dropna().copy()
    if len(train) < 4:
        return 0.0, 0.0, float('nan')

    train_p = pd.DataFrame({'ds': pd.to_datetime([str(y) for y in train.index]),
                             'y': train[target].values})
    disc_t = 0.0
    with Timer() as t:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                import logging
                logging.getLogger('prophet').setLevel(logging.WARNING)
                logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
                m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                            daily_seasonality=False)
                m.fit(train_p)
                future = m.make_future_dataframe(periods=1, freq='YE')
                fc = m.predict(future)
                pred = float(fc['yhat'].iloc[-1])
        except Exception:
            pred = float(train[target].mean())
    return disc_t, t.elapsed, pred


def time_xgb_blind(df: pd.DataFrame, target: str, feature_cols: list, cutoff: int) -> tuple:
    try:
        import xgboost as xgb
    except ImportError:
        return 0.0, 0.0, float('nan')

    train = df[df.index <= cutoff].copy()
    X_cols = [c for c in feature_cols if c in train.columns]
    if not X_cols or len(train) < MIN_TRAIN:
        return 0.0, 0.0, float('nan')

    X = train[X_cols].values[:-1]
    y = train[target].values[1:]
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    if valid.sum() < MIN_TRAIN:
        return 0.0, 0.0, float('nan')

    disc_t = 0.0
    with Timer() as t:
        m = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                              subsample=0.8, random_state=42, verbosity=0)
        m.fit(X[valid], y[valid])
        last = train[X_cols].iloc[-1].values.reshape(1, -1)
        pred = float(m.predict(last)[0]) if not np.any(np.isnan(last)) else float(y[valid].mean())
    return disc_t, t.elapsed, pred


def _run_engine_discovery(train: pd.DataFrame, cols: list) -> dict:
    """Run Scarcity engine and return graph dict {target: [parents]}."""
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    from scarcity.engine.graph_extractor import extract_graph
    from benchmark.scripts.benchmark_forecasting_horizons import _top_k_graph

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    schema = {'fields': [{'name': v} for v in cols]}
    engine.initialize_v2(schema, use_causal=True)
    for yr in sorted(train.index):
        row = {k: float(v) for k, v in train.loc[yr].items()
               if k in cols and pd.notna(v)}
        if row:
            engine.process_row(row)
    graph, edges = extract_graph(engine, conf_threshold=0.35, min_evidence=5)
    return _top_k_graph(graph, edges, max_parents=5)


def time_engine_plus_xgb(df: pd.DataFrame, target: str, cutoff: int,
                          country_data: dict, country: str) -> tuple:
    """Time full discovery + XGBoost prediction."""
    try:
        import xgboost as xgb
    except ImportError:
        return 0.0, 0.0, float('nan')

    train = df[df.index <= cutoff].copy()
    cols = [c for c in TARGETS if c in train.columns and not train[c].isna().all()]
    if len(cols) < 2 or len(train) < 10:
        return 0.0, 0.0, float('nan')

    # Time discovery separately (shared across all targets at this cutoff)
    with Timer() as disc:
        try:
            graph = _run_engine_discovery(train, cols)
        except Exception:
            graph = {}

    parents   = graph.get(target, [])
    feat_cols = [c for c in parents if c in train.columns] or \
                [c for c in cols if c != target]

    with Timer() as pred_t:
        X = train[feat_cols].values[:-1]
        y = train[target].values[1:]
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if valid.sum() < MIN_TRAIN:
            return disc.elapsed, 0.0, float('nan')
        m = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                              subsample=0.8, random_state=42, verbosity=0)
        m.fit(X[valid], y[valid])
        last = train[feat_cols].iloc[-1].values.reshape(1, -1)
        pred = float(m.predict(last)[0]) if not np.any(np.isnan(last)) else float(y[valid].mean())

    return disc.elapsed, pred_t.elapsed, pred


def time_lgbm_plus_engine(df: pd.DataFrame, target: str, cutoff: int,
                           country_data: dict, country: str) -> tuple:
    """Time full discovery + LightGBM prediction."""
    try:
        import lightgbm as lgb
    except ImportError:
        return 0.0, 0.0, float('nan')

    train = df[df.index <= cutoff].copy()
    cols = [c for c in TARGETS if c in train.columns and not train[c].isna().all()]
    if len(cols) < 2 or len(train) < 10:
        return 0.0, 0.0, float('nan')

    with Timer() as disc:
        try:
            graph = _run_engine_discovery(train, cols)
        except Exception:
            graph = {}

    parents   = graph.get(target, [])
    feat_cols = [c for c in parents if c in train.columns] or \
                [c for c in cols if c != target]

    with Timer() as pred_t:
        X = train[feat_cols].values[:-1]
        y = train[target].values[1:]
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if valid.sum() < MIN_TRAIN:
            return disc.elapsed, 0.0, float('nan')
        m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                               subsample=0.8, random_state=42, verbose=-1)
        m.fit(X[valid], y[valid])
        last = train[feat_cols].iloc[-1].values.reshape(1, -1)
        pred = float(m.predict(last)[0]) if not np.any(np.isnan(last)) else float(y[valid].mean())

    return disc.elapsed, pred_t.elapsed, pred


# ---------------------------------------------------------------------------
# Main timing loop
# ---------------------------------------------------------------------------

def run_timing(df: pd.DataFrame, country: str, country_data: dict,
               cutoff_years: list) -> pd.DataFrame:
    feature_cols = [c for c in TARGETS if c in df.columns]
    rows = []

    methods = {
        'ARIMA':             lambda t, c: time_arima(df, t, c),
        'Prophet':           lambda t, c: time_prophet(df, t, c),
        'XGBoost-blind':     lambda t, c: time_xgb_blind(df, t, feature_cols, c),
        'XGBoost+Scarcity':  lambda t, c: time_engine_plus_xgb(df, t, c, country_data, country),
        'LightGBM+Scarcity': lambda t, c: time_lgbm_plus_engine(df, t, c, country_data, country),
    }

    for cutoff in cutoff_years:
        print(f"  cutoff={cutoff}", flush=True)
        for method, fn in methods.items():
            for target in [c for c in TARGETS if c in df.columns]:
                disc_s, pred_s, prediction = fn(target, cutoff)
                rows.append({
                    'country': country, 'cutoff': cutoff, 'target': target,
                    'method': method,
                    'discovery_s': disc_s,
                    'predict_s': pred_s,
                    'total_s': disc_s + pred_s,
                })

    return pd.DataFrame(rows)


def print_timing_table(df: pd.DataFrame):
    print()
    print('=' * 80)
    print('COMPUTATIONAL COST COMPARISON')
    print(f'  Country: {COUNTRY}  |  Cutoffs: {len(df["cutoff"].unique())}  '
          f'|  Targets: {len(df["target"].unique())}')
    print('  Times are mean per (cutoff × target) pair')
    print('=' * 80)

    agg = df.groupby('method').agg(
        disc_mean=('discovery_s', 'mean'),
        pred_mean=('predict_s', 'mean'),
        total_mean=('total_s', 'mean'),
    ).reset_index()

    arima_total = float(agg[agg['method'] == 'ARIMA']['total_mean'].iloc[0]) if 'ARIMA' in agg['method'].values else 1.0

    print(f"\n  {'Method':<22}  {'Discovery(s)':>13}  {'Predict(ms)':>12}  "
          f"{'Total(ms)':>10}  {'vs ARIMA':>9}")
    print('  ' + '-' * 72)
    order = ['ARIMA', 'Prophet', 'XGBoost-blind', 'XGBoost+Scarcity', 'LightGBM+Scarcity']
    for m in order:
        row = agg[agg['method'] == m]
        if row.empty:
            continue
        disc  = float(row['disc_mean'].iloc[0])
        pred  = float(row['pred_mean'].iloc[0])
        total = float(row['total_mean'].iloc[0])
        ratio = total / arima_total if arima_total > 0 else float('nan')
        print(f"  {m:<22}  {disc:13.4f}  {pred*1000:12.1f}  "
              f"{total*1000:10.1f}  {ratio:>8.1f}x")

    # Full backtest extrapolation (24 cutoffs × 10 targets)
    print()
    print('=' * 80)
    print('EXTRAPOLATED: full rolling backtest (24 cutoffs × 10 targets)')
    print('=' * 80)
    print(f"  {'Method':<22}  {'Total (min)':>12}  {'vs ARIMA':>9}")
    print('  ' + '-' * 50)
    arima_full_s = None
    for m in order:
        row = agg[agg['method'] == m]
        if row.empty:
            continue
        total_per = float(row['total_mean'].iloc[0])
        full_s = total_per * 24 * 10
        if m == 'ARIMA':
            arima_full_s = full_s
        ratio = full_s / arima_full_s if arima_full_s else float('nan')
        print(f"  {m:<22}  {full_s/60:12.1f}  {ratio:>8.1f}x")

    print()
    print('Note: XGBoost+Scarcity "discovery" = full graph engine run.')
    print('      For h≥3 the engine runs once per cutoff across all targets.')
    print('      Amortized discovery cost per target = discovery_s / n_targets.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute cost benchmark')
    parser.add_argument('--cutoffs', type=int, default=len(CUTOFF_YEARS),
                        help='Number of cutoffs to use (default: all)')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    cutoffs = CUTOFF_YEARS[:args.cutoffs]

    print('=' * 80)
    print('COMPUTATIONAL COST BENCHMARK')
    print(f'  Country:  {COUNTRY}')
    print(f'  Cutoffs:  {cutoffs}')
    print(f'  Targets:  {len(TARGETS)}')
    print(f'  Methods:  ARIMA | Prophet | XGBoost-blind | XGBoost+Scarcity | LightGBM+Scarcity')
    print('=' * 80)

    print(f"\nLoading data: {[COUNTRY]} ...")
    country_data = load_countries([COUNTRY])
    df = country_data[COUNTRY]

    print(f"\nTiming {len(cutoffs)} cutoffs × {len(TARGETS)} targets × 5 methods...")
    t0 = time.time()
    results = run_timing(df, COUNTRY, country_data, cutoffs)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    print_timing_table(results)

    if not args.no_save:
        out_dir = _ROOT / 'artifacts' / 'benchmark_extended'
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / 'compute_cost.csv'
        results.to_csv(path, index=False, float_format='%.6f')
        print(f'\n  Saved to {path.relative_to(_ROOT)}')


if __name__ == '__main__':
    main()

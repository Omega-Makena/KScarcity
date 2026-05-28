"""
Multi-Target Multi-Horizon Forecasting Benchmark

Research questions:
  1. Does Prophet's data-scarce advantage hold across longer horizons (h=5, h=10)?
  2. Does graph feature selection (XGBoost+Scarcity) benefit persist across horizons and targets?
  3. Which targets are hardest to forecast at short vs long horizon?
  4. Does federation help more at long horizons where structural knowledge matters most?

Methods (9):
  persistence, arima, prophet, prophet+scarcity,
  xgboost blind/graph, lightgbm blind/graph, tft-lite

Targets (10):
  gdp_growth, inflation_cpi, unemployment,
  exports_gdp, imports_gdp, current_account,
  real_interest_rate, broad_money, private_credit, govt_consumption

Horizons:
  h=1  — 1-year-ahead  (short)
  h=3  — 3-year-ahead  (medium-short)
  h=5  — 5-year-ahead  (medium-long)
  h=10 — 10-year-ahead (long)

Direct multi-step:
  At cutoff year C, direct training pairs are (X[t], y[t+h]) for t in [t_start, C-h].
  Features at prediction time = variable values at C (lag-h in absolute terms).
  Falls back to ARIMA when fewer than MIN_PAIRS training pairs are available.
  Prophet and ARIMA: fit once per (cutoff, target), forecast all h simultaneously.
  Tree models and TFT: fit separately per h (direct pairs differ by horizon).

Conditions:
  single — Kenya only (N=34)
  federated — KEN+TZA+UGA (N_eff≈102, graph from all three; models trained on Kenya only)

Rolling origin:
  Initial train: INITIAL_TRAIN years (1990–1999)
  Cutoff years: 1999, 2000, ..., 2022
  Prediction years: cutoff + h  (must exist in Kenya data)
    h=1:  24 test points  (predict 2000–2023)
    h=3:  22 test points  (predict 2002–2023)
    h=5:  20 test points  (predict 2004–2023)
    h=10: 15 test points  (predict 2009–2023)

Usage:
    python benchmark/scripts/benchmark_forecasting_horizons.py
    python benchmark/scripts/benchmark_forecasting_horizons.py --no-fed
    python benchmark/scripts/benchmark_forecasting_horizons.py --targets gdp_growth inflation_cpi
    python benchmark/scripts/benchmark_forecasting_horizons.py --horizons 1 5
"""

import argparse
import io
import sys
import warnings
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph
from scarcity.engine.gpu_engine import GPUDiscoveryEngine, gpu_extract_graph
from benchmark.real_data.world_bank_loader import prepare_multi_country_data

import torch as _torch
_USE_GPU = _torch.cuda.is_available()
if _USE_GPU:
    print(f"  GPU discovery: CUDA available ({_torch.cuda.get_device_name(0)})", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TARGETS = [
    'gdp_growth', 'inflation_cpi', 'unemployment',
    'exports_gdp', 'imports_gdp', 'current_account',
    'real_interest_rate', 'broad_money', 'private_credit', 'govt_consumption',
]

HORIZONS      = [1, 3, 5, 10]
INITIAL_TRAIN = 10
CONF_THRESHOLD = 0.35
MIN_EVIDENCE   = 5
MAX_PARENTS    = 5
MIN_PAIRS      = 4   # minimum direct training pairs for tree/TFT; fall back to ARIMA otherwise

METHODS = [
    'persistence', 'arima', 'prophet', 'prophet_graph',
    'xgb_blind', 'xgb_graph', 'lgbm_blind', 'lgbm_graph', 'tft',
]

METHOD_LABELS = {
    'persistence':   'Persistence',
    'arima':         'ARIMA(1,1,0)',
    'prophet':       'Prophet',
    'prophet_graph': 'Prophet+Scarcity',
    'xgb_blind':     'XGBoost blind',
    'xgb_graph':     'XGBoost+Scarcity',
    'lgbm_blind':    'LightGBM blind',
    'lgbm_graph':    'LightGBM+Scarcity',
    'tft':           'TFT-lite',
}

GRAPH_METHODS = {'prophet_graph', 'xgb_graph', 'lgbm_graph'}
HORIZON_GROUPS = [('short',  [1, 3]), ('long', [5, 10])]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_countries(countries):
    print(f"  Loading data: {', '.join(countries)} ...", flush=True)
    data = prepare_multi_country_data(countries)
    cleaned = {}
    for cc, df in data.items():
        df = df.ffill().bfill()
        # Drop columns with no data at all — they break feature engineering
        fully_missing = [c for c in df.columns if df[c].isnull().all()]
        if fully_missing:
            print(f"  {cc}: dropping {len(fully_missing)} fully-missing cols: {fully_missing}")
            df = df.drop(columns=fully_missing)
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        cleaned[cc] = df
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Direct multi-step feature builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_direct_pairs(train_df, target, feature_cols, h):
    """
    Build (X[t], y[t+h]) training pairs for direct h-step prediction.
    X[t] = all feature variables at time t.
    y[t+h] = target at t+h (must be within training window).
    Returns (X, y, X_last, used_cols) or (None, None, None, []).
    """
    cols = [c for c in feature_cols if c in train_df.columns and c != target]
    if not cols:
        return None, None, None, []
    needed = cols + [target]
    sub = train_df[needed].copy()
    for c in needed:
        sub[c] = sub[c].fillna(sub[c].mean())
    sub = sub.dropna()
    n = len(sub)
    if n < h + MIN_PAIRS:
        return None, None, None, []
    # Pair: features at t predicting target at t+h
    X      = sub[cols].iloc[:n - h].values   # rows 0 .. n-h-1
    y      = sub[target].iloc[h:].values      # rows h .. n-1
    X_last = sub[cols].iloc[-1].values        # features at cutoff (last row)
    return X, y, X_last, cols


# ─────────────────────────────────────────────────────────────────────────────
# Graph helper (type-diverse top-K)
# ─────────────────────────────────────────────────────────────────────────────

def _top_k_graph(graph, edges, max_parents):
    parent_type_conf = {}
    for e in edges:
        tgt   = e['target']
        src   = e['source']
        rtype = e.get('type', 'unknown')
        conf  = float(e['confidence'])
        parent_type_conf.setdefault(tgt, {}).setdefault(src, {})
        parent_type_conf[tgt][src][rtype] = max(
            parent_type_conf[tgt][src].get(rtype, 0.0), conf)

    filtered = {}
    for tgt, parents in graph.items():
        pt = parent_type_conf.get(tgt, {})
        type_champ = {}
        for src in parents:
            for rtype, conf in pt.get(src, {}).items():
                if conf > type_champ.get(rtype, ('', 0.0))[1]:
                    type_champ[rtype] = (src, conf)
        selected = {src for src, _ in type_champ.values()}
        all_conf = {src: max(pt.get(src, {}).values(), default=0.0) for src in parents}
        for src in sorted(parents, key=lambda p: all_conf.get(p, 0.0), reverse=True):
            if len(selected) >= max_parents:
                break
            selected.add(src)
        filtered[tgt] = sorted(selected, key=lambda p: all_conf.get(p, 0.0), reverse=True)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Individual predictors
# ─────────────────────────────────────────────────────────────────────────────

def _arima_multi(series, max_h):
    """Fit ARIMA(1,1,0) and return forecasts for steps 1..max_h."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = np.array(series, dtype=float)
            if len(s) < 4:
                return {}
            m = ARIMA(s, order=(1, 1, 0)).fit()
            fc = m.forecast(steps=max_h)
            return {h: float(fc[h - 1]) for h in range(1, max_h + 1)}
    except Exception:
        return {}


def _prophet_multi(train_df, target, horizons, regressors=None):
    """
    Fit Prophet on training series, return predictions at each cutoff+h date.
    regressors: list of parent column names (filled with last known value in future).
    """
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_p = pd.DataFrame({
                'ds': pd.to_datetime(train_df.index.astype(str), format='%Y'),
                'y':  train_df[target].values,
            })
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False)
            reg_cols = []
            if regressors:
                for col in regressors:
                    if col in train_df.columns:
                        df_p[col] = train_df[col].values
                        m.add_regressor(col)
                        reg_cols.append(col)
            m.fit(df_p)
            last_year = int(train_df.index[-1])
            future_years = sorted(set(last_year + h for h in horizons))
            future_ds = pd.DataFrame({
                'ds': pd.to_datetime([str(y) for y in future_years], format='%Y')
            })
            for col in reg_cols:
                future_ds[col] = float(train_df[col].iloc[-1])
            fc = m.predict(future_ds)
            year_to_pred = {int(row['ds'].year): float(row['yhat'])
                            for _, row in fc.iterrows()}
            return {h: year_to_pred.get(last_year + h, np.nan) for h in horizons}
    except Exception:
        return {h: np.nan for h in horizons}


def _predict_xgb(train_df, target, h, feature_cols, graph_parents=None):
    try:
        import xgboost as xgb
        cols = graph_parents if graph_parents else feature_cols
        X, y, X_last, used = _build_direct_pairs(train_df, target, cols, h)
        if X is None:
            return np.nan
        mdl = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
        )
        mdl.fit(X, y)
        return float(mdl.predict(X_last.reshape(1, -1))[0])
    except Exception:
        return np.nan


def _predict_lgbm(train_df, target, h, feature_cols, graph_parents=None):
    try:
        import lightgbm as lgb
        cols = graph_parents if graph_parents else feature_cols
        X, y, X_last, used = _build_direct_pairs(train_df, target, cols, h)
        if X is None:
            return np.nan
        mdl = lgb.LGBMRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            num_leaves=7, verbose=-1, random_state=42,
        )
        mdl.fit(X, y)
        return float(mdl.predict(X_last.reshape(1, -1))[0])
    except Exception:
        return np.nan


def _predict_tft(train_df, target, h, feature_cols):
    try:
        import torch
        import torch.nn as nn
        cols = [c for c in feature_cols if c in train_df.columns and c != target]
        X, y, X_last, used = _build_direct_pairs(train_df, target, cols, h)
        if X is None:
            return np.nan

        X_mean = X.mean(axis=0);  X_std = X.std(axis=0) + 1e-8
        y_mean = float(y.mean()); y_std = float(y.std()) + 1e-8
        Xt = torch.tensor((X - X_mean) / X_std, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
        n_f, h_dim = Xt.shape[2], 16

        class _TFT(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(n_f, h_dim)
                self.attn = nn.MultiheadAttention(h_dim, num_heads=1, batch_first=True)
                self.norm = nn.LayerNorm(h_dim)
                self.out  = nn.Linear(h_dim, 1)
            def forward(self, x):
                z = torch.relu(self.proj(x))
                a, _ = self.attn(z, z, z)
                return self.out(self.norm(z + a).squeeze(1))

        model = _TFT()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
        for _ in range(50):
            loss = ((model(Xt) - yt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        X_last_t = torch.tensor(
            ((X_last - X_mean) / X_std), dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            raw = model(X_last_t).item()
        return float(raw * y_std + y_mean)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Core: run all methods for one (train_df, target) across all horizons
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_all(train_df, target, horizons, feature_cols, graph_parents):
    """
    Returns dict: {h: {method: predicted_scalar}}.
    Prophet and ARIMA fitted once; tree/TFT fitted per h.
    Falls back to ARIMA scalar when tree/TFT have insufficient pairs.
    """
    results = {h: {} for h in horizons}
    max_h   = max(horizons)
    series  = train_df[target].dropna().values if target in train_df.columns else np.array([])

    # ── Persistence (same for all h) ──────────────────────────────────────────
    last_val = float(series[-1]) if len(series) > 0 else np.nan
    for h in horizons:
        results[h]['persistence'] = last_val

    # ── ARIMA (fit once, extract at each h) ───────────────────────────────────
    arima_fc = _arima_multi(series, max_h)
    for h in horizons:
        results[h]['arima'] = arima_fc.get(h, np.nan)

    # ── Prophet (fit once, predict at all h) ──────────────────────────────────
    prophet_fc      = _prophet_multi(train_df, target, horizons)
    prophet_graph_fc = _prophet_multi(train_df, target, horizons,
                                       regressors=graph_parents if graph_parents else None)
    for h in horizons:
        results[h]['prophet']       = prophet_fc.get(h, np.nan)
        # If no graph parents, prophet_graph == prophet
        pg = prophet_graph_fc.get(h, np.nan)
        results[h]['prophet_graph'] = pg if not np.isnan(pg) else results[h]['prophet']

    # ── Tree models and TFT (per h — different direct training pairs) ─────────
    arima_fallback = {h: arima_fc.get(h, np.nan) for h in horizons}

    for h in horizons:
        # XGBoost blind
        p = _predict_xgb(train_df, target, h, feature_cols)
        results[h]['xgb_blind'] = p if not np.isnan(p) else arima_fallback[h]

        # XGBoost graph
        if graph_parents:
            p = _predict_xgb(train_df, target, h, feature_cols, graph_parents=graph_parents)
        else:
            p = results[h]['xgb_blind']
        results[h]['xgb_graph'] = p if not np.isnan(p) else arima_fallback[h]

        # LightGBM blind
        p = _predict_lgbm(train_df, target, h, feature_cols)
        results[h]['lgbm_blind'] = p if not np.isnan(p) else arima_fallback[h]

        # LightGBM graph
        if graph_parents:
            p = _predict_lgbm(train_df, target, h, feature_cols, graph_parents=graph_parents)
        else:
            p = results[h]['lgbm_blind']
        results[h]['lgbm_graph'] = p if not np.isnan(p) else arima_fallback[h]

        # TFT-lite
        p = _predict_tft(train_df, target, h, feature_cols)
        results[h]['tft'] = p if not np.isnan(p) else arima_fallback[h]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rolling backtest (multi-horizon)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_backtest(ken_df, aux_dfs, label, horizons, targets, initial_train,
                     conf_threshold, min_evidence):
    var_names    = sorted(ken_df.columns.tolist())
    years        = sorted(ken_df.index.tolist())
    all_years    = set(years)
    feature_cols = ken_df.columns.tolist()

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    schema = {'fields': [{'name': v} for v in var_names]}
    engine.initialize_v2(schema, use_causal=True)

    print(f"\n  [{label}] Streaming initial {initial_train} training years ...", flush=True)
    for yr in years[:initial_train]:
        engine.process_row({k: float(v) for k, v in ken_df.loc[yr].items() if pd.notna(v)})
        for cc, aux in aux_dfs.items():
            if yr in aux.index:
                rd = {k: float(v) for k, v in aux.loc[yr].reindex(var_names).items()
                      if pd.notna(v)}
                if rd:
                    engine.process_row(rd)

    records = []
    max_h   = max(horizons)

    # First cutoff = end of initial training window (index initial_train-1)
    for ci in range(initial_train - 1, len(years)):
        cutoff_yr  = years[ci]
        train_data = ken_df[ken_df.index <= cutoff_yr]

        # Skip if no evaluation horizon falls within available data
        valid_hs = [h for h in horizons if (cutoff_yr + h) in all_years]
        if not valid_hs:
            _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names)
            continue

        graph, edges  = extract_graph(engine, conf_threshold=conf_threshold,
                                       min_evidence=min_evidence)
        graph_topk    = _top_k_graph(graph, edges, max_parents=MAX_PARENTS)
        n_edges       = sum(len(v) for v in graph.values())
        print(f"  [{label}] cutoff={cutoff_yr}  N={len(train_data)}  "
              f"edges={n_edges}  valid_h={valid_hs}", flush=True)

        for target in targets:
            if target not in ken_df.columns:
                continue
            parents = graph_topk.get(target, [])
            preds   = _forecast_all(train_data, target, valid_hs, feature_cols, parents)

            for h in valid_hs:
                pred_yr = cutoff_yr + h
                actual  = ken_df.loc[pred_yr, target] if pred_yr in all_years else np.nan
                if pd.isna(actual):
                    continue
                actual = float(actual)

                for method in METHODS:
                    pred = preds[h].get(method, np.nan)
                    try:
                        ae = abs(actual - float(pred)) if not np.isnan(float(pred)) else np.nan
                    except (TypeError, ValueError):
                        ae = np.nan
                    records.append({
                        'label': label, 'cutoff': cutoff_yr, 'h': h,
                        'target': target, 'pred_yr': pred_yr,
                        'method': method, 'actual': actual, 'ae': ae,
                    })

        _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names)

    return records, engine


def _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names):
    """Stream the next calendar year into the engine."""
    if ci + 1 >= len(years):
        return
    next_yr = years[ci + 1]
    engine.process_row({k: float(v) for k, v in ken_df.loc[next_yr].items()
                        if pd.notna(v)})
    for cc, aux in aux_dfs.items():
        if next_yr in aux.index:
            rd = {k: float(v) for k, v in aux.loc[next_yr].reindex(var_names).items()
                  if pd.notna(v)}
            if rd:
                engine.process_row(rd)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mae(records, label, target, h, method):
    vals = [r['ae'] for r in records
            if r['label'] == label and r['target'] == target
            and r['h'] == h and r['method'] == method
            and not np.isnan(r['ae'])]
    return round(float(np.mean(vals)), 4) if vals else np.nan


def _count(records, label, target, h, method):
    return sum(1 for r in records
               if r['label'] == label and r['target'] == target
               and r['h'] == h and r['method'] == method
               and not np.isnan(r['ae']))


def _mean_across_targets(records, label, h, method, targets):
    vals = []
    for t in targets:
        v = _mae(records, label, t, h, method)
        if not np.isnan(v):
            vals.append(v)
    return round(float(np.mean(vals)), 4) if vals else np.nan


def _fmt(v, w=7):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ' ' * w + 'N/A'
    return f'{v:{w}.4f}'


def _delta_str(new_v, ref_v):
    if np.isnan(new_v) or np.isnan(ref_v):
        return '   N/A'
    d = new_v - ref_v
    return f'{d:+.4f}'


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

def print_results(all_records, conditions, targets, horizons):
    W = 112
    cond_labels = [lbl for lbl, _ in conditions]
    single_lbl  = cond_labels[0]
    fed_lbl     = cond_labels[1] if len(cond_labels) > 1 else None

    # ── Table 1: Mean MAE across all targets, by (method, h) ─────────────────
    print('\n' + '=' * W)
    print('TABLE 1 — AGGREGATE MAE  (mean across all targets; lower = better)')
    print(f"  Condition: {single_lbl}" + (f" | {fed_lbl}" if fed_lbl else ''))
    print('=' * W)

    h_cols = [f'  h={h:2d}' for h in horizons]
    header = f"  {'Method':<22}" + ''.join(f'{c:>12}' for c in h_cols)
    if fed_lbl:
        header += ''.join(f'  fed_h={h}' for h in horizons)
    print(header)
    print('  ' + '─' * (len(header) - 2))

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
                row += f'  {_delta_str(fv, single_maes[h]):>8}'
        print(row)

    # ── Table 2: Best method per (target, h) ─────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 2 — BEST METHOD PER TARGET AND HORIZON  (single-country MAE)')
    print('=' * W)
    th_header = f"  {'Target':<22}" + ''.join(f'  h={h}:best(MAE)' for h in horizons)
    print(th_header)
    print('  ' + '─' * (len(th_header) - 2))

    for target in targets:
        row = f"  {target:<22}"
        for h in horizons:
            best_mae = np.inf
            best_m   = '—'
            for method in METHODS:
                v = _mae(all_records, single_lbl, target, h, method)
                if not np.isnan(v) and v < best_mae:
                    best_mae, best_m = v, method
            if best_m != '—':
                short = METHOD_LABELS[best_m].replace('Scarcity', 'Scar').replace('blind', 'bln')[:12]
                row += f'  {short:<12}({best_mae:.3f})'
            else:
                row += f'  {"N/A":<12}(  N/A  )'
        print(row)

    # ── Table 3: Prophet vs tree+graph leader board ───────────────────────────
    print('\n' + '=' * W)
    print('TABLE 3 — PROPHET vs XGBOOST+SCARCITY  (single-country MAE; winner bolded)')
    print('=' * W)

    col_heads = ''.join(f'  h={h}  P-XgS' for h in horizons)
    print(f"  {'Target':<22}" + col_heads)
    print('  ' + '─' * (22 + 14 * len(horizons)))

    for target in targets:
        row = f"  {target:<22}"
        for h in horizons:
            p   = _mae(all_records, single_lbl, target, h, 'prophet')
            xgs = _mae(all_records, single_lbl, target, h, 'xgb_graph')
            if np.isnan(p) and np.isnan(xgs):
                row += '   N/A    N/A'
            elif np.isnan(xgs):
                row += f'  {p:.3f} Prophet'
            elif np.isnan(p):
                row += f'  N/A   {xgs:.3f}'
            else:
                winner = 'Prph' if p <= xgs else 'XgS '
                diff   = xgs - p
                row += f'  {p:.3f}/{xgs:.3f}({diff:+6.3f}){winner}'
        print(row)

    # ── Table 4: Graph selection benefit (blind→graph MAE delta) by horizon ──
    print('\n' + '=' * W)
    print('TABLE 4 — GRAPH FEATURE SELECTION BENEFIT  (negative = graph helps)')
    print(f"  blind_MAE − graph_MAE  |  single-country")
    print('=' * W)

    for model_pair, model_name in [('xgb', 'XGBoost'), ('lgbm', 'LightGBM')]:
        blind_k = f'{model_pair}_blind'
        graph_k = f'{model_pair}_graph'
        print(f"\n  {model_name}")
        print(f"  {'Target':<22}" + ''.join(f'  h={h:2d}   ' for h in horizons))
        print('  ' + '─' * (22 + 10 * len(horizons)))
        for target in targets:
            row = f"  {target:<22}"
            for h in horizons:
                b = _mae(all_records, single_lbl, target, h, blind_k)
                g = _mae(all_records, single_lbl, target, h, graph_k)
                if np.isnan(b) or np.isnan(g):
                    row += '     N/A  '
                else:
                    d = g - b
                    tag = ' <' if d < -0.1 else ('  >' if d > 0.1 else '  ~')
                    row += f'  {d:+6.3f}{tag}'
            print(row)
        # Aggregate row
        agg_row = f"  {'[mean across targets]':<22}"
        for h in horizons:
            deltas = []
            for t in targets:
                b = _mae(all_records, single_lbl, t, h, blind_k)
                g = _mae(all_records, single_lbl, t, h, graph_k)
                if not (np.isnan(b) or np.isnan(g)):
                    deltas.append(g - b)
            if deltas:
                d = float(np.mean(deltas))
                tag = ' <' if d < -0.1 else ('  >' if d > 0.1 else '  ~')
                agg_row += f'  {d:+6.3f}{tag}'
            else:
                agg_row += '     N/A  '
        print(agg_row)

    # ── Table 5: Federation benefit by horizon ────────────────────────────────
    if fed_lbl:
        print('\n' + '=' * W)
        print('TABLE 5 — FEDERATION BENEFIT  (single_MAE − fed_MAE; positive = fed helps)')
        print(f"  Comparing: {single_lbl} vs {fed_lbl}")
        print('=' * W)
        for method in ['prophet', 'xgb_graph', 'tft']:
            print(f"\n  {METHOD_LABELS[method]}")
            print(f"  {'Target':<22}" + ''.join(f'  h={h:2d}   ' for h in horizons))
            print('  ' + '─' * (22 + 10 * len(horizons)))
            for target in targets:
                row = f"  {target:<22}"
                for h in horizons:
                    s = _mae(all_records, single_lbl, target, h, method)
                    f_ = _mae(all_records, fed_lbl,    target, h, method)
                    if np.isnan(s) or np.isnan(f_):
                        row += '     N/A  '
                    else:
                        d   = s - f_   # positive = federation helps
                        tag = ' >' if d > 0.1 else ('  <' if d < -0.1 else '  ~')
                        row += f'  {d:+6.3f}{tag}'
                print(row)

    # ── Table 6: Short vs long horizon aggregate summary ─────────────────────
    print('\n' + '=' * W)
    print('TABLE 6 — SHORT vs LONG HORIZON SUMMARY  (mean MAE across all targets; single-country)')
    print('=' * W)
    short_h = [h for h in horizons if h <= 3]
    long_h  = [h for h in horizons if h > 3]

    def _group_mean(label, method, hs):
        vals = []
        for h in hs:
            v = _mean_across_targets(all_records, label, h, method, targets)
            if not np.isnan(v):
                vals.append(v)
        return round(float(np.mean(vals)), 4) if vals else np.nan

    print(f"\n  {'Method':<22}  {'Short (h≤3)':>12}  {'Long (h>3)':>12}  {'Degradation':>12}")
    print('  ' + '─' * 65)
    for method in METHODS:
        s = _group_mean(single_lbl, method, short_h)
        l = _group_mean(single_lbl, method, long_h)
        deg = _delta_str(l, s) if not (np.isnan(s) or np.isnan(l)) else '   N/A'
        print(f"  {METHOD_LABELS[method]:<22}  {_fmt(s):>12}  {_fmt(l):>12}  {deg:>12}")

    print('\n' + '=' * W)
    n_single = sum(1 for r in all_records if r['label'] == single_lbl)
    n_fed    = sum(1 for r in all_records if fed_lbl and r['label'] == fed_lbl)
    test_pts = {h: _count(all_records, single_lbl, targets[0], h, 'arima') for h in horizons}
    print(f"  Records: {n_single} single-country" +
          (f" | {n_fed} federated" if fed_lbl else ''))
    print(f"  Test points per horizon (first target): " +
          '  '.join(f'h={h}:{n}' for h, n in test_pts.items()))
    print('=' * W + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Multi-target multi-horizon forecasting benchmark')
    parser.add_argument('--no-fed',    action='store_true', help='Skip federated condition')
    parser.add_argument('--targets',   nargs='+', default=None, help='Subset of targets')
    parser.add_argument('--horizons',  nargs='+', type=int, default=None, help='Subset of horizons')
    parser.add_argument('--initial-train', type=int, default=INITIAL_TRAIN)
    args = parser.parse_args()

    tgts = args.targets  if args.targets  else TARGETS
    hors = sorted(args.horizons) if args.horizons else HORIZONS
    i_tr = args.initial_train

    print("=" * 80)
    print("MULTI-TARGET MULTI-HORIZON FORECASTING BENCHMARK")
    print(f"  Targets : {tgts}")
    print(f"  Horizons: {hors}")
    print(f"  Initial training: {i_tr} years")
    print("=" * 80)

    # Load data
    countries = ['KEN', 'TZA', 'UGA']
    data = load_countries(countries)
    ken_df = data.get('KEN')
    if ken_df is None:
        print("ERROR: Kenya data not loaded"); sys.exit(1)

    aux_dfs = {cc: df for cc, df in data.items() if cc != 'KEN'}

    # Filter targets to those actually in ken_df
    tgts = [t for t in tgts if t in ken_df.columns]
    if not tgts:
        print("ERROR: None of the specified targets found in Kenya data"); sys.exit(1)
    print(f"  Active targets ({len(tgts)}): {tgts}")

    conditions = []

    # ── Single-country condition ──────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("CONDITION: single-country (KEN only)")
    print("─" * 80)
    single_records, _ = rolling_backtest(
        ken_df, {}, 'single',
        horizons=hors, targets=tgts,
        initial_train=i_tr,
        conf_threshold=CONF_THRESHOLD,
        min_evidence=MIN_EVIDENCE,
    )
    conditions.append(('single', single_records))
    all_records = list(single_records)

    # ── Federated condition ───────────────────────────────────────────────────
    if not args.no_fed:
        print("\n" + "─" * 80)
        print("CONDITION: federated (KEN + TZA + UGA)")
        print("─" * 80)
        fed_records, _ = rolling_backtest(
            ken_df, aux_dfs, 'federated',
            horizons=hors, targets=tgts,
            initial_train=i_tr,
            conf_threshold=CONF_THRESHOLD,
            min_evidence=MIN_EVIDENCE,
        )
        conditions.append(('federated', fed_records))
        all_records.extend(fed_records)

    # ── Print results ─────────────────────────────────────────────────────────
    print_results(all_records, conditions, tgts, hors)


if __name__ == '__main__':
    main()

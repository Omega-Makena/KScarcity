"""
Extended Forecasting Benchmark: BVAR Minnesota Prior + Chronos Zero-Shot + Bootstrap CIs

Adds three peer-reviewed baselines to the existing multi-horizon rolling-origin backtest:

  1. BVAR (Minnesota / Litterman 1986 prior)
     Canonical macro-econometrics baseline for short series with many variables.
     Implemented via Bańbura-Giannone-Reichlin (2010) dummy observation approach:
     prior encoded as augmented rows, OLS on stacked system.
     λ=0.2 tightness, δ=1 (random-walk own-lag), µ=5 (co-persistence), p=1 lag.

  2. Chronos zero-shot (Amazon, T5-based, apache-2.0)
     Foundation model pretrained on millions of time series.
     No fine-tuning — pure zero-shot inference on Kenya macro series.
     Uses amazon/chronos-t5-small (fastest tier, ~50M parameters).

  3. Bootstrap 95% CIs on every MAE number
     1000 bootstrap resamples of the rolling-origin fold AEs.
     CI format: MAE [lower, upper].

Methods (11 total):
  persistence, arima, prophet, prophet_graph,
  bvar,
  xgb_blind, xgb_graph, lgbm_blind, lgbm_graph, tft,
  chronos

Conditions:
  single    — Kenya only (N=34)
  federated — KEN+TZA+UGA (N_eff≈102, graph from all three; models fit on Kenya only)

Rolling origin: cutoffs 1999–2022, h ∈ {1, 3, 5, 10}

Usage:
    python benchmark/scripts/benchmark_forecasting_extended.py
    python benchmark/scripts/benchmark_forecasting_extended.py --no-fed
    python benchmark/scripts/benchmark_forecasting_extended.py --targets gdp_growth inflation_cpi
    python benchmark/scripts/benchmark_forecasting_extended.py --horizons 1 3
    python benchmark/scripts/benchmark_forecasting_extended.py --bootstrap-samples 500
    python benchmark/scripts/benchmark_forecasting_extended.py --no-chronos  # skip if HF Hub blocked

Chronos model download:
    Chronos requires downloading weights from HuggingFace Hub on first run (~50MB for t5-tiny).
    If HuggingFace is blocked, pre-download on another machine and copy to:
      %USERPROFILE%\.cache\huggingface\hub\models--amazon--chronos-t5-tiny\
    Then run normally — the pipeline uses the local cache automatically.
    Alternatively: huggingface-cli download amazon/chronos-t5-tiny --local-dir ./chronos_weights
    and set: export HF_HUB_CACHE=./chronos_weights
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
for _name in ('prophet', 'cmdstanpy'):
    _lg = logging.getLogger(_name)
    _lg.propagate = False
    for _h in _lg.handlers[:]:
        _lg.removeHandler(_h)

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph
from benchmark.real_data.world_bank_loader import prepare_multi_country_data

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TARGETS = [
    'gdp_growth', 'inflation_cpi', 'unemployment',
    'exports_gdp', 'imports_gdp', 'current_account',
    'real_interest_rate', 'broad_money', 'private_credit', 'govt_consumption',
]

HORIZONS       = [1, 3, 5, 10]
INITIAL_TRAIN  = 10
CONF_THRESHOLD = 0.35
MIN_EVIDENCE   = 5
MAX_PARENTS    = 5
MIN_PAIRS      = 4

# BVAR Minnesota prior hyperparameters (Litterman 1986 defaults)
BVAR_LAGS    = 1
BVAR_LAMBDA  = 0.2   # overall tightness (smaller = tighter prior)
BVAR_DELTA   = 1.0   # own-lag 1 target (1 = random walk prior)
BVAR_MU      = 5.0   # co-persistence (sum-of-coefficients dummy weight)

# Bootstrap CI
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_ALPHA   = 0.05  # 95% CI

# GPU detection
try:
    import torch as _torch
    _DEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'
except ImportError:
    _DEVICE = 'cpu'

# Chronos availability (installed via pip install chronos-forecasting)
_CHRONOS_PIPE = None  # loaded lazily on first call

METHODS = [
    'persistence', 'arima', 'prophet', 'prophet_graph',
    'bvar',
    'xgb_blind', 'xgb_graph', 'lgbm_blind', 'lgbm_graph',
    'tft',
    'chronos',
]

METHOD_LABELS = {
    'persistence':   'Persistence',
    'arima':         'ARIMA(1,1,0)',
    'prophet':       'Prophet',
    'prophet_graph': 'Prophet+Scarcity',
    'bvar':          'BVAR-Minnesota',
    'xgb_blind':     'XGBoost blind',
    'xgb_graph':     'XGBoost+Scarcity',
    'lgbm_blind':    'LightGBM blind',
    'lgbm_graph':    'LightGBM+Scarcity',
    'tft':           'TFT-lite',
    'chronos':       'Chronos-T5-small',
}

GRAPH_METHODS  = {'prophet_graph', 'xgb_graph', 'lgbm_graph'}
HORIZON_GROUPS = [('short', [1, 3]), ('long', [5, 10])]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_countries(countries):
    print(f"  Loading data: {', '.join(countries)} ...", flush=True)
    data = prepare_multi_country_data(countries)
    cleaned = {}
    for cc, df in data.items():
        df = df.ffill().bfill()
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        cleaned[cc] = df
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Direct multi-step feature builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_direct_pairs(train_df, target, feature_cols, h):
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
    X      = sub[cols].iloc[:n - h].values
    y      = sub[target].iloc[h:].values
    X_last = sub[cols].iloc[-1].values
    return X, y, X_last, cols


# ─────────────────────────────────────────────────────────────────────────────
# Graph helper
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
# BVAR with Minnesota (Litterman 1986) prior
# Encoding: Bańbura, Giannone, Reichlin (2010) dummy observations
# ─────────────────────────────────────────────────────────────────────────────

def _bvar_fit(train_df, lags=BVAR_LAGS,
              lambda_=BVAR_LAMBDA, delta=BVAR_DELTA, mu=BVAR_MU):
    """
    Fit BVAR with Minnesota prior via Bańbura-Giannone-Reichlin (2010) dummy observations.

    Returns (B, cols, Y, T, ti_map) where:
      B   — coefficient matrix (K*p+1, K), rows: [lag1_var0, ..., lag1_varK, lag2..., intercept]
      cols — ordered list of column names
      Y   — cleaned data array (T, K)
      T   — number of time steps

    Returns None on failure.
    """
    try:
        cols = list(train_df.columns)
        sub  = train_df[cols].copy()
        for c in cols:
            sub[c] = sub[c].ffill().bfill().fillna(sub[c].mean())
        sub = sub.dropna(how='any')
        if len(sub) < lags + 4:
            return None

        K  = len(cols)
        p  = lags
        Y  = sub.values.astype(float)   # (T, K)
        T  = Y.shape[0]

        sigma = np.std(np.diff(Y, axis=0), axis=0) if T > 1 else np.ones(K)
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)

        # Dummy set 1: own-lag Minnesota shrinkage (own toward δσ/λ, cross toward 0)
        Yd1 = np.zeros((K * p, K))
        Xd1 = np.zeros((K * p, K * p + 1))
        for j in range(K):
            for l in range(p):
                row   = j * p + l
                scale = delta * sigma[j] / (lambda_ * (l + 1))
                Yd1[row, j]          = scale
                Xd1[row, j * p + l] = scale

        # Dummy set 2: sums-of-coefficients / co-persistence
        sig_diag = np.diag(sigma * mu)
        Yd2 = sig_diag
        Xd2 = np.column_stack([np.tile(sig_diag, (1, p)), np.zeros((K, 1))])

        # Dummy set 3: diffuse intercept
        Yd3 = np.zeros((1, K))
        Xd3 = np.zeros((1, K * p + 1))
        Xd3[0, -1] = 1.0 / lambda_

        # Actual data matrices
        Y_actual = Y[p:, :]
        X_rows   = []
        for t in range(p, T):
            row = []
            for l in range(1, p + 1):
                row.extend(Y[t - l, :])
            row.append(1.0)
            X_rows.append(row)
        X_actual = np.array(X_rows)

        Y_star = np.vstack([Y_actual, Yd1, Yd2, Yd3])
        X_star = np.vstack([X_actual, Xd1, Xd2, Xd3])

        B, _, _, _ = np.linalg.lstsq(X_star, Y_star, rcond=None)
        return B, cols, Y, T

    except Exception:
        return None


def _bvar_forecast_multi(train_df, target, horizons,
                         lags=BVAR_LAGS, lambda_=BVAR_LAMBDA,
                         delta=BVAR_DELTA, mu=BVAR_MU):
    """
    Fit BVAR once, return recursive h-step forecasts for all horizons.
    Fits the coefficient matrix once per (cutoff, target) call.
    Returns dict {h: scalar} — uses NaN fallback if fit fails.
    """
    result = _bvar_fit(train_df, lags, lambda_, delta, mu)
    if result is None:
        return {h: np.nan for h in horizons}

    B, cols, Y, T = result
    if target not in cols:
        return {h: np.nan for h in horizons}
    ti = cols.index(target)
    p  = lags

    max_h     = max(horizons)
    state_init = [Y[T - l - 1, :].copy() for l in range(p)]

    # Run recursion to max_h, snapshot at each requested horizon
    state   = [s.copy() for s in state_init]
    preds   = {}
    h_set   = set(horizons)
    for step in range(1, max_h + 1):
        x_vec  = np.concatenate(state)
        x_vec  = np.append(x_vec, 1.0)
        y_next = B.T @ x_vec
        state  = [y_next] + state[:-1]
        if step in h_set:
            preds[step] = float(state[0][ti])

    return {h: preds.get(h, np.nan) for h in horizons}


# ─────────────────────────────────────────────────────────────────────────────
# Chronos zero-shot (Amazon T5-based foundation model)
# ─────────────────────────────────────────────────────────────────────────────

def _load_chronos_pipeline():
    global _CHRONOS_PIPE
    if _CHRONOS_PIPE is not None:
        return _CHRONOS_PIPE
    try:
        import torch
        import threading
        from chronos import ChronosPipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        result = [None]
        error  = [None]

        def _download(model_id):
            try:
                result[0] = ChronosPipeline.from_pretrained(
                    model_id,
                    device_map=device,
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                error[0] = e

        for model_id in ['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small']:
            t = threading.Thread(target=_download, args=(model_id,), daemon=True)
            t.start()
            t.join(timeout=120)   # 2-minute timeout per model
            if result[0] is not None:
                _CHRONOS_PIPE = result[0]
                print(f"  [Chronos] Loaded {model_id} on {device}", flush=True)
                break
            elif not t.is_alive():
                print(f"  [Chronos] {model_id} failed: {error[0]}", flush=True)
            else:
                print(f"  [Chronos] Download timed out for {model_id} — check HuggingFace Hub access", flush=True)
                break  # don't retry if it's a network hang

        return _CHRONOS_PIPE
    except Exception as exc:
        print(f"  [Chronos] Not available: {exc}", flush=True)
        return None


def _chronos_predict(series, max_h):
    """
    Zero-shot Chronos forecast on a univariate numpy array.
    Returns dict {h: scalar} for h in 1..max_h.
    Uses median of the predictive distribution.
    """
    try:
        import torch
        pipe = _load_chronos_pipeline()
        if pipe is None:
            return {}
        s = np.array(series, dtype=float)
        s = s[~np.isnan(s)]
        if len(s) < 3:
            return {}
        context = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            forecast = pipe.predict(context, prediction_length=max_h, num_samples=20)
        # forecast: (1, num_samples, max_h) — take median across samples
        median_fc = forecast.squeeze(0).median(dim=0).values.cpu().numpy()
        return {h: float(median_fc[h - 1]) for h in range(1, max_h + 1)}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Standard predictors
# ─────────────────────────────────────────────────────────────────────────────

def _arima_multi(series, max_h):
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
    try:
        from prophet import Prophet
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
            last_year    = int(train_df.index[-1])
            future_years = sorted(set(last_year + h for h in horizons))
            future_ds    = pd.DataFrame({
                'ds': pd.to_datetime([str(y) for y in future_years], format='%Y')
            })
            for col in reg_cols:
                future_ds[col] = float(train_df[col].iloc[-1])
            fc            = m.predict(future_ds)
            year_to_pred  = {int(row['ds'].year): float(row['yhat'])
                             for _, row in fc.iterrows()}
            return {h: year_to_pred.get(last_year + h, np.nan) for h in horizons}
    except Exception:
        return {h: np.nan for h in horizons}


def _predict_xgb(train_df, target, h, feature_cols, graph_parents=None):
    try:
        import xgboost as xgb
        cols = graph_parents if graph_parents else feature_cols
        X, y, X_last, _ = _build_direct_pairs(train_df, target, cols, h)
        if X is None:
            return np.nan
        gpu_params = {'device': 'cuda'} if _DEVICE == 'cuda' else {}
        mdl = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
            **gpu_params,
        )
        mdl.fit(X, y)
        return float(mdl.predict(X_last.reshape(1, -1))[0])
    except Exception:
        return np.nan


def _predict_lgbm(train_df, target, h, feature_cols, graph_parents=None):
    try:
        import lightgbm as lgb
        cols = graph_parents if graph_parents else feature_cols
        X, y, X_last, _ = _build_direct_pairs(train_df, target, cols, h)
        if X is None:
            return np.nan
        gpu_params = {'device': 'gpu'} if _DEVICE == 'cuda' else {}
        mdl = lgb.LGBMRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            num_leaves=7, verbose=-1, random_state=42,
            **gpu_params,
        )
        mdl.fit(X, y)
        return float(mdl.predict(X_last.reshape(1, -1))[0])
    except Exception:
        return np.nan


def _predict_tft(train_df, target, h, feature_cols):
    try:
        import torch
        import torch.nn as nn
        device = torch.device(_DEVICE)
        cols   = [c for c in feature_cols if c in train_df.columns and c != target]
        X, y, X_last, _ = _build_direct_pairs(train_df, target, cols, h)
        if X is None:
            return np.nan

        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0) + 1e-8
        y_mean = float(y.mean())
        y_std  = float(y.std()) + 1e-8
        Xt = torch.tensor((X - X_mean) / X_std, dtype=torch.float32).unsqueeze(1).to(device)
        yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32).unsqueeze(1).to(device)
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

        model = _TFT().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
        for _ in range(50):
            loss = ((model(Xt) - yt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        X_last_t = torch.tensor(
            (X_last - X_mean) / X_std, dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(X_last_t).item()
        return float(raw * y_std + y_mean)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Core: run all methods for one (train_df, target) across all horizons
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_all(train_df, target, horizons, feature_cols, graph_parents):
    results = {h: {} for h in horizons}
    max_h   = max(horizons)
    series  = (train_df[target].dropna().values
               if target in train_df.columns else np.array([]))

    # Persistence
    last_val = float(series[-1]) if len(series) > 0 else np.nan
    for h in horizons:
        results[h]['persistence'] = last_val

    # ARIMA (fit once, all h)
    arima_fc = _arima_multi(series, max_h)
    for h in horizons:
        results[h]['arima'] = arima_fc.get(h, np.nan)
    arima_fallback = {h: arima_fc.get(h, np.nan) for h in horizons}

    # BVAR Minnesota — fit once, forecast all horizons recursively
    bvar_fc = _bvar_forecast_multi(train_df, target, horizons)
    for h in horizons:
        v = bvar_fc.get(h, np.nan)
        results[h]['bvar'] = v if not np.isnan(v) else arima_fallback[h]

    # Prophet (fit once, all h)
    prophet_fc       = _prophet_multi(train_df, target, horizons)
    prophet_graph_fc = _prophet_multi(train_df, target, horizons,
                                      regressors=graph_parents if graph_parents else None)
    for h in horizons:
        results[h]['prophet']       = prophet_fc.get(h, np.nan)
        pg = prophet_graph_fc.get(h, np.nan)
        results[h]['prophet_graph'] = pg if not np.isnan(pg) else results[h]['prophet']

    # Chronos zero-shot — only if method is active (skipped with --no-chronos)
    if 'chronos' in METHODS:
        chronos_fc = _chronos_predict(series, max_h)
        for h in horizons:
            results[h]['chronos'] = chronos_fc.get(h, np.nan)

    # Tree models and TFT (per h)
    for h in horizons:
        p = _predict_xgb(train_df, target, h, feature_cols)
        results[h]['xgb_blind'] = p if not np.isnan(p) else arima_fallback[h]

        if graph_parents:
            p = _predict_xgb(train_df, target, h, feature_cols,
                             graph_parents=graph_parents)
        else:
            p = results[h]['xgb_blind']
        results[h]['xgb_graph'] = p if not np.isnan(p) else arima_fallback[h]

        p = _predict_lgbm(train_df, target, h, feature_cols)
        results[h]['lgbm_blind'] = p if not np.isnan(p) else arima_fallback[h]

        if graph_parents:
            p = _predict_lgbm(train_df, target, h, feature_cols,
                              graph_parents=graph_parents)
        else:
            p = results[h]['lgbm_blind']
        results[h]['lgbm_graph'] = p if not np.isnan(p) else arima_fallback[h]

        p = _predict_tft(train_df, target, h, feature_cols)
        results[h]['tft'] = p if not np.isnan(p) else arima_fallback[h]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Engine advance helper
# ─────────────────────────────────────────────────────────────────────────────

def _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names):
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
# Rolling backtest
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
        engine.process_row({k: float(v) for k, v in ken_df.loc[yr].items()
                            if pd.notna(v)})
        for cc, aux in aux_dfs.items():
            if yr in aux.index:
                rd = {k: float(v) for k, v in aux.loc[yr].reindex(var_names).items()
                      if pd.notna(v)}
                if rd:
                    engine.process_row(rd)

    records = []

    for ci in range(initial_train - 1, len(years)):
        cutoff_yr  = years[ci]
        train_data = ken_df[ken_df.index <= cutoff_yr]

        valid_hs = [h for h in horizons if (cutoff_yr + h) in all_years]
        if not valid_hs:
            _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names)
            continue

        graph, edges = extract_graph(engine, conf_threshold=conf_threshold,
                                     min_evidence=min_evidence)
        graph_topk   = _top_k_graph(graph, edges, max_parents=MAX_PARENTS)
        n_edges      = sum(len(v) for v in graph.values())
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
                        ae = (abs(actual - float(pred))
                              if not np.isnan(float(pred)) else np.nan)
                    except (TypeError, ValueError):
                        ae = np.nan
                    records.append({
                        'label':   label,
                        'cutoff':  cutoff_yr,
                        'h':       h,
                        'target':  target,
                        'pred_yr': pred_yr,
                        'method':  method,
                        'actual':  actual,
                        'ae':      ae,
                    })

        _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(ae_vals, n_samples=BOOTSTRAP_SAMPLES, alpha=BOOTSTRAP_ALPHA, rng=None):
    """
    Non-parametric bootstrap 95% CI on mean MAE from per-fold absolute errors.
    Returns (mean, lower, upper).  Returns (nan, nan, nan) if fewer than 2 folds.
    """
    vals = [v for v in ae_vals if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        m = float(np.mean(vals)) if vals else np.nan
        return m, np.nan, np.nan
    arr = np.array(vals, dtype=float)
    if rng is None:
        rng = np.random.default_rng(42)
    boots = rng.choice(arr, size=(n_samples, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def _get_ae_vals(records, label, target, h, method):
    return [r['ae'] for r in records
            if r['label'] == label and r['target'] == target
            and r['h'] == h and r['method'] == method]


def _get_ae_vals_across_targets(records, label, h, method, targets):
    """Pool per-fold AEs across all targets for aggregate CI."""
    vals = []
    for t in targets:
        vals.extend(_get_ae_vals(records, label, t, h, method))
    return vals


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ci(mean, lo, hi, w=7):
    if np.isnan(mean):
        return ' ' * (w + 16) + 'N/A'
    if np.isnan(lo):
        return f'{mean:{w}.4f} [  —  ]'
    return f'{mean:{w}.4f} [{lo:.3f},{hi:.3f}]'


def _fmt(v, w=7):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ' ' * w + 'N/A'
    return f'{v:{w}.4f}'


def _delta_str(new_v, ref_v):
    if np.isnan(new_v) or np.isnan(ref_v):
        return '   N/A'
    return f'{new_v - ref_v:+.4f}'


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

def print_results(all_records, conditions, targets, horizons, bootstrap_n):
    W            = 130
    cond_labels  = [lbl for lbl, _ in conditions]
    single_lbl   = cond_labels[0]
    fed_lbl      = cond_labels[1] if len(cond_labels) > 1 else None
    rng          = np.random.default_rng(42)

    # ── Table 1: Aggregate MAE + 95% CI across all targets ───────────────────
    print('\n' + '=' * W)
    print('TABLE 1 — AGGREGATE MAE with 95% Bootstrap CI  '
          f'(mean across all targets; B={bootstrap_n}; lower = better)')
    print(f"  Condition: {single_lbl}" + (f" | {fed_lbl}" if fed_lbl else ''))
    print('  CI columns: MAE [2.5%, 97.5%] — non-parametric bootstrap resample of rolling-origin folds')
    print('=' * W)

    col_w = 24
    h_cols = [f'h={h}' for h in horizons]
    header = f"  {'Method':<22}" + ''.join(f'{c:^{col_w}}' for c in h_cols)
    print(header)
    print('  ' + '-' * (W - 2))

    for method in METHODS:
        label_str = METHOD_LABELS.get(method, method)
        row = f"  {label_str:<22}"
        for h in horizons:
            ae_vals = _get_ae_vals_across_targets(all_records, single_lbl, h, method, targets)
            mean_, lo, hi = _bootstrap_ci(ae_vals, n_samples=bootstrap_n, rng=rng)
            cell = _fmt_ci(mean_, lo, hi, w=6)
            row += f'{cell:^{col_w}}'
        print(row)

    if fed_lbl:
        print('\n  Federated condition:')
        print('  ' + '-' * (W - 2))
        for method in METHODS:
            label_str = METHOD_LABELS.get(method, method)
            row = f"  {label_str:<22}"
            for h in horizons:
                ae_vals = _get_ae_vals_across_targets(all_records, fed_lbl, h, method, targets)
                mean_, lo, hi = _bootstrap_ci(ae_vals, n_samples=bootstrap_n, rng=rng)
                cell = _fmt_ci(mean_, lo, hi, w=6)
                row += f'{cell:^{col_w}}'
            print(row)

    # ── Table 2: Per-target MAE at h=1 with CI, single condition ─────────────
    print('\n' + '=' * W)
    print(f'TABLE 2 — PER-TARGET MAE h=1 with 95% Bootstrap CI  [{single_lbl}]')
    print(f'  B={bootstrap_n} bootstrap resamples; CI format: MAE [2.5%, 97.5%]')
    print('=' * W)

    new_methods = ['bvar', 'chronos']
    compare_methods = ['persistence', 'arima', 'prophet', 'bvar', 'chronos', 'xgb_graph']
    col_w2 = 22
    hdr2 = f"  {'Target':<22}" + ''.join(f'{METHOD_LABELS.get(m, m):^{col_w2}}' for m in compare_methods)
    print(hdr2)
    print('  ' + '-' * (W - 2))

    for tgt in targets:
        row = f"  {tgt:<22}"
        for method in compare_methods:
            ae_vals = _get_ae_vals(all_records, single_lbl, tgt, 1, method)
            mean_, lo, hi = _bootstrap_ci(ae_vals, n_samples=bootstrap_n, rng=rng)
            if np.isnan(mean_):
                cell = f"{'N/A':^{col_w2}}"
            elif np.isnan(lo):
                cell = f"{mean_:.4f} [—]"
                cell = f"{cell:^{col_w2}}"
            else:
                cell = f"{mean_:.4f} [{lo:.3f},{hi:.3f}]"
                cell = f"{cell:^{col_w2}}"
            row += cell
        print(row)

    # ── Table 3: BVAR vs ARIMA vs Persistence — all horizons, aggregate ───────
    print('\n' + '=' * W)
    print(f'TABLE 3 — BVAR Minnesota vs Classical Baselines  '
          f'[{single_lbl}]  (aggregate across all targets)')
    print('  Delta columns: MAE(BVAR) − MAE(baseline); negative = BVAR better')
    print('=' * W)

    baselines = ['persistence', 'arima', 'prophet']
    h_w = 12
    hdr3 = (f"  {'h':>4}  {'BVAR MAE':>12}  " +
            ''.join(f'  vs {METHOD_LABELS[b]:>22}' for b in baselines))
    print(hdr3)
    print('  ' + '-' * (W - 2))

    for h in horizons:
        bvar_ae  = _get_ae_vals_across_targets(all_records, single_lbl, h, 'bvar', targets)
        bvar_m, bvar_lo, bvar_hi = _bootstrap_ci(bvar_ae, n_samples=bootstrap_n, rng=rng)
        row = (f"  h={h:<2}  {bvar_m:>8.4f} [{bvar_lo:.3f},{bvar_hi:.3f}]  ")
        for baseline in baselines:
            base_ae = _get_ae_vals_across_targets(all_records, single_lbl, h, baseline, targets)
            base_m, _, _ = _bootstrap_ci(base_ae, n_samples=bootstrap_n, rng=rng)
            delta = bvar_m - base_m
            sign  = '+' if delta >= 0 else ''
            row  += f"  {sign}{delta:>8.4f} (vs {base_m:.4f})             "
        print(row)

    # ── Table 4: Chronos vs best ML — all horizons, aggregate ────────────────
    print('\n' + '=' * W)
    print(f'TABLE 4 — Chronos Zero-Shot vs Best ML Baselines  '
          f'[{single_lbl}]  (aggregate across all targets)')
    print('  Key comparison: foundation model zero-shot vs graph-conditioned methods')
    print('=' * W)

    compare_vs_chronos = ['arima', 'prophet', 'xgb_graph', 'lgbm_graph']
    hdr4 = (f"  {'h':>4}  {'Chronos MAE':>20}  " +
            ''.join(f'  {METHOD_LABELS[m]:>20}' for m in compare_vs_chronos))
    print(hdr4)
    print('  ' + '-' * (W - 2))

    for h in horizons:
        chr_ae = _get_ae_vals_across_targets(all_records, single_lbl, h, 'chronos', targets)
        chr_m, chr_lo, chr_hi = _bootstrap_ci(chr_ae, n_samples=bootstrap_n, rng=rng)
        if np.isnan(chr_m):
            row = f"  h={h:<2}  {'N/A':^20}"
        else:
            row = f"  h={h:<2}  {chr_m:>8.4f} [{chr_lo:.3f},{chr_hi:.3f}]  "
        for m in compare_vs_chronos:
            ae  = _get_ae_vals_across_targets(all_records, single_lbl, h, m, targets)
            mv, lo, hi = _bootstrap_ci(ae, n_samples=bootstrap_n, rng=rng)
            if np.isnan(chr_m) or np.isnan(mv):
                row += f"  {'N/A':>20}"
            else:
                delta = chr_m - mv
                sign  = '+' if delta >= 0 else ''
                row  += f"  {mv:.4f} [{lo:.3f},{hi:.3f}]  ({sign}{delta:.4f})"
        print(row)

    # ── Table 5: CI overlap analysis — which deltas are significant ───────────
    print('\n' + '=' * W)
    print(f'TABLE 5 — Delta Significance (do 95% CIs overlap?)  [{single_lbl}]  h=1')
    print('  SIGNIFICANT: CIs of two methods do NOT overlap (robust difference)')
    print('  OVERLAP:     CIs overlap (cannot conclude one is better at 95% level)')
    print('=' * W)

    sig_pairs = [
        ('bvar',    'arima'),
        ('bvar',    'prophet'),
        ('chronos', 'arima'),
        ('chronos', 'xgb_graph'),
        ('chronos', 'bvar'),
        ('xgb_graph', 'prophet'),
    ]
    print(f"  {'Pair':<32}  {'Method A MAE':>18}  {'Method B MAE':>18}  {'Overlap?':>10}")
    print('  ' + '-' * (W - 2))
    for (ma, mb) in sig_pairs:
        ae_a = _get_ae_vals_across_targets(all_records, single_lbl, 1, ma, targets)
        ae_b = _get_ae_vals_across_targets(all_records, single_lbl, 1, mb, targets)
        ma_mean, ma_lo, ma_hi = _bootstrap_ci(ae_a, n_samples=bootstrap_n, rng=rng)
        mb_mean, mb_lo, mb_hi = _bootstrap_ci(ae_b, n_samples=bootstrap_n, rng=rng)
        if np.isnan(ma_mean) or np.isnan(mb_mean):
            overlap_str = 'N/A'
        elif ma_hi < mb_lo or mb_hi < ma_lo:
            overlap_str = 'SIGNIFICANT'
        else:
            overlap_str = 'overlap'
        pair_str = f'{METHOD_LABELS.get(ma, ma)} vs {METHOD_LABELS.get(mb, mb)}'
        a_str = (f'{ma_mean:.4f} [{ma_lo:.3f},{ma_hi:.3f}]'
                 if not np.isnan(ma_mean) else 'N/A')
        b_str = (f'{mb_mean:.4f} [{mb_lo:.3f},{mb_hi:.3f}]'
                 if not np.isnan(mb_mean) else 'N/A')
        print(f"  {pair_str:<32}  {a_str:>18}  {b_str:>18}  {overlap_str:>10}")

    # ── Table 6: Best method per target (h=1, single), with CI ───────────────
    print('\n' + '=' * W)
    print(f'TABLE 6 — Best Method per Target  [h=1, {single_lbl}]')
    print('  Winner: lowest mean bootstrap MAE; CI shown for winner and ARIMA baseline')
    print('=' * W)
    print(f"  {'Target':<22}  {'Winner':<20}  {'Winner MAE (95% CI)':>26}  "
          f"{'ARIMA MAE (95% CI)':>26}  {'Delta':>8}")
    print('  ' + '-' * (W - 2))

    for tgt in targets:
        best_method = None
        best_mean   = np.inf
        for method in METHODS:
            ae_v = _get_ae_vals(all_records, single_lbl, tgt, 1, method)
            m_, _, _ = _bootstrap_ci(ae_v, n_samples=bootstrap_n, rng=rng)
            if not np.isnan(m_) and m_ < best_mean:
                best_mean   = m_
                best_method = method

        if best_method is None:
            print(f"  {tgt:<22}  {'N/A'}")
            continue

        ae_best = _get_ae_vals(all_records, single_lbl, tgt, 1, best_method)
        b_m, b_lo, b_hi = _bootstrap_ci(ae_best, n_samples=bootstrap_n, rng=rng)
        ae_arima = _get_ae_vals(all_records, single_lbl, tgt, 1, 'arima')
        a_m, a_lo, a_hi = _bootstrap_ci(ae_arima, n_samples=bootstrap_n, rng=rng)

        b_str = (f'{b_m:.4f} [{b_lo:.3f},{b_hi:.3f}]'
                 if not np.isnan(b_m) else 'N/A')
        a_str = (f'{a_m:.4f} [{a_lo:.3f},{a_hi:.3f}]'
                 if not np.isnan(a_m) else 'N/A')
        delta = (b_m - a_m) if not (np.isnan(b_m) or np.isnan(a_m)) else np.nan
        d_str = f'{delta:+.4f}' if not np.isnan(delta) else 'N/A'
        print(f"  {tgt:<22}  {METHOD_LABELS.get(best_method, best_method):<20}  "
              f"{b_str:>26}  {a_str:>26}  {d_str:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extended benchmark: BVAR + Chronos + Bootstrap CIs'
    )
    parser.add_argument('--no-fed', action='store_true',
                        help='Skip federated condition')
    parser.add_argument('--targets', nargs='+', default=None,
                        help='Subset of targets')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Subset of horizons')
    parser.add_argument('--initial-train', type=int, default=INITIAL_TRAIN,
                        help='Initial training window (years)')
    parser.add_argument('--bootstrap-samples', type=int, default=BOOTSTRAP_SAMPLES,
                        help='Number of bootstrap resamples for CIs')
    parser.add_argument('--no-chronos', action='store_true',
                        help='Skip Chronos (faster run without downloading model)')
    args = parser.parse_args()

    targets  = args.targets  if args.targets  else TARGETS
    horizons = args.horizons if args.horizons else HORIZONS
    boot_n   = args.bootstrap_samples

    # Validate
    invalid_targets = [t for t in targets if t not in TARGETS]
    if invalid_targets:
        print(f"Unknown targets: {invalid_targets}. Valid: {TARGETS}")
        sys.exit(1)
    invalid_h = [h for h in horizons if h not in HORIZONS]
    if invalid_h:
        print(f"Unknown horizons: {invalid_h}. Valid: {HORIZONS}")
        sys.exit(1)

    if args.no_chronos:
        METHODS.remove('chronos') if 'chronos' in METHODS else None
        METHOD_LABELS.pop('chronos', None)

    print('=' * 80)
    print('EXTENDED FORECASTING BENCHMARK')
    print('  New methods: BVAR-Minnesota, Chronos-T5-small (zero-shot)')
    print('  Bootstrap CIs: non-parametric, B={}, alpha={:.0%}'.format(boot_n, BOOTSTRAP_ALPHA))
    print(f"  Targets ({len(targets)}): {', '.join(targets)}")
    print(f"  Horizons: {horizons}")
    print(f"  Initial train: {args.initial_train} years")
    print(f"  GPU device: {_DEVICE}")
    print('=' * 80, flush=True)

    # Pre-load Chronos so download message appears at top
    if 'chronos' in METHODS:
        print('\nPre-loading Chronos pipeline ...', flush=True)
        _load_chronos_pipeline()

    print('\nLoading World Bank data ...', flush=True)
    countries = ['KEN', 'TZA', 'UGA']
    country_data = load_countries(countries)
    ken_df = country_data.get('KEN')
    if ken_df is None or ken_df.empty:
        print('ERROR: Kenya data not available.'); sys.exit(1)

    # Ensure targets exist
    missing = [t for t in targets if t not in ken_df.columns]
    if missing:
        print(f"WARNING: Targets missing from Kenya data: {missing}")
        targets = [t for t in targets if t in ken_df.columns]

    aux_dfs = {cc: df for cc, df in country_data.items() if cc != 'KEN'}

    all_records = []
    conditions  = []

    # ── Single condition (Kenya only) ─────────────────────────────────────────
    single_lbl = 'KEN-single'
    print(f'\nRunning single-country condition [{single_lbl}] ...')
    recs = rolling_backtest(
        ken_df, {}, single_lbl, horizons, targets,
        args.initial_train, CONF_THRESHOLD, MIN_EVIDENCE,
    )
    all_records.extend(recs)
    conditions.append((single_lbl, {}))
    print(f"  [{single_lbl}] {len(recs)} records", flush=True)

    # ── Federated condition (KEN+TZA+UGA) ────────────────────────────────────
    if not args.no_fed:
        fed_lbl = 'KEN+TZA+UGA'
        print(f'\nRunning federated condition [{fed_lbl}] ...')
        recs_fed = rolling_backtest(
            ken_df, aux_dfs, fed_lbl, horizons, targets,
            args.initial_train, CONF_THRESHOLD, MIN_EVIDENCE,
        )
        all_records.extend(recs_fed)
        conditions.append((fed_lbl, aux_dfs))
        print(f"  [{fed_lbl}] {len(recs_fed)} records", flush=True)

    n_recs = len(all_records)
    chronos_ok = sum(1 for r in all_records
                     if r['method'] == 'chronos' and not np.isnan(r.get('ae', np.nan)))
    bvar_ok    = sum(1 for r in all_records
                     if r['method'] == 'bvar' and not np.isnan(r.get('ae', np.nan)))
    print(f'\nTotal records: {n_recs}')
    print(f'  BVAR-Minnesota valid AE:  {bvar_ok}')
    print(f'  Chronos valid AE:         {chronos_ok}')

    print('\nComputing bootstrap CIs and printing tables ...', flush=True)
    print_results(all_records, conditions, targets, horizons, boot_n)

    # Save records CSV
    out_dir = _ROOT / 'artifacts' / 'benchmark_extended'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'results.csv'
    pd.DataFrame(all_records).to_csv(csv_path, index=False)
    print(f'\nRecords saved to {csv_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()

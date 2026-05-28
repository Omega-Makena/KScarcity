"""
Scarcity Benchmark v2 — Proper evaluation of the 15 relationship types.

Key differences from v1:
  - Type-aware feature engineering: each of the 15 relationship types maps
    to its correct mathematical transformation (interaction terms, ECM,
    ratio features, compositional sums, etc.) instead of raw lag features.
  - Chronos-T5 zero-shot baseline: the correct null hypothesis.
  - Edge type validation: statistical check of every discovered edge.
  - Typed GNN: message-passing model that consumes edge types as architecture.
  - Comparison table: blind vs lag-graph (v1) vs typed-graph (v2) vs Chronos vs GNN.

Methods:
  persistence       — carry-forward last known value
  arima             — ARIMA(1,1,0)
  chronos           — Chronos-T5 zero-shot (no graph, no training)
  xgb_blind         — XGBoost, all variables as lag features
  xgb_lag           — XGBoost, v1-style graph (parent lag features only)
  xgb_typed         — XGBoost, type-aware graph features  ← KEY NEW METHOD
  lgbm_blind        — LightGBM, all variables as lag features
  lgbm_lag          — LightGBM, v1-style graph (parent lag features only)
  lgbm_typed        — LightGBM, type-aware graph features
  tft               — TFT-lite (attention-based, no graph)
  tgcn              — Typed-edge temporal GNN (requires torch-geometric)

Usage:
  python benchmark/v2/run_benchmark_v2.py --country RWA
  python benchmark/v2/run_benchmark_v2.py --country KEN --pool TZA UGA
  python benchmark/v2/run_benchmark_v2.py --country ETH --no-fed --no-tgcn
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

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
from benchmark.real_data.world_bank_loader import prepare_multi_country_data
from benchmark.v2.feature_translation import (
    select_typed_edges, build_type_aware_matrix, summarise_edge_types,
)
from benchmark.v2.edge_validator import validate_edges, validation_summary, print_validation_table
from benchmark.v2.chronos_wrapper import chronos_forecast, is_available as chronos_available
from benchmark.v2.gnn_model import (
    predict_tgcn, predict_tgcn_all_targets, is_available as pyg_available,
    predict_tgcn_all_targets_torch_only, is_available_torch_only as torch_gnn_available,
)

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
MAX_PARENTS    = 6
MIN_PAIRS      = 4

METHODS = [
    'persistence', 'arima', 'chronos',
    'xgb_blind', 'xgb_lag', 'xgb_typed',
    'lgbm_blind', 'lgbm_lag', 'lgbm_typed',
    'tft', 'tgcn',
    # New hybrid methods
    'persistence_scarcity',   # persistence level + XGB-predicted h-step change
    'chronos_scarcity',       # stacked: Ridge meta-learner on (Chronos, XGB+typed)
    'gnn_scarcity',           # TypedEdgeGNN, full graph, pure PyTorch (no pyg needed)
]

METHOD_LABELS = {
    'persistence':          'Persistence',
    'arima':                'ARIMA(1,1,0)',
    'chronos':              'Chronos-T5',
    'xgb_blind':            'XGB-blind',
    'xgb_lag':              'XGB+lag(v1)',
    'xgb_typed':            'XGB+typed(v2)',
    'lgbm_blind':           'LGBM-blind',
    'lgbm_lag':             'LGBM+lag(v1)',
    'lgbm_typed':           'LGBM+typed(v2)',
    'tft':                  'NF-NHITS/TFT',
    'tgcn':                 'TGCN-typed',
    'persistence_scarcity': 'Persist+Scarcity',
    'chronos_scarcity':     'Chronos+Scarcity',
    'gnn_scarcity':         'GNN+Scarcity',
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_countries(countries: List[str]) -> Dict[str, pd.DataFrame]:
    print(f"  Loading data: {', '.join(countries)} ...", flush=True)
    data = prepare_multi_country_data(countries)
    cleaned = {}
    for cc, df in data.items():
        df = df.ffill().bfill()
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
# Individual predictors
# ─────────────────────────────────────────────────────────────────────────────

def _arima(series: np.ndarray, max_h: int) -> Dict[int, float]:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if len(series) < 4:
                return {}
            m = ARIMA(series, order=(1, 1, 0)).fit()
            fc = m.forecast(steps=max_h)
            return {h: float(fc[h - 1]) for h in range(1, max_h + 1)}
    except Exception:
        return {}


def _prophet(train_df: pd.DataFrame, target: str, horizons: List[int],
             regressors: Optional[List[str]] = None) -> Dict[int, float]:
    try:
        from prophet import Prophet
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_p = pd.DataFrame({
                'ds': pd.to_datetime(train_df.index.astype(str), format='%Y'),
                'y': train_df[target].values,
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
            future_ds = pd.DataFrame({
                'ds': pd.to_datetime([str(last_year + h) for h in horizons], format='%Y')
            })
            for col in reg_cols:
                future_ds[col] = float(train_df[col].iloc[-1])
            fc = m.predict(future_ds)
            return {h: float(row['yhat']) for h, (_, row) in zip(horizons, fc.iterrows())}
    except Exception:
        return {h: np.nan for h in horizons}


def _xgb(train_df: pd.DataFrame, target: str, h: int,
         X: Optional[np.ndarray], y: Optional[np.ndarray],
         X_last: Optional[np.ndarray]) -> float:
    if X is None or len(y) < MIN_PAIRS:
        return np.nan
    try:
        import xgboost as xgb
        mdl = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
        )
        mdl.fit(X, y)
        return float(mdl.predict(X_last.reshape(1, -1))[0])
    except Exception:
        return np.nan


def _lgbm(train_df: pd.DataFrame, target: str, h: int,
          X: Optional[np.ndarray], y: Optional[np.ndarray],
          X_last: Optional[np.ndarray]) -> float:
    if X is None or len(y) < MIN_PAIRS:
        return np.nan
    try:
        import lightgbm as lgb
        mdl = lgb.LGBMRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            num_leaves=7, verbose=-1, random_state=42,
        )
        mdl.fit(X, y)
        return float(mdl.predict(X_last.reshape(1, -1))[0])
    except Exception:
        return np.nan


def _nf_tft_all_h(
    train_df: pd.DataFrame,
    target: str,
    horizons: List[int],
) -> Dict[int, float]:
    """
    NeuralForecast TFT trained on all variables simultaneously as multiple time series.
    Returns predictions for all requested horizons in one GPU-accelerated forward pass.

    Uses TFT (Temporal Fusion Transformer) with small input_size=4 for short annual
    series (10–30 observations).  Falls back to NHITS if TFT fails, then to NaN.
    """
    max_h = max(horizons)
    T = len(train_df)
    input_sz = 4  # TFT minimum stable context for annual macro data

    if T < input_sz + max_h + 2:
        return {h: np.nan for h in horizons}

    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TFT, NHITS

        # Long-format: every column becomes a separate time series
        records = []
        for col in train_df.columns:
            vals = train_df[col].dropna()
            if len(vals) < input_sz + 2:
                continue
            for yr, v in vals.items():
                records.append({
                    'unique_id': col,
                    'ds': pd.Timestamp(f'{yr}-01-01'),
                    'y': float(v),
                })

        if not records:
            return {h: np.nan for h in horizons}

        df_nf = pd.DataFrame(records)
        if target not in df_nf['unique_id'].values:
            return {h: np.nan for h in horizons}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Try TFT first; fall back to NHITS on failure
            for ModelCls, col_name in [(TFT, 'TFT'), (NHITS, 'NHITS')]:
                try:
                    model = ModelCls(
                        h=max_h,
                        input_size=input_sz,
                        max_steps=80,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                    )
                    nf = NeuralForecast(models=[model], freq='YS')
                    nf.fit(df_nf, verbose=False)
                    preds_df = nf.predict()
                    break
                except Exception:
                    preds_df = None
                    col_name = None

        if preds_df is None or col_name is None:
            return {h: np.nan for h in horizons}

        tgt_preds = preds_df[preds_df['unique_id'] == target].sort_values('ds')
        return {
            h: float(tgt_preds.iloc[h - 1][col_name])
            if len(tgt_preds) >= h else np.nan
            for h in horizons
        }

    except Exception as exc:
        warnings.warn(f'NeuralForecast TFT failed: {exc}')
        return {h: np.nan for h in horizons}


def _tft_lite(X: Optional[np.ndarray], y: Optional[np.ndarray],
              X_last: Optional[np.ndarray]) -> float:
    """Lightweight attention model — fallback when NeuralForecast is unavailable."""
    if X is None or len(y) < MIN_PAIRS:
        return np.nan
    try:
        import torch
        import torch.nn as nn

        X_mean = X.mean(axis=0); X_std = X.std(axis=0) + 1e-8
        y_mean = float(y.mean()); y_std = float(y.std()) + 1e-8
        Xt = torch.tensor((X - X_mean) / X_std, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
        n_f = Xt.shape[2]

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                h_dim = max(8, n_f)
                self.proj = nn.Linear(n_f, h_dim)
                self.attn = nn.MultiheadAttention(h_dim, num_heads=1, batch_first=True)
                self.norm = nn.LayerNorm(h_dim)
                self.out = nn.Linear(h_dim, 1)
            def forward(self, x):
                z = torch.relu(self.proj(x))
                a, _ = self.attn(z, z, z)
                return self.out(self.norm(z + a).squeeze(1))

        model = _Attn()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
        for _ in range(60):
            loss = ((model(Xt) - yt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        X_last_t = torch.tensor(
            (X_last - X_mean) / X_std, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return float(model(X_last_t).item() * y_std + y_mean)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Build blind feature matrix (all non-target variables as lag features)
# ─────────────────────────────────────────────────────────────────────────────

def _blind_matrix(train_df: pd.DataFrame, target: str, h: int):
    cols = [c for c in train_df.columns if c != target]
    if not cols or target not in train_df.columns:
        return None, None, None
    n = len(train_df)
    n_pairs = n - h
    if n_pairs < MIN_PAIRS:
        return None, None, None
    X = train_df[cols].values[:n_pairs].astype(float)
    y = train_df[target].values[h:].astype(float)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]
    valid = ~np.isnan(y)
    if valid.sum() < MIN_PAIRS:
        return None, None, None
    X_last = train_df[cols].values[-1].astype(float)
    X_last = np.where(np.isnan(X_last), col_means, X_last)
    return X[valid], y[valid], X_last


# ─────────────────────────────────────────────────────────────────────────────
# Build v1-style lag graph matrix (parent variables as raw lag features)
# ─────────────────────────────────────────────────────────────────────────────

def _lag_graph_matrix(train_df: pd.DataFrame, target: str, h: int,
                      parents: List[str]):
    if not parents:
        return None, None, None
    return _blind_matrix_cols(train_df, target, h, parents)


def _blind_matrix_cols(train_df: pd.DataFrame, target: str, h: int,
                       cols: List[str]):
    cols = [c for c in cols if c in train_df.columns and c != target]
    if not cols:
        return None, None, None
    n = len(train_df)
    n_pairs = n - h
    if n_pairs < MIN_PAIRS:
        return None, None, None
    X = train_df[cols].values[:n_pairs].astype(float)
    y = train_df[target].values[h:].astype(float)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_means[j]
    valid = ~np.isnan(y)
    if valid.sum() < MIN_PAIRS:
        return None, None, None
    X_last = train_df[cols].values[-1].astype(float)
    X_last = np.where(np.isnan(X_last), col_means, X_last)
    return X[valid], y[valid], X_last


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid predictors: Persistence+Scarcity, Chronos+Scarcity
# ─────────────────────────────────────────────────────────────────────────────

def _persistence_scarcity(
    train_df: pd.DataFrame,
    target: str,
    h: int,
    typed_edges: List[dict],
    fallback: float,
) -> float:
    """
    Persistence + Scarcity residual correction.

    Persistence baseline: Y(t+h) = Y(t)  [zero-change assumption]
    Scarcity correction:  learn h-step delta = Y(t+h) − Y(t) from typed graph features

    This decomposition is principled: persistence captures the level signal
    (strong for annual macro series) while the typed graph features capture
    structural change drivers.  If the graph is uninformative the XGBoost
    model will shrink toward 0 delta, leaving persistence unchanged.
    """
    raw = train_df[target].values.astype(float)
    n   = len(train_df)
    n_pairs = n - h

    if n_pairs < MIN_PAIRS:
        return fallback

    last_val = float(raw[-1]) if not np.isnan(raw[-1]) else fallback

    # Build typed feature matrix (reuse existing logic for NaN handling & type transforms)
    X_typed, y_typed, X_last, _ = build_type_aware_matrix(
        train_df, target, typed_edges, h, MIN_PAIRS
    )
    if X_typed is None:
        return last_val

    # Reconstruct h-step changes for the same valid rows build_type_aware_matrix used.
    # valid mask: rows where y[t+h] is not NaN (same filter the matrix applied)
    y_ahead   = raw[h:n_pairs + h]          # y[t+h] for t=0..n_pairs-1
    y_current = raw[:n_pairs]               # y[t]
    valid_mask = ~np.isnan(y_ahead[:len(y_typed)])  # align to matrix row count

    y_change_all = (y_ahead - y_current)[:len(y_typed)]  # h-step deltas, same length
    y_change     = y_change_all[valid_mask]
    X_change     = X_typed[valid_mask]

    # Secondary filter: rows where y_current is also non-NaN (delta is computable)
    current_valid = ~np.isnan(y_current[:len(y_typed)][valid_mask])
    y_change = y_change[current_valid]
    X_change = X_change[current_valid]

    if len(y_change) < MIN_PAIRS:
        return last_val

    try:
        import xgboost as xgb
        mdl = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
        )
        mdl.fit(X_change, y_change)
        delta_pred = float(mdl.predict(X_last.reshape(1, -1))[0])
        return last_val + delta_pred
    except Exception:
        return last_val


def _chronos_scarcity(
    train_df: pd.DataFrame,
    target: str,
    h: int,
    typed_edges: List[dict],
    current_chronos_pred: float,
    current_xgb_typed_pred: float,
    fallback: float,
) -> float:
    """
    Stacked Chronos + Scarcity meta-learner.

    Level-0 forecasters (complementary information sources):
      - Chronos-T5:  pretrained univariate temporal prior (global macro patterns)
      - XGB+typed:   cross-variable structural features from Scarcity graph

    Level-1 meta-learner (Ridge regression, 2 inputs):
      Trained on out-of-sample predictions via temporal train/holdout split:
        - First 60% of valid pairs  → train level-0 models
        - Remaining 40%             → generate OOS predictions for meta-training
        - Fit Ridge on (c_oos, x_oos) → y_oos
      Prediction: Ridge([current_chronos_pred, current_xgb_typed_pred])

    Falls back to equal-weight blend if meta-training fails.
    Returns NaN if Chronos is unavailable (do not fake the result).
    """
    if not chronos_available():
        return np.nan
    if np.isnan(current_chronos_pred):
        return np.nan

    series = train_df[target].values.astype(float)
    n      = len(train_df)

    X_typed, y_typed, X_last, _ = build_type_aware_matrix(
        train_df, target, typed_edges, h, MIN_PAIRS
    )
    if X_typed is None:
        # No typed features: can't do proper stacking, fall back to Chronos alone
        return current_chronos_pred

    n_valid = len(y_typed)
    split   = max(MIN_PAIRS, int(n_valid * 0.6))

    if n_valid - split < 2:
        # Not enough holdout data for meta-training: equal blend
        if np.isnan(current_xgb_typed_pred):
            return current_chronos_pred
        return 0.5 * current_chronos_pred + 0.5 * current_xgb_typed_pred

    # ── Level-0: train XGB on first `split` pairs ────────────────────────────
    try:
        import xgboost as xgb
        xgb_meta = xgb.XGBRegressor(
            n_estimators=30, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
        )
        xgb_meta.fit(X_typed[:split], y_typed[:split])
        xgb_oos = xgb_meta.predict(X_typed[split:]).astype(float)
    except Exception:
        if np.isnan(current_xgb_typed_pred):
            return current_chronos_pred
        return 0.5 * current_chronos_pred + 0.5 * current_xgb_typed_pred

    # ── Level-0: Chronos rolling predictions on holdout pairs ────────────────
    # Reconstruct which time-step each holdout row corresponds to.
    # build_type_aware_matrix filters on ~isnan(y_ahead); since data is imputed
    # the valid positions are approximately 0..n_valid-1 within the pair window.
    # We use the imputed series and step index `split + i` as the Chronos context.
    chronos_oos = np.full(n_valid - split, np.nan)
    for oos_i in range(n_valid - split):
        t_ctx = split + oos_i + 1          # series length at this pseudo-cutoff
        sub_s = series[:min(t_ctx, len(series))]
        sub_s_clean = sub_s[~np.isnan(sub_s)]
        if len(sub_s_clean) < 4:
            continue
        chronos_oos[oos_i] = chronos_forecast(sub_s_clean, [h]).get(h, np.nan)

    # ── Level-1: fit Ridge meta-learner ──────────────────────────────────────
    y_oos   = y_typed[split:]
    valid   = ~(np.isnan(chronos_oos) | np.isnan(xgb_oos) | np.isnan(y_oos))

    if valid.sum() < 2:
        if np.isnan(current_xgb_typed_pred):
            return current_chronos_pred
        return 0.5 * current_chronos_pred + 0.5 * current_xgb_typed_pred

    try:
        from sklearn.linear_model import Ridge
        meta_X = np.column_stack([chronos_oos[valid], xgb_oos[valid]])
        meta   = Ridge(alpha=1.0, fit_intercept=True)
        meta.fit(meta_X, y_oos[valid])

        if np.isnan(current_xgb_typed_pred):
            return current_chronos_pred

        return float(meta.predict(
            np.array([[current_chronos_pred, current_xgb_typed_pred]])
        )[0])
    except Exception:
        if np.isnan(current_xgb_typed_pred):
            return current_chronos_pred
        return 0.5 * current_chronos_pred + 0.5 * current_xgb_typed_pred


# ─────────────────────────────────────────────────────────────────────────────
# Forecast all methods for one (train_df, target, h)
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_one(
    train_df: pd.DataFrame,
    target: str,
    horizons: List[int],
    typed_edges: List[dict],
    all_edges: List[dict],
    var_names: List[str],
    use_chronos: bool,
    use_tgcn: bool,
    nf_cache: Optional[Dict[int, float]] = None,
    tgcn_cache: Optional[Dict[int, float]] = None,
    gnn_scarcity_cache: Optional[Dict[int, float]] = None,
) -> Dict[int, Dict[str, float]]:
    results = {h: {} for h in horizons}
    max_h = max(horizons)

    series = (train_df[target].dropna().values
              if target in train_df.columns else np.array([]))

    # Persistence
    last_val = float(series[-1]) if len(series) > 0 else np.nan
    for h in horizons:
        results[h]['persistence'] = last_val

    # ARIMA
    arima_fc = _arima(series, max_h)
    for h in horizons:
        results[h]['arima'] = arima_fc.get(h, np.nan)

    # Chronos (fit once, all horizons)
    if use_chronos and chronos_available():
        ch_fc = chronos_forecast(series, horizons)
    else:
        ch_fc = {h: np.nan for h in horizons}
    for h in horizons:
        results[h]['chronos'] = ch_fc.get(h, np.nan)

    # v1 lag parents (for comparison)
    lag_parents = list({e['source'] for e in typed_edges})

    # Use pre-computed NF-TFT, TGCN, and GNN+Scarcity caches
    nf_preds          = nf_cache          or {}
    tgcn_preds        = tgcn_cache        or {}
    gnn_scarcity_preds = gnn_scarcity_cache or {}

    # Per-horizon tree models
    arima_fb = {h: arima_fc.get(h, np.nan) for h in horizons}

    for h in horizons:
        # Build matrices
        X_blind, y_blind, Xl_blind = _blind_matrix(train_df, target, h)
        X_lag, y_lag, Xl_lag = _lag_graph_matrix(train_df, target, h, lag_parents)
        X_typed, y_typed, Xl_typed, _ = build_type_aware_matrix(
            train_df, target, typed_edges, h, MIN_PAIRS
        )

        fb = arima_fb[h]

        # XGBoost
        p = _xgb(train_df, target, h, X_blind, y_blind, Xl_blind)
        results[h]['xgb_blind'] = p if not np.isnan(p) else fb
        p = _xgb(train_df, target, h, X_lag, y_lag, Xl_lag)
        results[h]['xgb_lag'] = p if not np.isnan(p) else fb
        p = _xgb(train_df, target, h, X_typed, y_typed, Xl_typed)
        results[h]['xgb_typed'] = p if not np.isnan(p) else fb

        # LightGBM
        p = _lgbm(train_df, target, h, X_blind, y_blind, Xl_blind)
        results[h]['lgbm_blind'] = p if not np.isnan(p) else fb
        p = _lgbm(train_df, target, h, X_lag, y_lag, Xl_lag)
        results[h]['lgbm_lag'] = p if not np.isnan(p) else fb
        p = _lgbm(train_df, target, h, X_typed, y_typed, Xl_typed)
        results[h]['lgbm_typed'] = p if not np.isnan(p) else fb

        # NF-TFT: from cutoff-level cache; fallback to TFT-lite per target
        nf_p = nf_preds.get(h, np.nan)
        if np.isnan(nf_p):
            nf_p = _tft_lite(X_typed, y_typed, Xl_typed)
        results[h]['tft'] = nf_p if not np.isnan(nf_p) else fb

        # TGCN: from cutoff-level cache
        tgcn_p = tgcn_preds.get(h, np.nan)
        results[h]['tgcn'] = tgcn_p if not np.isnan(tgcn_p) else np.nan

        # ── GNN+Scarcity: TypedEdgeGNN full graph, pure PyTorch ──────────────
        gnn_p = gnn_scarcity_preds.get(h, np.nan)
        results[h]['gnn_scarcity'] = gnn_p if not np.isnan(gnn_p) else np.nan

        # ── Persistence+Scarcity: persistence level + XGB-predicted delta ────
        results[h]['persistence_scarcity'] = _persistence_scarcity(
            train_df, target, h, typed_edges, fallback=fb
        )

        # ── Chronos+Scarcity: Ridge meta-learner on (Chronos, XGB+typed) ─────
        ch_pred  = results[h]['chronos']
        xgb_pred = results[h]['xgb_typed']
        results[h]['chronos_scarcity'] = _chronos_scarcity(
            train_df, target, h, typed_edges,
            current_chronos_pred=ch_pred,
            current_xgb_typed_pred=xgb_pred,
            fallback=fb,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rolling backtest
# ─────────────────────────────────────────────────────────────────────────────

def rolling_backtest_v2(
    primary_df: pd.DataFrame,
    aux_dfs: Dict[str, pd.DataFrame],
    label: str,
    horizons: List[int],
    targets: List[str],
    use_chronos: bool = True,
    use_tgcn: bool = False,
) -> List[Dict]:
    var_names = sorted(primary_df.columns.tolist())
    years = sorted(primary_df.index.tolist())
    all_years = set(years)

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    schema = {'fields': [{'name': v} for v in var_names]}
    engine.initialize_v2(schema, use_causal=True)

    print(f"\n  [{label}] Streaming initial {INITIAL_TRAIN} training years ...", flush=True)
    for yr in years[:INITIAL_TRAIN]:
        engine.process_row({k: float(v) for k, v in primary_df.loc[yr].items()
                            if pd.notna(v)})
        for aux in aux_dfs.values():
            if yr in aux.index:
                rd = {k: float(v) for k, v in aux.loc[yr].reindex(var_names).items()
                      if pd.notna(v)}
                if rd:
                    engine.process_row(rd)

    records = []
    validated_edge_sample = []   # collect edges at mid-run for validation report

    for ci in range(INITIAL_TRAIN - 1, len(years)):
        cutoff_yr = years[ci]
        train_data = primary_df[primary_df.index <= cutoff_yr]
        valid_hs = [h for h in horizons if (cutoff_yr + h) in all_years]

        if not valid_hs:
            _advance(engine, primary_df, aux_dfs, years, ci, var_names)
            continue

        _, edges = extract_graph(engine, conf_threshold=CONF_THRESHOLD,
                                 min_evidence=MIN_EVIDENCE)
        n_edges = len(edges)
        print(f"  [{label}] cutoff={cutoff_yr}  N={len(train_data)}  "
              f"edges={n_edges}  valid_h={valid_hs}  "
              f"types={summarise_edge_types(edges)}", flush=True)

        # Collect edge sample for validation (once, at midpoint)
        if ci == INITIAL_TRAIN + (len(years) - INITIAL_TRAIN) // 2:
            validated_edge_sample = edges

        # NF-TFT: train once per cutoff on all variables → cache per (target, h)
        # Avoids retraining the same model 10× (once per target)
        _nf_full: Dict[str, Dict[int, float]] = {}
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import TFT, NHITS
            max_h = max(valid_hs)
            input_sz = 4
            T_cut = len(train_data)
            if T_cut >= input_sz + max_h + 2:
                nf_records = []
                for col in train_data.columns:
                    vals = train_data[col].dropna()
                    if len(vals) >= input_sz + 2:
                        for yr, v in vals.items():
                            nf_records.append({'unique_id': col,
                                               'ds': pd.Timestamp(f'{yr}-01-01'),
                                               'y': float(v)})
                if nf_records:
                    df_nf = pd.DataFrame(nf_records)
                    for ModelCls, col_name in [(TFT, 'TFT'), (NHITS, 'NHITS')]:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                model = ModelCls(h=max_h, input_size=input_sz,
                                                 max_steps=80,
                                                 enable_progress_bar=False,
                                                 enable_model_summary=False)
                                nf = NeuralForecast(models=[model], freq='YS')
                                nf.fit(df_nf, verbose=False)
                                preds_df = nf.predict()
                            for uid in preds_df['unique_id'].unique():
                                row = preds_df[preds_df['unique_id'] == uid].sort_values('ds')
                                _nf_full[uid] = {h: float(row.iloc[h-1][col_name])
                                                 if len(row) >= h else np.nan
                                                 for h in valid_hs}
                            break
                        except Exception:
                            pass
        except Exception:
            pass

        # TGCN: train once per (cutoff × h) — requires torch-geometric
        tgcn_cutoff_cache: Dict[int, Dict[str, float]] = {}
        if use_tgcn and pyg_available() and edges:
            for h in valid_hs:
                tgcn_cutoff_cache[h] = predict_tgcn_all_targets(
                    train_data, targets, h, edges, var_names,
                    hidden=16, epochs=15,
                )

        # GNN+Scarcity: TypedEdgeGNN, full graph, pure PyTorch (no pyg needed)
        # Uses all edges (not per-target), hidden=32, epochs=30 for more capacity.
        gnn_scarcity_cache: Dict[int, Dict[str, float]] = {}
        if torch_gnn_available() and edges:
            for h in valid_hs:
                gnn_scarcity_cache[h] = predict_tgcn_all_targets_torch_only(
                    train_data, targets, h, edges, var_names,
                    hidden=32, epochs=30,
                )

        for target in targets:
            if target not in primary_df.columns:
                continue

            typed_edges = select_typed_edges(edges, target, max_parents=MAX_PARENTS)
            nf_cache = _nf_full.get(target, {})
            tgcn_cache = {h: tgcn_cutoff_cache.get(h, {}).get(target, np.nan)
                          for h in valid_hs}
            gnn_sc_cache = {h: gnn_scarcity_cache.get(h, {}).get(target, np.nan)
                            for h in valid_hs}
            preds = _forecast_one(
                train_data, target, valid_hs, typed_edges, edges, var_names,
                use_chronos=use_chronos, use_tgcn=use_tgcn,
                nf_cache=nf_cache, tgcn_cache=tgcn_cache,
                gnn_scarcity_cache=gnn_sc_cache,
            )

            for h in valid_hs:
                pred_yr = cutoff_yr + h
                if pred_yr not in all_years:
                    continue
                actual = primary_df.loc[pred_yr, target] if pred_yr in all_years else np.nan
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

        _advance(engine, primary_df, aux_dfs, years, ci, var_names)

    return records, validated_edge_sample


def _advance(engine, primary_df, aux_dfs, years, ci, var_names):
    if ci + 1 >= len(years):
        return
    yr = years[ci + 1]
    engine.process_row({k: float(v) for k, v in primary_df.loc[yr].items()
                        if pd.notna(v)})
    for aux in aux_dfs.values():
        if yr in aux.index:
            rd = {k: float(v) for k, v in aux.loc[yr].reindex(var_names).items()
                  if pd.notna(v)}
            if rd:
                engine.process_row(rd)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _mae(records, label, target, h, method):
    vals = [r['ae'] for r in records
            if r['label'] == label and r['target'] == target
            and r['h'] == h and r['method'] == method
            and r['ae'] is not None and not np.isnan(r['ae'])]
    return round(float(np.mean(vals)), 4) if vals else np.nan


def _mean_across(records, label, h, method, targets):
    vals = [_mae(records, label, t, h, method) for t in targets]
    vals = [v for v in vals if not np.isnan(v)]
    return round(float(np.mean(vals)), 4) if vals else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

def print_results_v2(records, label, targets, horizons):
    W = 110

    # ── Table 1: Aggregate MAE ────────────────────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 1 — AGGREGATE MAE  (mean across all targets; lower = better)')
    print(f"  Condition: {label}")
    print('=' * W)
    print(f"  {'Method':<22}" + ''.join(f'  h={h:2d}     ' for h in horizons))
    print('  ' + '─' * (22 + 12 * len(horizons)))
    for method in METHODS:
        row = f"  {METHOD_LABELS[method]:<22}"
        for h in horizons:
            v = _mean_across(records, label, h, method, targets)
            row += f'  {v:8.4f}' if not np.isnan(v) else '       N/A'
        print(row)

    # ── Table 2: v1 lag vs v2 typed (THE KEY COMPARISON) ─────────────────────
    print('\n' + '=' * W)
    print('TABLE 2 — LAG (v1) vs TYPED (v2)  (negative delta = typed helps)')
    print(f"  [typed_MAE − lag_MAE] per target and horizon")
    print('=' * W)
    for model_base in ('xgb', 'lgbm'):
        lag_k = f'{model_base}_lag'
        typed_k = f'{model_base}_typed'
        print(f"\n  {model_base.upper()}")
        print(f"  {'Target':<22}" + ''.join(f'  h={h:2d}  ' for h in horizons))
        print('  ' + '─' * (22 + 8 * len(horizons)))
        for target in targets:
            row = f"  {target:<22}"
            for h in horizons:
                lag_v = _mae(records, label, target, h, lag_k)
                typ_v = _mae(records, label, target, h, typed_k)
                if np.isnan(lag_v) or np.isnan(typ_v):
                    row += '     N/A'
                else:
                    d = typ_v - lag_v
                    tag = '<' if d < -0.05 else ('>' if d > 0.05 else '~')
                    row += f'  {d:+5.2f}{tag}'
            print(row)
        # Aggregate
        agg = f"  {'[aggregate]':<22}"
        for h in horizons:
            deltas = []
            for t in targets:
                l = _mae(records, label, t, h, lag_k)
                v = _mae(records, label, t, h, typed_k)
                if not (np.isnan(l) or np.isnan(v)):
                    deltas.append(v - l)
            if deltas:
                d = float(np.mean(deltas))
                tag = '<' if d < -0.05 else ('>' if d > 0.05 else '~')
                agg += f'  {d:+5.2f}{tag}'
            else:
                agg += '     N/A'
        print(agg)

    # ── Table 3: Chronos vs best Scarcity ─────────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 3 — CHRONOS vs SCARCITY  (negative = Chronos beats Scarcity)')
    print(f"  [chronos_MAE − xgb_typed_MAE] per target")
    print('=' * W)
    print(f"  {'Target':<22}" + ''.join(f'  h={h:2d}  ' for h in horizons))
    print('  ' + '─' * (22 + 8 * len(horizons)))
    for target in targets:
        row = f"  {target:<22}"
        for h in horizons:
            ch = _mae(records, label, target, h, 'chronos')
            sc = _mae(records, label, target, h, 'xgb_typed')
            if np.isnan(ch) or np.isnan(sc):
                row += '     N/A'
            else:
                d = ch - sc
                tag = 'C>' if d < -0.05 else ('S>' if d > 0.05 else ' ~')
                row += f'  {d:+5.2f}{tag}'
        print(row)

    # ── Table 4: New hybrid methods vs their baselines ────────────────────────
    print('\n' + '=' * W)
    print('TABLE 4 — HYBRID METHODS vs BASELINES')
    print(f"  Negative delta = hybrid beats baseline. N/A = method unavailable.")
    print('=' * W)

    hybrid_comparisons = [
        ('persistence_scarcity', 'persistence', 'Persist+Scarcity vs Persistence'),
        ('chronos_scarcity',     'chronos',     'Chronos+Scarcity  vs Chronos'),
        ('gnn_scarcity',         'xgb_typed',   'GNN+Scarcity      vs XGB+typed'),
    ]

    for hybrid_k, base_k, title in hybrid_comparisons:
        print(f"\n  {title}  [hybrid_MAE − base_MAE]")
        print(f"  {'Target':<22}" + ''.join(f'  h={h:2d}  ' for h in horizons))
        print('  ' + '─' * (22 + 8 * len(horizons)))
        for target in targets:
            row = f"  {target:<22}"
            for h in horizons:
                h_v = _mae(records, label, target, h, hybrid_k)
                b_v = _mae(records, label, target, h, base_k)
                if np.isnan(h_v) or np.isnan(b_v):
                    row += '     N/A'
                else:
                    d = h_v - b_v
                    tag = '<' if d < -0.05 else ('>' if d > 0.05 else '~')
                    row += f'  {d:+5.2f}{tag}'
            print(row)
        # Aggregate row
        agg_row = f"  {'[aggregate]':<22}"
        for h in horizons:
            deltas = []
            for t in targets:
                h_v = _mae(records, label, t, h, hybrid_k)
                b_v = _mae(records, label, t, h, base_k)
                if not (np.isnan(h_v) or np.isnan(b_v)):
                    deltas.append(h_v - b_v)
            if deltas:
                d = float(np.mean(deltas))
                tag = '<' if d < -0.05 else ('>' if d > 0.05 else '~')
                agg_row += f'  {d:+5.2f}{tag}'
            else:
                agg_row += '     N/A'
        print(agg_row)

    # ── Table 5: Best method per target ───────────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 5 — BEST METHOD PER TARGET AND HORIZON')
    print('=' * W)
    print(f"  {'Target':<22}" + ''.join(f'  h={h}' for h in horizons))
    print('  ' + '─' * (22 + 10 * len(horizons)))
    for target in targets:
        row = f"  {target:<22}"
        for h in horizons:
            best_mae, best_m = np.inf, '—'
            for method in METHODS:
                v = _mae(records, label, target, h, method)
                if not np.isnan(v) and v < best_mae:
                    best_mae, best_m = v, method
            short = METHOD_LABELS.get(best_m, best_m)[:12] if best_m != '—' else '—'
            row += f'  {short:<12}({best_mae:.2f})' if best_m != '—' else '  N/A'
        print(row)

    print('\n' + '=' * W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', default='KEN', help='Primary country ISO3')
    parser.add_argument('--pool', nargs='+', default=['TZA', 'UGA'],
                        help='Federation pool countries')
    parser.add_argument('--no-fed', action='store_true')
    parser.add_argument('--no-chronos', action='store_true')
    parser.add_argument('--no-tgcn', action='store_true')
    parser.add_argument('--horizons', nargs='+', type=int, default=None)
    parser.add_argument('--targets', nargs='+', default=None)
    args = parser.parse_args()

    hors = sorted(args.horizons) if args.horizons else HORIZONS
    use_chronos = not args.no_chronos
    use_tgcn = not args.no_tgcn and pyg_available()

    print('=' * 80)
    print(f'SCARCITY BENCHMARK v2 — {args.country}')
    print(f'  Pool:     {args.pool}')
    print(f'  Horizons: {hors}')
    print(f'  Chronos:  {"YES" if use_chronos and chronos_available() else "NO (unavailable)"}')
    print(f'  TGCN:     {"YES" if use_tgcn else "NO (unavailable)"}')
    print(f'  GNN+Scarcity (torch-only): {"YES" if torch_gnn_available() else "NO (torch not installed)"}')
    print(f'  Persist+Scarcity: YES (always enabled)')
    print(f'  Chronos+Scarcity: {"YES (stacked meta-learner)" if use_chronos and chronos_available() else "NO (requires Chronos)"}')
    print('=' * 80)

    # Load data
    all_countries = [args.country] + (args.pool if not args.no_fed else [])
    data = load_countries(all_countries)

    primary_df = data.get(args.country)
    if primary_df is None:
        print(f'ERROR: {args.country} data not loaded'); sys.exit(1)

    aux_dfs = ({cc: df for cc, df in data.items() if cc != args.country}
               if not args.no_fed else {})

    tgts = [t for t in (args.targets or TARGETS) if t in primary_df.columns]
    print(f"  Active targets ({len(tgts)}): {tgts}")

    pool_str = '+'.join(sorted(aux_dfs.keys())) if aux_dfs else 'none'
    label = f'{args.country}-fed({pool_str})' if aux_dfs else f'{args.country}-single'

    # Run backtest
    records, edge_sample = rolling_backtest_v2(
        primary_df, aux_dfs, label, hors, tgts,
        use_chronos=use_chronos, use_tgcn=use_tgcn,
    )

    # Print results
    print_results_v2(records, label, tgts, hors)

    # Edge type validation
    if edge_sample:
        print('\n' + '=' * 80)
        print('EDGE TYPE VALIDATION (mid-run snapshot)')
        print('=' * 80)
        validated = validate_edges(edge_sample, primary_df)
        print(validation_summary(validated))
        print()
        print_validation_table(validated, top_n=20)

    # Save results
    out_dir = _ROOT / 'artifacts' / 'benchmark_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_tag = '+'.join(sorted(aux_dfs.keys())) if aux_dfs else 'none'
    stem = f'v2_{args.country}_pool_{pool_tag}'

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_dir / f'{stem}.csv', index=False)

    # Summary JSON
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for h in hors:
            v = _mean_across(records, label, h, method, tgts)
            summary[method][f'h{h}'] = v if not np.isnan(v) else None
    with open(out_dir / f'{stem}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {out_dir / f'{stem}.csv'}")
    print(f"  Summary saved to {out_dir / f'{stem}_summary.json'}")


if __name__ == '__main__':
    main()

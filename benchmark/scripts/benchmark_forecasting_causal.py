"""
Multi-Target Multi-Horizon Forecasting with Causal Identification

Extends benchmark_forecasting_horizons.py with a causal identification layer:
for each (target, candidate_parent) pair from Scarcity's discovery engine, run
DoWhy's production causal pipeline to validate genuine causal effects before
using parents as forecasting features.

Estimands applied per candidate parent:
  ATE           Always (N >= CAUSAL_MIN_N=15) — backdoor.linear_regression
  ATT           Always — target_units="att"
  ATC           Always — target_units="atc"
  CATE          N >= CATE_MIN_N=25 — EconML CausalForestDML
  LATE          When instrument found in discovered graph (IV)
  MEDIATION_NDE When mediator found in discovered graph (Natural Direct Effect)
  MEDIATION_NIE When mediator found in discovered graph (Natural Indirect Effect)

Causal identification rule:
  support = (# significant estimands) / (# applicable estimands)
  Parent is "causally identified" if support >= CAUSAL_VOTE_THRESHOLD (0.5)
  Significance: CI excludes zero  OR  |estimate| > EFFECT_THRESHOLD (0.5)
  Fallback: if N < CAUSAL_MIN_N or no parents survive, use graph parents unchanged

New methods (12 total, vs 9 in horizons benchmark):
  prophet_causal — Prophet conditioned on causally-validated regressors
  xgb_causal     — XGBoost with causally-validated parents
  lgbm_causal    — LightGBM with causally-validated parents

Research questions:
  1. Does causal identification improve over raw graph-conditioned forecasting?
  2. Which estimands agree most often across targets?
  3. Which targets have the most causally-validated vs spurious parents?
  4. Does causal filtering hurt at long horizons (fewer N for identification)?
  5. Does LATE/MEDIATION add signal beyond ATE/ATT/ATC majority vote?

Expected runtime: 35-70 min (full run, all conditions)
  Quick test: --no-fed --targets gdp_growth inflation_cpi --horizons 1 3

Usage:
    python benchmark/scripts/benchmark_forecasting_causal.py
    python benchmark/scripts/benchmark_forecasting_causal.py --no-fed
    python benchmark/scripts/benchmark_forecasting_causal.py --fast
    python benchmark/scripts/benchmark_forecasting_causal.py --targets gdp_growth inflation_cpi
    python benchmark/scripts/benchmark_forecasting_causal.py --horizons 1 3
"""

import argparse
import concurrent.futures
import io
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import logging

def _silence(name):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.propagate = False
    for h in lg.handlers[:]:
        lg.removeHandler(h)

_silence('prophet')
_silence('cmdstanpy')
_silence('scarcity.causal')
_silence('dowhy')
_silence('econml')

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

# Causal identification parameters
CAUSAL_MIN_N         = 15    # minimum N for any DoWhy estimation
CATE_MIN_N           = 25    # minimum N for EconML CausalForestDML
CAUSAL_VOTE_THRESHOLD = 0.5  # fraction of significant estimands needed
EFFECT_THRESHOLD     = 0.5   # fallback: |estimate| > this when CI unavailable
MAX_CONFOUNDERS      = 3     # confounders per spec (controls overfit at small N)
CAUSAL_ARTIFACT_ROOT = 'artifacts/causal_benchmark'

# Parallelism
N_CAUSAL_WORKERS  = min(4, os.cpu_count() or 1)   # parallel specs inside run_causal
N_TARGET_WORKERS  = min(4, os.cpu_count() or 1)   # parallel targets within a cutoff

# GPU — detect once at import time; tree models & TFT will use it when available
try:
    import torch as _torch
    _DEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'
except ImportError:
    _DEVICE = 'cpu'

METHODS = [
    'persistence', 'arima',
    'prophet', 'prophet_graph', 'prophet_causal',
    'xgb_blind', 'xgb_graph', 'xgb_causal',
    'lgbm_blind', 'lgbm_graph', 'lgbm_causal',
    'tft',
]

METHOD_LABELS = {
    'persistence':    'Persistence',
    'arima':          'ARIMA(1,1,0)',
    'prophet':        'Prophet',
    'prophet_graph':  'Prophet+Graph',
    'prophet_causal': 'Prophet+Causal',
    'xgb_blind':      'XGBoost blind',
    'xgb_graph':      'XGBoost+Graph',
    'xgb_causal':     'XGBoost+Causal',
    'lgbm_blind':     'LightGBM blind',
    'lgbm_graph':     'LightGBM+Graph',
    'lgbm_causal':    'LightGBM+Causal',
    'tft':            'TFT-lite',
}

GRAPH_METHODS  = {'prophet_graph', 'xgb_graph', 'lgbm_graph'}
CAUSAL_METHODS = {'prophet_causal', 'xgb_causal', 'lgbm_causal'}
HORIZON_GROUPS = [('short', [1, 3]), ('long', [5, 10])]

ALL_ESTIMAND_NAMES = ['ATE', 'ATT', 'ATC', 'CATE', 'LATE', 'MEDIATION_NDE', 'MEDIATION_NIE']

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
# Graph helper (type-diverse top-K, same as horizons benchmark)
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
# Causal identification — graph structure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_edge_index(edges):
    """Build directed adjacency sets from edge list."""
    parents_of  = defaultdict(set)  # parents_of[X]  = {nodes with edge -> X}
    children_of = defaultdict(set)  # children_of[X] = {nodes X has edge -> }
    for e in edges:
        src = e['source']
        tgt = e['target']
        parents_of[tgt].add(src)
        children_of[src].add(tgt)
    return parents_of, children_of


def _find_instrument(treatment, outcome, parents_of):
    """Find Z: Z -> treatment in graph AND Z not -> outcome directly."""
    treatment_parents = parents_of.get(treatment, set())
    outcome_parents   = parents_of.get(outcome, set())
    candidates = treatment_parents - outcome_parents - {outcome, treatment}
    return next(iter(sorted(candidates)), None)


def _find_mediator(treatment, outcome, children_of, parents_of):
    """Find M: treatment -> M in graph AND M -> outcome in graph."""
    treatment_children = children_of.get(treatment, set())
    outcome_parents    = parents_of.get(outcome, set())
    candidates = (treatment_children & outcome_parents) - {treatment, outcome}
    return next(iter(sorted(candidates)), None)


def _is_significant(artifact):
    """Return True if the causal estimate is significantly non-zero."""
    est = artifact.estimate
    ci  = artifact.confidence_intervals

    # CATE/ITE can return per-obs arrays — take mean
    if isinstance(est, list):
        est = float(np.mean([x for x in est if x is not None and not np.isnan(x)] or [0.0]))
    try:
        est = float(est)
    except (TypeError, ValueError):
        return False
    if np.isnan(est):
        return False

    if ci is not None:
        try:
            lower, upper = ci
            if isinstance(lower, list):
                lower = float(np.mean(lower))
            if isinstance(upper, list):
                upper = float(np.mean(upper))
            lower, upper = float(lower), float(upper)
            if not (np.isnan(lower) or np.isnan(upper)):
                return lower > 0 or upper < 0   # CI excludes zero
        except Exception:
            pass

    return abs(est) > EFFECT_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Causal identification — build specs and run
# ─────────────────────────────────────────────────────────────────────────────

def _build_causal_specs(train_df, target, candidate_parents, parents_of, children_of, N,
                        fast_mode=False):
    """
    Build a flat list of (EstimandSpec, parent_str, estimand_name) triples.
    Applicable estimands depend on N and graph structure.
    """
    from scarcity.causal.specs import EstimandSpec, EstimandType

    triples = []
    other_cols = [c for c in train_df.columns
                  if c != target and c not in candidate_parents]

    for parent in candidate_parents:
        if parent not in train_df.columns:
            continue
        confounders = [c for c in other_cols if c != parent][:MAX_CONFOUNDERS]

        base_types = [EstimandType.ATE]
        if not fast_mode:
            base_types = [EstimandType.ATE, EstimandType.ATT, EstimandType.ATC]

        for etype in base_types:
            triples.append((
                EstimandSpec(treatment=parent, outcome=target,
                             confounders=confounders, type=etype),
                parent, etype.value,
            ))

        # CATE — needs enough data for forest
        if N >= CATE_MIN_N and not fast_mode:
            effect_mods = confounders[:2]
            triples.append((
                EstimandSpec(treatment=parent, outcome=target,
                             confounders=confounders,
                             effect_modifiers=effect_mods,
                             type=EstimandType.CATE),
                parent, 'CATE',
            ))

        # LATE — IV if instrument exists in graph and in data
        instrument = _find_instrument(parent, target, parents_of)
        if instrument and instrument in train_df.columns and not fast_mode:
            triples.append((
                EstimandSpec(treatment=parent, outcome=target,
                             confounders=confounders,
                             instrument=instrument,
                             type=EstimandType.LATE),
                parent, 'LATE',
            ))

        # MEDIATION — NDE + NIE if mediator exists
        mediator = _find_mediator(parent, target, children_of, parents_of)
        if mediator and mediator in train_df.columns and not fast_mode:
            for etype in [EstimandType.MEDIATION_NDE, EstimandType.MEDIATION_NIE]:
                triples.append((
                    EstimandSpec(treatment=parent, outcome=target,
                                 confounders=confounders,
                                 mediator=mediator,
                                 type=etype),
                    parent, etype.value,
                ))

    return triples


def _identify_causal_parents(train_df, target, candidate_parents, all_edges, N,
                              cutoff_yr, fast_mode=False):
    """
    Run causal engine on all candidate parents for a single target.
    Returns:
      causal_parents: list of parents with causal support >= threshold
      agreement_table: {parent: {estimand_name: 'sig'|'ns'|'err'}}
    """
    if N < CAUSAL_MIN_N or not candidate_parents:
        return list(candidate_parents), {}

    try:
        from scarcity.causal.engine import run_causal
        from scarcity.causal.specs import RuntimeSpec, ParallelismMode, FailPolicy
    except ImportError:
        return list(candidate_parents), {}

    parents_of, children_of = _build_edge_index(all_edges)

    # Filter to parents that exist in the data
    valid_parents = [p for p in candidate_parents if p in train_df.columns]
    if not valid_parents:
        return list(candidate_parents), {}

    triples = _build_causal_specs(
        train_df, target, valid_parents, parents_of, children_of, N, fast_mode=fast_mode
    )
    if not triples:
        return list(candidate_parents), {}

    specs    = [t[0] for t in triples]
    meta     = [(t[1], t[2]) for t in triples]  # (parent, estimand_name)

    # Keep causal engine serial (NONE) — outer ThreadPoolExecutor over targets
    # already provides parallelism. Nested ProcessPoolExecutor + ThreadPoolExecutor
    # crashes on Windows with spawn context.
    runtime = RuntimeSpec(
        refute_random_common_cause=False,
        refute_placebo_treatment=False,
        refute_data_subset=False,
        refutation_simulations=0,
        parallelism=ParallelismMode.NONE,
        n_jobs=1,
        fail_policy=FailPolicy.CONTINUE,
        export_graphs=False,
        artifact_root=CAUSAL_ARTIFACT_ROOT,
        run_id=f'bench_{cutoff_yr}_{target}',
    )

    try:
        df_causal = train_df.reset_index(drop=True).copy()
        result = run_causal(df_causal, specs, runtime)
    except Exception as exc:
        print(f"    [causal] run_causal exception: {exc}", flush=True)
        return list(candidate_parents), {}

    # Map spec index -> artifact
    artifact_map = {a.index: a for a in result.results}
    error_set    = {e.index for e in result.errors}

    parent_votes    = {p: {'total': 0, 'sig': 0} for p in valid_parents}
    agreement_table = {p: {} for p in valid_parents}

    for i, (parent, ename) in enumerate(meta):
        if i in artifact_map:
            sig = _is_significant(artifact_map[i])
            parent_votes[parent]['total'] += 1
            if sig:
                parent_votes[parent]['sig'] += 1
            agreement_table[parent][ename] = 'sig' if sig else 'ns'
        elif i in error_set:
            agreement_table[parent][ename] = 'err'
        else:
            agreement_table[parent][ename] = 'na'

    causal_parents = []
    for parent in valid_parents:
        v = parent_votes[parent]
        if v['total'] == 0:
            causal_parents.append(parent)
        elif v['sig'] / v['total'] >= CAUSAL_VOTE_THRESHOLD:
            causal_parents.append(parent)

    # Fallback: if all parents are filtered out, use original set
    if not causal_parents:
        return list(candidate_parents), agreement_table

    return causal_parents, agreement_table


# ─────────────────────────────────────────────────────────────────────────────
# Individual predictors (identical to horizons benchmark)
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
            last_year   = int(train_df.index[-1])
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


def _predict_xgb(train_df, target, h, feature_cols, parents=None):
    try:
        import xgboost as xgb
        cols = parents if parents else feature_cols
        X, y, X_last, used = _build_direct_pairs(train_df, target, cols, h)
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


def _predict_lgbm(train_df, target, h, feature_cols, parents=None):
    try:
        import lightgbm as lgb
        cols = parents if parents else feature_cols
        X, y, X_last, used = _build_direct_pairs(train_df, target, cols, h)
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
        cols = [c for c in feature_cols if c in train_df.columns and c != target]
        X, y, X_last, used = _build_direct_pairs(train_df, target, cols, h)
        if X is None:
            return np.nan

        X_mean = X.mean(axis=0);  X_std = X.std(axis=0) + 1e-8
        y_mean = float(y.mean()); y_std = float(y.std()) + 1e-8
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
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
        for _ in range(50):
            loss = ((model(Xt) - yt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        X_last_t = torch.tensor(
            ((X_last - X_mean) / X_std), dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(X_last_t).item()
        return float(raw * y_std + y_mean)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Core: run all methods for one (train_df, target) across all horizons
# ─────────────────────────────────────────────────────────────────────────────

def _forecast_all(train_df, target, horizons, feature_cols, graph_parents, causal_parents):
    """
    Returns dict: {h: {method: predicted_scalar}}.
    Includes both graph-conditioned and causally-validated methods.
    """
    results = {h: {} for h in horizons}
    max_h   = max(horizons)
    series  = train_df[target].dropna().values if target in train_df.columns else np.array([])

    last_val = float(series[-1]) if len(series) > 0 else np.nan
    for h in horizons:
        results[h]['persistence'] = last_val

    arima_fc = _arima_multi(series, max_h)
    for h in horizons:
        results[h]['arima'] = arima_fc.get(h, np.nan)

    prophet_fc       = _prophet_multi(train_df, target, horizons)
    prophet_graph_fc = _prophet_multi(train_df, target, horizons,
                                      regressors=graph_parents if graph_parents else None)
    prophet_causal_fc = _prophet_multi(train_df, target, horizons,
                                       regressors=causal_parents if causal_parents else None)
    for h in horizons:
        results[h]['prophet'] = prophet_fc.get(h, np.nan)
        pg = prophet_graph_fc.get(h, np.nan)
        results[h]['prophet_graph'] = pg if not np.isnan(pg) else results[h]['prophet']
        pc = prophet_causal_fc.get(h, np.nan)
        results[h]['prophet_causal'] = pc if not np.isnan(pc) else results[h]['prophet']

    arima_fallback = {h: arima_fc.get(h, np.nan) for h in horizons}

    for h in horizons:
        p = _predict_xgb(train_df, target, h, feature_cols)
        results[h]['xgb_blind'] = p if not np.isnan(p) else arima_fallback[h]

        p = (_predict_xgb(train_df, target, h, feature_cols, parents=graph_parents)
             if graph_parents else results[h]['xgb_blind'])
        results[h]['xgb_graph'] = p if not np.isnan(p) else arima_fallback[h]

        p = (_predict_xgb(train_df, target, h, feature_cols, parents=causal_parents)
             if causal_parents else results[h]['xgb_graph'])
        results[h]['xgb_causal'] = p if not np.isnan(p) else arima_fallback[h]

        p = _predict_lgbm(train_df, target, h, feature_cols)
        results[h]['lgbm_blind'] = p if not np.isnan(p) else arima_fallback[h]

        p = (_predict_lgbm(train_df, target, h, feature_cols, parents=graph_parents)
             if graph_parents else results[h]['lgbm_blind'])
        results[h]['lgbm_graph'] = p if not np.isnan(p) else arima_fallback[h]

        p = (_predict_lgbm(train_df, target, h, feature_cols, parents=causal_parents)
             if causal_parents else results[h]['lgbm_graph'])
        results[h]['lgbm_causal'] = p if not np.isnan(p) else arima_fallback[h]

        p = _predict_tft(train_df, target, h, feature_cols)
        results[h]['tft'] = p if not np.isnan(p) else arima_fallback[h]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-target worker (runs in thread, safe because GIL released in subprocesses)
# ─────────────────────────────────────────────────────────────────────────────

def _process_target(target, train_data, feature_cols, graph_parents, edges,
                    N, valid_hs, all_years, cutoff_yr, ken_df, fast_mode):
    """
    Causal identification + forecasting for one (cutoff, target) pair.
    Returns (local_records, agree_tbl, n_graph, n_causal).
    """
    if target not in ken_df.columns:
        return [], {}, 0, 0

    if N >= CAUSAL_MIN_N:
        causal_parents, agree_tbl = _identify_causal_parents(
            train_data, target, graph_parents, edges, N,
            cutoff_yr=cutoff_yr, fast_mode=fast_mode,
        )
        n_graph  = len(graph_parents)
        n_causal = len(causal_parents)
        if n_graph > 0:
            pct = 100 * n_causal / n_graph
            print(f"    {target}: graph={n_graph} causal={n_causal} ({pct:.0f}% retained)",
                  flush=True)
    else:
        causal_parents = graph_parents
        agree_tbl = {}
        n_graph = n_causal = len(graph_parents)

    preds = _forecast_all(
        train_data, target, valid_hs, feature_cols, graph_parents, causal_parents
    )

    local_records = []
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
            local_records.append({
                'label': None,  # filled in by caller
                'cutoff': cutoff_yr, 'h': h,
                'target': target, 'pred_yr': pred_yr,
                'method': method, 'actual': actual, 'ae': ae,
                'n_graph': n_graph, 'n_causal': n_causal,
            })

    return local_records, agree_tbl, n_graph, n_causal


# ─────────────────────────────────────────────────────────────────────────────
# Rolling backtest with causal identification
# ─────────────────────────────────────────────────────────────────────────────

def rolling_backtest(ken_df, aux_dfs, label, horizons, targets, initial_train,
                     conf_threshold, min_evidence, fast_mode=False):
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

    records       = []
    # Global estimand agreement tracking: {target: {estimand: [sig, total]}}
    global_agreement = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # Parent retention rates: {target: [n_causal, n_graph]}
    retention_stats  = defaultdict(lambda: [0, 0])

    for ci in range(initial_train - 1, len(years)):
        cutoff_yr  = years[ci]
        train_data = ken_df[ken_df.index <= cutoff_yr]
        N          = len(train_data)

        valid_hs = [h for h in horizons if (cutoff_yr + h) in all_years]
        if not valid_hs:
            _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names)
            continue

        graph, edges  = extract_graph(engine, conf_threshold=conf_threshold,
                                       min_evidence=min_evidence)
        graph_topk    = _top_k_graph(graph, edges, max_parents=MAX_PARENTS)
        n_edges       = sum(len(v) for v in graph.values())

        run_causal_flag = N >= CAUSAL_MIN_N
        print(f"  [{label}] cutoff={cutoff_yr}  N={N}  edges={n_edges}  "
              f"valid_h={valid_hs}  causal={'yes' if run_causal_flag else 'skip(N<15)'}",
              flush=True)

        # Process targets in parallel threads (GIL released during subprocess causal calls)
        futures_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_TARGET_WORKERS) as ex:
            for target in targets:
                gp = graph_topk.get(target, [])
                fut = ex.submit(
                    _process_target,
                    target, train_data, feature_cols, gp, edges,
                    N if run_causal_flag else 0,   # pass 0 to skip causal when N<15
                    valid_hs, all_years, cutoff_yr, ken_df, fast_mode,
                )
                futures_map[fut] = target

            for fut in concurrent.futures.as_completed(futures_map):
                target = futures_map[fut]
                try:
                    local_recs, agree_tbl, n_graph, n_causal = fut.result()
                except Exception as exc:
                    print(f"    [warn] {target} failed: {exc}", flush=True)
                    continue

                for r in local_recs:
                    r['label'] = label
                records.extend(local_recs)

                for parent, ename_map in agree_tbl.items():
                    for ename, status in ename_map.items():
                        if status in ('sig', 'ns'):
                            global_agreement[target][ename][1] += 1
                            if status == 'sig':
                                global_agreement[target][ename][0] += 1
                retention_stats[target][0] += n_causal
                retention_stats[target][1] += n_graph

        _advance_engine(engine, ken_df, aux_dfs, years, ci, var_names)

    return records, engine, dict(global_agreement), dict(retention_stats)


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


def _group_mean(records, label, method, hs, targets):
    vals = []
    for h in hs:
        v = _mean_across_targets(records, label, h, method, targets)
        if not np.isnan(v):
            vals.append(v)
    return round(float(np.mean(vals)), 4) if vals else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

def print_results(all_records, conditions, targets, horizons,
                  global_agreement, retention_stats):
    W = 120
    cond_labels = [lbl for lbl, _ in conditions]
    single_lbl  = cond_labels[0]
    fed_lbl     = cond_labels[1] if len(cond_labels) > 1 else None
    short_h     = [h for h in horizons if h <= 3]
    long_h      = [h for h in horizons if h > 3]

    # ── Table 1: Aggregate MAE by (method, h) ────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 1 — AGGREGATE MAE  (mean across all targets; lower = better)')
    print(f"  Condition: {single_lbl}" + (f" | {fed_lbl}" if fed_lbl else ''))
    print('=' * W)

    h_cols = [f'  h={h:2d}' for h in horizons]
    header = f"  {'Method':<24}" + ''.join(f'{c:>12}' for c in h_cols)
    if fed_lbl:
        header += ''.join(f'  Δfed_h={h}' for h in horizons)
    print(header)
    print('  ' + '─' * (len(header) - 2))

    for method in METHODS:
        row = f"  {METHOD_LABELS[method]:<24}"
        single_maes = {}
        for h in horizons:
            v = _mean_across_targets(all_records, single_lbl, h, method, targets)
            single_maes[h] = v
            row += f'{_fmt(v):>12}'
        if fed_lbl:
            for h in horizons:
                fv  = _mean_across_targets(all_records, fed_lbl, h, method, targets)
                row += f'  {_delta_str(fv, single_maes[h]):>8}'
        print(row)

    # ── Table 2: Best method per (target, h) ─────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 2 — BEST METHOD PER TARGET AND HORIZON  (single-country MAE)')
    print('=' * W)
    th_header = f"  {'Target':<24}" + ''.join(f'  h={h}:best(MAE)' for h in horizons)
    print(th_header)
    print('  ' + '─' * (len(th_header) - 2))

    for target in targets:
        row = f"  {target:<24}"
        for h in horizons:
            best_mae, best_m = np.inf, '—'
            for method in METHODS:
                v = _mae(all_records, single_lbl, target, h, method)
                if not np.isnan(v) and v < best_mae:
                    best_mae, best_m = v, method
            if best_m != '—':
                short = METHOD_LABELS[best_m].replace('Scarcity', 'Scar')[:14]
                row += f'  {short:<14}({best_mae:.3f})'
            else:
                row += f'  {"N/A":<14}(  N/A  )'
        print(row)

    # ── Table 3: Graph vs Causal — head-to-head by method family ─────────────
    print('\n' + '=' * W)
    print('TABLE 3 — GRAPH vs CAUSAL PARENT SELECTION  (single-country MAE)')
    print('  Δ = causal_MAE − graph_MAE  |  negative = causal wins')
    print('=' * W)

    for model_pair, model_name in [('xgb', 'XGBoost'), ('lgbm', 'LightGBM'), ('prophet', 'Prophet')]:
        graph_k  = f'{model_pair}_graph'
        causal_k = f'{model_pair}_causal'
        print(f"\n  {model_name}")
        print(f"  {'Target':<24}" + ''.join(f'  h={h:2d}   ' for h in horizons))
        print('  ' + '─' * (24 + 10 * len(horizons)))
        for target in targets:
            row = f"  {target:<24}"
            for h in horizons:
                g = _mae(all_records, single_lbl, target, h, graph_k)
                c = _mae(all_records, single_lbl, target, h, causal_k)
                if np.isnan(g) or np.isnan(c):
                    row += '     N/A  '
                else:
                    d = c - g
                    tag = ' <' if d < -0.05 else ('  >' if d > 0.05 else '  ~')
                    row += f'  {d:+6.3f}{tag}'
            print(row)
        # Aggregate row
        agg_row = f"  {'[mean across targets]':<24}"
        for h in horizons:
            deltas = []
            for t in targets:
                g = _mae(all_records, single_lbl, t, h, graph_k)
                c = _mae(all_records, single_lbl, t, h, causal_k)
                if not (np.isnan(g) or np.isnan(c)):
                    deltas.append(c - g)
            if deltas:
                d = float(np.mean(deltas))
                tag = ' <' if d < -0.05 else ('  >' if d > 0.05 else '  ~')
                agg_row += f'  {d:+6.3f}{tag}'
            else:
                agg_row += '     N/A  '
        print(agg_row)

    # ── Table 4: Short vs Long horizon summary ────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 4 — SHORT vs LONG HORIZON  (mean MAE across all targets; single-country)')
    print('=' * W)
    print(f"\n  {'Method':<24}  {'Short (h<=3)':>12}  {'Long (h>3)':>12}  {'Degradation':>12}")
    print('  ' + '─' * 67)
    for method in METHODS:
        s   = _group_mean(all_records, single_lbl, method, short_h, targets)
        l   = _group_mean(all_records, single_lbl, method, long_h,  targets)
        deg = _delta_str(l, s) if not (np.isnan(s) or np.isnan(l)) else '   N/A'
        print(f"  {METHOD_LABELS[method]:<24}  {_fmt(s):>12}  {_fmt(l):>12}  {deg:>12}")

    # ── Table 5: Causal parent retention rate per target ─────────────────────
    print('\n' + '=' * W)
    print('TABLE 5 — CAUSAL PARENT RETENTION  (causal / graph across all cutoffs)')
    print(f"  Vote threshold: {CAUSAL_VOTE_THRESHOLD:.0%}  |  Min N for causal: {CAUSAL_MIN_N}")
    print('=' * W)
    print(f"  {'Target':<24}  {'Graph parents':>14}  {'Causal parents':>14}  "
          f"{'Retention %':>12}  {'Impact (MAE)':>12}")
    print('  ' + '─' * 80)
    for target in targets:
        if target not in retention_stats or target not in global_agreement:
            continue
        n_c, n_g = retention_stats[target]
        pct = 100 * n_c / n_g if n_g > 0 else 0.0
        # Impact: mean MAE improvement from causal vs graph (aggregate across h)
        impacts = []
        for h in horizons:
            xg = _mae(all_records, single_lbl, target, h, 'xgb_graph')
            xc = _mae(all_records, single_lbl, target, h, 'xgb_causal')
            if not (np.isnan(xg) or np.isnan(xc)):
                impacts.append(xc - xg)
        impact_str = f'{float(np.mean(impacts)):+.4f}' if impacts else '   N/A'
        print(f"  {target:<24}  {n_g:>14}  {n_c:>14}  {pct:>11.1f}%  {impact_str:>12}")

    # ── Table 6: Estimand agreement matrix ───────────────────────────────────
    print('\n' + '=' * W)
    print('TABLE 6 — ESTIMAND AGREEMENT MATRIX')
    print('  sig_rate = (# significant) / (# estimated) across all cutoffs + parents')
    print(f"  Estimands: ATE ATT ATC CATE LATE NDE NIE")
    print('=' * W)
    enames = ALL_ESTIMAND_NAMES
    header = f"  {'Target':<24}" + ''.join(f'  {e:>8}' for e in enames)
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for target in targets:
        ag = global_agreement.get(target, {})
        row = f"  {target:<24}"
        for ename in enames:
            if ename in ag:
                sig, total = ag[ename]
                pct = 100 * sig / total if total > 0 else 0.0
                row += f'  {pct:>7.1f}%'
            else:
                row += f'  {"N/A":>8}'
        print(row)

    # ── Table 7: Federation benefit for causal methods ────────────────────────
    if fed_lbl:
        print('\n' + '=' * W)
        print('TABLE 7 — FEDERATION BENEFIT FOR CAUSAL METHODS')
        print(f"  single_MAE − fed_MAE  |  positive = federation helps")
        print(f"  Comparing: {single_lbl} vs {fed_lbl}")
        print('=' * W)
        for method in ['xgb_causal', 'lgbm_causal', 'prophet_causal']:
            print(f"\n  {METHOD_LABELS[method]}")
            print(f"  {'Target':<24}" + ''.join(f'  h={h:2d}   ' for h in horizons))
            print('  ' + '─' * (24 + 10 * len(horizons)))
            for target in targets:
                row = f"  {target:<24}"
                for h in horizons:
                    s  = _mae(all_records, single_lbl, target, h, method)
                    f_ = _mae(all_records, fed_lbl,    target, h, method)
                    if np.isnan(s) or np.isnan(f_):
                        row += '     N/A  '
                    else:
                        d   = s - f_
                        tag = ' >' if d > 0.1 else ('  <' if d < -0.1 else '  ~')
                        row += f'  {d:+6.3f}{tag}'
                print(row)

    # ── Summary footer ────────────────────────────────────────────────────────
    print('\n' + '=' * W)
    n_single = sum(1 for r in all_records if r['label'] == single_lbl)
    n_fed    = sum(1 for r in all_records if fed_lbl and r['label'] == fed_lbl)
    test_pts = {h: _count(all_records, single_lbl, targets[0], h, 'arima') for h in horizons}
    print(f"  Records: {n_single} single-country" +
          (f" | {n_fed} federated" if fed_lbl else ''))
    print(f"  Test points per horizon (first target): " +
          '  '.join(f'h={h}:{n}' for h, n in test_pts.items()))
    print(f"  Causal min N: {CAUSAL_MIN_N}  |  CATE min N: {CATE_MIN_N}  |  "
          f"Vote threshold: {CAUSAL_VOTE_THRESHOLD:.0%}  |  "
          f"Effect threshold: {EFFECT_THRESHOLD}")
    print('=' * W + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multi-target multi-horizon forecasting with causal identification')
    parser.add_argument('--no-fed',      action='store_true', help='Skip federated condition')
    parser.add_argument('--fast',        action='store_true',
                        help='Fast mode: ATE only, skip CATE/LATE/MEDIATION')
    parser.add_argument('--targets',     nargs='+', default=None)
    parser.add_argument('--horizons',    nargs='+', type=int, default=None)
    parser.add_argument('--initial-train', type=int, default=INITIAL_TRAIN)
    args = parser.parse_args()

    tgts = args.targets  if args.targets  else TARGETS
    hors = sorted(args.horizons) if args.horizons else HORIZONS
    i_tr = args.initial_train

    mode_str = 'FAST (ATE only)' if args.fast else 'FULL (ATE+ATT+ATC+CATE+LATE+MEDIATION)'
    print('=' * 80)
    print('MULTI-TARGET MULTI-HORIZON FORECASTING — CAUSAL IDENTIFICATION BENCHMARK')
    print(f'  Estimand mode : {mode_str}')
    print(f'  Targets       : {tgts}')
    print(f'  Horizons      : {hors}')
    print(f'  Initial train : {i_tr} years')
    print(f'  Causal min N  : {CAUSAL_MIN_N}  |  Vote threshold: {CAUSAL_VOTE_THRESHOLD:.0%}')
    print('=' * 80)

    countries = ['KEN', 'TZA', 'UGA']
    data = load_countries(countries)
    ken_df = data.get('KEN')
    if ken_df is None:
        print('ERROR: Kenya data not loaded'); sys.exit(1)

    aux_dfs = {cc: df for cc, df in data.items() if cc != 'KEN'}
    tgts = [t for t in tgts if t in ken_df.columns]
    if not tgts:
        print('ERROR: None of the specified targets found in Kenya data'); sys.exit(1)
    print(f'  Active targets ({len(tgts)}): {tgts}')

    conditions   = []
    all_records  = []
    all_agreement = {}
    all_retention = {}

    # ── Single-country condition ──────────────────────────────────────────────
    print('\n' + '─' * 80)
    print('CONDITION: single-country (KEN only)')
    print('─' * 80)
    single_records, _, s_agree, s_ret = rolling_backtest(
        ken_df, {}, 'single',
        horizons=hors, targets=tgts,
        initial_train=i_tr,
        conf_threshold=CONF_THRESHOLD,
        min_evidence=MIN_EVIDENCE,
        fast_mode=args.fast,
    )
    conditions.append(('single', single_records))
    all_records.extend(single_records)
    all_agreement.update(s_agree)
    all_retention.update(s_ret)

    # ── Federated condition ───────────────────────────────────────────────────
    if not args.no_fed:
        print('\n' + '─' * 80)
        print('CONDITION: federated (KEN + TZA + UGA)')
        print('─' * 80)
        fed_records, _, f_agree, f_ret = rolling_backtest(
            ken_df, aux_dfs, 'federated',
            horizons=hors, targets=tgts,
            initial_train=i_tr,
            conf_threshold=CONF_THRESHOLD,
            min_evidence=MIN_EVIDENCE,
            fast_mode=args.fast,
        )
        conditions.append(('federated', fed_records))
        all_records.extend(fed_records)
        # Merge federated agreement (accumulate on top of single)
        for tgt, ag in f_agree.items():
            if tgt not in all_agreement:
                all_agreement[tgt] = defaultdict(lambda: [0, 0])
            for ename, (sig, total) in ag.items():
                all_agreement[tgt][ename][0] += sig
                all_agreement[tgt][ename][1] += total

    print_results(all_records, conditions, tgts, hors,
                  all_agreement, all_retention)


if __name__ == '__main__':
    main()

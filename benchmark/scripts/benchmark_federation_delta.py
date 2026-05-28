"""
delta_coh Claim 4 Validation -- all 10 targets at h=1

Research question (Claim 4):
    Cross-country parent coherence (delta_coh) predicts the direction of
    federation benefit for downstream XGBoost+Scarcity forecasting at h=1.

Protocol:
    1. Build single-country engine (KEN only) and federated engine (KEN+TZA+UGA).
    2. Compute delta_coh per target (replicates §50 metric).
    3. Run h=1 rolling-origin backtest (24 cutoffs, 1999-2022) for both
       conditions using XGBoost+Scarcity only.
    4. Compute actual_h1_delta = MAE_single - MAE_fed per target
       (positive = federation reduces MAE = federation helps).
    5. Compute Spearman rho(delta_coh, actual_h1_delta) across all 10 targets.
    6. Confirm or refute Claim 4.

Usage:
    python benchmark/scripts/benchmark_federation_delta.py
    python benchmark/scripts/benchmark_federation_delta.py --skip-backtest
"""

import argparse
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

import logging
for _name in ('prophet', 'cmdstanpy'):
    logging.getLogger(_name).setLevel(logging.ERROR)
    logging.getLogger(_name).propagate = False

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph
from benchmark.real_data.world_bank_loader import prepare_multi_country_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGETS = [
    'gdp_growth', 'inflation_cpi', 'unemployment',
    'exports_gdp', 'imports_gdp', 'current_account',
    'real_interest_rate', 'broad_money', 'private_credit', 'govt_consumption',
]
COUNTRIES       = ['KEN', 'TZA', 'UGA']
CONF_THRESHOLD  = 0.35
MIN_EVIDENCE    = 5
INITIAL_TRAIN   = 10
MAX_PARENTS     = 5
MIN_PAIRS       = 4
DELTA_COH_GUARD = 0.02   # guard band for routing; positive = USE_FED

# Known h=1 MAE deltas from §46/§47 (single MAE - fed MAE; positive = fed helps)
KNOWN_DELTAS_H1 = {
    'gdp_growth':        +0.42,
    'inflation_cpi':     -1.23,
    'real_interest_rate': +1.71,
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    print("Loading World Bank data: KEN, TZA, UGA ...", flush=True)
    raw = prepare_multi_country_data(COUNTRIES)
    cleaned = {}
    for cc, df in raw.items():
        df = df.ffill().bfill()
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        cleaned[cc] = df
    print(f"  KEN: {len(cleaned['KEN'])} years  "
          f"TZA: {len(cleaned['TZA'])} years  "
          f"UGA: {len(cleaned['UGA'])} years", flush=True)
    return cleaned

# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def _build_engine(dfs, aux_countries):
    ken_df    = dfs['KEN']
    var_names = sorted(ken_df.columns.tolist())
    years     = sorted(ken_df.index.tolist())

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    engine.initialize_v2({'fields': [{'name': v} for v in var_names]}, use_causal=True)

    for yr in years:
        engine.process_row({k: float(v) for k, v in ken_df.loc[yr].items()
                            if pd.notna(v)})
        for cc in aux_countries:
            aux = dfs[cc]
            if yr in aux.index:
                rd = {k: float(v) for k, v in aux.loc[yr].reindex(var_names).items()
                      if pd.notna(v)}
                if rd:
                    engine.process_row(rd)

    graph, edges = extract_graph(engine, conf_threshold=CONF_THRESHOLD,
                                 min_evidence=MIN_EVIDENCE)
    return graph, edges


def build_graphs(dfs):
    print("\n[1/2] Single-country engine (KEN only) ...", flush=True)
    g_single, e_single = _build_engine(dfs, [])
    n_s = sum(len(v) for v in g_single.values())
    print(f"      {n_s} edges discovered", flush=True)

    print("[2/2] Federated engine (KEN + TZA + UGA) ...", flush=True)
    g_fed, e_fed = _build_engine(dfs, ['TZA', 'UGA'])
    n_f = sum(len(v) for v in g_fed.values())
    print(f"      {n_f} edges discovered", flush=True)

    return g_single, e_single, g_fed, e_fed

# ---------------------------------------------------------------------------
# Cross-country coherence (exact replica of §50 metric)
# ---------------------------------------------------------------------------

def _lag_corr(dfs, src, tgt, lag=1):
    out = {}
    for cc, df in dfs.items():
        if src not in df.columns or tgt not in df.columns:
            continue
        common = df[src].dropna().index.intersection(df[tgt].dropna().index)
        if len(common) < lag + 5:
            continue
        xs = df.loc[common, src].values[:-lag]
        yt = df.loc[common, tgt].values[lag:]
        if len(xs) < 5:
            continue
        try:
            r, _ = pearsonr(xs, yt)
            if not np.isnan(r):
                out[cc] = float(r)
        except Exception:
            pass
    return out


def _coherence(corrs):
    if len(corrs) < 2:
        return 0.0
    vals  = list(corrs.values())
    signs = [np.sign(v) for v in vals]
    maj   = np.sign(sum(signs))
    if maj == 0:
        maj = 1.0
    sign_agree = sum(1 for s in signs if s == maj) / len(signs)
    abs_v = [abs(v) for v in vals]
    cv    = np.std(abs_v) / (np.mean(abs_v) + 1e-4)
    str_agree = max(0.0, 1.0 - cv)
    return sign_agree * str_agree


def build_coh_table(dfs, e_single, e_fed):
    pairs = set()
    for e in e_single:
        pairs.add((e['source'], e['target']))
    for e in e_fed:
        pairs.add((e['source'], e['target']))
    table = {}
    for (src, tgt) in pairs:
        corrs = _lag_corr(dfs, src, tgt, lag=1)
        table[(src, tgt)] = _coherence(corrs)
    print(f"  Coherence computed for {len(table)} unique edges.", flush=True)
    return table


def compute_delta_coh(g_single, g_fed, coh_table):
    """Return {target: (delta_coh, s_coh, f_coh, rec)} for all TARGETS."""
    out = {}
    for tgt in TARGETS:
        single_p = g_single.get(tgt, [])
        fed_p    = g_fed.get(tgt, [])

        def mean_c(parents):
            if not parents:
                return float('nan')
            return float(np.mean([coh_table.get((p, tgt), 0.0) for p in parents]))

        c_s = mean_c(single_p)
        c_f = mean_c(fed_p)

        if np.isnan(c_s) or np.isnan(c_f):
            delta = float('nan')
            rec   = 'NO_CHANGE'
        else:
            delta = c_f - c_s
            if delta > DELTA_COH_GUARD:
                rec = 'USE_FED'
            elif delta < -DELTA_COH_GUARD:
                rec = 'NO_FED'
            else:
                rec = 'MARGINAL'
        out[tgt] = (delta, c_s, c_f, rec)
    return out

# ---------------------------------------------------------------------------
# Top-K graph helper
# ---------------------------------------------------------------------------

def _top_k_graph(graph, edges, max_parents=MAX_PARENTS):
    parent_type_conf = {}
    for e in edges:
        tgt  = e['target']; src = e['source']
        rt   = e.get('type', 'unknown')
        conf = float(e['confidence'])
        parent_type_conf.setdefault(tgt, {}).setdefault(src, {})
        parent_type_conf[tgt][src][rt] = max(
            parent_type_conf[tgt][src].get(rt, 0.0), conf)

    filtered = {}
    for tgt, parents in graph.items():
        pt        = parent_type_conf.get(tgt, {})
        type_champ = {}
        for src in parents:
            for rtype, conf in pt.get(src, {}).items():
                if conf > type_champ.get(rtype, ('', 0.0))[1]:
                    type_champ[rtype] = (src, conf)
        selected = {src for src, _ in type_champ.values()}
        all_conf = {src: max(pt.get(src, {}).values(), default=0.0)
                    for src in parents}
        for src in sorted(parents, key=lambda p: all_conf.get(p, 0.0), reverse=True):
            if len(selected) >= max_parents:
                break
            selected.add(src)
        filtered[tgt] = sorted(selected,
                                key=lambda p: all_conf.get(p, 0.0), reverse=True)
    return filtered

# ---------------------------------------------------------------------------
# XGBoost predictor
# ---------------------------------------------------------------------------

def _build_direct_pairs(train_df, target, feature_cols, h=1):
    cols = [c for c in feature_cols if c in train_df.columns and c != target]
    if not cols:
        return None, None, None
    needed = cols + [target]
    sub = train_df[needed].copy()
    for c in needed:
        sub[c] = sub[c].fillna(sub[c].mean())
    sub = sub.dropna()
    n = len(sub)
    if n < h + MIN_PAIRS:
        return None, None, None
    X      = sub[cols].iloc[:n - h].values
    y      = sub[target].iloc[h:].values
    X_last = sub[cols].iloc[-1].values
    return X, y, X_last


def _predict_xgb_graph(train_df, target, graph_parents, feature_cols):
    try:
        import xgboost as xgb
        cols = graph_parents if graph_parents else feature_cols
        X, y, X_last = _build_direct_pairs(train_df, target, cols, h=1)
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


def _arima_h1(series):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        s = np.array(series, dtype=float)
        if len(s) < 4:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = ARIMA(s, order=(1, 1, 0)).fit()
            return float(m.forecast(steps=1)[0])
    except Exception:
        return np.nan

# ---------------------------------------------------------------------------
# Rolling backtest (h=1, xgb_graph only, single and federated)
# ---------------------------------------------------------------------------

def _stream_engine(engine, ken_df, aux_dfs, year, var_names):
    engine.process_row({k: float(v) for k, v in ken_df.loc[year].items()
                        if pd.notna(v)})
    for cc, aux in aux_dfs.items():
        if year in aux.index:
            rd = {k: float(v) for k, v in aux.loc[year].reindex(var_names).items()
                  if pd.notna(v)}
            if rd:
                engine.process_row(rd)


def run_backtest(dfs, label, aux_countries):
    """
    Rolling-origin h=1 backtest with XGBoost+Scarcity.
    Returns list of {target, cutoff, actual, pred, ae}.
    """
    ken_df    = dfs['KEN']
    var_names = sorted(ken_df.columns.tolist())
    years     = sorted(ken_df.index.tolist())
    all_years = set(years)
    feature_cols = ken_df.columns.tolist()
    aux_dfs   = {cc: dfs[cc] for cc in aux_countries}

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    engine.initialize_v2({'fields': [{'name': v} for v in var_names]}, use_causal=True)

    print(f"\n  [{label}] Streaming initial {INITIAL_TRAIN} training years ...", flush=True)
    for yr in years[:INITIAL_TRAIN]:
        _stream_engine(engine, ken_df, aux_dfs, yr, var_names)

    records = []
    for ci in range(INITIAL_TRAIN - 1, len(years)):
        cutoff_yr = years[ci]
        pred_yr   = cutoff_yr + 1
        if pred_yr not in all_years:
            _stream_engine(engine, ken_df, aux_dfs, years[ci + 1] if ci + 1 < len(years) else cutoff_yr, var_names)
            continue

        graph, edges = extract_graph(engine, conf_threshold=CONF_THRESHOLD,
                                     min_evidence=MIN_EVIDENCE)
        graph_topk   = _top_k_graph(graph, edges)
        train_data   = ken_df[ken_df.index <= cutoff_yr]

        n_edges = sum(len(v) for v in graph.values())
        print(f"  [{label}] cutoff={cutoff_yr}  N={len(train_data)}  edges={n_edges}", flush=True)

        for target in TARGETS:
            if target not in ken_df.columns:
                continue
            parents = graph_topk.get(target, [])
            pred = _predict_xgb_graph(train_data, target, parents, feature_cols)
            if np.isnan(pred):
                pred = _arima_h1(train_data[target].dropna().values)

            actual = ken_df.loc[pred_yr, target] if pred_yr in all_years else np.nan
            if pd.isna(actual):
                continue
            ae = abs(float(actual) - float(pred)) if not np.isnan(float(pred)) else np.nan
            records.append({'label': label, 'target': target,
                            'cutoff': cutoff_yr, 'actual': float(actual),
                            'pred': float(pred) if not np.isnan(pred) else np.nan,
                            'ae': ae})

        if ci + 1 < len(years):
            _stream_engine(engine, ken_df, aux_dfs, years[ci + 1], var_names)

    return records

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_mae(records, label, target):
    vals = [r['ae'] for r in records
            if r['label'] == label and r['target'] == target
            and r['ae'] is not None and not np.isnan(r['ae'])]
    return (round(float(np.mean(vals)), 4), len(vals)) if vals else (np.nan, 0)

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_full_table(delta_coh_data, mae_single, mae_fed, n_single, n_fed):
    """Print the full validation table and Spearman rho."""
    print()
    print('=' * 112)
    print('CLAIM 4 VALIDATION -- delta_coh vs actual h=1 MAE delta (all 10 targets)')
    print('  actual_h1_delta = MAE_single - MAE_fed  (positive = federation helps)')
    print('=' * 112)
    print(f"  {'Target':<24} {'delta_coh':>9} {'Rec':<10} "
          f"{'MAE_s':>7} {'MAE_f':>7} {'actual_h1':>9} "
          f"{'Dir_match':>9}  {'N':>3}")
    print(f"  {'─'*108}")

    results = []
    for tgt in TARGETS:
        delta, c_s, c_f, rec = delta_coh_data[tgt]
        mae_s, ns = mae_single.get(tgt, (np.nan, 0))
        mae_f, nf = mae_fed.get(tgt, (np.nan, 0))

        if not np.isnan(mae_s) and not np.isnan(mae_f):
            actual_h1 = mae_s - mae_f   # positive = fed helps
        else:
            actual_h1 = np.nan

        known = KNOWN_DELTAS_H1.get(tgt)
        n = ns  # test points

        if np.isnan(actual_h1):
            dir_match = '  N/A'
        elif actual_h1 > 0 and rec == 'USE_FED':
            dir_match = '  YES'
        elif actual_h1 < 0 and rec == 'NO_FED':
            dir_match = '  YES'
        elif rec == 'MARGINAL':
            dir_match = '  --'
        else:
            dir_match = '   NO'

        dc_s  = f"{delta:+.3f}" if not np.isnan(delta) else '    --'
        ah_s  = f"{actual_h1:+.4f}" if not np.isnan(actual_h1) else '     --'
        ms_s  = f"{mae_s:.4f}"      if not np.isnan(mae_s)     else '   N/A'
        mf_s  = f"{mae_f:.4f}"      if not np.isnan(mae_f)     else '   N/A'

        star = ' *' if tgt in KNOWN_DELTAS_H1 else '  '
        print(f"  {tgt:<24} {dc_s:>9} {rec:<10} "
              f"{ms_s:>7} {mf_s:>7} {ah_s:>9} "
              f"{dir_match:>9}  {n:>3}{star}")

        results.append({'target': tgt, 'delta_coh': delta, 'actual_h1': actual_h1,
                        'rec': rec, 'mae_s': mae_s, 'mae_f': mae_f})

    print(f"\n  * = known from §46/§47 prior benchmarks\n")

    # Spearman correlation
    valid = [(r['delta_coh'], r['actual_h1']) for r in results
             if not np.isnan(r['delta_coh']) and not np.isnan(r['actual_h1'])]

    if valid:
        dc_v, ah_v = zip(*valid)
        rho, pval = spearmanr(dc_v, ah_v)
        print(f"  Spearman rho(delta_coh, actual_h1_delta) across {len(valid)} targets: "
              f"{rho:+.3f}  (p={pval:.3f})")

        dir_matches = sum(1 for r in results
                          if not np.isnan(r['actual_h1'])
                          and ((r['actual_h1'] > 0 and r['rec'] == 'USE_FED') or
                               (r['actual_h1'] < 0 and r['rec'] == 'NO_FED')))
        dir_total   = sum(1 for r in results
                          if not np.isnan(r['actual_h1']) and r['rec'] != 'MARGINAL')
        print(f"  Direction accuracy: {dir_matches}/{dir_total} "
              f"= {dir_matches/dir_total:.0%}" if dir_total else "  Direction accuracy: N/A")

        # Claim 4 verdict
        print()
        print('  VERDICT')
        print('  -------')
        if len(valid) < 5:
            print(f"  PRELIMINARY: only {len(valid)} data points — insufficient to confirm or refute.")
        elif abs(rho) >= 0.7 and dir_matches / max(dir_total, 1) >= 0.7:
            print(f"  CONFIRMED: rho={rho:+.3f}, direction accuracy={dir_matches}/{dir_total}.")
            print(f"  Claim 4 is supported on all {len(valid)} targets.")
        elif abs(rho) >= 0.4:
            print(f"  PARTIAL: rho={rho:+.3f}. delta_coh is predictive but not perfectly monotone.")
            print(f"  Downgrade Claim 4 to 'moderate evidence'.")
        else:
            print(f"  REFUTED: rho={rho:+.3f}. delta_coh does not predict federation benefit.")

    print('=' * 112)
    return results


def print_routing_comparison(delta_coh_data, results):
    """Compare §50 routing predictions against actual h=1 outcomes."""
    print()
    print('=' * 80)
    print('ROUTING COMPARISON: §50 predictions vs actual h=1 outcomes')
    print('  Predicted USE_FED: delta_coh > +0.02')
    print('  Predicted NO_FED:  delta_coh < -0.02')
    print('=' * 80)
    n_correct = 0
    n_total   = 0
    for r in results:
        if np.isnan(r['actual_h1']):
            continue
        delta = r['delta_coh']
        pred_helps = delta > DELTA_COH_GUARD
        pred_hurts = delta < -DELTA_COH_GUARD
        actual_helps = r['actual_h1'] > 0
        if (pred_helps and actual_helps) or (pred_hurts and not actual_helps):
            status = 'CORRECT'
            n_correct += 1
        elif abs(delta) <= DELTA_COH_GUARD:
            status = 'MARGINAL'
        else:
            status = 'WRONG  '
        n_total += 1
        known_tag = ' (§46/47 prior)' if r['target'] in KNOWN_DELTAS_H1 else ''
        print(f"  {r['target']:<24}  delta_coh={delta:+.3f}  "
              f"actual={r['actual_h1']:+.4f}  {status}{known_tag}")
    print(f"\n  {n_correct}/{n_total} non-marginal predictions correct")
    print('=' * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='delta_coh Claim 4 validation: all 10 targets at h=1')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip rolling backtest (only compute delta_coh table)')
    args = parser.parse_args()

    print('=' * 80)
    print('CLAIM 4 VALIDATION: delta_coh predicts federation benefit direction')
    print(f'  Countries: {COUNTRIES}')
    print(f'  Guard band: +/-{DELTA_COH_GUARD}')
    print(f'  Backtest: h=1, XGBoost+Scarcity, {INITIAL_TRAIN}-year initial train')
    print('=' * 80)

    dfs = load_data()

    print('\n-- Phase 1: Graph analysis and delta_coh --')
    g_single, e_single, g_fed, e_fed = build_graphs(dfs)
    print('\nComputing cross-country coherence ...', flush=True)
    coh_table = build_coh_table(dfs, e_single, e_fed)
    delta_coh_data = compute_delta_coh(g_single, g_fed, coh_table)

    print('\n  delta_coh summary:')
    for tgt in TARGETS:
        delta, c_s, c_f, rec = delta_coh_data[tgt]
        dc_s = f"{delta:+.3f}" if not np.isnan(delta) else '  N/A'
        print(f"    {tgt:<24}  delta_coh={dc_s}  ({rec})")

    if args.skip_backtest:
        print('\n  --skip-backtest: exiting before rolling backtest.')
        return

    print('\n-- Phase 2: Rolling h=1 backtest (single-country) --')
    records_single = run_backtest(dfs, 'single', aux_countries=[])

    print('\n-- Phase 3: Rolling h=1 backtest (federated) --')
    records_fed = run_backtest(dfs, 'federated', aux_countries=['TZA', 'UGA'])

    # Aggregate MAE per target
    mae_single = {tgt: compute_mae(records_single, 'single', tgt) for tgt in TARGETS}
    mae_fed    = {tgt: compute_mae(records_fed,    'federated', tgt) for tgt in TARGETS}

    # Build dict of (mae, n) for the table fn signature
    mae_s_dict = {tgt: mae_single[tgt] for tgt in TARGETS}
    mae_f_dict = {tgt: mae_fed[tgt]    for tgt in TARGETS}

    results = print_full_table(delta_coh_data, mae_s_dict, mae_f_dict,
                               {tgt: mae_single[tgt][1] for tgt in TARGETS},
                               {tgt: mae_fed[tgt][1]    for tgt in TARGETS})
    print_routing_comparison(delta_coh_data, results)

    # Save results CSV for documentation
    out_dir = _ROOT / 'artifacts' / 'benchmark_extended'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'delta_coh_validation.csv'
    rows = []
    for r, tgt in zip(results, TARGETS):
        delta, c_s, c_f, rec = delta_coh_data[tgt]
        ms, ns = mae_single[tgt]
        mf, nf = mae_fed[tgt]
        rows.append({
            'target': tgt, 'delta_coh': delta, 's_coh': c_s, 'f_coh': c_f,
            'rec': rec, 'mae_single': ms, 'mae_fed': mf, 'actual_h1_delta': r['actual_h1'],
            'n_test': ns,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format='%.4f')
    print(f"\n  Results saved to {out_path.relative_to(_ROOT)}")


if __name__ == '__main__':
    main()

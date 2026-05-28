"""
Forecasting Model Comparison — Prophet vs Tree Models vs TFT (Single & Federated)

Research question:
  Prophet is designed for data-scarce environments. Does Scarcity's graph feature
  selection help XGBoost / LightGBM compete with Prophet at N=34? Where does TFT
  break down?

Methods compared:
  Baseline:
    persistence          — naive last-value carry-forward
    arima                — ARIMA(1,1,0)

  Data-scarce reference:
    prophet              — Prophet (no seasonality, designed for small N)
    prophet_graph        — Prophet + Scarcity-discovered extra_regressors

  Tree models (blind — all 18 lag-1 features):
    xgb_blind            — XGBoost (n_est=50, max_depth=3)
    lgbm_blind           — LightGBM (n_est=50, max_depth=3)

  Tree models (graph-conditioned — only discovered parents as features):
    xgb_graph            — XGBoost + Scarcity graph feature selection
    lgbm_graph           — LightGBM + Scarcity graph feature selection

  Deep learning:
    tft                  — Temporal Fusion Transformer (reality check at N=34)

Two conditions:
  Single-country  — Kenya only (N=34, graph discovered from 34 rows)
  Federated       — KEN+TZA+UGA (N_eff=102, graph discovered from 102 rows;
                    models still trained on Kenya only)

Rolling-origin backtest:
  Initial training years: 10 (1990–1999)
  Test years: 2000–2023 (up to 24 one-step-ahead predictions)
  Graph: extracted at each test year boundary from all rows seen so far

Targets: gdp_growth, inflation_cpi

Usage:
    python benchmark/scripts/benchmark_forecasting_models.py
    python benchmark/scripts/benchmark_forecasting_models.py --targets gdp_growth
    python benchmark/scripts/benchmark_forecasting_models.py --no-fed
"""

import argparse
import io
import sys
import warnings
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph, graph_summary
from benchmark.real_data.world_bank_loader import prepare_multi_country_data
from benchmark.evaluation.forecasting import ForecastingEvaluator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGETS = ['gdp_growth', 'inflation_cpi']
INITIAL_TRAIN = 10          # years before rolling backtest starts
CONF_THRESHOLD = 0.35
MIN_EVIDENCE = 5
MAX_PARENTS = 5             # graph parents per target for tree/prophet models

# Methods in display order
METHODS = [
    'persistence',
    'arima',
    'prophet',
    'prophet_graph',
    'xgb_blind',
    'xgb_graph',
    'lgbm_blind',
    'lgbm_graph',
    'tft',
]

METHOD_LABELS = {
    'persistence':   'Persistence',
    'arima':         'ARIMA(1,1,0)',
    'prophet':       'Prophet',
    'prophet_graph': 'Prophet+Scarcity',
    'xgb_blind':     'XGBoost (blind)',
    'xgb_graph':     'XGBoost+Scarcity',
    'lgbm_blind':    'LightGBM (blind)',
    'lgbm_graph':    'LightGBM+Scarcity',
    'tft':           'TFT',
}

GRAPH_METHODS = {'prophet_graph', 'xgb_graph', 'lgbm_graph'}
TREE_METHODS  = {'xgb_blind', 'xgb_graph', 'lgbm_blind', 'lgbm_graph'}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_countries(countries):
    print(f"Loading data for: {', '.join(countries)} ...")
    data = prepare_multi_country_data(countries)
    cleaned = {}
    for cc, df in data.items():
        df = df.ffill().bfill()
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        cleaned[cc] = df
    return cleaned


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def rolling_backtest(ken_df, aux_dfs, label, conf_threshold, min_evidence,
                     targets=None, initial_train=None):
    """
    One-step-ahead rolling backtest.

    Training: all years <= T-1 (Kenya + aux countries streamed into engine).
    Test: predict Kenya year T.
    Graph: extracted at each test year from the engine state at that point.
    """
    tgts   = targets      if targets      is not None else TARGETS
    i_train = initial_train if initial_train is not None else INITIAL_TRAIN

    var_names = sorted(ken_df.columns.tolist())
    years     = sorted(ken_df.index.tolist())

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    schema = {'fields': [{'name': v} for v in var_names]}
    engine.initialize_v2(schema, use_causal=True)

    print(f"\n  [{label}] Initial training ({i_train} years) ...", flush=True)
    for yr in years[:i_train]:
        engine.process_row({k: float(v) for k, v in ken_df.loc[yr].items() if pd.notna(v)})
        for cc, aux in aux_dfs.items():
            if yr in aux.index:
                row = aux.loc[yr].reindex(var_names)
                rd  = {k: float(v) for k, v in row.items() if pd.notna(v)}
                if rd:
                    engine.process_row(rd)

    records = []

    for test_yr in years[i_train:]:
        train_data = ken_df[ken_df.index < test_yr]
        test_data  = ken_df[ken_df.index == test_yr]

        if len(test_data) == 0 or len(train_data) < 4:
            continue

        graph, edges = extract_graph(engine,
                                     conf_threshold=conf_threshold,
                                     min_evidence=min_evidence)

        # Trim graph to MAX_PARENTS per target (type-diverse top-K by confidence)
        graph_topk = _top_k_graph(graph, edges, max_parents=MAX_PARENTS)

        n_edges = sum(len(v) for v in graph.values())
        gdp_p   = len(graph_topk.get('gdp_growth', []))
        inf_p   = len(graph_topk.get('inflation_cpi', []))
        print(f"  [{label}] {test_yr}: {n_edges} edges | "
              f"gdp_parents={gdp_p} infl_parents={inf_p}", flush=True)

        for target in tgts:
            if target not in ken_df.columns:
                continue
            ev = ForecastingEvaluator(target_variable=target, horizon=1)

            row = {'year': test_yr, 'target': target, 'label': label,
                   'n_edges': n_edges,
                   'n_parents': len(graph_topk.get(target, []))}

            row['persistence']   = ev.evaluate_persistence(train_data, test_data)
            row['arima']         = ev.evaluate_arima(train_data, test_data)
            row['prophet']       = ev.evaluate_prophet(train_data, test_data)
            row['prophet_graph'] = ev.evaluate_prophet_with_graph(
                                       train_data, test_data, graph_topk)
            row['xgb_blind']     = ev.evaluate_xgboost(train_data, test_data)
            row['xgb_graph']     = ev.evaluate_xgboost_with_graph(
                                       train_data, test_data, graph_topk)
            row['lgbm_blind']    = ev.evaluate_lightgbm(train_data, test_data)
            row['lgbm_graph']    = ev.evaluate_lightgbm_with_graph(
                                       train_data, test_data, graph_topk)
            row['tft']           = ev.evaluate_tft(train_data, test_data)

            records.append(row)

        # Advance engine: stream test year
        engine.process_row({k: float(v) for k, v in ken_df.loc[test_yr].items()
                             if pd.notna(v)})
        for cc, aux in aux_dfs.items():
            if test_yr in aux.index:
                rd = {k: float(v) for k, v in aux.loc[test_yr].reindex(var_names).items()
                      if pd.notna(v)}
                if rd:
                    engine.process_row(rd)

    return records, engine


def _top_k_graph(graph, edges, max_parents):
    """Type-diverse top-K parent selection (mirrors run_scarcity_federation.py)."""
    parent_type_conf = {}
    for e in edges:
        tgt  = e['target']
        src  = e['source']
        rtype = e.get('type', 'unknown')
        conf  = float(e['confidence'])
        parent_type_conf.setdefault(tgt, {}).setdefault(src, {})
        parent_type_conf[tgt][src][rtype] = max(
            parent_type_conf[tgt][src].get(rtype, 0.0), conf)

    filtered = {}
    for tgt, parents in graph.items():
        pt = parent_type_conf.get(tgt, {})
        # Best parent per type
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def mean_mae(records, label, target, method):
    vals = [r[method]['mae'] for r in records
            if r['label'] == label and r['target'] == target
            and isinstance(r[method].get('mae'), float)
            and not np.isnan(r[method]['mae'])]
    return round(np.mean(vals), 4) if vals else float('nan')


def mean_dir(records, label, target, method):
    vals = [r[method]['dir_acc'] for r in records
            if r['label'] == label and r['target'] == target
            and isinstance(r[method].get('dir_acc'), float)
            and not np.isnan(r[method]['dir_acc'])]
    return round(np.mean(vals), 3) if vals else float('nan')


def tft_note(records, label, target):
    notes = [r['tft'].get('note', '') for r in records
             if r['label'] == label and r['target'] == target
             and 'note' in r.get('tft', {})]
    return notes[0] if notes else ''


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _fmt_mae(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '   N/A  '
    return f'{v:7.4f}'


def _delta(new_v, ref_v):
    if any(isinstance(x, float) and np.isnan(x) for x in [new_v, ref_v]):
        return '   N/A'
    d = new_v - ref_v
    sign = '+' if d >= 0 else ''
    return f'{sign}{d:.4f}'


def print_results(all_records, conditions, targets=None, initial_train=None):
    width = 100
    print("\n" + "=" * width)
    print("FORECASTING MODEL COMPARISON: Prophet vs Tree Models vs TFT")
    i_tr = initial_train if initial_train is not None else INITIAL_TRAIN
    tgts = targets if targets is not None else TARGETS
    print(f"Rolling backtest | Initial train: {i_tr} years | "
          f"Test: Kenya one-step-ahead MAE (lower=better)")
    print("=" * width)

    prophet_ref = {}  # (label, target) -> prophet MAE — used for delta columns

    for target in tgts:
        print(f"\n{'─'*width}")
        print(f"TARGET: {target.upper()}")
        print(f"{'─'*width}")

        # Header
        label_w = 22
        hdr = f"  {'Method':<{label_w}}"
        for label, _ in conditions:
            hdr += f"  {label[:14]:>14} MAE  DirAcc"
        if len(conditions) > 1:
            hdr += f"  {'Fed lift':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for method in METHODS:
            ml = METHOD_LABELS[method]
            tag = " <graph>" if method in GRAPH_METHODS else \
                  " <tree>" if method in TREE_METHODS else ""
            line = f"  {(ml + tag):<{label_w}}"

            maes = {}
            for label, _ in conditions:
                mae = mean_mae(all_records, label, target, method)
                da  = mean_dir(all_records, label, target, method)
                maes[label] = mae

                if method == 'prophet' and label == conditions[0][0]:
                    prophet_ref[(label, target)] = mae

                da_str = f'{da:.3f}' if not np.isnan(da) else '  N/A'
                line += f"  {_fmt_mae(mae)}  {da_str:>6}"

            if len(conditions) > 1:
                m1 = maes.get(conditions[0][0], float('nan'))
                m2 = maes.get(conditions[1][0], float('nan'))
                line += f"  {_delta(m2, m1):>10}"

            print(line)

        # TFT note
        for label, _ in conditions:
            note = tft_note(all_records, label, target)
            if note:
                print(f"    TFT [{label}]: {note}")

        # Summary rows
        print()
        for label, _ in conditions:
            prop_mae = mean_mae(all_records, label, target, 'prophet')
            xg_g     = mean_mae(all_records, label, target, 'xgb_graph')
            lg_g     = mean_mae(all_records, label, target, 'lgbm_graph')
            xg_b     = mean_mae(all_records, label, target, 'xgb_blind')
            lg_b     = mean_mae(all_records, label, target, 'lgbm_blind')

            best_tree_graph = min(x for x in [xg_g, lg_g] if not np.isnan(x)) \
                if not all(np.isnan(x) for x in [xg_g, lg_g]) else float('nan')
            best_tree_blind = min(x for x in [xg_b, lg_b] if not np.isnan(x)) \
                if not all(np.isnan(x) for x in [xg_b, lg_b]) else float('nan')

            print(f"  [{label}] Prophet MAE={_fmt_mae(prop_mae).strip()} | "
                  f"Best tree+graph={_fmt_mae(best_tree_graph).strip()} | "
                  f"Best tree blind={_fmt_mae(best_tree_blind).strip()}")

            if not np.isnan(prop_mae) and not np.isnan(best_tree_graph):
                gap = best_tree_graph - prop_mae
                if gap > 0.2:
                    verdict = "Prophet wins: tree+graph cannot match at this N"
                elif gap < -0.2:
                    verdict = "Tree+Scarcity wins: graph feature selection beats Prophet"
                else:
                    verdict = "Competitive: tree+graph within 0.2 MAE of Prophet"
                print(f"          -> {verdict}")

    # Graph feature selection benefit (blind vs graph, per tree model)
    print(f"\n{'─'*width}")
    print("GRAPH FEATURE SELECTION BENEFIT (tree blind - tree+graph MAE; negative = graph helps)")
    print(f"{'─'*width}")
    hdr2 = f"  {'Model':<20}"
    for target in tgts:
        hdr2 += f"  {target[:20]:>20}"
    for label, _ in conditions:
        print(f"\n  [{label}]")
        print(f"  {hdr2}")
        for tree_pair in [('xgb_blind', 'xgb_graph', 'XGBoost'),
                          ('lgbm_blind', 'lgbm_graph', 'LightGBM')]:
            blind_k, graph_k, name = tree_pair
            line = f"    {name:<18}"
            for target in tgts:
                b = mean_mae(all_records, label, target, blind_k)
                g = mean_mae(all_records, label, target, graph_k)
                line += f"  {_delta(g, b):>20}"
            print(line)

    print(f"\n{'─'*width}")
    print("INTERPRETATION")
    print(f"{'─'*width}")
    print("  Prophet: designed for data-scarce time series — strong prior, additive model,")
    print("           no feature dimensionality risk at N=34.")
    print("  XGBoost/LightGBM (blind): 18 lag-1 features, N_train starting at 10 — severe")
    print("           overfitting expected (18 features / 9 training pairs on first backtest).")
    print("  XGBoost/LightGBM + Scarcity: 3-5 graph-selected features — reduced feature/sample")
    print("           ratio; tests whether Scarcity's graph acts as effective regularisation.")
    print("  TFT: minimum 15 rows for even marginal functionality; fails early in backtest.")
    print("  Federated condition: same Kenya-only test, but graph from KEN+TZA+UGA (N_eff=102)")
    print("           improves graph quality — richer, more stable economic parent sets.")
    print("=" * width)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Forecasting comparison: Prophet vs tree models vs TFT')
    parser.add_argument('--targets', nargs='+', default=TARGETS,
                        help='Target variables (default: gdp_growth inflation_cpi)')
    parser.add_argument('--no-fed', action='store_true',
                        help='Skip federated condition (Kenya only)')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--min-evidence', type=int, default=MIN_EVIDENCE)
    parser.add_argument('--initial-train', type=int, default=INITIAL_TRAIN)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_targets   = args.targets
    run_init      = args.initial_train

    np.random.seed(args.seed)

    # Load data
    countries_needed = ['KEN'] if args.no_fed else ['KEN', 'TZA', 'UGA']
    country_dfs = load_countries(countries_needed)
    ken_df = country_dfs['KEN']

    all_records = []
    conditions  = []

    # ── Condition 1: Single-country (Kenya only) ───────────────────────────
    print("\n" + "="*60)
    print("CONDITION 1: Single-country (Kenya only, N=34)")
    print("="*60)
    records_single, _ = rolling_backtest(
        ken_df, {}, 'Single (N=34)',
        conf_threshold=args.conf, min_evidence=args.min_evidence,
        targets=run_targets, initial_train=run_init)
    all_records.extend(records_single)
    conditions.append(('Single (N=34)', records_single))

    # ── Condition 2: Federated (KEN+TZA+UGA) ──────────────────────────────
    if not args.no_fed:
        print("\n" + "="*60)
        print("CONDITION 2: Federated (KEN+TZA+UGA, N_eff up to 102)")
        print("="*60)
        aux = {cc: df for cc, df in country_dfs.items() if cc != 'KEN'}
        records_fed, _ = rolling_backtest(
            ken_df, aux, 'Federated (N=102)',
            conf_threshold=args.conf, min_evidence=args.min_evidence,
            targets=run_targets, initial_train=run_init)
        all_records.extend(records_fed)
        conditions.append(('Federated (N=102)', records_fed))

    print_results(all_records, conditions, run_targets, run_init)


if __name__ == '__main__':
    main()

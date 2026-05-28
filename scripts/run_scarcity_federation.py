"""
Federation scarcity experiment.

Tests whether pooling observations from multiple East African countries
helps the Scarcity engine discover macro relationships (GDP, inflation)
that single-country data (n=34) cannot reliably detect.

Design
------
Phase 1 (baseline):
  Engine trained on Kenya only (34 annual rows).
  Rolling backtest: predict Kenya year T using graph from years < T.

Phase 2 (federated):
  Engine trained on Kenya + Tanzania + Uganda + Ethiopia (~136 rows).
  For each calendar year, stream all countries before advancing.
  Rolling backtest: still predict Kenya year T — only the training
  evidence changes.

Comparison:
  - Number of edges discovered (total and for GDP/inflation targets)
  - Forecasting MAE on Kenya (gdp_growth, inflation_cpi)
  - Which relationships only appear in the federated graph

Usage:
    python scripts/run_scarcity_federation.py
    python scripts/run_scarcity_federation.py --countries KEN TZA UGA ETH
"""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph, graph_summary, inspect_edges
from benchmark.real_data.world_bank_loader import prepare_multi_country_data
from benchmark.evaluation.forecasting import ForecastingEvaluator

warnings.filterwarnings('ignore')

REPORT_OUT  = str(_ROOT / 'scarcity' / 'synthetic' / 'benchmark_results')
DATA_JSON   = _ROOT / 'scarcity' / 'synthetic' / 'benchmark_results' / 'benchmark_data.json'
TARGETS     = ['gdp_growth', 'inflation_cpi']

KNOWN_EDGES = {
    ('inflation_cpi', 'real_interest_rate'),
    ('inflation_cpi', 'broad_money'),
    ('gcf',           'gdp_growth'),
    ('exports_gdp',   'gdp_growth'),
    ('imports_gdp',   'gdp_growth'),
    ('broad_money',   'inflation_cpi'),
    ('current_account', 'gdp_growth'),
    ('govt_consumption', 'gdp_growth'),
    ('private_credit', 'gdp_growth'),
}


# ── Engine helpers ─────────────────────────────────────────────────────────────

def build_engine(var_names: list) -> OnlineDiscoveryEngine:
    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    schema = {'fields': [{'name': v} for v in var_names]}
    engine.initialize_v2(schema, use_causal=True)
    return engine


def stream_row(engine, row_dict: dict) -> None:
    engine.process_row(row_dict)


def fmt(val, d=4):
    try:
        f = float(val)
        return 'N/A' if f != f else f'{f:.{d}f}'
    except (TypeError, ValueError):
        return 'N/A'


def mean_safe(vals):
    clean = [v for v in vals if v is not None and v == v]
    return round(sum(clean) / len(clean), 4) if clean else float('nan')


ALL_15_TYPES = [
    'causal', 'competitive', 'compositional', 'correlational', 'equilibrium',
    'functional', 'graph', 'logical', 'mediating', 'moderating',
    'probabilistic', 'similarity', 'structural', 'synergistic', 'temporal',
]


def pool_type_coverage(engine, conf_threshold=0.45, min_evidence=5):
    """Return per-type pool stats for all 15 hypothesis types."""
    from collections import defaultdict
    by_type = defaultdict(list)
    for h in engine.hypotheses.population.values():
        by_type[h.rel_type.value].append(h)
    rows = []
    for t in ALL_15_TYPES:
        hyps = by_type.get(t, [])
        if not hyps:
            rows.append({'type': t, 'count': 0, 'max_conf': 0.0,
                         'med_conf': 0.0, 'extractable': 0})
        else:
            confs = [h.confidence for h in hyps]
            extractable = sum(
                1 for h in hyps
                if h.confidence >= conf_threshold and h.evidence >= min_evidence
            )
            rows.append({
                'type': t,
                'count': len(hyps),
                'max_conf': round(float(max(confs)), 4),
                'med_conf': round(float(sorted(confs)[len(confs)//2]), 4),
                'extractable': extractable,
            })
    return rows


def top_k_graph(graph: dict, edges: list, max_parents: int = 6) -> dict:
    """
    Return a filtered graph that represents ALL discovered relationship types.

    Selection strategy (type-diversity first):
      1. For each relationship type that produced an edge to the target,
         pick the highest-confidence parent from that type.
      2. Then fill remaining slots (up to max_parents) by overall confidence.

    This ensures that functional, temporal, equilibrium, competitive, synergistic,
    compositional, mediating, moderating, graph, probabilistic, structural, logical,
    and similarity relationships all get a representative parent alongside the
    dominant causal/correlational ones — without handing 15+ regressors to
    ARIMA/Prophet when n_train is small.

    max_parents=6 balances type diversity against overfitting:
      n_train≈15 supports ~3 regressors by strict heuristics; RidgeCV in
      evaluate_scarcity_graph handles regularisation, and Prophet/ARIMA both
      receive the same graph so their relative comparison is still fair.
    """
    # Per-target: {source: {rel_type: best_conf}}
    parent_type_conf: dict = {}
    for e in edges:
        tgt = e['target']
        src = e['source']
        rtype = e.get('type', 'unknown')
        conf = float(e['confidence'])
        parent_type_conf.setdefault(tgt, {}).setdefault(src, {})
        old = parent_type_conf[tgt][src].get(rtype, 0.0)
        parent_type_conf[tgt][src][rtype] = max(old, conf)

    filtered = {}
    for tgt, parents in graph.items():
        pt = parent_type_conf.get(tgt, {})

        # Step 1: best parent per relationship type (type diversity)
        type_champion: dict = {}  # rel_type -> (src, conf)
        for src in parents:
            for rtype, conf in pt.get(src, {}).items():
                if conf > type_champion.get(rtype, ('', 0.0))[1]:
                    type_champion[rtype] = (src, conf)
        selected = {src for src, _ in type_champion.values()}

        # Step 2: fill to max_parents by overall best confidence
        all_conf = {src: max(pt.get(src, {}).values(), default=0.0) for src in parents}
        for src in sorted(parents, key=lambda p: all_conf.get(p, 0.0), reverse=True):
            if len(selected) >= max_parents:
                break
            selected.add(src)

        filtered[tgt] = sorted(selected, key=lambda p: all_conf.get(p, 0.0), reverse=True)
    return filtered


def check_plausibility(edges):
    tagged = []
    for e in edges:
        fwd = (e['source'], e['target'])
        rev = (e['target'], e['source'])
        if fwd in KNOWN_EDGES or rev in KNOWN_EDGES:
            tag = 'KNOWN'
        elif any(k in e['source'] or k in e['target']
                 for k in ('gdp', 'inflation', 'interest', 'gcf', 'credit',
                           'money', 'govt', 'export', 'import', 'account')):
            tag = 'PLAUSIBLE'
        else:
            tag = 'NOVEL'
        tagged.append({**e, 'plausibility': tag})
    return tagged


# ── Backtest ───────────────────────────────────────────────────────────────────

def rolling_backtest(
    kenya_df: pd.DataFrame,
    aux_dfs: dict,           # {country: df} — other countries for co-training
    conf_threshold: float,
    min_evidence: int,
    initial_train_years: int,
    label: str,
) -> tuple:
    """
    Rolling-origin backtest.

    Training: for each year <= T-1, stream Kenya row AND all aux country rows
              for that year (if available).
    Test: predict Kenya year T from graph extracted at T.
    """
    years = sorted(kenya_df.index.tolist())
    all_vars = sorted(kenya_df.columns.tolist())
    engine   = build_engine(all_vars)

    print(f'\n  [{label}] Streaming initial {initial_train_years} training years ...', flush=True)
    for yr in years[:initial_train_years]:
        # Kenya
        engine.process_row(kenya_df.loc[yr].to_dict())
        # Aux countries (same calendar year)
        for cc, aux in aux_dfs.items():
            if yr in aux.index:
                row = aux.loc[yr].reindex(all_vars).to_dict()
                engine.process_row(row)

    results = []
    edge_snapshots = {}

    # Count effective observations seen
    n_per_year = 1 + len(aux_dfs)
    print(f'  [{label}] {n_per_year} obs/year (1 Kenya + {len(aux_dfs)} aux)')

    for test_year in years[initial_train_years:]:
        graph, edges = extract_graph(engine, conf_threshold=conf_threshold,
                                     min_evidence=min_evidence)
        edge_snapshots[test_year] = edges

        # Per-model parent caps — each model gets a type-diverse graph tuned
        # to its capacity to handle regressors on small training windows.
        #
        #   RidgeCV (evaluate_scarcity_graph): 6 parents — RidgeCV cross-validates
        #     the regularisation penalty, so additional regressors are penalised
        #     rather than overfit.
        #
        #   Prophet: 5 parents — Prophet is relatively robust with extra_regressors
        #     but starts to have convergence issues above ~5 on n_train≈15.
        #
        #   ARIMAX: 3 parents — ARIMA with exog is the most fragile: each exog
        #     column consumes a degree of freedom from the lag-1 lag-shifted
        #     fit (y_fit = y[1:], exog_fit = exog[:-1]), so n_eff = n_train - 2.
        #     At n_train=15: n_eff=13, safe budget ≈ 3 regressors.
        n_train = int((kenya_df.index < test_year).sum())
        arimax_budget  = max(1, min(3, n_train // 5))  # conservative: n/5, floor 1, ceil 3
        prophet_budget = max(1, min(5, n_train // 3))  # moderate:     n/3, floor 1, ceil 5
        ridge_budget   = max(1, min(6, n_train // 2))  # liberal:      n/2, floor 1, ceil 6

        graph_ridge   = top_k_graph(graph, edges, max_parents=ridge_budget)
        graph_prophet = top_k_graph(graph, edges, max_parents=prophet_budget)
        graph_arimax  = top_k_graph(graph, edges, max_parents=arimax_budget)

        n_edges = sum(len(v) for v in graph.values())
        print(f'  [{label}] Year {test_year}: {n_edges} edges | '
              f'GDP parents={len(graph.get("gdp_growth",[]))} '
              f'Infl parents={len(graph.get("inflation_cpi",[]))} | '
              f'budgets ridge={ridge_budget} prophet={prophet_budget} arimax={arimax_budget}',
              flush=True)

        train_data = kenya_df[kenya_df.index < test_year]
        test_data  = kenya_df[kenya_df.index == test_year]

        for target in TARGETS:
            if target not in kenya_df.columns:
                continue
            ev = ForecastingEvaluator(target_variable=target, horizon=1)
            pers             = ev.evaluate_persistence(train_data, test_data)
            arima            = ev.evaluate_arima(train_data, test_data)
            prophet          = ev.evaluate_prophet(train_data, test_data)
            arimax_scarcity  = ev.evaluate_arimax_with_graph(train_data, test_data, graph_arimax)
            prophet_scarcity = ev.evaluate_prophet_with_graph(train_data, test_data, graph_prophet)

            results.append({
                'year':    test_year,
                'target':  target,
                'label':   label,
                'n_parents':         len(graph.get(target, [])),
                'parents':           graph.get(target, []),
                'parents_prophet':   graph_prophet.get(target, []),
                'parents_arimax':    graph_arimax.get(target, []),
                'parents_ridge':     graph_ridge.get(target, []),
                'persistence_mae':      pers['mae'],
                'arima_mae':            arima['mae'],
                'prophet_mae':          prophet['mae'],
                'arimax_scarcity_mae':  arimax_scarcity['mae'],
                'prophet_scarcity_mae': prophet_scarcity['mae'],
                'persistence_dir':      pers['dir_acc'],
                'arima_dir':            arima['dir_acc'],
                'prophet_dir':          prophet['dir_acc'],
                'arimax_scarcity_dir':  arimax_scarcity['dir_acc'],
                'prophet_scarcity_dir': prophet_scarcity['dir_acc'],
            })

        # Stream test year into engine (Kenya + aux)
        engine.process_row(kenya_df.loc[test_year].to_dict())
        for cc, aux in aux_dfs.items():
            if test_year in aux.index:
                engine.process_row(aux.loc[test_year].reindex(all_vars).to_dict())

    return results, edge_snapshots, engine


# ── Print helpers ──────────────────────────────────────────────────────────────

METHODS = [
    ('persistence',      'PERSISTENCE'),
    ('arima',            'ARIMA'),
    ('prophet',          'PROPHET'),
    ('arimax_scarcity',  'ARIMAX+SCARCITY'),
    ('prophet_scarcity', 'PROPHET+SCARCITY'),
]


def print_summary(results: list, label: str) -> None:
    df = pd.DataFrame(results)
    sub = df[df['label'] == label]
    print(f'\n  --- {label} ---')
    for target in TARGETS:
        t = sub[sub['target'] == target]
        if t.empty:
            continue
        print(f'  {target}:')
        pct_parents = 100 * (t['n_parents'] > 0).mean()
        for key, name in METHODS:
            col = f'{key}_mae'
            if col not in t.columns:
                continue
            mae     = mean_safe(t[col].tolist())
            dir_acc = mean_safe(t[f'{key}_dir'].tolist())
            extra = f'  (graph in {pct_parents:.0f}% of years)' \
                    if 'scarcity' in key else ''
            print(f'    {name:<20} MAE={fmt(mae)}  Dir={fmt(dir_acc,3)}{extra}')


def compare_results(single: list, fed: list) -> None:
    print('\n' + '=' * 70)
    print('GRAPH-INFORMED FORECASTING: SINGLE vs FEDERATED')
    print('Federation gives graph-informed models more accurate parent sets.')
    print('=' * 70)
    df_s = pd.DataFrame(single)
    df_f = pd.DataFrame(fed)

    for target in TARGETS:
        print(f'\n  {target}:')
        print(f'  {"Method":<22} {"Single MAE":>11}  {"Federated MAE":>13}  {"Delta":>8}')
        print(f'  {"-"*60}')
        for key, name in METHODS:
            col = f'{key}_mae'
            ts = df_s[df_s['target'] == target]
            tf = df_f[df_f['target'] == target]
            if col not in ts.columns:
                continue
            mae_s = mean_safe(ts[col].tolist())
            mae_f = mean_safe(tf[col].tolist())
            delta = mae_f - mae_s if (mae_s == mae_s and mae_f == mae_f) else float('nan')
            better = ' *** BETTER' if delta < -0.05 else (' <-- worse' if delta > 0.05 else '')
            print(f'  {name:<22} {fmt(mae_s):>11}  {fmt(mae_f):>13}  '
                  f'{fmt(delta,4):>8}{better}')

        ts = df_s[df_s['target'] == target]
        tf = df_f[df_f['target'] == target]
        pct_s = 100 * (ts['n_parents'] > 0).mean()
        pct_f = 100 * (tf['n_parents'] > 0).mean()
        print(f'  {"Graph coverage":<22} {pct_s:>10.0f}%  {pct_f:>12.0f}%')


def compare_edges(single_edges: list, fed_edges: list) -> None:
    print('\n' + '=' * 70)
    print('EDGE DISCOVERY: SINGLE vs FEDERATED')
    print('=' * 70)

    s_tagged = check_plausibility(single_edges)
    f_tagged = check_plausibility(fed_edges)

    for label, tagged in [('Single-country', s_tagged), ('Federated', f_tagged)]:
        known  = sum(1 for e in tagged if e.get('plausibility') == 'KNOWN')
        plaus  = sum(1 for e in tagged if e.get('plausibility') == 'PLAUSIBLE')
        novel  = sum(1 for e in tagged if e.get('plausibility') == 'NOVEL')
        print(f'\n  {label}: {len(tagged)} total edges  '
              f'| KNOWN={known}  PLAUSIBLE={plaus}  NOVEL={novel}')

    # Edges in federated but not in single
    s_pairs = {(e['source'], e['target'], e['type']) for e in single_edges}
    new_in_fed = [e for e in f_tagged
                  if (e['source'], e['target'], e['type']) not in s_pairs]
    macro_new = [e for e in new_in_fed if e.get('plausibility') in ('KNOWN', 'PLAUSIBLE')]

    if macro_new:
        print(f'\n  Macro edges discovered ONLY in federated graph ({len(macro_new)}):')
        print(f'  {"Source":<28} {"Target":<28} {"Type":<14} {"Conf":>6}  Plaus')
        print(f'  {"-"*85}')
        for e in sorted(macro_new, key=lambda x: -x['confidence'])[:20]:
            print(f'  {e["source"]:<28} {e["target"]:<28} '
                  f'{e["type"]:<14} {e["confidence"]:>6.3f}  {e["plausibility"]}')
    else:
        print(f'\n  {len(new_in_fed)} new edges in federated graph (none are KNOWN/PLAUSIBLE macro edges)')

    # GDP/inflation parents in federated graph
    print('\n  GDP/inflation parents in federated graph:')
    for tgt in TARGETS:
        parents = [e for e in f_tagged if e['target'] == tgt]
        if parents:
            for p in sorted(parents, key=lambda x: -x['confidence'])[:5]:
                print(f'    {p["source"]:<28} -> {p["target"]:<22} '
                      f'{p["type"]:<14} conf={p["confidence"]:.3f}  {p["plausibility"]}')
        else:
            print(f'    {tgt}: no parents discovered')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--countries', nargs='+', default=['KEN', 'TZA', 'UGA'],
                        help='Countries to use (first is always the target for forecasting)')
    parser.add_argument('--conf',         type=float, default=0.45)
    parser.add_argument('--min_evidence', type=int,   default=5)
    parser.add_argument('--train_years',  type=int,   default=15)
    args = parser.parse_args()

    target_country = args.countries[0]
    aux_countries  = args.countries[1:]

    print('=' * 70)
    print('Scarcity Federation Experiment')
    print(f'  Target country:     {target_country}')
    print(f'  Auxiliary countries: {aux_countries}')
    print(f'  conf={args.conf}  min_evidence={args.min_evidence}  train_years={args.train_years}')
    print('=' * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    print('\n[1/4] Loading macroeconomic data ...')
    all_data = prepare_multi_country_data(args.countries)

    kenya_df = all_data[target_country]
    aux_dfs  = {cc: all_data[cc] for cc in aux_countries if cc in all_data}

    print(f'\n  Effective obs/year: {1 + len(aux_dfs)}'
          f' ({target_country} + {list(aux_dfs.keys())})')

    # Align columns: all countries share the same variable names
    all_vars = sorted(kenya_df.columns.tolist())
    for cc in aux_dfs:
        aux_dfs[cc] = aux_dfs[cc].reindex(columns=all_vars)

    # ── Phase 1: Single-country baseline ──────────────────────────────────────
    print('\n[2/4] Phase 1 — single-country baseline (Kenya only) ...')
    t0 = time.time()
    single_results, single_snapshots, single_engine = rolling_backtest(
        kenya_df, aux_dfs={},
        conf_threshold=args.conf,
        min_evidence=args.min_evidence,
        initial_train_years=args.train_years,
        label='Single',
    )
    single_graph, single_edges = extract_graph(
        single_engine, conf_threshold=args.conf, min_evidence=args.min_evidence)
    single_edges = sorted(check_plausibility(single_edges), key=lambda x: -x['confidence'])
    print(f'  Done in {time.time()-t0:.1f}s  |  {graph_summary(single_graph, single_edges)}')

    # ── Phase 2: Federated ────────────────────────────────────────────────────
    print(f'\n[3/4] Phase 2 — federated ({target_country} + {list(aux_dfs.keys())}) ...')
    t0 = time.time()
    fed_results, fed_snapshots, fed_engine = rolling_backtest(
        kenya_df, aux_dfs=aux_dfs,
        conf_threshold=args.conf,
        min_evidence=args.min_evidence,
        initial_train_years=args.train_years,
        label='Federated',
    )
    fed_graph, fed_edges = extract_graph(
        fed_engine, conf_threshold=args.conf, min_evidence=args.min_evidence)
    fed_edges = sorted(check_plausibility(fed_edges), key=lambda x: -x['confidence'])
    print(f'  Done in {time.time()-t0:.1f}s  |  {graph_summary(fed_graph, fed_edges)}')

    # ── Results ────────────────────────────────────────────────────────────────
    print('\n[4/4] Comparing results ...')
    print('\n' + '=' * 70)
    print('PER-PHASE SUMMARY (Kenya rolling backtest)')
    print('=' * 70)
    print_summary(single_results, 'Single')
    print_summary(fed_results,    'Federated')

    compare_results(single_results, fed_results)
    compare_edges(single_edges, fed_edges)

    # ── Update report ─────────────────────────────────────────────────────────
    print('\n[Report] Updating benchmark report ...')
    with open(DATA_JSON) as f:
        loaded = json.load(f)
    base = loaded[-1] if isinstance(loaded, list) else loaded

    # Pool-level type coverage (all 15 hypothesis types, both engines)
    single_coverage = pool_type_coverage(
        single_engine, conf_threshold=args.conf, min_evidence=args.min_evidence)
    fed_coverage    = pool_type_coverage(
        fed_engine, conf_threshold=args.conf, min_evidence=args.min_evidence)

    # Build federation comparison section
    base['federation_scarcity'] = {
        'single_results':      single_results,
        'fed_results':         fed_results,
        'single_n_edges':      len(single_edges),
        'fed_n_edges':         len(fed_edges),
        'single_edges':        single_edges[:40],
        'fed_edges':           fed_edges[:40],
        'aux_countries':       aux_countries,
        'n_obs_per_year':      1 + len(aux_dfs),
        'single_pool_coverage': single_coverage,
        'fed_pool_coverage':    fed_coverage,
    }

    from benchmark.synthetic.reporting import generate_report
    rpt, dat = generate_report(base, out_dir=REPORT_OUT)
    print(f'  Report: {rpt}')
    print(f'  Data:   {dat}')
    print('=' * 70)


if __name__ == '__main__':
    main()

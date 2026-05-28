"""
Engine-driven benchmark.

Phases:
  1. Stream Kenya data through OnlineDiscoveryEngine (all 15 hypothesis types).
  2. Rolling-origin backtest: extract graph at each year, forecast next year.
  3. Compare Scarcity vs Persistence, ARIMA, VAR, Prophet.
  4. Anomaly detection using engine residuals.
  5. Inspect discovered edges for economic sensibility.
  6. If Scarcity underperforms: diagnose discovered graph.
  7. Update benchmark report with engine results.

Usage:
    python scripts/run_engine_benchmark.py
    python scripts/run_engine_benchmark.py --conf 0.4 --min_evidence 8
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

# Force UTF-8 output on Windows (avoids cp1252 errors for box/arrow chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph, graph_summary, inspect_edges
from benchmark.real_data.world_bank_loader import prepare_multi_country_data
from benchmark.evaluation.forecasting import ForecastingEvaluator
from benchmark.evaluation.anomaly_detection import AnomalyDetectionEvaluator
from benchmark.synthetic.reporting import generate_report

warnings.filterwarnings("ignore")

REPORT_OUT = str(_ROOT / "scarcity" / "synthetic" / "benchmark_results")
DATA_JSON   = _ROOT / "scarcity" / "synthetic" / "benchmark_results" / "benchmark_data.json"

# Known Kenya macroeconomic relationships from literature (for edge plausibility check)
KNOWN_KENYA_EDGES = {
    # inflation → interest_rate (central bank reaction function)
    ('inflation_cpi', 'interest_rate'),
    ('inflation_cpi', 'real_interest_rate'),
    # gdp_growth ↔ trade / investment
    ('gdp_growth', 'gross_capital_formation'),
    ('exports_pct_gdp', 'gdp_growth'),
    ('imports_pct_gdp', 'gdp_growth'),
    # money supply → inflation
    ('money_supply_m2', 'inflation_cpi'),
    # exchange rate → inflation (pass-through)
    ('exchange_rate', 'inflation_cpi'),
    # unemployment → gdp (Okun's law)
    ('unemployment_rate', 'gdp_growth'),
    # government spending → gdp
    ('govt_expenditure', 'gdp_growth'),
    # population growth → gdp
    ('population_growth', 'gdp_growth'),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_engine(var_names: list, small_dataset_mode: bool = True) -> OnlineDiscoveryEngine:
    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=small_dataset_mode)
    schema = {"fields": [{"name": v} for v in var_names]}
    engine.initialize_v2(schema, use_causal=True)
    return engine


def stream_dataframe(engine: OnlineDiscoveryEngine, df: pd.DataFrame,
                     label: str = "") -> None:
    for i, year in enumerate(df.index):
        row = df.loc[year].to_dict()
        engine.process_row(row)
        if label and (i + 1) % 5 == 0:
            print(f"    {label}: streamed {i+1}/{len(df)} rows", flush=True)


def mean_or_nan(vals):
    clean = [v for v in vals if v is not None and v == v]
    return round(sum(clean) / len(clean), 4) if clean else float('nan')


def fmt(val, d=4):
    try:
        f = float(val)
        if f != f:   # NaN
            return "N/A"
        return f"{f:.{d}f}"
    except (TypeError, ValueError):
        return "N/A"


def check_plausibility(edges, known=KNOWN_KENYA_EDGES):
    """Tag edges as known/plausible/novel relative to macroeconomic literature."""
    tagged = []
    for e in edges:
        pair_fwd = (e['source'], e['target'])
        pair_rev = (e['target'], e['source'])
        if pair_fwd in known or pair_rev in known:
            tag = "KNOWN"
        elif any(k in e['source'] or k in e['target']
                 for k in ('gdp', 'inflation', 'interest', 'exchange', 'trade',
                           'export', 'import', 'money', 'govt', 'fiscal')):
            tag = "PLAUSIBLE"
        else:
            tag = "NOVEL"
        tagged.append({**e, 'plausibility': tag})
    return tagged


# ── Phase 1: Hypothesis pool summary ─────────────────────────────────────────

def phase1_pool_summary(engine, label="") -> dict:
    from scarcity.engine.discovery import HypothesisState
    counts = defaultdict(int)
    type_counts = defaultdict(int)
    for h in engine.hypotheses.population.values():
        counts[h.meta.state.value] += 1
        type_counts[h.rel_type.value] += 1

    total = sum(counts.values())
    print(f"\n  Hypothesis Pool {label}:")
    print(f"    Total: {total}")
    for state, n in sorted(counts.items()):
        print(f"    {state:<12}: {n:>4}")
    print(f"  Types:")
    for t, n in sorted(type_counts.items()):
        print(f"    {t:<20}: {n:>4}")
    return dict(counts)


# ── Phase 2: Rolling-origin backtest with engine graph ────────────────────────

def phase2_rolling_backtest(kenya_df: pd.DataFrame, targets: list,
                             conf_threshold: float, min_evidence: int,
                             initial_train_years: int = 15,
                             small_dataset_mode: bool = True) -> tuple:
    """
    Incremental rolling-origin backtest.

    For each test year T:
      - Engine has been trained on all years < T (incremental, no re-init)
      - Extract graph at T
      - Forecast year T using graph features + RidgeCV on train data
      - Stream year T into engine
    """
    years = sorted(kenya_df.index.tolist())
    if len(years) <= initial_train_years:
        raise ValueError("Not enough data for rolling backtest.")

    engine = build_engine(list(kenya_df.columns), small_dataset_mode=small_dataset_mode)

    # Stream initial training window
    print(f"\n  Streaming {initial_train_years} initial training years ...", flush=True)
    stream_dataframe(engine, kenya_df.iloc[:initial_train_years],
                     label=f"init({initial_train_years}yr)")

    results = []
    edge_snapshots = {}

    test_years = years[initial_train_years:]
    for test_year in test_years:
        # 1. Extract graph BEFORE streaming test year
        graph, edges = extract_graph(engine, conf_threshold=conf_threshold,
                                     min_evidence=min_evidence)
        edge_snapshots[test_year] = edges

        n_edges = sum(len(v) for v in graph.values())
        print(f"  Year {test_year}: {n_edges} graph edges, "
              f"{len(graph)} target vars with parents", flush=True)

        train_data = kenya_df[kenya_df.index < test_year]
        test_data  = kenya_df[kenya_df.index == test_year]

        for target in targets:
            if target not in kenya_df.columns:
                continue
            ev = ForecastingEvaluator(target_variable=target, horizon=1)

            pers   = ev.evaluate_persistence(train_data, test_data)
            arima  = ev.evaluate_arima(train_data, test_data)
            var    = ev.evaluate_var(train_data, test_data)
            proph  = ev.evaluate_prophet(train_data, test_data)
            scarcity = ev.evaluate_scarcity_graph(train_data, test_data, graph)

            results.append({
                'year':           test_year,
                'target':         target,
                'n_parents':      len(graph.get(target, [])),
                'parents':        graph.get(target, []),
                'persistence_mae': pers['mae'],
                'arima_mae':      arima['mae'],
                'var_mae':        var['mae'],
                'prophet_mae':    proph['mae'],
                'scarcity_mae':   scarcity['mae'],
                'persistence_dir': pers['dir_acc'],
                'arima_dir':      arima['dir_acc'],
                'var_dir':        var['dir_acc'],
                'prophet_dir':    proph['dir_acc'],
                'scarcity_dir':   scarcity['dir_acc'],
            })

        # 2. Stream test year into engine (temporal integrity maintained)
        engine.process_row(kenya_df.loc[test_year].to_dict())

    return results, edge_snapshots, engine


# ── Phase 3: Anomaly detection with engine graph ──────────────────────────────

def phase3_anomaly_detection(kenya_df: pd.DataFrame, graph: dict,
                              targets: list) -> dict:
    """
    Build graph predictions using Ridge on full data, then detect anomalies
    as large residuals.  Inject synthetic shocks to measure precision/recall.
    """
    from sklearn.linear_model import Ridge

    pred_cols = {}
    for target, parents in graph.items():
        valid_parents = [p for p in parents if p in kenya_df.columns]
        if not valid_parents or target not in kenya_df.columns:
            continue
        sub = kenya_df[[target] + valid_parents].dropna()
        if len(sub) < 5:
            continue
        X = sub[valid_parents].values[:-1]
        y = sub[target].values[1:]
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        preds = pd.Series(index=sub.index[1:], data=model.predict(X),
                          name=target)
        pred_cols[target] = preds

    if not pred_cols:
        return {'error': 'No graph predictions available (graph too sparse)'}

    graph_preds = pd.DataFrame(pred_cols)
    aligned_data = kenya_df.reindex(graph_preds.index)[graph_preds.columns]

    # Inject synthetic anomalies
    rng = np.random.RandomState(42)
    anomaly_rate = 0.10   # higher rate for small dataset
    n_anom = max(1, int(anomaly_rate * len(aligned_data)))
    mask = pd.DataFrame(False, index=aligned_data.index, columns=aligned_data.columns)
    data_anom = aligned_data.copy()
    for col in aligned_data.columns:
        idx = rng.choice(len(aligned_data), n_anom, replace=False)
        std = float(aligned_data[col].std()) or 1.0
        data_anom.iloc[idx, aligned_data.columns.get_loc(col)] += (
            rng.choice([-1, 1], n_anom) * 4.0 * std)
        mask.iloc[idx, mask.columns.get_loc(col)] = True

    ev = AnomalyDetectionEvaluator(data_anom, mask)
    return {
        'zscore':            ev.evaluate_zscore(threshold=2.5),
        'isolation_forest':  ev.evaluate_isolation_forest(),
        'scarcity_residuals': ev.evaluate_scarcity_residuals(graph_preds, threshold=2.5),
        'n_graph_vars':      len(pred_cols),
        'anomaly_rate':      anomaly_rate,
    }


# ── Phase 4: Underperformance diagnosis ──────────────────────────────────────

def phase4_diagnose(results: list, edge_snapshots: dict,
                    targets: list, conf_threshold: float) -> None:
    print("\n" + "=" * 70)
    print("UNDERPERFORMANCE DIAGNOSIS")
    print("=" * 70)

    df = pd.DataFrame(results)
    for target in targets:
        sub = df[df['target'] == target].copy()
        if sub.empty:
            continue
        s_mae  = mean_or_nan(sub['scarcity_mae'].tolist())
        a_mae  = mean_or_nan(sub['arima_mae'].tolist())
        p_mae  = mean_or_nan(sub['persistence_mae'].tolist())
        valid  = [x for x in [a_mae, p_mae] if x == x]
        best   = min(valid) if valid else float('nan')

        underperforms = s_mae == s_mae and best == best and s_mae > best * 1.10
        marker = " <-- UNDERPERFORMS" if underperforms else ""
        print(f"\n  {target}:")
        print(f"    Scarcity MAE={fmt(s_mae)}  ARIMA={fmt(a_mae)}  "
              f"Persistence={fmt(p_mae)}{marker}")

        if underperforms:
            # Show what parents were discovered for this target
            all_parents = defaultdict(int)
            for year, edges in edge_snapshots.items():
                for e in edges:
                    if e['target'] == target:
                        all_parents[e['source']] += 1

            if all_parents:
                print(f"    Discovered parents (frequency across years):")
                for src, freq in sorted(all_parents.items(), key=lambda x: -x[1]):
                    print(f"      {src:<30} appeared in {freq}/{len(edge_snapshots)} years")
            else:
                print(f"    No parents discovered for {target} at "
                      f"conf_threshold={conf_threshold}.")
                print(f"    → Consider lowering --conf or increasing data.")

            # Check n_parents per year
            parents_per_year = sub[['year', 'n_parents']].set_index('year')
            sparse_years = parents_per_year[parents_per_year['n_parents'] == 0]
            if len(sparse_years) > 0:
                pct = 100 * len(sparse_years) / len(parents_per_year)
                print(f"    Graph was empty for {len(sparse_years)}/{len(parents_per_year)} "
                      f"test years ({pct:.0f}%) — falls back to persistence")


# ── Reporting ─────────────────────────────────────────────────────────────────

def build_engine_report_section(results: list, anomaly: dict,
                                 final_edges: list) -> dict:
    """Build a dict that can be merged into benchmark_data.json."""
    df = pd.DataFrame(results)
    targets = df['target'].unique().tolist()

    summary_rows = []
    for target in targets:
        sub = df[df['target'] == target]
        for method in ('persistence', 'arima', 'var', 'prophet', 'scarcity'):
            mae_col = f'{method}_mae'
            dir_col = f'{method}_dir'
            summary_rows.append({
                'target': target,
                'method': method.upper(),
                'mae':    mean_or_nan(sub[mae_col].tolist() if mae_col in sub else []),
                'dir_acc': mean_or_nan(sub[dir_col].tolist() if dir_col in sub else []),
            })

    # Tag edge plausibility
    tagged_edges = check_plausibility(final_edges)

    return {
        'engine_backtest': summary_rows,
        'engine_anomaly':  anomaly,
        'engine_edges':    tagged_edges,
        'n_total_edges':   len(final_edges),
    }


def print_results_table(results: list) -> None:
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("ROLLING BACKTEST RESULTS (mean MAE across all test years)")
    print("=" * 70)
    for target in df['target'].unique():
        sub = df[df['target'] == target]
        print(f"\n  {target}:")
        for method in ('persistence', 'arima', 'var', 'prophet', 'scarcity'):
            mae = mean_or_nan(sub[f'{method}_mae'].tolist())
            dir_acc = mean_or_nan(sub[f'{method}_dir'].tolist())
            marker = " <-- BEST" if mae == mae and all(
                mae <= mean_or_nan(sub[f'{m}_mae'].tolist())
                for m in ('persistence', 'arima', 'var', 'prophet', 'scarcity')
                if m != method
            ) else ""
            print(f"    {method.upper():<12}  MAE={fmt(mae)}  Dir={fmt(dir_acc, 3)}{marker}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",         type=float, default=0.45,
                        help="Confidence threshold for graph extraction")
    parser.add_argument("--min_evidence", type=int,   default=5,
                        help="Minimum evidence count for graph extraction")
    parser.add_argument("--train_years",  type=int,   default=15,
                        help="Initial training window (years)")
    parser.add_argument("--countries",    nargs="+",  default=["KEN"],
                        help="Countries to run backtest on")
    parser.add_argument("--small_dataset_mode", action="store_true", default=True,
                        help="Use small-dataset tuned MetaController (lower min_evidence, λ=0.94)")
    parser.add_argument("--no_small_dataset_mode", dest="small_dataset_mode",
                        action="store_false",
                        help="Disable small-dataset mode (use default thresholds)")
    args = parser.parse_args()

    print("=" * 70)
    print("Scarcity Engine Benchmark")
    print(f"  conf_threshold={args.conf}  min_evidence={args.min_evidence}")
    print(f"  train_years={args.train_years}  countries={args.countries}")
    print(f"  small_dataset_mode={args.small_dataset_mode}")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/5] Loading macroeconomic data ...", flush=True)
    data_dict = prepare_multi_country_data(args.countries)

    all_results = []
    all_edge_snapshots = {}
    final_engine = None
    final_kenya_df = None

    for country, df in data_dict.items():
        print(f"\n  {country}: {df.shape[0]} years × {df.shape[1]} variables")
        targets = [t for t in ['gdp_growth', 'inflation_cpi'] if t in df.columns]
        if not targets:
            targets = [df.columns[0]]
        print(f"  Targets: {targets}")

        # ── Phase 1: Pool summary after initial stream ──
        print(f"\n[2/5] Initialising engine and streaming training data ...", flush=True)
        engine_preview = build_engine(list(df.columns),
                                      small_dataset_mode=args.small_dataset_mode)
        stream_dataframe(engine_preview, df.iloc[:args.train_years], label="preview")
        phase1_pool_summary(engine_preview, label=f"after {args.train_years} years")

        # ── Phase 2: Rolling backtest ──
        print(f"\n[3/5] Rolling-origin backtest ({country}) ...", flush=True)
        t0 = time.time()
        results, edge_snapshots, engine = phase2_rolling_backtest(
            df, targets,
            conf_threshold=args.conf,
            min_evidence=args.min_evidence,
            initial_train_years=args.train_years,
            small_dataset_mode=args.small_dataset_mode,
        )
        print(f"  Backtest complete in {time.time()-t0:.1f}s ({len(results)} evaluations)")

        all_results.extend(results)
        all_edge_snapshots.update(edge_snapshots)
        final_engine = engine
        final_kenya_df = df

    # ── Extract final graph ──
    print(f"\n[4/5] Extracting final discovered graph ...", flush=True)
    final_graph, final_edges = extract_graph(
        final_engine,
        conf_threshold=args.conf,
        min_evidence=args.min_evidence,
    )
    final_edges_tagged = sorted(
        check_plausibility(final_edges), key=lambda x: -x['confidence']
    )
    print(f"  {graph_summary(final_graph, final_edges)}")

    print("\n  --- Top Discovered Edges ---")
    inspect_edges(final_edges_tagged, top_n=30)

    # Plausibility breakdown
    for tag in ('KNOWN', 'PLAUSIBLE', 'NOVEL'):
        n = sum(1 for e in final_edges_tagged if e.get('plausibility') == tag)
        print(f"  {tag}: {n} edges")

    # ── Print results table ──
    print_results_table(all_results)

    # ── Phase 3: Anomaly detection ──
    print(f"\n[5/5] Anomaly detection with engine graph ...", flush=True)
    targets_all = [t for t in ['gdp_growth', 'inflation_cpi']
                   if final_kenya_df is not None and t in final_kenya_df.columns]
    anomaly_results = {}
    if final_kenya_df is not None:
        anomaly_results = phase3_anomaly_detection(
            final_kenya_df, final_graph, targets_all)
        z = anomaly_results.get('zscore', {})
        sr = anomaly_results.get('scarcity_residuals', {})
        print(f"  Z-Score:            P={fmt(z.get('precision'))}  "
              f"R={fmt(z.get('recall'))}  F1={fmt(z.get('f1'))}")
        print(f"  Scarcity Residuals: P={fmt(sr.get('precision'))}  "
              f"R={fmt(sr.get('recall'))}  F1={fmt(sr.get('f1'))}")

    # ── Underperformance diagnosis ──
    phase4_diagnose(all_results, all_edge_snapshots, targets_all, args.conf)

    # ── Merge and update report ──
    print(f"\n[Report] Updating benchmark report ...", flush=True)
    with open(DATA_JSON) as f:
        loaded = json.load(f)
    base = loaded[-1] if isinstance(loaded, list) else loaded

    engine_section = build_engine_report_section(all_results, anomaly_results,
                                                  final_edges_tagged)
    base['engine_driven'] = engine_section

    # Rebuild real_world_backtest with engine rows included
    if all_results:
        engine_rows = []
        df_res = pd.DataFrame(all_results)
        for target in df_res['target'].unique():
            sub = df_res[df_res['target'] == target]
            for method in ('persistence', 'arima', 'var', 'prophet', 'scarcity'):
                engine_rows.append({
                    'country': 'KEN',
                    'target': target,
                    'method': method.upper(),
                    'mae': mean_or_nan(sub[f'{method}_mae'].tolist()),
                    'dir_acc': mean_or_nan(sub[f'{method}_dir'].tolist()),
                })
        base['real_world_backtest'] = engine_rows

    from benchmark.synthetic.reporting import generate_report
    rpt, dat = generate_report(base, out_dir=REPORT_OUT)

    print(f"\n{'=' * 70}")
    print(f"Report:  {rpt}")
    print(f"Data:    {dat}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

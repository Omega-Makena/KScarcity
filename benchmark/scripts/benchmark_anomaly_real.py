"""
Anomaly Detection on Real Data: Kenya World Bank (N=34, 1990-2023)

Tests whether Scarcity's structural knowledge graph improves anomaly detection
on real macroeconomic data with limited observations (N=34).

This is a harder test than the synthetic benchmark (N=300):
  - Shorter series: graph discovery is more uncertain
  - No ground truth causal structure: discovered edges may be spurious or incomplete
  - Real noise: economic relationships are weak signals in annual macro data
  - TYPE_2 detection depends on whether the right edge was actually discovered

Anomaly injection on real Kenya data:
  TYPE_1a  -- GDP growth spike at 1997 (+4 sigma) [univariate, all methods should catch]
  TYPE_1b  -- Inflation spike at 2005 (+4 sigma) [univariate, all methods should catch]
  TYPE_2a  -- Relationship break: top discovered edge; parent spikes 2002, child flat 2003
  TYPE_2b  -- Relationship break: 2nd edge; parent spikes 2012, child flat 2013

Methods compared:
  zscore           -- per-column Z-score (baseline)
  isof_blind       -- IsolationForest on raw variables (blind)
  rrcf_blind       -- Production RRCF (window=10, recalibrated threshold)
  graph_residuals  -- lag-1 Ridge residuals using discovered graph
  isof_graph       -- IsolationForest on graph-residual space
  rrcf_graph       -- Production RRCF on graph-residual space

Usage:
    python benchmark/scripts/benchmark_anomaly_real.py
    python benchmark/scripts/benchmark_anomaly_real.py --conf 0.30 --min-evidence 3
"""

import argparse
import io
import sys
import warnings
from pathlib import Path

# Force UTF-8 on Windows consoles that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.graph_extractor import extract_graph, graph_summary, inspect_edges
from benchmark.real_data.world_bank_loader import prepare_multi_country_data
from benchmark.evaluation.anomaly_detection import AnomalyDetectionEvaluator

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# RRCF: recalibrated for N=34 (production uses window=256, threshold=6.0)
RRCF_WINDOW = 10
RRCF_TREES = 50
RRCF_THRESHOLD = 3.0

ZSCORE_THRESHOLD = 3.0
GRAPH_THRESHOLD = 2.5    # slightly lower than 3.0 — N=34 std estimates are noisy

# Injection design
TYPE1A_COL, TYPE1A_YEAR = 'gdp_growth', 1997
TYPE1B_COL, TYPE1B_YEAR = 'inflation_cpi', 2005
TYPE2A_PARENT_YR, TYPE2A_CHILD_YR = 2002, 2003
TYPE2B_PARENT_YR, TYPE2B_CHILD_YR = 2012, 2013
TYPE2_SPIKE_SIGMA = 3.0

# Known economic edges used as fallback if discovery is too sparse
KNOWN_FALLBACKS = [
    ('broad_money',    'inflation_cpi'),
    ('gcf',            'gdp_growth'),
    ('exports_gdp',    'gdp_growth'),
    ('inflation_cpi',  'real_interest_rate'),
    ('govt_consumption', 'gdp_growth'),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_clean() -> pd.DataFrame:
    """Load Kenya World Bank data and fill NaN with ffill -> bfill -> column mean."""
    print("Loading Kenya World Bank data (1990-2023)...")
    country_data = prepare_multi_country_data(['KEN'])
    df = country_data['KEN'].copy()

    # ffill -> bfill covers most structural gaps at series boundaries
    df = df.ffill().bfill()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    n_nan = df.isnull().sum().sum()
    print(f"  Shape: {df.shape}  |  NaN remaining after fill: {n_nan}")
    return df


# ---------------------------------------------------------------------------
# Graph discovery
# ---------------------------------------------------------------------------

def discover_graph(df_clean: pd.DataFrame,
                   conf_threshold: float = 0.30,
                   min_evidence: int = 3):
    """
    Run OnlineDiscoveryEngine on clean Kenya data.

    Uses lenient thresholds (conf=0.30, evidence=3) because N=34 limits
    how much statistical evidence each hypothesis can accumulate.

    Returns:
        graph  -- Dict[str, List[str]]: target -> [parents]
        edges  -- List[Dict]: full edge metadata sorted by confidence
    """
    var_names = df_clean.columns.tolist()
    print(f"\nDiscovering graph: {len(df_clean)} rows x {len(var_names)} variables "
          f"(conf>={conf_threshold}, min_evidence>={min_evidence})...")

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    schema = {'fields': [{'name': v} for v in var_names]}
    engine.initialize_v2(schema, use_causal=True)

    for yr, row in df_clean.iterrows():
        row_dict = {k: float(v) for k, v in row.items() if pd.notna(v)}
        engine.process_row(row_dict)

    graph, edges = extract_graph(engine,
                                 conf_threshold=conf_threshold,
                                 min_evidence=min_evidence)

    summary = graph_summary(graph, edges)
    print(f"  {summary}")

    if edges:
        print("\n  Top discovered edges (by confidence):")
        inspect_edges(edges, top_n=10)

    return graph, edges


# ---------------------------------------------------------------------------
# Edge selection for TYPE_2 injection
# ---------------------------------------------------------------------------

def find_injection_edges(edges, df_clean: pd.DataFrame, n_edges: int = 2):
    """
    Select up to n_edges high-confidence discovered edges for TYPE_2 injection.
    Falls back to KNOWN_FALLBACKS if discovery is too sparse.

    Returns list of (parent, child) tuples.
    """
    cols = set(df_clean.columns)

    # From discovered edges: filter valid pairs, sort by confidence
    valid = []
    for e in sorted(edges, key=lambda x: -x['confidence']):
        src, tgt = e['source'], e['target']
        if src in cols and tgt in cols and src != tgt:
            valid.append((src, tgt))

    selected = valid[:n_edges]

    # Fill from known fallbacks if needed
    for p, c in KNOWN_FALLBACKS:
        if len(selected) >= n_edges:
            break
        if p in cols and c in cols and (p, c) not in selected:
            selected.append((p, c))

    return selected[:n_edges]


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------

def inject_anomalies(df_clean: pd.DataFrame, type2_edges):
    """
    Inject labeled anomalies into a copy of df_clean.

    Returns:
        df_anom  -- DataFrame with injected values
        mask     -- bool DataFrame (True = injected anomaly)
        notes    -- human-readable descriptions of each injection
    """
    df = df_clean.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    notes = []

    def _inject_type1(col, year, sigma=4.0):
        if year in df.index and col in df.columns:
            mu, s = df[col].mean(), df[col].std()
            injected = mu + sigma * s
            df.loc[year, col] = injected
            mask.loc[year, col] = True
            notes.append(f"TYPE_1  {col:25s} spike at {year}: "
                         f"{injected:.2f} (+{sigma:.0f}sigma, mean={mu:.2f})")

    def _inject_type2(parent, child, par_year, chi_year):
        if par_year not in df.index or chi_year not in df.index:
            return
        if parent not in df.columns or child not in df.columns:
            return
        mu_p, s_p = df[parent].mean(), df[parent].std()
        mu_c = df[child].mean()
        # parent spikes at par_year
        df.loc[par_year, parent] = mu_p + TYPE2_SPIKE_SIGMA * s_p
        # child flat-lines at chi_year (relationship break)
        df.loc[chi_year, child] = mu_c
        mask.loc[chi_year, child] = True
        notes.append(f"TYPE_2  {parent:25s} -> {child:25s}: "
                     f"parent +{TYPE2_SPIKE_SIGMA:.0f}sig at {par_year}, "
                     f"child forced to mean {mu_c:.2f} at {chi_year}")

    _inject_type1(TYPE1A_COL, TYPE1A_YEAR)
    _inject_type1(TYPE1B_COL, TYPE1B_YEAR)

    if len(type2_edges) >= 1:
        p, c = type2_edges[0]
        _inject_type2(p, c, TYPE2A_PARENT_YR, TYPE2A_CHILD_YR)

    if len(type2_edges) >= 2:
        p, c = type2_edges[1]
        _inject_type2(p, c, TYPE2B_PARENT_YR, TYPE2B_CHILD_YR)

    return df, mask, notes


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(df_anom: pd.DataFrame, mask: pd.DataFrame, graph):
    """Evaluate all methods on anomaly-injected real data."""
    ev = AnomalyDetectionEvaluator(df_anom, mask)
    results = {}

    print("\nRunning evaluators...")

    print("  [1/6] Z-score (blind, threshold=3.0)...")
    results['zscore'] = ev.evaluate_zscore(threshold=ZSCORE_THRESHOLD)

    print("  [2/6] IsolationForest (blind)...")
    results['isof_blind'] = ev.evaluate_isolation_forest()

    print(f"  [3/6] RRCF blind (window={RRCF_WINDOW}, threshold={RRCF_THRESHOLD})...")
    results['rrcf_blind'] = ev.evaluate_rrcf_engine(
        window_size=RRCF_WINDOW, num_trees=RRCF_TREES, threshold=RRCF_THRESHOLD)

    print(f"  [4/6] GraphResiduals (discovered graph, threshold={GRAPH_THRESHOLD})...")
    results['graph_residuals'] = ev.evaluate_scarcity_graph_anomaly(
        graph, threshold=GRAPH_THRESHOLD)

    print("  [5/6] IsoForest on graph-residual space...")
    results['isof_graph'] = ev.evaluate_rrcf_graph_conditioned(
        graph, contamination=0.05)

    print(f"  [6/6] RRCF on graph-residual space (window={RRCF_WINDOW})...")
    results['rrcf_graph'] = ev.evaluate_rrcf_graph_conditioned_engine(
        graph, window_size=RRCF_WINDOW, num_trees=RRCF_TREES, threshold=RRCF_THRESHOLD)

    return results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _fmt(v, d=4):
    if v is None or (isinstance(v, float) and v != v):
        return ' N/A  '
    return f'{float(v):.{d}f}'


def print_results(results, notes, type2_edges, n_total, n_anomalies):
    METHOD_LABELS = {
        'zscore':          'Z-score (blind)',
        'isof_blind':      'IsoForest (blind)',
        'rrcf_blind':      'RRCF (blind)',
        'graph_residuals': 'GraphResiduals (disc.)',
        'isof_graph':      'IsoForest+Graph (disc.)',
        'rrcf_graph':      'RRCF+Graph (disc.)',
    }
    GRAPH_METHODS = {'graph_residuals', 'isof_graph', 'rrcf_graph'}

    width = 78
    print("\n" + "=" * width)
    print("ANOMALY DETECTION: REAL KENYA DATA (N=34 years, 19 variables, 1990-2023)")
    print("=" * width)
    print(f"Total cells: {n_total}  |  Injected anomalies: {n_anomalies}")

    print("\nInjected anomalies:")
    for note in notes:
        print(f"  {note}")

    if type2_edges:
        print("\nEdges selected for TYPE_2 injection:")
        for i, (p, c) in enumerate(type2_edges, 1):
            source = 'discovered' if i <= len(type2_edges) else 'fallback'
            print(f"  {i}. {p} -> {c}  [{source}]")

    hdr = f"{'Method':<24}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'FPR':>7}  {'TP':>3}  {'FP':>3}  {'FN':>3}"
    sep = "-" * len(hdr)
    print(f"\n{hdr}")
    print(sep)

    for key, label in METHOD_LABELS.items():
        m = results.get(key, {})
        prec = _fmt(m.get('precision'))
        rec  = _fmt(m.get('recall'))
        f1   = _fmt(m.get('f1'))
        fpr  = _fmt(m.get('fpr'))
        tp   = m.get('tp', '-')
        fp   = m.get('fp', '-')
        fn   = m.get('fn', '-')
        tag  = " <-- graph" if key in GRAPH_METHODS else ""
        print(f"{label:<24}  {prec}  {rec}  {f1}  {fpr}  {tp:>3}  {fp:>3}  {fn:>3}{tag}")

    print(sep)

    # Summary
    gr_f1  = results.get('graph_residuals', {}).get('f1', 0) or 0
    zs_f1  = results.get('zscore', {}).get('f1', 0) or 0
    delta  = gr_f1 - zs_f1
    gr_fpr = results.get('graph_residuals', {}).get('fpr', 0) or 0
    zs_fpr = results.get('zscore', {}).get('fpr', 0) or 0
    gr_rec = results.get('graph_residuals', {}).get('recall', 0) or 0
    zs_rec = results.get('zscore', {}).get('recall', 0) or 0

    best_key = max(METHOD_LABELS, key=lambda k: results.get(k, {}).get('f1', 0) or 0)
    best_f1  = (results.get(best_key, {}).get('f1', 0) or 0)

    print(f"\nBest overall: {METHOD_LABELS[best_key]}  F1={best_f1:.4f}")
    print(f"GraphResiduals vs Z-score: F1 {'+' if delta >= 0 else ''}{delta:.4f}")
    print(f"  GraphResiduals  Prec={_fmt(results.get('graph_residuals',{}).get('precision'))}  "
          f"Rec={gr_rec:.4f}  FPR={gr_fpr:.4f}")
    print(f"  Z-score         Prec={_fmt(results.get('zscore',{}).get('precision'))}  "
          f"Rec={zs_rec:.4f}  FPR={zs_fpr:.4f}")

    print("\nINTERPRETATION")
    print("-" * 40)
    if delta > 0.05:
        print("  Graph-conditioning IMPROVES detection on real data (N=34).")
        print("  Structural knowledge helps even with a short annual series.")
    elif delta < -0.05:
        print("  Graph-conditioning HURTS at N=34: discovered graph too sparse/noisy.")
        print("  Real macro data at this length limits relationship discovery quality.")
    else:
        print("  Graph-conditioning is NEUTRAL at N=34 (F1 within +/-0.05 of Z-score).")
        print("  Short real series limits TYPE_2 detectability via graph residuals.")

    print()
    print("  NOTE: TYPE_2 anomalies are only detectable if the parent->child edge")
    print("  was discovered at conf>=0.30. If not found, GraphResiduals falls back")
    print("  to Z-score for that variable -- TYPE_2 is then invisible to all methods.")
    print("  RRCF threshold=3.0 is recalibrated for window=10 (production uses 6.0")
    print("  for 256-pt windows); both are fixed thresholds, not adaptive.")
    print("=" * width)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Real-data anomaly detection benchmark (Kenya WB N=34)')
    parser.add_argument('--conf', type=float, default=0.30,
                        help='Minimum confidence for graph extraction (default 0.30)')
    parser.add_argument('--min-evidence', type=int, default=3,
                        help='Minimum evidence for graph extraction (default 3)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. Load real data
    df_clean = load_and_clean()

    # 2. Discover graph on clean data (do NOT use anomaly-contaminated data for discovery)
    graph, edges = discover_graph(df_clean,
                                  conf_threshold=args.conf,
                                  min_evidence=args.min_evidence)

    # 3. Select edges for TYPE_2 injection
    type2_edges = find_injection_edges(edges, df_clean, n_edges=2)
    print(f"\nEdges selected for TYPE_2 injection:")
    for i, (p, c) in enumerate(type2_edges, 1):
        print(f"  {i}. {p} -> {c}")

    # 4. Inject anomalies into a COPY of the data
    df_anom, mask, notes = inject_anomalies(df_clean, type2_edges)
    n_total     = mask.size
    n_anomalies = int(mask.values.sum())
    print(f"\nInjected {n_anomalies} anomaly cells across {n_total} total:")
    for note in notes:
        print(f"  {note}")

    # 5. Run all evaluators
    results = run_benchmark(df_anom, mask, graph)

    # 6. Print results table
    print_results(results, notes, type2_edges, n_total, n_anomalies)


if __name__ == '__main__':
    main()

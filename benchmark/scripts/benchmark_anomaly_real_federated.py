"""
Federated Anomaly Detection on Real Data — KEN + TZA + UGA (N_eff=102)

Tests whether pooling East Africa observations (KEN + TZA + UGA, 34 years each)
gives the discovery engine enough evidence to improve anomaly detection on
Kenya-specific data, compared to single-country (N=34) and synthetic (N=300).

Federation design:
  - Discovery engine trains on KEN + TZA + UGA simultaneously.
  - For each calendar year, the Kenya row is streamed first, then TZA, then UGA
    (same pattern as run_scarcity_federation.py and the §42 benchmark).
  - N_eff = 34 years x 3 countries = 102 effective observations.
  - Anomalies are injected into Kenya data ONLY.
  - Evaluators use the federated graph (discovered from all 3 countries) on
    Kenya data (the evaluation dataset).

Expected improvement over §44 (single-country N=34):
  The §42 federation benchmark showed federated graph discovers:
    - 198 edges (vs 114 single-country)
    - 13 KNOWN economic relationships (vs 0)
    - Mean conf 0.735 (vs 0.574)
  With better, stable economic edges (GDP/inflation/money) rather than just
  trend correlations (internet/mobile), TYPE_2 relationship breaks based on
  genuine macro relationships should be more cleanly detectable.

Usage:
    python benchmark/scripts/benchmark_anomaly_real_federated.py
    python benchmark/scripts/benchmark_anomaly_real_federated.py --countries KEN TZA UGA ETH
    python benchmark/scripts/benchmark_anomaly_real_federated.py --conf 0.35 --min-evidence 5
"""

import argparse
import io
import sys
import warnings
from pathlib import Path

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

DEFAULT_COUNTRIES = ['KEN', 'TZA', 'UGA']

# RRCF: same recalibrated params as single-country benchmark
RRCF_WINDOW = 10
RRCF_TREES = 50
RRCF_THRESHOLD = 3.0

ZSCORE_THRESHOLD = 3.0
GRAPH_THRESHOLD = 2.5

# Injection design — same years as §44 for apples-to-apples comparison
TYPE1A_COL, TYPE1A_YEAR = 'gdp_growth', 1997
TYPE1B_COL, TYPE1B_YEAR = 'inflation_cpi', 2005
TYPE2A_PARENT_YR, TYPE2A_CHILD_YR = 2002, 2003
TYPE2B_PARENT_YR, TYPE2B_CHILD_YR = 2012, 2013
TYPE2_SPIKE_SIGMA = 3.0

# Known economic edges as preferred injection targets and fallback
KNOWN_EDGES = [
    ('broad_money',      'inflation_cpi'),
    ('gcf',              'gdp_growth'),
    ('exports_gdp',      'gdp_growth'),
    ('inflation_cpi',    'real_interest_rate'),
    ('govt_consumption', 'gdp_growth'),
    ('imports_gdp',      'gdp_growth'),
    ('private_credit',   'gdp_growth'),
]

# §44 single-country reference numbers (for comparison table)
SINGLE_COUNTRY_REF = {
    'zscore':          {'f1': 0.4444, 'fpr': 0.0047, 'prec': 0.4000, 'rec': 0.5000},
    'isof_blind':      {'f1': 0.0000, 'fpr': 0.0592, 'prec': 0.0000, 'rec': 0.0000},
    'rrcf_blind':      {'f1': 0.0130, 'fpr': 0.7056, 'prec': 0.0066, 'rec': 0.7500},
    'graph_residuals': {'f1': 0.1905, 'fpr': 0.0234, 'prec': 0.1176, 'rec': 0.5000},
    'isof_graph':      {'f1': 0.0000, 'fpr': 0.0592, 'prec': 0.0000, 'rec': 0.0000},
    'rrcf_graph':      {'f1': 0.0130, 'fpr': 0.7056, 'prec': 0.0066, 'rec': 0.7500},
}


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_and_clean(countries):
    """
    Load World Bank data for all countries. KEN from local CSV; TZA/UGA from API cache.
    Returns {country_code: cleaned DataFrame}.
    """
    print(f"Loading World Bank data for: {', '.join(countries)}...")
    country_data = prepare_multi_country_data(countries)

    cleaned = {}
    for cc, df in country_data.items():
        df = df.ffill().bfill()
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        n_nan = df.isnull().sum().sum()
        cleaned[cc] = df
        print(f"  {cc}: shape={df.shape}  NaN after fill={n_nan}")

    return cleaned


# ---------------------------------------------------------------------------
# Federated graph discovery
# ---------------------------------------------------------------------------

def discover_federated_graph(country_dfs, conf_threshold, min_evidence):
    """
    Stream all countries for each calendar year into a single engine.

    For each year: Kenya row first, then auxiliary countries (TZA, UGA, ...).
    This matches the §42 federation benchmark streaming order.

    Returns:
        engine   -- trained OnlineDiscoveryEngine
        graph    -- Dict[str, List[str]]: target -> [parents]
        edges    -- flat edge list with metadata
        n_eff    -- effective number of observations streamed
    """
    ken_df = country_dfs['KEN']
    aux_dfs = {cc: df for cc, df in country_dfs.items() if cc != 'KEN'}

    var_names = sorted(ken_df.columns.tolist())
    years = sorted(ken_df.index.tolist())
    n_countries = 1 + len(aux_dfs)

    print(f"\nFederated graph discovery: {len(years)} years x {n_countries} countries "
          f"= {len(years) * n_countries} effective observations")
    print(f"Variables: {len(var_names)}  |  conf>={conf_threshold}  min_evidence>={min_evidence}")

    engine = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    schema = {'fields': [{'name': v} for v in var_names]}
    engine.initialize_v2(schema, use_causal=True)

    n_eff = 0
    for yr in years:
        # Kenya row
        row_ken = {k: float(v) for k, v in ken_df.loc[yr].items() if pd.notna(v)}
        engine.process_row(row_ken)
        n_eff += 1

        # Auxiliary countries (same calendar year, same variables)
        for cc, aux_df in aux_dfs.items():
            if yr in aux_df.index:
                row_aux = aux_df.loc[yr].reindex(var_names)
                row_dict = {k: float(v) for k, v in row_aux.items() if pd.notna(v)}
                if row_dict:
                    engine.process_row(row_dict)
                    n_eff += 1

    graph, edges = extract_graph(engine,
                                 conf_threshold=conf_threshold,
                                 min_evidence=min_evidence)

    summary = graph_summary(graph, edges)
    known_set = set(KNOWN_EDGES)
    n_known_pairs = sum(
        1 for (p, c) in known_set
        if any(e['source'] == p and e['target'] == c for e in edges)
    )

    print(f"  {summary}")
    print(f"  Effective observations: {n_eff}")
    print(f"  KNOWN economic edge pairs recovered: {n_known_pairs} of {len(KNOWN_EDGES)}")

    if edges:
        print("\n  Top discovered edges (by confidence):")
        inspect_edges(edges, top_n=12)

    return engine, graph, edges, n_eff


# ---------------------------------------------------------------------------
# Edge selection for TYPE_2 injection
# ---------------------------------------------------------------------------

def find_injection_edges(edges, df_clean, n_edges=2):
    """
    Select edges for TYPE_2 injection, preferring KNOWN economic relationships.

    Priority order:
      1. KNOWN economic edges that were discovered (stable, stationary variables)
      2. Other high-confidence discovered edges
      3. KNOWN edges used as fallback even if not discovered

    Returns list of (parent, child) tuples.
    """
    cols = set(df_clean.columns)
    known_set = set(KNOWN_EDGES)

    # Score edges: KNOWN discovered first, then by confidence
    def priority(e):
        pair = (e['source'], e['target'])
        rev  = (e['target'], e['source'])
        is_known = (pair in known_set or rev in known_set)
        return (int(is_known), e['confidence'])

    valid = []
    seen = set()
    for e in sorted(edges, key=priority, reverse=True):
        src, tgt = e['source'], e['target']
        if src in cols and tgt in cols and src != tgt and (src, tgt) not in seen:
            valid.append((src, tgt))
            seen.add((src, tgt))

    selected = valid[:n_edges]

    # Fallback: known edges not yet in list
    for p, c in KNOWN_EDGES:
        if len(selected) >= n_edges:
            break
        if p in cols and c in cols and (p, c) not in selected:
            selected.append((p, c))

    return selected[:n_edges]


# ---------------------------------------------------------------------------
# Anomaly injection (same design as §44 for comparability)
# ---------------------------------------------------------------------------

def inject_anomalies(df_clean, type2_edges):
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
                         f"{injected:.2f} (+{sigma:.0f}sig, mean={mu:.2f})")

    def _inject_type2(parent, child, par_year, chi_year):
        if par_year not in df.index or chi_year not in df.index:
            return
        if parent not in df.columns or child not in df.columns:
            return
        mu_p, s_p = df[parent].mean(), df[parent].std()
        mu_c = df[child].mean()
        df.loc[par_year, parent] = mu_p + TYPE2_SPIKE_SIGMA * s_p
        df.loc[chi_year, child] = mu_c
        mask.loc[chi_year, child] = True
        notes.append(f"TYPE_2  {parent:25s} -> {child:25s}: "
                     f"parent +{TYPE2_SPIKE_SIGMA:.0f}sig at {par_year}, "
                     f"child forced to mean {mu_c:.2f} at {chi_year}")

    _inject_type1(TYPE1A_COL, TYPE1A_YEAR)
    _inject_type1(TYPE1B_COL, TYPE1B_YEAR)

    if len(type2_edges) >= 1:
        _inject_type2(*type2_edges[0], TYPE2A_PARENT_YR, TYPE2A_CHILD_YR)
    if len(type2_edges) >= 2:
        _inject_type2(*type2_edges[1], TYPE2B_PARENT_YR, TYPE2B_CHILD_YR)

    return df, mask, notes


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(df_anom, mask, graph):
    ev = AnomalyDetectionEvaluator(df_anom, mask)
    results = {}

    print("\nRunning evaluators on Kenya data (with federated graph)...")

    print("  [1/6] Z-score (blind, threshold=3.0)...")
    results['zscore'] = ev.evaluate_zscore(threshold=ZSCORE_THRESHOLD)

    print("  [2/6] IsolationForest (blind)...")
    results['isof_blind'] = ev.evaluate_isolation_forest()

    print(f"  [3/6] RRCF blind (window={RRCF_WINDOW}, threshold={RRCF_THRESHOLD})...")
    results['rrcf_blind'] = ev.evaluate_rrcf_engine(
        window_size=RRCF_WINDOW, num_trees=RRCF_TREES, threshold=RRCF_THRESHOLD)

    print(f"  [4/6] GraphResiduals (federated graph, threshold={GRAPH_THRESHOLD})...")
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


def _delta(new_val, ref_val):
    if new_val is None or ref_val is None:
        return '   N/A'
    d = float(new_val) - float(ref_val)
    sign = '+' if d >= 0 else ''
    return f'{sign}{d:.4f}'


def print_results(results, notes, type2_edges, n_total, n_anomalies, n_eff, countries):
    METHOD_LABELS = {
        'zscore':          'Z-score (blind)',
        'isof_blind':      'IsoForest (blind)',
        'rrcf_blind':      'RRCF (blind)',
        'graph_residuals': 'GraphResiduals (fed.)',
        'isof_graph':      'IsoForest+Graph (fed.)',
        'rrcf_graph':      'RRCF+Graph (fed.)',
    }
    GRAPH_METHODS = {'graph_residuals', 'isof_graph', 'rrcf_graph'}
    width = 90

    print("\n" + "=" * width)
    print(f"FEDERATED ANOMALY DETECTION: {'+'.join(countries)} -> KEN evaluation")
    print(f"N_eff={n_eff} training obs ({len(countries)} countries x 34 years) | "
          f"Eval: Kenya only ({n_total} cells, {n_anomalies} anomalies)")
    print("=" * width)

    print("\nInjected anomalies (Kenya data):")
    for note in notes:
        print(f"  {note}")

    if type2_edges:
        print("\nEdges selected for TYPE_2 injection:")
        for i, (p, c) in enumerate(type2_edges, 1):
            kn = "(KNOWN)" if (p, c) in set(KNOWN_EDGES) else ""
            print(f"  {i}. {p} -> {c}  {kn}")

    hdr = (f"{'Method':<24}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'FPR':>7}  "
           f"{'TP':>3}  {'FP':>3}  {'FN':>3}  {'vs §44 F1':>10}")
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
        ref  = SINGLE_COUNTRY_REF.get(key, {})
        df1  = _delta(m.get('f1'), ref.get('f1'))
        tag  = " <-- fed.graph" if key in GRAPH_METHODS else ""
        print(f"{label:<24}  {prec}  {rec}  {f1}  {fpr}  {tp:>3}  {fp:>3}  {fn:>3}  {df1:>10}{tag}")

    print(sep)

    # Key comparisons
    gr_f1  = results.get('graph_residuals', {}).get('f1', 0) or 0
    zs_f1  = results.get('zscore', {}).get('f1', 0) or 0
    delta  = gr_f1 - zs_f1
    gr_fpr = results.get('graph_residuals', {}).get('fpr', 0) or 0
    zs_fpr = results.get('zscore', {}).get('fpr', 0) or 0

    ref_gr_f1 = SINGLE_COUNTRY_REF.get('graph_residuals', {}).get('f1', 0)
    fed_lift  = gr_f1 - ref_gr_f1

    best_key = max(METHOD_LABELS, key=lambda k: results.get(k, {}).get('f1', 0) or 0)
    best_f1  = results.get(best_key, {}).get('f1', 0) or 0

    print(f"\nBest overall: {METHOD_LABELS[best_key]}  F1={best_f1:.4f}")
    print(f"GraphResiduals vs Z-score (federated): F1 {'+' if delta >= 0 else ''}{delta:.4f}")
    print(f"GraphResiduals federation lift over single-country: {'+' if fed_lift >= 0 else ''}{fed_lift:.4f} F1")
    print(f"  Federated GraphResiduals  Prec={_fmt(results.get('graph_residuals',{}).get('precision'))}  "
          f"Rec={results.get('graph_residuals',{}).get('recall',0):.4f}  FPR={gr_fpr:.4f}")
    print(f"  Z-score (blind)           Prec={_fmt(results.get('zscore',{}).get('precision'))}  "
          f"Rec={results.get('zscore',{}).get('recall',0):.4f}  FPR={zs_fpr:.4f}")

    print("\nINTERPRETATION")
    print("-" * 50)
    if delta > 0.05:
        print("  Federation CLOSES THE GAP: graph-conditioning now helps.")
        print("  More observations (N_eff=102) produce stable economic edges")
        print("  that improve TYPE_2 relationship-break detection.")
    elif delta > -0.05:
        print("  Federation is NEUTRAL: graph-conditioning within +/-0.05 of Z-score.")
        print("  N_eff=102 improves graph quality but may not be enough for TYPE_2 detection.")
    else:
        print("  Graph-conditioning still HURTS even with federation (N_eff=102).")
        print("  Suggests break-even N > 102; synthetic N=300 remains the threshold.")

    if fed_lift > 0.05:
        print(f"  Federation IMPROVES GraphResiduals F1 by +{fed_lift:.4f} over single-country (§44).")
    elif fed_lift < -0.05:
        print(f"  Federation HURTS GraphResiduals F1 by {fed_lift:.4f} vs single-country (§44).")
    else:
        print(f"  Federation has minimal impact on GraphResiduals F1 ({fed_lift:+.4f} vs §44).")

    print()
    print("  CAVEAT: TYPE_2 anomalies are only detectable if the graph contains")
    print("  the parent->child edge used for injection. KNOWN economic edges")
    print("  (broad_money->inflation, gcf->gdp) are preferred injection targets")
    print("  because they are stationary (unlike trend variables in §44).")
    print("=" * width)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Federated anomaly detection benchmark (KEN+TZA+UGA -> Kenya eval)')
    parser.add_argument('--countries', nargs='+', default=DEFAULT_COUNTRIES,
                        help='Countries to include in federation (default: KEN TZA UGA)')
    parser.add_argument('--conf', type=float, default=0.35,
                        help='Min confidence for graph extraction (default 0.35)')
    parser.add_argument('--min-evidence', type=int, default=5,
                        help='Min evidence for graph extraction (default 5)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Ensure KEN is always in the country list
    countries = list(dict.fromkeys(['KEN'] + [c for c in args.countries if c != 'KEN']))

    # 1. Load all countries
    country_dfs = load_and_clean(countries)

    # 2. Discover federated graph on clean data
    engine, graph, edges, n_eff = discover_federated_graph(
        country_dfs, args.conf, args.min_evidence)

    # 3. Select TYPE_2 injection edges (prefer KNOWN economic relationships)
    ken_df = country_dfs['KEN']
    type2_edges = find_injection_edges(edges, ken_df, n_edges=2)
    print(f"\nEdges selected for TYPE_2 injection:")
    for i, (p, c) in enumerate(type2_edges, 1):
        kn = "(KNOWN economic relationship)" if (p, c) in set(KNOWN_EDGES) else ""
        print(f"  {i}. {p} -> {c}  {kn}")

    # 4. Inject anomalies into Kenya data only
    df_anom, mask, notes = inject_anomalies(ken_df, type2_edges)
    n_total     = mask.size
    n_anomalies = int(mask.values.sum())
    print(f"\nInjected {n_anomalies} anomaly cells across {n_total} Kenya cells:")
    for note in notes:
        print(f"  {note}")

    # 5. Run evaluators (Kenya data, federated graph)
    results = run_benchmark(df_anom, mask, graph)

    # 6. Print results with §44 comparison
    print_results(results, notes, type2_edges, n_total, n_anomalies, n_eff, countries)


if __name__ == '__main__':
    main()

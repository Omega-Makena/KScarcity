"""
Ablation study for typed relationship discovery.

5 variants compare how individual engine components contribute to
typed GT recall:

  full_system      : All components enabled (baseline).
  causal_only      : Only CausalHypothesis + TemporalHypothesis seeded.
  top5_types_only  : Temporal, Equilibrium, Correlational, Functional,
                     Causal -- no triple-variable hypotheses.
  no_exploration   : Exploration step disabled (BanditRouter-driven
                     dynamic seeding turned off).
  no_lifecycle     : Lifecycle management disabled (hypotheses never
                     marked DEAD/TENTATIVE -- all stay ACTIVE).

Each variant is run on KEN data, and the resulting discoveries are
evaluated against the typed GT.

Usage:
    python scripts/experiments/run_ablation_typed.py
    python scripts/experiments/run_ablation_typed.py --fast
    python scripts/experiments/run_ablation_typed.py --variant full_system top5_types_only
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

warnings.filterwarnings('ignore')

from scripts.experiments.data_loader import load_country_data
from scripts.experiments.ground_truth_typed import (
    get_typed_ground_truth,
    get_known_null_relationships,
)
from scripts.experiments.evaluation_typed import (
    compare_specialists,
    compute_per_type_recall,
    false_positive_analysis,
)
from scripts.experiments.run_kscarcity_typed import (
    _build_schema,
    _summary_to_eval_format,
)

GT_COLS = [
    'gdp_growth', 'inflation_cpi', 'unemployment', 'real_interest_rate',
    'private_credit', 'govt_consumption', 'exports_gdp', 'imports_gdp',
    'current_account', 'gcf', 'electricity_access', 'internet_users',
    'school_enrollment', 'life_expectancy', 'broad_money', 'govt_debt',
]

ALL_VARIANTS = [
    'full_system',
    'causal_only',
    'top5_types_only',
    'no_exploration',
    'no_lifecycle',
]


# ---------------------------------------------------------------------------
# Custom engine initialization helpers
# ---------------------------------------------------------------------------

def _init_causal_only(engine, var_names: list[str]) -> None:
    """
    Seed only TemporalHypothesis (for lifecycle health) and CausalHypothesis
    for each ordered pair. Tests whether Granger causality alone suffices
    for typed GT discovery.
    """
    from scarcity.engine.relationships import CausalHypothesis, TemporalHypothesis
    bs = engine.buffer_size
    for v in var_names:
        engine.hypotheses.add(TemporalHypothesis(v, lag=2, buffer_size=bs))
    for a, b in itertools.combinations(var_names, 2):
        engine.hypotheses.add(CausalHypothesis(a, b, lag=2, buffer_size=bs))
        engine.hypotheses.add(CausalHypothesis(b, a, lag=2, buffer_size=bs))


def _init_top5_types(engine, var_names: list[str]) -> None:
    """
    Seed the 5 core pairwise hypothesis types (no triple-variable hypotheses).
    Tests whether triple types (Compositional, Synergistic, Mediating,
    Moderating, Logical) contribute meaningful signal.
    """
    from scarcity.engine.relationships import (
        CausalHypothesis, CorrelationalHypothesis, TemporalHypothesis,
        FunctionalHypothesis, EquilibriumHypothesis,
    )
    bs = engine.buffer_size
    for v in var_names:
        engine.hypotheses.add(TemporalHypothesis(v, lag=2, buffer_size=bs))
        engine.hypotheses.add(EquilibriumHypothesis(v, buffer_size=bs))
    for a, b in itertools.combinations(var_names, 2):
        engine.hypotheses.add(CorrelationalHypothesis(a, b, buffer_size=bs))
        engine.hypotheses.add(CorrelationalHypothesis(b, a, buffer_size=bs))
        engine.hypotheses.add(FunctionalHypothesis(a, b, degree=1, buffer_size=bs))
        engine.hypotheses.add(FunctionalHypothesis(b, a, degree=1, buffer_size=bs))
        engine.hypotheses.add(CausalHypothesis(a, b, lag=2, buffer_size=bs))
        engine.hypotheses.add(CausalHypothesis(b, a, lag=2, buffer_size=bs))


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------

def run_ablation_variant(
    df: pd.DataFrame,
    variant: str,
    buffer_size: int = 30,
    min_conf: float = 0.15,
    verbose: bool = True,
) -> list[dict]:
    """
    Run one ablation variant. Returns discoveries in eval format.

    Args:
        df:         DataFrame with GT columns as cols, years as index.
        variant:    One of ALL_VARIANTS.
        buffer_size: Engine buffer size.
        min_conf:   Minimum confidence threshold for export.
        verbose:    Print progress.
    """
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    if variant not in ALL_VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}. Choose from: {ALL_VARIANTS}')

    engine = OnlineDiscoveryEngine(buffer_size=buffer_size)
    schema = _build_schema(df.columns.tolist())
    var_names = df.columns.tolist()

    if variant == 'full_system':
        engine.initialize_v2(schema, use_causal=True)

    elif variant == 'causal_only':
        # Manual init — skip the standard initialize_v2
        engine.grouper.initialize(var_names)
        engine._var_index = {name: idx for idx, name in enumerate(var_names)}
        _init_causal_only(engine, var_names)

    elif variant == 'top5_types_only':
        engine.grouper.initialize(var_names)
        engine._var_index = {name: idx for idx, name in enumerate(var_names)}
        _init_top5_types(engine, var_names)

    elif variant == 'no_exploration':
        engine.initialize_v2(schema, use_causal=True)
        engine.exploration_enabled = False

    elif variant == 'no_lifecycle':
        engine.initialize_v2(schema, use_causal=True)
        engine.lifecycle_interval = 10 ** 9  # never triggers

    n_hyp = len(engine.hypotheses.population)
    if verbose:
        print(f'  [{variant}] {n_hyp} hypotheses seeded, feeding {len(df)} rows...')

    t0 = time.time()
    rows_fed = 0
    for _, row in df.iterrows():
        row_dict = {k: float(v) for k, v in row.items() if pd.notna(v)}
        if row_dict:
            engine.process_row(row_dict)
            rows_fed += 1

    elapsed = time.time() - t0

    summaries = engine.export_hypothesis_summary(min_conf=min_conf)
    discoveries = _summary_to_eval_format(summaries)

    by_type: dict[str, int] = {}
    for d in discoveries:
        by_type[d['type']] = by_type.get(d['type'], 0) + 1

    if verbose:
        print(f'  [{variant}] {rows_fed} rows in {elapsed:.1f}s, '
              f'{len(discoveries)} discoveries: '
              + ', '.join(f'{t}={n}' for t, n in sorted(by_type.items())))

    return discoveries


# ---------------------------------------------------------------------------
# Ablation summary
# ---------------------------------------------------------------------------

def run_full_ablation(
    df: pd.DataFrame,
    variants: list[str] | None = None,
    buffer_size: int = 30,
    min_conf: float = 0.15,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run all variants and return a results dict.

    Returns:
        {variant: {'discoveries': list, 'recall': dict, 'overall': dict,
                   'null_fp': float}}
    """
    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    if variants is None:
        variants = ALL_VARIANTS

    results: dict[str, dict] = {}

    for var in variants:
        if verbose:
            print(f'\nVariant: {var}')
        disc = run_ablation_variant(df, var, buffer_size=buffer_size,
                                    min_conf=min_conf, verbose=verbose)
        disc_wrap = {var: disc}
        recall = compute_per_type_recall(disc_wrap, gt)
        overall = compare_specialists(disc_wrap, gt)[var]
        fp_info = false_positive_analysis(disc_wrap, gt, null_pairs)
        results[var] = {
            'discoveries': disc,
            'n_discoveries': len(disc),
            'recall': recall,
            'overall': overall,
            'null_fp_rate': fp_info['null_fp_rate'],
        }

    return results


def print_ablation_summary(results: dict[str, dict]) -> None:
    """Print a comparison table across all ablation variants."""
    print('\nAblation Summary Table')
    print('=' * 80)
    print(f'  {"Variant":20s} {"N_disc":>7s} {"TP":>4s} {"FP":>5s} {"FN":>5s}'
          f' {"P":>6s} {"R":>6s} {"F1":>6s} {"NullFP":>7s}')
    print(f'  {"-"*75}')
    for var, info in results.items():
        m = info['overall']
        print(f'  {var:20s} {info["n_discoveries"]:7d} {m["tp"]:4d} {m["fp"]:5d}'
              f' {m["fn"]:5d} {m["precision"]:6.3f} {m["recall"]:6.3f}'
              f' {m["f1"]:6.3f} {info["null_fp_rate"]:7.3f}')

    # Per-type recall rows (relative to full_system)
    all_types = sorted({
        t for info in results.values()
        for t in info['recall']
    })
    full_info = results.get('full_system', {})
    print(f'\n  Per-type recall (columns = variants):')
    header = f'  {"Type":15s}'
    for var in results:
        header += f' {var[:12]:>12s}'
    print(header)
    print(f'  {"-"*(16 + 13*len(results))}')
    for t in all_types:
        row = f'  {t:15s}'
        for var, info in results.items():
            r = info['recall'].get(t, {}).get('recall', 0.0)
            row += f' {r:12.3f}'
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print('K-Scarcity Typed Discovery Ablation Study')
    print('=' * 50)

    df_ken = load_country_data('KEN')
    avail = [c for c in GT_COLS if c in df_ken.columns]
    df_work = df_ken[avail].dropna()

    if args.fast:
        df_work = df_work.head(min(15, len(df_work)))

    print(f'Dataset: KEN {df_work.shape} '
          f'(years {df_work.index.min()}-{df_work.index.max()})')

    variants = args.variant if args.variant else ALL_VARIANTS

    results = run_full_ablation(
        df_work,
        variants=variants,
        buffer_size=args.buffer_size,
        min_conf=args.min_conf,
        verbose=not args.quiet,
    )

    print_ablation_summary(results)

    # Save
    out_dir = _ROOT / 'results' / 'typed_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    out_path = out_dir / 'ablation_typed_results.json'

    save = {}
    for var, info in results.items():
        save[var] = {
            'n_discoveries': info['n_discoveries'],
            'overall': info['overall'],
            'null_fp_rate': info['null_fp_rate'],
            'recall_by_type': {
                t: v['recall']
                for t, v in info['recall'].items()
            },
        }
    out_path.write_text(json.dumps(save, indent=2), encoding='utf-8')
    if not args.quiet:
        print(f'\nResults saved to {out_path.relative_to(_ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ablation study on typed relationship discovery'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Use only first 15 rows (smoke test)')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--variant', nargs='+', choices=ALL_VARIANTS,
                        help='Run specific variants only')
    parser.add_argument('--buffer-size', type=int, default=30)
    parser.add_argument('--min-conf', type=float, default=0.15)
    args = parser.parse_args()
    main(args)

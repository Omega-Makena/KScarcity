"""
Weakness Fix 9: Find the type crossover N.

The ablation study found that top5_types_only achieves higher overall recall
than full_system at N=20. This raises the question: at what N does the full
15-type system *overtake* the top-5-type ablation?

This is important because:
  - If crossover N > 100, then for real-world N≈20 datasets, running all 15
    hypothesis types is wasteful.
  - If crossover N ≈ 30-50, the full system starts paying off within a
    reasonable data collection horizon.

Method: sweep N from 10 to max available rows (≈34 for KEN), run both
variants, and record recall at each N. Find first N where full_system
recall >= top5_recall.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Ablation variants (reuse from run_ablation_typed logic)
# ---------------------------------------------------------------------------

def _run_ablation_variant(
    df: pd.DataFrame,
    variant: str,
    buffer_size: int = 30,
    min_conf: float = 0.15,
) -> list[dict]:
    """Run a single ablation variant, return discoveries."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    from scripts.experiments.run_kscarcity_typed import (
        _build_schema, _summary_to_eval_format,
    )

    engine = OnlineDiscoveryEngine(buffer_size=min(buffer_size, len(df)))
    schema = _build_schema(df.columns.tolist())
    var_names = df.columns.tolist()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if variant == 'full_system':
            engine.initialize_v2(schema, use_causal=False)

        elif variant == 'top5_types_only':
            from scarcity.engine.hypotheses.temporal import TemporalHypothesis
            from scarcity.engine.hypotheses.equilibrium import EquilibriumHypothesis
            from scarcity.engine.hypotheses.correlational import CorrelationalHypothesis
            from scarcity.engine.hypotheses.functional import FunctionalHypothesis
            from scarcity.engine.hypotheses.causal import CausalHypothesis
            engine.initialize_v2(schema, use_causal=False)
            all_hyps = list(engine.hypotheses.population.values())
            allowed_types = {
                'temporal', 'equilibrium', 'correlational', 'functional', 'causal',
            }
            to_remove = [h for h in all_hyps if h.hyp_type not in allowed_types]
            for h in to_remove:
                key = h.key if hasattr(h, 'key') else None
                if key and key in engine.hypotheses.population:
                    del engine.hypotheses.population[key]

        elif variant == 'causal_only':
            from scarcity.engine.hypotheses.temporal import TemporalHypothesis
            from scarcity.engine.hypotheses.causal import CausalHypothesis
            engine.initialize_v2(schema, use_causal=False)
            all_hyps = list(engine.hypotheses.population.values())
            allowed_types = {'temporal', 'causal'}
            to_remove = [h for h in all_hyps if h.hyp_type not in allowed_types]
            for h in to_remove:
                key = h.key if hasattr(h, 'key') else None
                if key and key in engine.hypotheses.population:
                    del engine.hypotheses.population[key]

        else:
            engine.initialize_v2(schema, use_causal=False)

        for _, row in df.iterrows():
            row_dict = {k: float(v) for k, v in row.items() if pd.notna(v)}
            if row_dict:
                engine.process_row(row_dict)

        summaries = engine.export_hypothesis_summary(min_conf=min_conf)

    return _summary_to_eval_format(summaries)


# ---------------------------------------------------------------------------
# N sweep for crossover
# ---------------------------------------------------------------------------

def find_type_crossover_n(
    df_full: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    n_values: list[int] | None = None,
    buffer_size: int = 30,
    min_conf: float = 0.15,
    variants: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Sweep N; run full_system and top5_types_only at each N.
    Find first N where full_system recall >= top5_recall.

    Returns:
        {
          'sweep': [{n, full_recall, top5_recall, winner}, ...],
          'crossover_n': int or None,
          'full_system_recall_at_n_max': float,
          'top5_recall_at_n_max': float,
        }
    """
    from scripts.experiments.evaluation_typed import compare_specialists

    df_complete = df_full.dropna()
    n_max = len(df_complete)

    if n_values is None:
        # Dense sweep from 10 to max available
        n_values = list(range(10, n_max + 1, 2))
        if n_max not in n_values:
            n_values.append(n_max)
        n_values = sorted(set(n_values))

    if variants is None:
        variants = ['full_system', 'top5_types_only']

    if verbose:
        print(f'  Crossover N sweep: {len(n_values)} points, '
              f'max_N={n_max}, variants={variants}')

    sweep_rows = []
    crossover_n = None

    for n in n_values:
        df_n = df_complete.head(n)
        actual_n = len(df_n)
        if actual_n < 8:
            continue

        row = {'n': actual_n}
        for variant in variants:
            try:
                disc = _run_ablation_variant(
                    df_n, variant,
                    buffer_size=min(buffer_size, actual_n),
                    min_conf=min_conf,
                )
                m = compare_specialists({variant: disc}, ground_truth).get(variant, {})
                row[f'{variant}_recall'] = round(m.get('recall', 0.0), 4)
                row[f'{variant}_f1'] = round(m.get('f1', 0.0), 4)
            except Exception as exc:
                row[f'{variant}_recall'] = 0.0
                row[f'{variant}_f1'] = 0.0

        sweep_rows.append(row)

        # Check crossover
        if ('full_system' in variants and 'top5_types_only' in variants
                and crossover_n is None):
            fs_rec = row.get('full_system_recall', 0.0)
            t5_rec = row.get('top5_types_only_recall', 0.0)
            if fs_rec >= t5_rec and fs_rec > 0:
                crossover_n = actual_n

        if verbose:
            parts = [f"N={actual_n}"]
            for v in variants:
                parts.append(f"{v[:8]}={row.get(f'{v}_recall', 0.0):.4f}")
            print(f"    {' | '.join(parts)}")

    if verbose:
        if crossover_n is not None:
            print(f'\n  Crossover at N={crossover_n}: '
                  f'full_system recall first >= top5_types_only recall')
        else:
            print('\n  No crossover found in available N range '
                  f'(max N={n_max})')

    last = sweep_rows[-1] if sweep_rows else {}
    return {
        'sweep': sweep_rows,
        'crossover_n': crossover_n,
        'n_max': n_max,
        'full_system_recall_at_n_max': last.get('full_system_recall', 0.0),
        'top5_recall_at_n_max': last.get('top5_types_only_recall', 0.0),
        'variants': variants,
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix9(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 9: type crossover N on KEN data."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
    )
    from scripts.experiments.run_federation_typed import GT_COLS

    df_raw = load_country_data('KEN')
    avail = [c for c in GT_COLS if c in df_raw.columns]
    df = df_raw[avail].dropna()

    if verbose:
        print(f'  KEN complete rows: {len(df)}  columns: {len(df.columns)}')

    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    if fast:
        # Only a few N values
        n_max = min(20, len(df))
        n_vals = [10, 12, 15, n_max]
    else:
        n_vals = None  # full sweep

    return find_type_crossover_n(
        df, gt, null_pairs, n_values=n_vals, verbose=verbose
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 9: Type crossover N')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix9(fast=args.fast, verbose=not args.quiet)

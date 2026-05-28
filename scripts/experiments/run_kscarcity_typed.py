"""
K-Scarcity engine runner for typed relationship discovery validation.

Wraps OnlineDiscoveryEngine so its output can be fed to the same
evaluation_typed.py functions that evaluate the specialist baselines.

Key mapping from engine hypothesis summary to evaluation format:
  - mediating: vars=[source, mediator, target]
  - synergistic: vars=[source, moderator, target]
  - all others: vars=[source, target]  (or [var] for self-loops)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

def _build_schema(columns: list[str]) -> dict:
    """Build engine schema dict from a list of variable names."""
    return {'fields': [{'name': c, 'type': 'float'} for c in columns]}


# ---------------------------------------------------------------------------
# Engine output → evaluation format converter
# ---------------------------------------------------------------------------

def _summary_to_eval_format(summaries: list[dict]) -> list[dict]:
    """
    Convert engine export_hypothesis_summary() output to the standardised
    discovery format expected by evaluation_typed.py.

    Engine format:
        {'vars': [...], 'type': str, 'conf': float, 'evidence': int}

    Eval format:
        {'source': str, 'target': str, 'type': str, 'confidence': float,
         'sign': int, 'statistic': float, 'p_value': float, 'method': str,
         [mediator: str], [moderator: str]}
    """
    result = []
    for s in summaries:
        vars_ = s.get('vars', [])
        rel_type = s.get('type', '')
        conf = float(s.get('conf', 0.0))
        evidence = int(s.get('evidence', 0))

        if len(vars_) == 0:
            continue

        base = {
            'type': rel_type,
            'confidence': round(conf, 4),
            'sign': 0,
            'statistic': float(evidence),
            'p_value': round(1.0 - conf, 4),
            'method': 'k_scarcity',
        }

        if rel_type == 'mediating' and len(vars_) == 3:
            base['source'] = vars_[0]
            base['target'] = vars_[2]
            base['mediator'] = vars_[1]
        elif rel_type == 'synergistic' and len(vars_) == 3:
            base['source'] = vars_[0]
            base['target'] = vars_[2]
            base['moderator'] = vars_[1]
        elif rel_type == 'moderating' and len(vars_) == 3:
            base['source'] = vars_[0]
            base['target'] = vars_[2]
            base['moderator'] = vars_[1]
        elif rel_type == 'compositional' and len(vars_) >= 2:
            base['source'] = vars_[0]
            base['target'] = vars_[-1]
        elif rel_type in ('temporal', 'equilibrium') and len(vars_) == 1:
            base['source'] = vars_[0]
            base['target'] = vars_[0]
        elif len(vars_) >= 2:
            base['source'] = vars_[0]
            base['target'] = vars_[1]
        elif len(vars_) == 1:
            base['source'] = vars_[0]
            base['target'] = vars_[0]
        else:
            continue

        result.append(base)

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_kscarcity_on_df(
    df: pd.DataFrame,
    buffer_size: int = 30,
    min_conf: float = 0.15,
    use_causal: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Feed a DataFrame row-by-row into the K-Scarcity engine and return
    all discovered relationships in evaluation format.

    Args:
        df: DataFrame with variables as columns, time steps as rows.
            Rows with NaN values are skipped.
        buffer_size: engine buffer size (number of rows engine keeps in memory).
        min_conf: minimum confidence threshold for hypothesis export.
        use_causal: whether to seed CausalHypothesis pairs (Granger-based;
                    slower but covers causal GT relationships).
        verbose: print progress.

    Returns:
        List of discovery dicts in evaluation format.
    """
    warnings.filterwarnings('ignore')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    engine = OnlineDiscoveryEngine(buffer_size=buffer_size)
    schema = _build_schema(df.columns.tolist())

    if verbose:
        print(f'  Initializing engine: {len(df.columns)} vars, buffer={buffer_size}, '
              f'use_causal={use_causal}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        engine.initialize_v2(schema, use_causal=use_causal)

    n_hyp = len(engine.hypotheses.population)
    if verbose:
        print(f'  Hypotheses after init: {n_hyp}')

    t0 = time.time()
    rows_fed = 0
    for _, row in df.iterrows():
        row_dict = {k: float(v) for k, v in row.items() if pd.notna(v)}
        if not row_dict:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine.process_row(row_dict)
        rows_fed += 1

    elapsed = time.time() - t0
    if verbose:
        print(f'  Fed {rows_fed} rows in {elapsed:.1f}s')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        summaries = engine.export_hypothesis_summary(min_conf=min_conf)

    discoveries = _summary_to_eval_format(summaries)

    if verbose:
        by_type: dict[str, int] = {}
        for d in discoveries:
            by_type[d['type']] = by_type.get(d['type'], 0) + 1
        print(f'  Exported {len(discoveries)} discoveries (conf>={min_conf}):')
        for t, n in sorted(by_type.items()):
            print(f'    {t:15s}: {n}')

    return discoveries


def run_kscarcity_n_sweep(
    df_full: pd.DataFrame,
    n_values: list[int],
    buffer_size: int = 30,
    min_conf: float = 0.15,
    use_causal: bool = True,
    verbose: bool = True,
) -> dict[int, list[dict]]:
    """
    Run K-Scarcity at each N (row count) and return discoveries per N.

    A fresh engine is created for each N to avoid state leakage.

    Returns:
        {N: [discovery_dict, ...]}
    """
    df_clean = df_full.dropna()
    results: dict[int, list[dict]] = {}

    for n in sorted(n_values):
        df_n = df_clean.head(n) if len(df_clean) >= n else df_clean
        actual_n = len(df_n)
        if verbose:
            print(f'\n  [N={n}] actual_rows={actual_n}')
        try:
            disc = run_kscarcity_on_df(
                df_n,
                buffer_size=min(buffer_size, actual_n),
                min_conf=min_conf,
                use_causal=use_causal,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f'    ERROR at N={n}: {exc}')
            disc = []
        results[n] = disc

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
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

    print('Loading Kenya data...')
    df_ken = load_country_data('KEN')
    gt_cols = [
        'gdp_growth', 'inflation_cpi', 'unemployment', 'real_interest_rate',
        'private_credit', 'govt_consumption', 'exports_gdp', 'imports_gdp',
        'current_account', 'gcf', 'electricity_access', 'internet_users',
        'school_enrollment', 'life_expectancy', 'broad_money',
    ]
    df = df_ken[[c for c in gt_cols if c in df_ken.columns]].dropna()
    print(f'  Shape: {df.shape}')

    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    print('\nRunning K-Scarcity engine (full dataset)...')
    disc = run_kscarcity_on_df(df, buffer_size=30, min_conf=0.15, verbose=True)

    kscarcity_as_dict = {'k_scarcity': disc}

    print('\n--- Q1: Per-type recall ---')
    recall = compute_per_type_recall(kscarcity_as_dict, gt)
    for t, info in sorted(recall.items()):
        print(f"  {t:15s}: {info['n_discovered']}/{info['n_gt']} recall={info['recall']:.3f}")

    print('\n--- Q2: Overall metrics ---')
    cmp = compare_specialists(kscarcity_as_dict, gt)
    m = cmp['k_scarcity']
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print('\n--- Q3: FP on null pairs ---')
    fp = false_positive_analysis(kscarcity_as_dict, gt, null_pairs)
    print(f"  Null FP rate: {fp['null_fp_rate']:.3f}")
    print(f"  Total FP: {fp['total_fp_all']}")

    print('\nSelf-test complete.')

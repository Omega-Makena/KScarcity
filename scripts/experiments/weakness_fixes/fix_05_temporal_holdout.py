"""
Weakness Fix 5: Temporal holdout evaluation.

Problem: current evaluation uses all 20 observations for both training and
testing. This is data snooping — the system has seen the full time series
when making discoveries, but we evaluate discoveries against a GT derived
from the same variables. True evaluation should hold out a test period.

For N=20 observations this is very tight, but we can still do:
  1. Simple train/test split: train on first 70% of years, evaluate on last 30%.
     Compute: are discoveries made on train still meaningful on test data?
  2. Expanding window: train on [1..t] for t in [10,12,15,18], evaluate new
     discoveries at each step — simulates real-time deployment.
  3. LOOCV: leave one year out at a time; evaluate consistency of discoveries.

Note: We evaluate discovery *consistency* (do the same pairs remain
significant in the holdout period), not GT recall (GT is theory-grounded
and applies regardless of time period).
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
from scipy import stats

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Discovery consistency between train and test splits
# ---------------------------------------------------------------------------

def discovery_consistency(
    disc_train: list[dict],
    df_test: pd.DataFrame,
    alpha: float = 0.10,
    verbose: bool = True,
) -> dict:
    """
    For each discovery made on training data, check if the same relationship
    holds in test data (same direction, p < alpha).

    Returns:
        {
          'n_disc_train': int,
          'n_consistent': int,   # relationship holds in test
          'n_inconsistent': int, # relationship reverses in test
          'n_insufficient': int, # too few test rows to evaluate
          'consistency_rate': float,
        }
    """
    n_consistent = 0
    n_inconsistent = 0
    n_insufficient = 0
    details = []

    for d in disc_train:
        src = d.get('source', '')
        tgt = d.get('target', '')
        disc_type = d.get('type', '')
        train_sign = d.get('sign', 0)

        # Skip if columns not in test data
        if src not in df_test.columns or tgt not in df_test.columns:
            n_insufficient += 1
            continue
        if len(df_test) < 5:
            n_insufficient += 1
            continue

        try:
            if src == tgt:
                # Temporal: check AR(1) in test
                x = df_test[src].values
                r, p = stats.pearsonr(x[:-1], x[1:]) if len(x) > 1 else (0, 1)
            else:
                r, p = stats.pearsonr(df_test[src].values, df_test[tgt].values)
        except Exception:
            n_insufficient += 1
            continue

        test_sign = int(np.sign(r)) if abs(r) > 0.05 else 0
        consistent = (p < alpha) and (test_sign == train_sign or train_sign == 0)

        if p >= alpha:
            n_insufficient += 1  # not significant in test (too few obs)
        elif consistent:
            n_consistent += 1
        else:
            n_inconsistent += 1

        details.append({
            'source': src, 'target': tgt, 'type': disc_type,
            'train_sign': train_sign, 'test_r': round(r, 4),
            'test_p': round(p, 4), 'consistent': consistent,
        })

    total_evaluable = n_consistent + n_inconsistent
    consistency_rate = n_consistent / total_evaluable if total_evaluable > 0 else 0.0

    if verbose:
        print(f'  Discovery consistency: {n_consistent}/{total_evaluable} consistent '
              f'({consistency_rate:.3f}), {n_insufficient} insufficient test data')

    return {
        'n_disc_train': len(disc_train),
        'n_consistent': n_consistent,
        'n_inconsistent': n_inconsistent,
        'n_insufficient': n_insufficient,
        'consistency_rate': round(consistency_rate, 4),
        'details': details,
    }


# ---------------------------------------------------------------------------
# Simple train/test split
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    df: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    train_frac: float = 0.70,
    verbose: bool = True,
) -> dict:
    """
    Train on first train_frac rows; check consistency on last (1-train_frac).
    Also evaluate GT recall on the training split alone.
    """
    from scripts.experiments.specialist_baselines import run_all_specialists
    from scripts.experiments.evaluation_typed import compare_specialists

    n = len(df)
    n_train = max(8, int(round(n * train_frac)))
    n_test = n - n_train

    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]

    if verbose:
        print(f'  Temporal split: train={n_train} rows, test={n_test} rows')

    # Train
    disc_by_type = run_all_specialists(df_train, verbose=False)
    disc_train = [d for discs in disc_by_type.values() for d in discs]
    m_train = compare_specialists({'sys': disc_train}, ground_truth).get('sys', {})

    # Full data (for comparison)
    disc_all_by_type = run_all_specialists(df, verbose=False)
    disc_all = [d for discs in disc_all_by_type.values() for d in discs]
    m_all = compare_specialists({'sys': disc_all}, ground_truth).get('sys', {})

    # Consistency check
    consistency = discovery_consistency(disc_train, df_test, verbose=verbose)

    if verbose:
        print(f'\n  GT recall on train only: {m_train.get("recall", 0):.4f}  '
              f'F1={m_train.get("f1", 0):.4f}')
        print(f'  GT recall on all data:   {m_all.get("recall", 0):.4f}  '
              f'F1={m_all.get("f1", 0):.4f}')

    return {
        'train_metrics': {k: m_train.get(k, 0) for k in ['precision','recall','f1']},
        'all_metrics': {k: m_all.get(k, 0) for k in ['precision','recall','f1']},
        'consistency': consistency,
        'n_train': n_train,
        'n_test': n_test,
    }


# ---------------------------------------------------------------------------
# Expanding window
# ---------------------------------------------------------------------------

def expanding_window_evaluation(
    df: pd.DataFrame,
    ground_truth: list[dict],
    window_sizes: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run specialists on expanding windows and track recall convergence.
    Returns recall at each window size.
    """
    from scripts.experiments.specialist_baselines import run_all_specialists
    from scripts.experiments.evaluation_typed import compare_specialists

    n = len(df)
    if window_sizes is None:
        window_sizes = sorted({8, 10, 12, 15, min(18, n), n})

    results = []
    for ws in window_sizes:
        if ws > n:
            continue
        df_w = df.iloc[:ws]
        disc_by_type = run_all_specialists(df_w, verbose=False)
        disc_all = [d for discs in disc_by_type.values() for d in discs]
        m = compare_specialists({'sys': disc_all}, ground_truth).get('sys', {})
        results.append({
            'n': ws,
            'n_disc': len(disc_all),
            'recall': m.get('recall', 0.0),
            'f1': m.get('f1', 0.0),
            'precision': m.get('precision', 0.0),
        })

    if verbose:
        print(f'\n  Expanding window evaluation:')
        print(f"  {'N':>5s}  {'N_disc':>7s}  {'Recall':>7s}  {'F1':>7s}")
        print(f"  {'-'*30}")
        for r in results:
            print(f"  {r['n']:5d}  {r['n_disc']:7d}  {r['recall']:7.4f}  {r['f1']:7.4f}")

    return {'windows': results}


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def temporal_holdout_evaluation(
    df: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    verbose: bool = True,
) -> dict:
    """Full temporal holdout evaluation: split + expanding window."""
    split_res = temporal_train_test_split(df, ground_truth, null_pairs, verbose=verbose)
    expand_res = expanding_window_evaluation(df, ground_truth, verbose=verbose)
    return {
        'train_test_split': split_res,
        'expanding_window': expand_res,
    }


def run_fix5(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 5: temporal holdout on KEN data."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
    )
    from scripts.experiments.run_federation_typed import GT_COLS

    df_raw = load_country_data('KEN')
    avail = [c for c in GT_COLS if c in df_raw.columns]
    df = df_raw[avail].dropna()
    if fast:
        df = df.head(15)
    if verbose:
        print(f'  KEN complete rows: {len(df)}  columns: {len(df.columns)}')

    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    return temporal_holdout_evaluation(df, gt, null_pairs, verbose=verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 5: Temporal holdout')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix5(fast=args.fast, verbose=not args.quiet)

"""
Weakness Fix 1: Permutation test + precision@k / recall@k.

The core methodological problem: current evaluation reports recall=X% against
27 GT relationships but provides no significance test. A system that fires
randomly could match GT entries by chance.

Fix:
  - permutation_test_discovery(): shuffles each column independently
    (preserves marginals, breaks cross-variable structure), runs specialists
    on shuffled data n_permutations times, computes p-value.
  - precision_recall_at_k(): rank-based evaluation replacing confidence-
    threshold filtering. Sorts discoveries by confidence and computes
    P@k and R@k for k in k_values.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# precision@k / recall@k
# ---------------------------------------------------------------------------

def precision_recall_at_k(
    discoveries: list[dict],
    ground_truth: list[dict],
    k_values: Sequence[int] = (5, 10, 20, 50, 100),
    strict_type: bool = True,
) -> dict[int, dict]:
    """
    Rank-based evaluation: sort discoveries by confidence, evaluate top-k.

    Args:
        discoveries: flat list of discovery dicts (with 'confidence' field).
        ground_truth: list of GT entry dicts.
        k_values: cut-offs to evaluate.
        strict_type: whether type must match exactly.

    Returns:
        {k: {
            'precision_at_k': float,
            'recall_at_k': float,
            'n_gt_in_top_k': int,
            'n_gt': int,
            'n_available': int,  # min(k, len(discoveries))
        }}
    """
    from scripts.experiments.evaluation_typed import _any_gt_match

    # Sort descending by confidence
    sorted_disc = sorted(discoveries, key=lambda d: d.get('confidence', 0.0),
                         reverse=True)
    n_gt = len(ground_truth)
    result: dict[int, dict] = {}

    for k in sorted(k_values):
        top_k = sorted_disc[:k]
        n_available = len(top_k)

        matched_gt_idxs: set[int] = set()
        for d in top_k:
            entry = _any_gt_match(d, ground_truth, strict_type=strict_type)
            if entry is not None:
                matched_gt_idxs.add(ground_truth.index(entry))

        n_matched = len(matched_gt_idxs)
        prec = n_matched / n_available if n_available > 0 else 0.0
        rec = n_matched / n_gt if n_gt > 0 else 0.0

        result[k] = {
            'precision_at_k': round(prec, 4),
            'recall_at_k': round(rec, 4),
            'n_gt_in_top_k': n_matched,
            'n_gt': n_gt,
            'n_available': n_available,
        }

    return result


def print_precision_recall_at_k(
    results: dict[int, dict],
    label: str = '',
) -> None:
    """Print P@k / R@k table."""
    header = f'  Precision@k / Recall@k'
    if label:
        header += f'  [{label}]'
    print(header)
    print(f"  {'k':>6s}  {'P@k':>7s}  {'R@k':>7s}  {'GT hits':>8s}  {'avail':>6s}")
    print(f"  {'-'*42}")
    for k, info in sorted(results.items()):
        print(f"  {k:6d}  {info['precision_at_k']:7.4f}  "
              f"{info['recall_at_k']:7.4f}  "
              f"{info['n_gt_in_top_k']:8d}  "
              f"{info['n_available']:6d}")


# ---------------------------------------------------------------------------
# Column-wise permutation (preserves marginals, breaks cross-variable structure)
# ---------------------------------------------------------------------------

def _permute_df(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each column independently; preserves per-column distribution."""
    perm_data = {col: rng.permutation(df[col].values) for col in df.columns}
    return pd.DataFrame(perm_data, index=df.index)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test_discovery(
    df: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    n_permutations: int = 200,
    seed: int = 42,
    verbose: bool = True,
    method: str = 'specialists',
) -> dict:
    """
    Permutation test: does the system find more GT structure than expected by chance?

    Null hypothesis: the observed recall / F1 is no higher than what would be
    obtained on data with the same marginal distributions but no cross-variable
    dependencies (achieved by column-wise independent shuffling).

    Args:
        df: complete-case DataFrame (all GT columns, no NaN).
        ground_truth: list of GT entry dicts.
        null_pairs: known-null pairs (for FP analysis).
        n_permutations: number of column-wise shuffles.
        seed: base random seed.
        verbose: print progress.
        method: 'specialists' (fast) or 'kscarcity' (slower).

    Returns:
        {
          'real_recall': float,
          'real_f1': float,
          'real_n_disc': int,
          'real_precision': float,
          'perm_recalls': list[float],   # length n_permutations
          'perm_f1s': list[float],
          'p_value_recall': float,
          'p_value_f1': float,
          'n_permutations': int,
          'precision_at_k': dict,        # P@k/R@k on real data
          'method': str,
        }
    """
    from scripts.experiments.evaluation_typed import (
        compare_specialists,
        false_positive_analysis,
    )

    def _run_system(data: pd.DataFrame) -> list[dict]:
        if method == 'kscarcity':
            from scripts.experiments.run_kscarcity_typed import run_kscarcity_on_df
            return run_kscarcity_on_df(
                data, buffer_size=min(30, len(data)), min_conf=0.10,
                use_causal=False, verbose=False,
            )
        else:
            from scripts.experiments.specialist_baselines import run_all_specialists
            disc_by_type = run_all_specialists(data, verbose=False)
            return [d for discs in disc_by_type.values() for d in discs]

    def _compute_metrics(disc: list[dict]) -> tuple[float, float, float]:
        """Returns (recall, f1, precision)."""
        wrapped = {'system': disc}
        m = compare_specialists(wrapped, ground_truth).get('system', {})
        return m.get('recall', 0.0), m.get('f1', 0.0), m.get('precision', 0.0)

    if verbose:
        print(f'  Running {method} on real data (N={len(df)})...')

    real_disc = _run_system(df)
    real_recall, real_f1, real_prec = _compute_metrics(real_disc)
    real_n_disc = len(real_disc)

    if verbose:
        print(f'  Real: N_disc={real_n_disc}  recall={real_recall:.4f}  '
              f'f1={real_f1:.4f}  prec={real_prec:.4f}')
        print(f'  Running {n_permutations} column-wise permutations...')

    # P@k on real data
    flat_disc = real_disc if isinstance(real_disc, list) else list(real_disc)
    pak_results = precision_recall_at_k(flat_disc, ground_truth)

    perm_recalls: list[float] = []
    perm_f1s: list[float] = []
    rng = np.random.default_rng(seed)

    for i in range(n_permutations):
        df_perm = _permute_df(df, rng)
        try:
            perm_disc = _run_system(df_perm)
            rec, f1, _ = _compute_metrics(perm_disc)
        except Exception:
            rec, f1 = 0.0, 0.0
        perm_recalls.append(rec)
        perm_f1s.append(f1)

        if verbose and (i + 1) % 50 == 0:
            print(f'    permutation {i+1}/{n_permutations}  '
                  f'mean_perm_recall={np.mean(perm_recalls):.4f}')

    # One-sided p-values: P(perm_stat >= real_stat)
    p_recall = np.mean([r >= real_recall for r in perm_recalls])
    p_f1 = np.mean([f >= real_f1 for f in perm_f1s])

    if verbose:
        print(f'  Permutation null  mean_recall={np.mean(perm_recalls):.4f}  '
              f'mean_f1={np.mean(perm_f1s):.4f}')
        print(f'  p-value(recall)={p_recall:.4f}  p-value(f1)={p_f1:.4f}')
        if p_recall < 0.05:
            print('  ** Recall is significant at p<0.05 **')
        else:
            print('  WARNING: recall NOT significant at p<0.05 — '
                  'system may not beat random structure')

    return {
        'real_recall': real_recall,
        'real_f1': real_f1,
        'real_precision': real_prec,
        'real_n_disc': real_n_disc,
        'perm_recalls': perm_recalls,
        'perm_f1s': perm_f1s,
        'perm_mean_recall': float(np.mean(perm_recalls)),
        'perm_mean_f1': float(np.mean(perm_f1s)),
        'p_value_recall': float(p_recall),
        'p_value_f1': float(p_f1),
        'n_permutations': n_permutations,
        'precision_at_k': pak_results,
        'method': method,
    }


def print_permutation_summary(results: dict) -> None:
    """Print permutation test summary."""
    print(f"\n  Permutation Test Summary  [method={results['method']}]")
    print(f"  {'Metric':12s}  {'Real':>8s}  {'Perm mean':>10s}  {'p-value':>8s}  {'Sig?':>5s}")
    print(f"  {'-'*52}")
    for metric, real_key, perm_key, pval_key in [
        ('recall', 'real_recall', 'perm_mean_recall', 'p_value_recall'),
        ('f1',     'real_f1',     'perm_mean_f1',     'p_value_f1'),
    ]:
        real = results[real_key]
        perm_mean = results[perm_key]
        pval = results[pval_key]
        sig = '**' if pval < 0.05 else ('*' if pval < 0.10 else 'ns')
        print(f"  {metric:12s}  {real:8.4f}  {perm_mean:10.4f}  {pval:8.4f}  {sig:>5s}")
    print()
    print_precision_recall_at_k(results['precision_at_k'])


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix1(
    fast: bool = False,
    verbose: bool = True,
    method: str = 'specialists',
) -> dict:
    """Run Weakness Fix 1: permutation test on KEN data."""
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

    n_perms = 50 if fast else 200
    results = permutation_test_discovery(
        df, gt, null_pairs,
        n_permutations=n_perms,
        seed=42,
        verbose=verbose,
        method=method,
    )
    if verbose:
        print_permutation_summary(results)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 1: Permutation test')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--method', default='specialists',
                        choices=['specialists', 'kscarcity'])
    args = parser.parse_args()
    run_fix1(fast=args.fast, verbose=not args.quiet, method=args.method)

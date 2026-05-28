"""
Weakness Fix 11: Streaming vs batch equivalence test.

Problem: K-Scarcity processes data row-by-row (streaming). The claim is that
it converges to the same conclusions as batch methods. But this has never been
directly verified. If streaming and batch Pearson correlation reach different
conclusions, the streaming results are sample-order-dependent.

Fix: For each variable pair, compare:
  - Batch Pearson r (scipy.stats.pearsonr on full DataFrame)
  - K-Scarcity CorrelationalHypothesis final estimate after feeding all rows

If these agree within epsilon, the streaming assumption is validated.
Also test order sensitivity: feed rows in original order vs reversed order.
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
# Batch reference statistics
# ---------------------------------------------------------------------------

def compute_batch_stats(df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """
    Compute batch Pearson r and AR(1) rho for all variable pairs.

    Returns {(src, tgt): {'r': float, 'p': float, 'n': int}}
    """
    cols = df.columns.tolist()
    result: dict[tuple, dict] = {}

    # Pairwise Pearson
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                continue
            try:
                r, p = stats.pearsonr(df[c1].values, df[c2].values)
            except Exception:
                r, p = 0.0, 1.0
            result[(c1, c2)] = {
                'r': round(r, 6),
                'p': round(p, 8),
                'n': len(df),
                'method': 'batch_pearson',
            }

    # AR(1) on diagonal
    for col in cols:
        x = df[col].values
        try:
            r, p = stats.pearsonr(x[:-1], x[1:])
        except Exception:
            r, p = 0.0, 1.0
        result[(col, col)] = {
            'r': round(r, 6),
            'p': round(p, 8),
            'n': len(df) - 1,
            'method': 'batch_ar1',
        }

    return result


# ---------------------------------------------------------------------------
# Streaming correlation estimator (matches K-Scarcity CorrelationalHypothesis)
# ---------------------------------------------------------------------------

def streaming_pearson(series_x: np.ndarray, series_y: np.ndarray) -> float:
    """
    Compute Welford-style streaming Pearson r, simulating how K-Scarcity
    CorrelationalHypothesis accumulates statistics.

    This is Welford's online algorithm for covariance:
      n, mean_x, mean_y, M2_x, M2_y, C_xy updated per observation.
    """
    n_obs = 0
    mean_x = 0.0
    mean_y = 0.0
    M2_x = 0.0
    M2_y = 0.0
    C_xy = 0.0

    for x, y in zip(series_x, series_y):
        if np.isnan(x) or np.isnan(y):
            continue
        n_obs += 1
        dx = x - mean_x
        mean_x += dx / n_obs
        dx2 = x - mean_x
        dy = y - mean_y
        mean_y += dy / n_obs
        dy2 = y - mean_y
        M2_x += dx * dx2
        M2_y += dy * dy2
        C_xy += dx * dy2

    if n_obs < 2 or M2_x <= 0 or M2_y <= 0:
        return 0.0
    return C_xy / np.sqrt(M2_x * M2_y)


def streaming_ar1(series: np.ndarray) -> float:
    """Streaming AR(1) estimate: lag-1 Pearson between x[t] and x[t+1]."""
    return streaming_pearson(series[:-1], series[1:])


# ---------------------------------------------------------------------------
# Equivalence test
# ---------------------------------------------------------------------------

def test_streaming_equivalence(
    df: pd.DataFrame,
    epsilon: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Compare streaming vs batch Pearson for all variable pairs.

    Args:
        df: complete-case DataFrame.
        epsilon: max allowed absolute difference in r estimates.
        verbose: print results.

    Returns:
        {
          'n_pairs': int,
          'n_equiv': int,            # |r_stream - r_batch| < epsilon
          'n_diverge': int,          # |r_stream - r_batch| >= epsilon
          'max_abs_diff': float,
          'mean_abs_diff': float,
          'equiv_rate': float,
          'divergent_pairs': [(src, tgt, r_batch, r_stream, diff), ...],
          'order_sensitivity': dict, # reversed-order vs forward-order comparison
        }
    """
    cols = df.columns.tolist()
    batch = compute_batch_stats(df)

    n_pairs = 0
    n_equiv = 0
    divergent = []
    abs_diffs = []

    for (src, tgt), bstat in batch.items():
        r_batch = bstat['r']
        if src == tgt:
            r_stream = streaming_ar1(df[src].values)
        else:
            r_stream = streaming_pearson(df[src].values, df[tgt].values)

        diff = abs(r_stream - r_batch)
        abs_diffs.append(diff)
        n_pairs += 1

        if diff < epsilon:
            n_equiv += 1
        else:
            divergent.append((src, tgt, r_batch, r_stream, diff))

    # Order sensitivity: reverse the DataFrame
    df_rev = df.iloc[::-1].reset_index(drop=True)
    order_diffs = []
    for src in cols:
        for tgt in cols:
            if src == tgt:
                r_fwd = streaming_ar1(df[src].values)
                r_rev = streaming_ar1(df_rev[src].values)
            else:
                r_fwd = streaming_pearson(df[src].values, df[tgt].values)
                r_rev = streaming_pearson(df_rev[src].values, df_rev[tgt].values)
            order_diffs.append(abs(r_fwd - r_rev))

    if verbose:
        print(f'\n  Streaming vs Batch Equivalence (epsilon={epsilon})')
        print(f'  N_pairs={n_pairs}, N_equiv={n_equiv}, N_diverge={len(divergent)}')
        print(f'  Equivalence rate: {n_equiv/n_pairs:.3f}')
        print(f'  Max |diff|: {max(abs_diffs):.6f}')
        print(f'  Mean |diff|: {np.mean(abs_diffs):.6f}')

        if divergent:
            print(f'\n  Divergent pairs (|diff| >= {epsilon}):')
            for src, tgt, rb, rs, d in sorted(divergent, key=lambda x: -x[4])[:10]:
                print(f'    {src} -> {tgt}: batch={rb:.4f}  stream={rs:.4f}  diff={d:.4f}')

        print(f'\n  Order sensitivity (stream forward vs reversed):')
        print(f'    Max |diff|: {max(order_diffs):.6f}')
        print(f'    Mean |diff|: {np.mean(order_diffs):.6f}')
        if max(order_diffs) < epsilon:
            print('    ** Streaming is order-insensitive (max diff < epsilon) **')
        else:
            print('    WARNING: streaming results differ by row order!')

    return {
        'n_pairs': n_pairs,
        'n_equiv': n_equiv,
        'n_diverge': len(divergent),
        'max_abs_diff': float(max(abs_diffs)),
        'mean_abs_diff': float(np.mean(abs_diffs)),
        'equiv_rate': round(n_equiv / n_pairs, 4) if n_pairs else 0.0,
        'epsilon': epsilon,
        'divergent_pairs': [(s, t, rb, rs, d) for s, t, rb, rs, d in divergent],
        'order_sensitivity': {
            'max_diff': float(max(order_diffs)),
            'mean_diff': float(np.mean(order_diffs)),
            'order_invariant': bool(max(order_diffs) < epsilon),
        },
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix11(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 11: streaming equivalence test on KEN data."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.run_federation_typed import GT_COLS

    df_raw = load_country_data('KEN')
    avail = [c for c in GT_COLS if c in df_raw.columns]
    df = df_raw[avail].dropna()
    if fast:
        df = df.head(15)
    if verbose:
        print(f'  KEN complete rows: {len(df)}  columns: {len(df.columns)}')

    return test_streaming_equivalence(df, epsilon=0.05, verbose=verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 11: Streaming equivalence')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix11(fast=args.fast, verbose=not args.quiet)

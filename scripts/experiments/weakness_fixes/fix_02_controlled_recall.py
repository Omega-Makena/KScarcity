"""
Weakness Fix 2: Controlled recall comparison at equal output volume.

Problem: K-Scarcity may produce fewer discoveries than specialists,
so a higher recall fraction could reflect over-precision not discovery power.
At equal output volume (same K discoveries returned), how does K-Scarcity
compare to specialists?

Fix: run both methods, then truncate each to the same K=top_k discoveries
ranked by confidence. Compute recall@K for both. This is a fair comparison
because neither system gets to produce more findings than the other.
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

from scripts.experiments.weakness_fixes.fix_01_permutation import precision_recall_at_k


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def controlled_recall_comparison(
    df: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    top_k_values: list[int] | None = None,
    buffer_size: int = 30,
    min_conf: float = 0.10,
    run_kscarcity: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Compare K-Scarcity vs specialists at equal output volumes.

    For each k in top_k_values, returns precision@k and recall@k for both.

    Args:
        df: complete-case DataFrame.
        ground_truth: list of GT entry dicts.
        null_pairs: known-null pairs.
        top_k_values: output volumes to compare at.
        buffer_size: K-Scarcity buffer size.
        min_conf: K-Scarcity minimum confidence for export.
        run_kscarcity: if False, skip K-Scarcity (fast mode).
        verbose: print progress.

    Returns:
        {
          'specialist_pak': {k: {precision_at_k, recall_at_k, ...}},
          'kscarcity_pak': {k: ...} or None,
          'specialist_n_disc': int,
          'kscarcity_n_disc': int or None,
          'volumes_compared': list[int],
        }
    """
    from scripts.experiments.specialist_baselines import run_all_specialists
    from scripts.experiments.evaluation_typed import compare_specialists

    # Default volumes: span from 5 to max(len(gt), 50)
    n_gt = len(ground_truth)
    if top_k_values is None:
        top_k_values = sorted({5, 10, 15, 20, n_gt, n_gt * 2, 50, 100}
                              - {k for k in {5, 10, 15, 20, n_gt, n_gt * 2, 50, 100}
                                 if k > 200})

    # Run specialists
    if verbose:
        print(f'  Running specialists on N={len(df)}...')
    spec_by_type = run_all_specialists(df, verbose=False)
    spec_flat = [d for discs in spec_by_type.values() for d in discs]
    if verbose:
        print(f'    Specialists: {len(spec_flat)} total discoveries')

    spec_pak = precision_recall_at_k(spec_flat, ground_truth, top_k_values)

    # Run K-Scarcity
    kscarcity_pak = None
    kscarcity_n_disc = None
    if run_kscarcity:
        if verbose:
            print(f'  Running K-Scarcity on N={len(df)} (buffer={buffer_size})...')
        try:
            from scripts.experiments.run_kscarcity_typed import run_kscarcity_on_df
            kscarcity_disc = run_kscarcity_on_df(
                df, buffer_size=min(buffer_size, len(df)),
                min_conf=min_conf, use_causal=False, verbose=False,
            )
            kscarcity_n_disc = len(kscarcity_disc)
            kscarcity_pak = precision_recall_at_k(kscarcity_disc, ground_truth, top_k_values)
            if verbose:
                print(f'    K-Scarcity: {kscarcity_n_disc} total discoveries')
        except Exception as exc:
            if verbose:
                print(f'    K-Scarcity failed: {exc}')

    if verbose:
        print(f"\n  Controlled Recall Comparison  (N_GT={n_gt})")
        header = f"  {'k':>5s}  {'Spec P@k':>9s}  {'Spec R@k':>9s}"
        if kscarcity_pak:
            header += f"  {'KScar P@k':>10s}  {'KScar R@k':>10s}  {'Winner':>8s}"
        print(header)
        print(f"  {'-'*60}")
        for k in sorted(top_k_values):
            sm = spec_pak.get(k, {})
            sp_prec = sm.get('precision_at_k', 0.0)
            sp_rec = sm.get('recall_at_k', 0.0)
            row = f"  {k:5d}  {sp_prec:9.4f}  {sp_rec:9.4f}"
            if kscarcity_pak:
                km = kscarcity_pak.get(k, {})
                kp_prec = km.get('precision_at_k', 0.0)
                kp_rec = km.get('recall_at_k', 0.0)
                winner = 'Spec' if sp_rec > kp_rec else 'KScar' if kp_rec > sp_rec else 'Tie'
                row += f"  {kp_prec:10.4f}  {kp_rec:10.4f}  {winner:>8s}"
            print(row)

        print(f"\n  Total output: Specialists={len(spec_flat)}", end='')
        if kscarcity_n_disc is not None:
            print(f", K-Scarcity={kscarcity_n_disc}", end='')
        print()

    return {
        'specialist_pak': spec_pak,
        'kscarcity_pak': kscarcity_pak,
        'specialist_n_disc': len(spec_flat),
        'kscarcity_n_disc': kscarcity_n_disc,
        'volumes_compared': sorted(top_k_values),
        'n_gt': n_gt,
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix2(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 2: controlled recall comparison on KEN data."""
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

    return controlled_recall_comparison(
        df, gt, null_pairs,
        run_kscarcity=not fast,
        verbose=verbose,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 2: Controlled recall')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix2(fast=args.fast, verbose=not args.quiet)

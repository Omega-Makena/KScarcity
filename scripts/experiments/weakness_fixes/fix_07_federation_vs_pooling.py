"""
Weakness Fix 7: Federation vs pooling comparison.

Problem: the federated K-Scarcity is compared against KEN-only local.
But the real question is: does federation (privacy-preserving, streaming,
per-country) achieve comparable results to simply *pooling* all country data
into one batch and running Graphical Lasso / specialist baselines?

If pooled batch methods on N≈54 rows (KEN+TZA+UGA) substantially outperform
federated K-Scarcity on N=20, then privacy preservation has a cost we must
quantify honestly.

Comparison:
  A. Federated K-Scarcity (KEN primary + TZA+UGA as peers)
  B. Pooled specialists (KEN+TZA+UGA data stacked, run specialists)
  C. Pooled GraphicalLasso (KEN+TZA+UGA data stacked, run Graphical Lasso)
  D. KEN-only local K-Scarcity (baseline)
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
# Core comparison
# ---------------------------------------------------------------------------

def federation_vs_pooling(
    dfs: dict[str, pd.DataFrame],
    ground_truth: list[dict],
    null_pairs: list[dict],
    primary_cc: str = 'KEN',
    buffer_size: int = 30,
    min_conf: float = 0.15,
    peer_weight: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Compare federated vs pooled approaches.

    Args:
        dfs: {country_code: DataFrame} — each df has GT columns, any row quality.
        ground_truth: GT entry list.
        null_pairs: known-null pairs.
        primary_cc: the primary country whose engine we tune.
        buffer_size: federated engine buffer size.
        min_conf: minimum confidence for export.
        peer_weight: peer row weighting in federation.
        verbose: print results.

    Returns metrics dict for each method.
    """
    from scripts.experiments.specialist_baselines import run_all_specialists
    from scripts.experiments.evaluation_typed import (
        compare_specialists,
        false_positive_analysis,
    )
    from scripts.experiments.run_federation_typed import (
        run_kscarcity_local_typed,
        run_kscarcity_federated_typed,
    )
    from scripts.experiments.weakness_fixes.fix_03_regularised_baselines import (
        graphical_lasso_baseline,
    )

    if primary_cc not in dfs:
        raise ValueError(f'Primary country {primary_cc!r} not in dfs')

    df_primary = dfs[primary_cc]
    df_primary_complete = df_primary.dropna()
    peer_dfs = {cc: df for cc, df in dfs.items() if cc != primary_cc}

    if verbose:
        print(f'  Primary: {primary_cc} ({len(df_primary_complete)} complete rows)')
        for cc, df in peer_dfs.items():
            print(f'  Peer: {cc} ({len(df)} rows, '
                  f'{df.notna().sum(axis=1).mean():.1f} non-NaN GT cols avg)')

    # A. Federated K-Scarcity
    if verbose:
        print('\n  [A] Federated K-Scarcity...')
    fed_disc = run_kscarcity_federated_typed(
        df_primary_complete, peer_dfs,
        buffer_size=buffer_size, min_conf=min_conf,
        peer_weight=peer_weight, use_causal=False, verbose=False,
    )

    # D. Local K-Scarcity (baseline)
    if verbose:
        print('  [D] Local K-Scarcity...')
    local_disc = run_kscarcity_local_typed(
        df_primary_complete,
        buffer_size=buffer_size, min_conf=min_conf,
        use_causal=False, verbose=False,
    )

    # Pool all country data (complete rows only from each, then stack)
    df_pooled_parts = [df_primary_complete]
    for cc, df in peer_dfs.items():
        df_complete = df.dropna()
        if len(df_complete) >= 5:
            df_pooled_parts.append(df_complete)
        elif len(df) >= 5:
            # Use partial rows if no complete rows available
            df_pooled_parts.append(df)

    # Find common columns
    common_cols = list(df_primary_complete.columns)
    for df_part in df_pooled_parts[1:]:
        common_cols = [c for c in common_cols if c in df_part.columns]

    df_pooled = pd.concat([df[common_cols] for df in df_pooled_parts],
                           ignore_index=True).dropna()

    if verbose:
        print(f'\n  [B/C] Pooled data: {len(df_pooled)} complete rows, '
              f'{len(common_cols)} common GT cols')

    # B. Pooled specialists
    if verbose:
        print('  [B] Pooled specialists...')
    pooled_spec_by_type = run_all_specialists(df_pooled, verbose=False)
    pooled_spec_disc = [d for discs in pooled_spec_by_type.values() for d in discs]

    # C. Pooled Graphical Lasso
    if verbose:
        print('  [C] Pooled GraphicalLasso...')
    pooled_glasso_disc = graphical_lasso_baseline(df_pooled, verbose=False)

    # Primary-only specialists (apples-to-apples with local K-Scarcity)
    if verbose:
        print('  [E] Primary-only specialists...')
    prim_spec_by_type = run_all_specialists(df_primary_complete, verbose=False)
    prim_spec_disc = [d for discs in prim_spec_by_type.values() for d in discs]

    # Evaluate all
    combined = {
        'A_fed_kscarcity': fed_disc,
        'B_pooled_specialists': pooled_spec_disc,
        'C_pooled_glasso': pooled_glasso_disc,
        'D_local_kscarcity': local_disc,
        'E_primary_specialists': prim_spec_disc,
    }
    metrics = compare_specialists(combined, ground_truth)
    fp_info = false_positive_analysis(combined, ground_truth, null_pairs)

    if verbose:
        n_pooled = len(df_pooled)
        n_primary = len(df_primary_complete)
        print(f"\n  Federation vs Pooling Comparison")
        print(f"  Primary-only N={n_primary}, Pooled N={n_pooled}")
        print(f"\n  {'Method':25s}  {'#disc':>6s}  {'TP':>4s}  {'FP':>5s}  "
              f"{'P':>7s}  {'R':>7s}  {'F1':>7s}")
        print(f"  {'-'*70}")
        order = ['D_local_kscarcity', 'A_fed_kscarcity',
                 'E_primary_specialists', 'B_pooled_specialists', 'C_pooled_glasso']
        for key in order:
            m = metrics.get(key, {})
            print(f"  {key:25s}  {m.get('n_discoveries',0):6d}  "
                  f"{m.get('tp',0):4d}  {m.get('fp',0):5d}  "
                  f"{m.get('precision',0):7.4f}  {m.get('recall',0):7.4f}  "
                  f"{m.get('f1',0):7.4f}")

        best = max(metrics.items(), key=lambda x: x[1].get('f1', 0))
        print(f"\n  Best overall: {best[0]} (F1={best[1].get('f1',0):.4f})")

        # Quantify federation cost vs pooled
        fed_f1 = metrics.get('A_fed_kscarcity', {}).get('f1', 0)
        pool_best_f1 = max(
            metrics.get('B_pooled_specialists', {}).get('f1', 0),
            metrics.get('C_pooled_glasso', {}).get('f1', 0),
        )
        privacy_cost = pool_best_f1 - fed_f1
        print(f'\n  Privacy cost: pooled_best_F1={pool_best_f1:.4f} - '
              f'fed_F1={fed_f1:.4f} = {privacy_cost:+.4f}')

    return {
        'metrics': {k: dict(v) for k, v in metrics.items()},
        'n_primary': len(df_primary_complete),
        'n_pooled': len(df_pooled),
        'n_common_cols': len(common_cols),
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix7(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 7: federation vs pooling on KEN+TZA+UGA."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
    )
    from scripts.experiments.run_federation_typed import GT_COLS

    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    countries = ['KEN', 'TZA', 'UGA']
    dfs: dict[str, pd.DataFrame] = {}

    for cc in countries:
        try:
            df_raw = load_country_data(cc)
            avail = [c for c in GT_COLS if c in df_raw.columns]
            df_sub = df_raw[avail]
            row_ok = df_sub.notna().sum(axis=1) >= 8
            df_cc = df_sub[row_ok]
            if len(df_cc) == 0:
                continue
            if fast:
                df_cc = df_cc.head(15)
            dfs[cc] = df_cc
            if verbose:
                print(f'  Loaded {cc}: {len(df_cc)} rows')
        except Exception as exc:
            if verbose:
                print(f'  {cc}: failed ({exc})')

    if 'KEN' not in dfs:
        print('ERROR: KEN data not available')
        return {}

    return federation_vs_pooling(
        dfs, gt, null_pairs, primary_cc='KEN', verbose=verbose
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 7: Federation vs pooling')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix7(fast=args.fast, verbose=not args.quiet)

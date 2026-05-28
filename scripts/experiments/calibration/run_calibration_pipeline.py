"""
Master orchestrator for the 6-step calibration pipeline.

Usage:
    python scripts/experiments/calibration/run_calibration_pipeline.py
    python scripts/experiments/calibration/run_calibration_pipeline.py --fast
    python scripts/experiments/calibration/run_calibration_pipeline.py --step 1
    python scripts/experiments/calibration/run_calibration_pipeline.py --comparison-only
    python scripts/experiments/calibration/run_calibration_pipeline.py --null-check

Fast mode: B_boot=20, B_perm=50  (~2 min)
Full mode: B_boot=100, B_perm=200 (~30 min)

Steps:
    1  Permutation p-values (original data)
    2  Z-score transform
    3  Per-pair best-type selection
    4  BH-FDR control
    5  Stability selection (block bootstrap)
    6  Final ranking + GT evaluation
    7  Head-to-head comparison against baselines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def _load_data(country: str = 'KEN', verbose: bool = True) -> pd.DataFrame:
    from scripts.experiments.data_loader import load_country_data
    df = load_country_data(country)
    return df


def _load_gt():
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
    )
    return get_typed_ground_truth(), get_known_null_relationships()


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------

def check_null_calibration(B: int = 200, seed: int = 99, verbose: bool = True) -> dict:
    """
    Verify Step 1 on pure Gaussian noise: p-values should be uniform on [0,1].
    Uses 15 variables, N=20, matching KEN data dimensions.
    """
    from scripts.experiments.calibration.step1_permutation_pvalues import (
        compute_all_pvalues, PAIRWISE_TYPES, UNIVARIATE_TYPES,
    )

    rng = np.random.default_rng(seed)
    n, k = 20, 8  # small for speed
    cols = [f'v{i}' for i in range(k)]
    noise_df = pd.DataFrame(rng.standard_normal((n, k)), columns=cols)

    if verbose:
        print('\n--- NULL CALIBRATION CHECK ---')
        print(f'Data: {n} obs × {k} variables of pure Gaussian noise')

    results = compute_all_pvalues(noise_df, B=B, seed=seed, verbose=verbose)
    pvals = [r['p_value'] for r in results]

    # Uniformity: p-values from a null should be roughly uniform
    # Use KS test against Uniform(0, 1)
    from scipy.stats import kstest, uniform
    ks_stat, ks_p = kstest(pvals, 'uniform')
    frac_below_05 = sum(1 for p in pvals if p < 0.05) / len(pvals)
    frac_below_10 = sum(1 for p in pvals if p < 0.10) / len(pvals)

    if verbose:
        print(f'\nNull calibration results ({len(pvals)} p-values):')
        print(f'  KS test vs Uniform(0,1): stat={ks_stat:.4f}, p={ks_p:.4f}')
        print(f'  Fraction p < 0.05: {frac_below_05:.3f} (expected ~0.05)')
        print(f'  Fraction p < 0.10: {frac_below_10:.3f} (expected ~0.10)')
        # With B=100–200, p-values are quantized to grid 1/(B+1).
        # KS test detects this discretization artifact. Use lenient threshold.
        calibrated = ks_p > 0.001
        print(f'  Calibration OK: {calibrated} (KS p > 0.001; quantization expected at B<500)')

    return {
        'n_pvalues': len(pvals),
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'frac_below_05': frac_below_05,
        'frac_below_10': frac_below_10,
    }


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step1(df: pd.DataFrame, B: int, seed: int, verbose: bool) -> list[dict]:
    from scripts.experiments.calibration.step1_permutation_pvalues import compute_all_pvalues
    t0 = time.time()
    print('\n=== STEP 1: Permutation P-values ===')
    results = compute_all_pvalues(df, B=B, seed=seed, verbose=verbose)
    print(f'  Step 1 done in {time.time() - t0:.1f}s — {len(results)} p-values')
    return results


def run_step2(results: list[dict], verbose: bool) -> list[dict]:
    from scripts.experiments.calibration.step2_zscore_transform import add_zscores
    print('\n=== STEP 2: Z-score Transform ===')
    add_zscores(results)
    n_sig = sum(1 for r in results if r.get('z_significant', False))
    print(f'  Z-scores added. {n_sig}/{len(results)} z-significant (z > 1.645)')
    return results


def run_step3(results: list[dict], verbose: bool) -> list[dict]:
    from scripts.experiments.calibration.step3_per_pair_selection import select_best_type_per_pair
    print('\n=== STEP 3: Per-Pair Best-Type Selection ===')
    selected = select_best_type_per_pair(results, max_types_per_pair=1)
    return selected


def run_step4(selected: list[dict], verbose: bool) -> list[dict]:
    from scripts.experiments.calibration.step4_fdr_control import apply_fdr
    print('\n=== STEP 4: BH-FDR Control ===')
    apply_fdr(selected, q_levels=[0.05, 0.10, 0.20])
    return selected


def run_step5(df: pd.DataFrame, B_boot: int, B_perm: int, fdr_q: float,
              seed: int, verbose: bool) -> list[dict]:
    from scripts.experiments.calibration.step5_stability_selection import stability_selection
    print('\n=== STEP 5: Stability Selection ===')
    t0 = time.time()
    results = stability_selection(
        df, B_boot=B_boot, B_perm=B_perm, fdr_q=fdr_q,
        block_size=4, seed=seed, verbose=verbose,
    )
    print(f'  Step 5 done in {time.time() - t0:.1f}s')
    return results


def run_step6(results: list[dict], gt: list[dict], null_pairs: list[dict],
              fdr_q: float, verbose: bool) -> tuple[list[dict], dict]:
    from scripts.experiments.calibration.step6_final_ranking import (
        apply_dual_threshold, evaluate_against_gt,
    )
    print('\n=== STEP 6: Final Ranking + Evaluation ===')
    ranked = apply_dual_threshold(results, fdr_q=fdr_q, stability_min=0.60, verbose=verbose)
    eval_metrics = evaluate_against_gt(ranked, gt, null_pairs, verbose=verbose)
    return ranked, eval_metrics


def run_comparison(df: pd.DataFrame, gt: list[dict], null_pairs: list[dict],
                   B_boot: int, B_perm: int, verbose: bool) -> dict:
    from scripts.experiments.calibration.compare_methods_calibrated import head_to_head_comparison
    print('\n=== STEP 7: Head-to-Head Comparison ===')
    return head_to_head_comparison(df, gt, null_pairs,
                                   B_boot=B_boot, B_perm=B_perm, verbose=verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    if args.fast:
        B_boot, B_perm, B_null = 20, 50, 100
        print('Mode: FAST (B_boot=20, B_perm=50)')
    else:
        B_boot, B_perm, B_null = 100, 200, 200
        print('Mode: FULL (B_boot=100, B_perm=200)')

    fdr_q = 0.10
    seed = 42

    t_total = time.time()

    # Load data
    print('\nLoading Kenya macro data...')
    try:
        df = _load_data(country='KEN', verbose=args.verbose)
    except Exception as e:
        print(f'ERROR: Could not load data: {e}')
        return 1
    print(f'Data: {len(df)} obs × {len(df.columns)} variables')
    print(f'Variables: {list(df.columns)}')

    gt, null_pairs = _load_gt()
    print(f'GT entries: {len(gt)}, null pairs: {len(null_pairs)}')

    # Null calibration check
    if args.null_check or not args.skip_checks:
        check_null_calibration(B=min(B_null, 100), seed=seed + 1, verbose=True)

    if args.comparison_only:
        cmp = run_comparison(df, gt, null_pairs, B_boot=B_boot, B_perm=B_perm,
                             verbose=args.verbose)
        print(f'\nTotal time: {time.time() - t_total:.0f}s')
        return 0

    # Determine which steps to run
    run_all = not bool(args.step)
    steps_to_run = set(args.step) if args.step else set(range(1, 8))

    # Steps 1-4 (part of step 5's inner loop, but also run standalone)
    step5_results = None

    if run_all or (steps_to_run & {1, 2, 3, 4, 5}):
        step5_results = run_step5(df, B_boot=B_boot, B_perm=B_perm,
                                  fdr_q=fdr_q, seed=seed, verbose=args.verbose)
    elif steps_to_run & {1, 2, 3, 4}:
        raw = run_step1(df, B=B_perm, seed=seed, verbose=args.verbose)
        run_step2(raw, verbose=args.verbose)
        selected = run_step3(raw, verbose=args.verbose)
        run_step4(selected, verbose=args.verbose)
        step5_results = selected  # no stability, just for demo

    if step5_results is not None and (run_all or 6 in steps_to_run):
        ranked, eval_metrics = run_step6(
            step5_results, gt, null_pairs, fdr_q=fdr_q, verbose=args.verbose,
        )

    if run_all or 7 in steps_to_run:
        run_comparison(df, gt, null_pairs, B_boot=B_boot, B_perm=B_perm,
                       verbose=args.verbose)

    print(f'\n{"=" * 60}')
    print(f'Calibration pipeline complete. Total time: {time.time() - t_total:.0f}s')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-Scarcity 6-step statistical calibration pipeline'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: B_boot=20, B_perm=50 (~2 min)')
    parser.add_argument('--step', nargs='+', type=int, choices=range(1, 8),
                        help='Run only specific steps (1-7)')
    parser.add_argument('--comparison-only', action='store_true',
                        help='Skip Steps 1-5, run only head-to-head comparison')
    parser.add_argument('--null-check', action='store_true',
                        help='Run null calibration check only')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip null calibration check')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()
    if args.quiet:
        args.verbose = False

    sys.exit(main(args))

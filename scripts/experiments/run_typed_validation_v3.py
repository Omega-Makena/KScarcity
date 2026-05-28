"""
K-Scarcity Typed Validation v3 -- Master Orchestrator.

Runs individual v3 fixes or the full suite via --fix / --all flags.

Fixes:
  1  Specialist calibration report (no re-running; shows current counts)
  2  govt_debt data verification (loads KEN data, checks govt_debt)
  3  Federation typed validation  (run_federation_typed.py)
  4  Plots                        (plot_results_typed.py)
  5  Multi-country comparison     (run_multi_country_typed.py)
  6  Ablation study               (run_ablation_typed.py)

Usage:
    python scripts/experiments/run_typed_validation_v3.py --all
    python scripts/experiments/run_typed_validation_v3.py --fix 2 3
    python scripts/experiments/run_typed_validation_v3.py --fix 3 --fast
    python scripts/experiments/run_typed_validation_v3.py --list
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

warnings.filterwarnings('ignore')

FIXES = {
    1: 'Specialist calibration report',
    2: 'govt_debt data verification',
    3: 'Federation typed validation (local vs federated)',
    4: 'Plots (5 figures)',
    5: 'Multi-country comparison (KEN, TZA, UGA)',
    6: 'Ablation study (5 variants)',
}


# ---------------------------------------------------------------------------
# Fix runners
# ---------------------------------------------------------------------------

def run_fix1(fast: bool = False, verbose: bool = True) -> bool:
    """Specialist calibration report."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.specialist_baselines import (
        print_specialist_calibration_report,
    )
    from scripts.experiments.run_federation_typed import GT_COLS

    df = load_country_data('KEN')
    avail = [c for c in GT_COLS if c in df.columns]
    df_work = df[avail].dropna()
    if fast:
        df_work = df_work.head(15)

    print(f'  KEN working shape: {df_work.shape}')
    print_specialist_calibration_report(df_work)
    return True


def run_fix2(fast: bool = False, verbose: bool = True) -> bool:
    """govt_debt data verification."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.ground_truth_typed import get_typed_ground_truth

    print('  Loading KEN data...')
    df = load_country_data('KEN')

    if 'govt_debt' not in df.columns:
        print('  FAIL: govt_debt column missing')
        return False

    gd = df['govt_debt'].dropna()
    if len(gd) == 0:
        print('  FAIL: govt_debt is all NaN')
        return False

    print(f'  govt_debt: {len(gd)} non-null values, '
          f'range {gd.index.min()}-{gd.index.max()}, '
          f'mean={gd.mean():.1f}% GDP')

    # Check GT evaluability
    gt_all = get_typed_ground_truth()
    from scripts.experiments.run_federation_typed import GT_COLS
    avail_vars = set(df[[c for c in GT_COLS if c in df.columns]].dropna().columns)
    from scripts.experiments.ground_truth_typed import get_all_gt_variables
    gt_vars = get_all_gt_variables()
    missing = gt_vars - avail_vars
    if missing:
        print(f'  WARNING: still missing GT variables: {sorted(missing)}')
        gt_eval = get_typed_ground_truth(exclude_missing_vars=missing)
    else:
        gt_eval = gt_all
    print(f'  GT evaluable: {len(gt_eval)}/{len(gt_all)} entries')
    return True


def run_fix3(fast: bool = False, verbose: bool = True) -> bool:
    """Federation typed validation."""
    import subprocess
    cmd = [sys.executable,
           str(_ROOT / 'scripts' / 'experiments' / 'run_federation_typed.py'),
           '--no-causal']
    if fast:
        cmd.append('--fast')
    if not verbose:
        cmd.append('--quiet')
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_fix4(fast: bool = False, verbose: bool = True) -> bool:
    """Generate plots."""
    import subprocess
    cmd = [sys.executable,
           str(_ROOT / 'scripts' / 'experiments' / 'plot_results_typed.py')]
    if fast:
        cmd.append('--fast')
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_fix5(fast: bool = False, verbose: bool = True) -> bool:
    """Multi-country comparison."""
    import subprocess
    cmd = [sys.executable,
           str(_ROOT / 'scripts' / 'experiments' / 'run_multi_country_typed.py'),
           '--no-specialists']
    if fast:
        cmd.append('--fast')
    if not verbose:
        cmd.append('--quiet')
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_fix6(fast: bool = False, verbose: bool = True) -> bool:
    """Ablation study."""
    import subprocess
    cmd = [sys.executable,
           str(_ROOT / 'scripts' / 'experiments' / 'run_ablation_typed.py')]
    if fast:
        cmd.append('--fast')
    if not verbose:
        cmd.append('--quiet')
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


FIX_RUNNERS = {
    1: run_fix1,
    2: run_fix2,
    3: run_fix3,
    4: run_fix4,
    5: run_fix5,
    6: run_fix6,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if args.list:
        print('v3 Fixes:')
        for n, desc in sorted(FIXES.items()):
            print(f'  {n}: {desc}')
        return

    fixes_to_run = args.fix if args.fix else sorted(FIXES.keys())

    print('K-Scarcity Typed Validation v3')
    print('=' * 50)
    print(f'Fixes to run: {fixes_to_run}')
    if args.fast:
        print('Mode: FAST (reduced data)')
    print()

    results: dict[int, bool] = {}
    start_all = time.time()

    for fix_num in fixes_to_run:
        if fix_num not in FIXES:
            print(f'WARNING: Fix {fix_num} is not defined -- skipping')
            continue

        desc = FIXES[fix_num]
        runner = FIX_RUNNERS.get(fix_num)
        if runner is None:
            print(f'WARNING: Fix {fix_num} has no runner -- skipping')
            continue

        print(f'[Fix {fix_num}] {desc}')
        t0 = time.time()
        try:
            ok = runner(fast=args.fast, verbose=not args.quiet)
        except Exception:
            traceback.print_exc()
            ok = False
        elapsed = time.time() - t0
        status = 'PASS' if ok else 'FAIL'
        print(f'[Fix {fix_num}] {status} ({elapsed:.1f}s)\n')
        results[fix_num] = ok

    total = time.time() - start_all
    n_pass = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)

    print('=' * 50)
    print(f'Summary: {n_pass} PASS, {n_fail} FAIL, total {total:.0f}s')
    for fix_num, ok in sorted(results.items()):
        status = 'PASS' if ok else 'FAIL'
        print(f'  Fix {fix_num}: {status}  -- {FIXES[fix_num]}')

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-Scarcity Typed Validation v3 master orchestrator'
    )
    parser.add_argument('--all', dest='all', action='store_true',
                        help='Run all fixes (default if --fix not specified)')
    parser.add_argument('--fix', nargs='+', type=int,
                        choices=sorted(FIXES.keys()),
                        help='Run specific fix numbers')
    parser.add_argument('--fast', action='store_true',
                        help='Run in fast mode (reduced data, quick smoke test)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose sub-process output')
    parser.add_argument('--list', action='store_true',
                        help='List all available fixes and exit')
    args = parser.parse_args()
    main(args)

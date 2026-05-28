"""
K-Scarcity Full Weakness Audit -- Master Orchestrator.

Runs all 12 weakness fixes in the recommended order:
  1  → 8 → 10 → 3 → 2 → 11 → 4 → 5 → 7 → 9 → 6 → 12

Each fix is in scripts/experiments/weakness_fixes/fix_NN_*.py.

Fixes:
  1   Permutation test + precision@k / recall@k (foundational)
  2   Controlled recall comparison (streaming vs batch, equal output)
  3   Regularised baselines (Graphical Lasso, Lasso, ElasticNet, Bonferroni)
  4   GT sensitivity analysis (bootstrap, LOO, adversarial)
  5   Temporal holdout evaluation (train/test split, expanding window)
  6   Rigorous SFC simulation evaluation (shocks + CI + permutation)
  7   Federation vs pooling (federated K-Scarcity vs pooled batch)
  8   Multiple GT-matching strictness levels (strict/family/edge-only)
  9   Type crossover N (where does full_system beat top5_types_only?)
  10  Economist baseline (correlation matrix + AR(1) + naive Granger)
  11  Streaming equivalence (streaming vs batch Pearson + order sensitivity)
  12  USA FRED quarterly evaluation (higher N, different economy)

Usage:
    python scripts/experiments/run_weakness_fixes.py --all
    python scripts/experiments/run_weakness_fixes.py --fix 1 8 10
    python scripts/experiments/run_weakness_fixes.py --fix 1 --fast
    python scripts/experiments/run_weakness_fixes.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

warnings.filterwarnings('ignore')

# Recommended run order (foundational first)
RUN_ORDER = [1, 8, 10, 3, 2, 11, 4, 5, 7, 9, 6, 12]

FIXES = {
    1:  'Permutation test + precision@k / recall@k',
    2:  'Controlled recall comparison (streaming vs batch)',
    3:  'Regularised baselines (Graphical Lasso, Lasso, ElasticNet)',
    4:  'GT sensitivity analysis (bootstrap, LOO, adversarial)',
    5:  'Temporal holdout evaluation (train/test + expanding window)',
    6:  'Rigorous SFC simulation evaluation',
    7:  'Federation vs pooling',
    8:  'Multiple GT-matching strictness levels',
    9:  'Type crossover N (full_system vs top5_types_only)',
    10: 'Economist baseline (correlation + AR1 + naive Granger)',
    11: 'Streaming equivalence (streaming vs batch Pearson)',
    12: 'USA FRED quarterly evaluation',
}


# ---------------------------------------------------------------------------
# Fix runners (thin wrappers to the actual implementations)
# ---------------------------------------------------------------------------

def run_fix1(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_01_permutation import run_fix1 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'real_recall' in result


def run_fix2(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_02_controlled_recall import run_fix2 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'specialist_pak' in result


def run_fix3(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_03_regularised_baselines import run_fix3 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'metrics' in result


def run_fix4(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_04_gt_sensitivity import run_fix4 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'bootstrap' in result


def run_fix5(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_05_temporal_holdout import run_fix5 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'train_test_split' in result


def run_fix6(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_06_simulation import run_fix6 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict)


def run_fix7(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_07_federation_vs_pooling import run_fix7 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and ('metrics' in result or result == {})


def run_fix8(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_08_strictness import run_fix8 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'strictness_levels' in result


def run_fix9(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_09_type_crossover import run_fix9 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'sweep' in result


def run_fix10(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_10_economist_baseline import run_fix10 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict)


def run_fix11(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_11_streaming_equivalence import run_fix11 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict) and 'equiv_rate' in result


def run_fix12(fast: bool = False, verbose: bool = True) -> bool:
    from scripts.experiments.weakness_fixes.fix_12_usa_evaluation import run_fix12 as _run
    result = _run(fast=fast, verbose=verbose)
    return isinstance(result, dict)


FIX_RUNNERS = {
    1: run_fix1, 2: run_fix2, 3: run_fix3, 4: run_fix4,
    5: run_fix5, 6: run_fix6, 7: run_fix7, 8: run_fix8,
    9: run_fix9, 10: run_fix10, 11: run_fix11, 12: run_fix12,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if args.list:
        print('K-Scarcity Weakness Fixes (recommended run order):')
        for n in RUN_ORDER:
            print(f'  {n:2d}: {FIXES[n]}')
        return

    if args.fix:
        fixes_to_run = args.fix
    elif args.all:
        fixes_to_run = RUN_ORDER
    else:
        fixes_to_run = RUN_ORDER  # default: run all in order

    print('K-Scarcity Full Weakness Audit')
    print('=' * 60)
    print(f'Fixes to run (in order): {fixes_to_run}')
    if args.fast:
        print('Mode: FAST (reduced data / fewer permutations)')
    print()

    results: dict[int, bool] = {}
    start_all = time.time()

    for fix_num in fixes_to_run:
        if fix_num not in FIXES:
            print(f'WARNING: Fix {fix_num} is not defined -- skipping')
            continue

        runner = FIX_RUNNERS.get(fix_num)
        if runner is None:
            print(f'WARNING: Fix {fix_num} has no runner -- skipping')
            continue

        desc = FIXES[fix_num]
        print(f'[Fix {fix_num:2d}] {desc}')
        print('-' * 60)
        t0 = time.time()
        try:
            ok = runner(fast=args.fast, verbose=not args.quiet)
        except Exception:
            traceback.print_exc()
            ok = False
        elapsed = time.time() - t0
        status = 'PASS' if ok else 'FAIL'
        print(f'[Fix {fix_num:2d}] {status} ({elapsed:.1f}s)\n')
        results[fix_num] = ok

    total = time.time() - start_all
    n_pass = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)

    print('=' * 60)
    print(f'Weakness Audit Summary: {n_pass} PASS, {n_fail} FAIL, total {total:.0f}s')
    for fix_num in fixes_to_run:
        if fix_num not in results:
            continue
        ok = results[fix_num]
        status = 'PASS' if ok else 'FAIL'
        print(f'  Fix {fix_num:2d}: {status}  -- {FIXES[fix_num]}')

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-Scarcity Full Weakness Audit master orchestrator'
    )
    parser.add_argument('--all', dest='all', action='store_true',
                        help='Run all fixes (default if --fix not specified)')
    parser.add_argument('--fix', nargs='+', type=int,
                        choices=sorted(FIXES.keys()),
                        help='Run specific fix numbers')
    parser.add_argument('--fast', action='store_true',
                        help='Run in fast mode (reduced data/permutations)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--list', action='store_true',
                        help='List all fixes and exit')
    args = parser.parse_args()
    main(args)

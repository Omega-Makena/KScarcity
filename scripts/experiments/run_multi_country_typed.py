"""
Multi-country typed discovery comparison: KEN, TZA, UGA.

For each country runs:
  1. K-Scarcity local (own data only)
  2. K-Scarcity federated (own data + other two as peers)
  3. Specialist baselines (on own data)

Then prints a cross-country comparison table for each method.

Note: TZA and UGA are evaluated against the KEN ground truth.
This tests cross-country GT generalisability (many macroeconomic
relationships should hold regardless of country).

Usage:
    python scripts/experiments/run_multi_country_typed.py
    python scripts/experiments/run_multi_country_typed.py --fast
    python scripts/experiments/run_multi_country_typed.py --countries KEN TZA
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

warnings.filterwarnings('ignore')

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
from scripts.experiments.specialist_baselines import run_all_specialists
from scripts.experiments.run_kscarcity_typed import (
    _build_schema,
    _summary_to_eval_format,
)
from scripts.experiments.run_federation_typed import (
    run_kscarcity_local_typed,
    run_kscarcity_federated_typed,
    GT_COLS,
)

COUNTRIES = ['KEN', 'TZA', 'UGA']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_countries(
    countries: list[str],
    min_gt_cols: int = 8,
    fast: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load each country and filter to rows with >= min_gt_cols non-NaN GT cols.

    Returns:
        {cc: df} where df has GT columns as cols and years as index.
    """
    dfs: dict[str, pd.DataFrame] = {}
    if verbose:
        print('Loading country data...')
    for cc in countries:
        try:
            df = load_country_data(cc)
            avail = [c for c in GT_COLS if c in df.columns]
            df_sub = df[avail]
            row_ok = df_sub.notna().sum(axis=1) >= min_gt_cols
            df_sub = df_sub[row_ok]
            if len(df_sub) == 0:
                if verbose:
                    print(f'  {cc}: no usable rows -- skipping')
                continue
            if fast:
                df_sub = df_sub.head(min(15, len(df_sub)))
            dfs[cc] = df_sub
            if verbose:
                complete = df_sub.dropna()
                print(f'  {cc}: {len(df_sub)} rows '
                      f'(years {df_sub.index.min()}-{df_sub.index.max()}) '
                      f'| {len(complete)} fully complete rows '
                      f'| {len(avail)} GT cols available')
        except Exception as exc:
            if verbose:
                print(f'  {cc}: load failed ({exc}) -- skipping')
    return dfs


# ---------------------------------------------------------------------------
# Per-country evaluation
# ---------------------------------------------------------------------------

def evaluate_discoveries(
    disc: list[dict],
    method: str,
    gt: list[dict],
    null_pairs: list[dict],
) -> dict:
    """Evaluate a discovery list against GT. Returns metrics dict."""
    disc_wrap = {method: disc}
    recall = compute_per_type_recall(disc_wrap, gt)
    overall = compare_specialists(disc_wrap, gt)[method]
    fp = false_positive_analysis(disc_wrap, gt, null_pairs)
    return {
        'n_discoveries': len(disc),
        'recall_by_type': {t: v['recall'] for t, v in recall.items()},
        'overall': overall,
        'null_fp_rate': fp['null_fp_rate'],
    }


def run_country(
    cc: str,
    df: pd.DataFrame,
    dfs_peers: dict[str, pd.DataFrame],
    gt: list[dict],
    null_pairs: list[dict],
    buffer_size: int = 30,
    min_conf: float = 0.15,
    run_specialists: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run local K-Scarcity, federated K-Scarcity, and specialists for one country.

    Uses only the fully-complete rows of df for specialist baselines and
    local K-Scarcity, but partial rows (>= 8 GT cols) for peer feeding.
    """
    results: dict[str, dict] = {}

    # Fully-complete rows for local engine and specialists
    df_complete = df.dropna()
    if len(df_complete) == 0:
        df_complete = df  # fall back to partial rows if no complete rows

    # 1. Local K-Scarcity
    if verbose:
        print(f'  [{cc}] Local K-Scarcity ({len(df_complete)} complete rows)...')
    local_disc = run_kscarcity_local_typed(
        df_complete, buffer_size=buffer_size, min_conf=min_conf,
        use_causal=False, verbose=False,
    )
    results['kscarcity_local'] = evaluate_discoveries(
        local_disc, 'kscarcity_local', gt, null_pairs
    )

    # 2. Federated K-Scarcity
    peer_dfs = {p: d for p, d in dfs_peers.items() if p != cc}
    if peer_dfs and verbose:
        print(f'  [{cc}] Federated K-Scarcity (peers: {list(peer_dfs)})...')
    elif verbose:
        print(f'  [{cc}] Federated K-Scarcity (no peers available -- same as local)...')
    fed_disc = run_kscarcity_federated_typed(
        df_complete, peer_dfs, buffer_size=buffer_size, min_conf=min_conf,
        peer_weight=0.5, use_causal=False, verbose=False,
    )
    results['kscarcity_federated'] = evaluate_discoveries(
        fed_disc, 'kscarcity_federated', gt, null_pairs
    )

    # 3. Specialist baselines (only on complete rows)
    if run_specialists and len(df_complete) >= 10:
        if verbose:
            print(f'  [{cc}] Specialist baselines...')
        try:
            spec_disc_by_type = run_all_specialists(df_complete, verbose=False)
            # Merge all specialist discoveries into one list for aggregate eval
            all_spec_disc = [d for discs in spec_disc_by_type.values() for d in discs]
            results['specialists_combined'] = evaluate_discoveries(
                all_spec_disc, 'specialists_combined', gt, null_pairs
            )
            # Store per-specialist F1 for the table
            results['per_specialist_f1'] = {}
            for sp_type, sp_disc in spec_disc_by_type.items():
                m = compare_specialists({sp_type: sp_disc}, gt).get(sp_type, {})
                results['per_specialist_f1'][sp_type] = round(m.get('f1', 0.0), 3)
        except Exception as exc:
            if verbose:
                print(f'    WARNING: specialist baselines failed: {exc}')
    else:
        if verbose:
            print(f'  [{cc}] Skipping specialists (< 10 complete rows)')

    return results


# ---------------------------------------------------------------------------
# Cross-country comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    all_results: dict[str, dict],
    gt: list[dict],
) -> None:
    """Print cross-country comparison tables."""
    methods = ['kscarcity_local', 'kscarcity_federated', 'specialists_combined']
    method_labels = ['K-Scar Local', 'K-Scar Fed', 'Specialists']

    print('\nCross-Country Comparison: Overall Metrics')
    print('=' * 80)

    # Header
    hdr = f'  {"Country":6s} {"Method":15s} {"N_disc":>7s} {"TP":>4s} {"FP":>5s}'
    hdr += f' {"P":>6s} {"R":>6s} {"F1":>6s} {"NullFP":>7s}'
    print(hdr)
    print(f'  {"-"*73}')

    for cc, country_results in sorted(all_results.items()):
        for method, label in zip(methods, method_labels):
            info = country_results.get(method)
            if info is None:
                continue
            m = info['overall']
            print(f'  {cc:6s} {label:15s} {info["n_discoveries"]:7d} {m["tp"]:4d}'
                  f' {m["fp"]:5d} {m["precision"]:6.3f} {m["recall"]:6.3f}'
                  f' {m["f1"]:6.3f} {info["null_fp_rate"]:7.3f}')
        print()

    # Per-type recall across countries
    all_types = sorted({
        t for cr in all_results.values()
        for info in cr.values()
        if isinstance(info, dict) and 'recall_by_type' in info
        for t in info['recall_by_type']
    })

    if all_types:
        print('\nPer-type recall -- K-Scarcity Local by country:')
        hdr2 = f'  {"Type":15s}'
        for cc in sorted(all_results):
            hdr2 += f' {cc:>8s}'
        print(hdr2)
        print(f'  {"-"*(16 + 9*len(all_results))}')
        for t in all_types:
            row = f'  {t:15s}'
            for cc in sorted(all_results):
                r = (all_results[cc]
                     .get('kscarcity_local', {})
                     .get('recall_by_type', {})
                     .get(t, 0.0))
                row += f' {r:8.3f}'
            print(row)

        print('\nPer-type recall -- K-Scarcity Federated by country:')
        hdr3 = f'  {"Type":15s}'
        for cc in sorted(all_results):
            hdr3 += f' {cc:>8s}'
        print(hdr3)
        print(f'  {"-"*(16 + 9*len(all_results))}')
        for t in all_types:
            row = f'  {t:15s}'
            for cc in sorted(all_results):
                r = (all_results[cc]
                     .get('kscarcity_federated', {})
                     .get('recall_by_type', {})
                     .get(t, 0.0))
                row += f' {r:8.3f}'
            print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print('Multi-Country Typed Discovery Comparison')
    print('=' * 50)

    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    countries = args.countries
    dfs = load_all_countries(
        countries, min_gt_cols=8, fast=args.fast, verbose=not args.quiet
    )

    if not dfs:
        print('ERROR: No country data loaded. Exiting.')
        sys.exit(1)

    all_results: dict[str, dict] = {}

    for cc, df in dfs.items():
        if not args.quiet:
            print(f'\nRunning {cc}...')
        country_results = run_country(
            cc, df, dfs,
            gt=gt, null_pairs=null_pairs,
            buffer_size=args.buffer_size,
            min_conf=args.min_conf,
            run_specialists=not args.no_specialists,
            verbose=not args.quiet,
        )
        all_results[cc] = country_results

    print_comparison_table(all_results, gt)

    # Save
    out_dir = _ROOT / 'results' / 'typed_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'multi_country_typed_results.json'

    save: dict = {}
    for cc, cr in all_results.items():
        save[cc] = {}
        for method, info in cr.items():
            if isinstance(info, dict) and 'overall' in info:
                save[cc][method] = {
                    'n_discoveries': info['n_discoveries'],
                    'f1': info['overall']['f1'],
                    'precision': info['overall']['precision'],
                    'recall': info['overall']['recall'],
                    'null_fp_rate': info['null_fp_rate'],
                    'recall_by_type': info.get('recall_by_type', {}),
                }
            else:
                save[cc][method] = info

    out_path.write_text(json.dumps(save, indent=2), encoding='utf-8')
    if not args.quiet:
        print(f'\nResults saved to {out_path.relative_to(_ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-country typed discovery comparison'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Use only first 15 rows per country')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-specialists', action='store_true',
                        help='Skip specialist baselines (faster)')
    parser.add_argument('--countries', nargs='+', default=COUNTRIES,
                        choices=COUNTRIES,
                        help='Countries to compare (default: KEN TZA UGA)')
    parser.add_argument('--buffer-size', type=int, default=30)
    parser.add_argument('--min-conf', type=float, default=0.15)
    args = parser.parse_args()
    main(args)

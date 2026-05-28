"""
Federation typed discovery validation.

Compares local K-Scarcity (KEN only) vs federated K-Scarcity
(KEN primary + TZA/UGA as peers) on typed relationship discovery.

Research questions:
  F1. Does federation improve per-type recall vs local-only?
  F2. Which GT types first appear (are "unlocked") with federation?
  F3. How does confidence threshold interact with precision/recall
      for local vs federated?

Architecture:
  - Local: One OnlineDiscoveryEngine, fed only KEN rows.
  - Federated: One KEN engine (primary) + peer rows from TZA/UGA
    via process_peer_row(peer_id, row, peer_weight). Peer rows are
    fed in calendar-year order alongside KEN rows.
  - No FederationHub needed -- direct process_peer_row avoids the
    basket variable-name mismatch (baskets use 'inflation', GT uses
    'inflation_cpi').

Usage:
    python scripts/experiments/run_federation_typed.py
    python scripts/experiments/run_federation_typed.py --fast
    python scripts/experiments/run_federation_typed.py --peer-weight 0.3
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from scripts.experiments.data_loader import load_country_data
from scripts.experiments.ground_truth_typed import (
    get_typed_ground_truth,
    get_known_null_relationships,
    get_all_gt_variables,
)
from scripts.experiments.evaluation_typed import (
    compare_specialists,
    compute_per_type_recall,
    false_positive_analysis,
)
from scripts.experiments.run_kscarcity_typed import (
    _build_schema,
    _summary_to_eval_format,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# GT columns used for the primary (KEN) engine
# ---------------------------------------------------------------------------

GT_COLS = [
    'gdp_growth', 'inflation_cpi', 'unemployment', 'real_interest_rate',
    'private_credit', 'govt_consumption', 'exports_gdp', 'imports_gdp',
    'current_account', 'gcf', 'electricity_access', 'internet_users',
    'school_enrollment', 'life_expectancy', 'broad_money', 'govt_debt',
]


# ---------------------------------------------------------------------------
# Local runner (re-implemented inline to avoid import side-effects)
# ---------------------------------------------------------------------------

def run_kscarcity_local_typed(
    df: pd.DataFrame,
    buffer_size: int = 30,
    min_conf: float = 0.15,
    use_causal: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Run K-Scarcity on a single country's data (no federation)."""
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    engine = OnlineDiscoveryEngine(buffer_size=buffer_size)
    schema = _build_schema(df.columns.tolist())
    engine.initialize_v2(schema, use_causal=use_causal)

    rows_fed = 0
    for _, row in df.iterrows():
        row_dict = {k: float(v) for k, v in row.items() if pd.notna(v)}
        if row_dict:
            engine.process_row(row_dict)
            rows_fed += 1

    if verbose:
        print(f'  [local] Fed {rows_fed} KEN rows')

    summaries = engine.export_hypothesis_summary(min_conf=min_conf)
    discoveries = _summary_to_eval_format(summaries)

    if verbose:
        by_type: dict[str, int] = {}
        for d in discoveries:
            by_type[d['type']] = by_type.get(d['type'], 0) + 1
        print(f'  [local] {len(discoveries)} discoveries (conf>={min_conf}): '
              + ', '.join(f'{t}={n}' for t, n in sorted(by_type.items())))

    return discoveries


# ---------------------------------------------------------------------------
# Federated runner
# ---------------------------------------------------------------------------

def run_kscarcity_federated_typed(
    df_primary: pd.DataFrame,
    dfs_peers: dict[str, pd.DataFrame],
    buffer_size: int = 30,
    min_conf: float = 0.15,
    peer_weight: float = 0.5,
    use_causal: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Run K-Scarcity on KEN (primary) + TZA/UGA peer rows.

    For each calendar year present in the primary data, the primary
    engine processes the KEN row. Then, for each peer country, if that
    year is present in the peer data, a peer row is fed via
    process_peer_row(peer_id, row, peer_weight).

    Peer rows include only non-NaN variables, so columns missing from a
    peer country (e.g. govt_debt for TZA) are silently skipped by the
    engine's sanitize step.

    Args:
        df_primary: KEN DataFrame (years as index, GT cols as columns).
        dfs_peers:  {country_code: DataFrame} for TZA and/or UGA.
        peer_weight: Trust weight for peer observations [0, 1].
    """
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

    engine = OnlineDiscoveryEngine(buffer_size=buffer_size)
    schema = _build_schema(df_primary.columns.tolist())
    engine.initialize_v2(schema, use_causal=use_causal)

    primary_years = sorted(df_primary.index)
    rows_fed_primary = 0
    rows_fed_peer: dict[str, int] = {cc: 0 for cc in dfs_peers}

    for year in primary_years:
        # Own row
        own_row = {
            k: float(v)
            for k, v in df_primary.loc[year].items()
            if pd.notna(v)
        }
        if own_row:
            engine.process_row(own_row)
            rows_fed_primary += 1

        # Peer rows for the same year
        for peer_cc, df_peer in dfs_peers.items():
            if year not in df_peer.index:
                continue
            peer_row = {
                k: float(v)
                for k, v in df_peer.loc[year].items()
                if pd.notna(v)
            }
            if peer_row:
                engine.process_peer_row(peer_cc, peer_row, peer_weight=peer_weight)
                rows_fed_peer[peer_cc] += 1

    if verbose:
        peer_summary = ', '.join(
            f'{cc}={n}' for cc, n in sorted(rows_fed_peer.items())
        )
        print(f'  [fed] Fed {rows_fed_primary} KEN rows; peer rows: {peer_summary}')

    summaries = engine.export_hypothesis_summary(min_conf=min_conf)
    discoveries = _summary_to_eval_format(summaries)

    if verbose:
        by_type: dict[str, int] = {}
        for d in discoveries:
            by_type[d['type']] = by_type.get(d['type'], 0) + 1
        print(f'  [fed] {len(discoveries)} discoveries (conf>={min_conf}): '
              + ', '.join(f'{t}={n}' for t, n in sorted(by_type.items())))

    return discoveries


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compare_local_vs_federated_typed(
    local_disc: list[dict],
    fed_disc: list[dict],
    gt: list[dict],
    null_pairs: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Full per-type comparison: local vs federated typed discovery.

    Returns a dict with:
        local / federated: {type: recall_info, ..., overall: P/R/F1}
        delta_recall:      {type: fed_recall - local_recall}
        unlocked:          GT entries discovered fed but NOT local
        lost:              GT entries discovered local but NOT fed
        null_fp:           {local: rate, federated: rate}
    """
    disc_sets = {'local': local_disc, 'federated': fed_disc}

    recall: dict[str, dict] = {}
    overall: dict[str, dict] = {}
    null_fp: dict[str, float] = {}

    for label, disc in disc_sets.items():
        disc_wrap = {label: disc}
        recall[label] = compute_per_type_recall(disc_wrap, gt)
        overall[label] = compare_specialists(disc_wrap, gt)[label]
        fp_info = false_positive_analysis(disc_wrap, gt, null_pairs)
        null_fp[label] = fp_info['null_fp_rate']

    # Delta recall per type
    all_types = sorted(set(recall['local']) | set(recall['federated']))
    delta_recall = {}
    for t in all_types:
        r_loc = recall['local'].get(t, {}).get('recall', 0.0)
        r_fed = recall['federated'].get(t, {}).get('recall', 0.0)
        delta_recall[t] = round(r_fed - r_loc, 4)

    # Capability unlock: GT entries found in federated but not local
    def _is_discovered(disc_list: list[dict], gt_entry: dict) -> bool:
        """Check if any discovery matches this GT entry."""
        from scripts.experiments.evaluation_typed import match_discovery_to_gt
        return any(
            match_discovery_to_gt(d, gt_entry, strict_type=True)
            for d in disc_list
        )

    unlocked = [
        e for e in gt
        if _is_discovered(fed_disc, e) and not _is_discovered(local_disc, e)
    ]
    lost = [
        e for e in gt
        if _is_discovered(local_disc, e) and not _is_discovered(fed_disc, e)
    ]

    if verbose:
        print('\n  LOCAL vs FEDERATED per-type recall:')
        print(f'  {"Type":15s} {"Local R":>8s} {"Fed R":>8s} {"Delta":>8s}')
        print(f'  {"-"*45}')
        for t in all_types:
            r_loc = recall['local'].get(t, {}).get('recall', 0.0)
            r_fed = recall['federated'].get(t, {}).get('recall', 0.0)
            delta = delta_recall[t]
            flag = ' +++' if delta > 0.1 else (' ---' if delta < -0.1 else '')
            print(f'  {t:15s} {r_loc:8.3f} {r_fed:8.3f} {delta:+8.3f}{flag}')

        print('\n  Overall metrics:')
        for label in ('local', 'federated'):
            m = overall[label]
            print(f'  {label:10s}: TP={m["tp"]} FP={m["fp"]} FN={m["fn"]} '
                  f'P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}')

        print(f'\n  Null FP rate -- local: {null_fp["local"]:.3f}  '
              f'federated: {null_fp["federated"]:.3f}')

        if unlocked:
            print(f'\n  Capability UNLOCKED by federation ({len(unlocked)}):')
            for e in unlocked:
                med = e.get('mediator', e.get('moderator', ''))
                via = f' via {med}' if med else ''
                print(f'    [{e["type"]}] {e["source"]} -> {e["target"]}{via}')
        else:
            print('\n  No new GT capabilities unlocked by federation.')

        if lost:
            print(f'\n  Capabilities LOST in federated ({len(lost)}):')
            for e in lost:
                print(f'    [{e["type"]}] {e["source"]} -> {e["target"]}')

    return {
        'recall': recall,
        'overall': overall,
        'delta_recall': delta_recall,
        'unlocked': unlocked,
        'lost': lost,
        'null_fp': null_fp,
    }


def run_confidence_threshold_sweep(
    local_disc: list[dict],
    fed_disc: list[dict],
    gt: list[dict],
    thresholds: list[float] | None = None,
    verbose: bool = True,
) -> dict[float, dict]:
    """
    Compute P/R/F1 for local and federated at each confidence threshold.

    Returns:
        {threshold: {'local': metrics, 'federated': metrics}}
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    results: dict[float, dict] = {}

    if verbose:
        print('\n  Confidence threshold sweep:')
        print(f'  {"Thresh":>7s} | {"Local P":>8s} {"Local R":>8s} {"Local F1":>9s}'
              f' | {"Fed P":>8s} {"Fed R":>8s} {"Fed F1":>9s}')
        print(f'  {"-"*67}')

    for thresh in thresholds:
        loc_filtered = [d for d in local_disc if d['confidence'] >= thresh]
        fed_filtered = [d for d in fed_disc if d['confidence'] >= thresh]

        loc_m = compare_specialists({'local': loc_filtered}, gt)['local']
        fed_m = compare_specialists({'federated': fed_filtered}, gt)['federated']

        results[thresh] = {'local': loc_m, 'federated': fed_m}

        if verbose:
            print(f'  {thresh:7.2f} | {loc_m["precision"]:8.3f} {loc_m["recall"]:8.3f}'
                  f' {loc_m["f1"]:9.3f} | {fed_m["precision"]:8.3f} {fed_m["recall"]:8.3f}'
                  f' {fed_m["f1"]:9.3f}')

    return results


def run_capability_unlock_analysis(
    local_disc: list[dict],
    fed_disc: list[dict],
    gt: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Capability unlock: for each GT type, report whether federation
    first enables discovery (local_recall=0, fed_recall>0).

    Returns:
        {type: {'local_recall': float, 'fed_recall': float, 'unlocked': bool}}
    """
    from scripts.experiments.evaluation_typed import match_discovery_to_gt

    def _is_found(disc_list: list[dict], gt_entry: dict) -> bool:
        return any(
            match_discovery_to_gt(d, gt_entry, strict_type=True)
            for d in disc_list
        )

    by_type: dict[str, list[dict]] = {}
    for e in gt:
        by_type.setdefault(e['type'], []).append(e)

    result: dict[str, dict] = {}
    for t, entries in sorted(by_type.items()):
        n_gt = len(entries)
        n_local = sum(1 for e in entries if _is_found(local_disc, e))
        n_fed = sum(1 for e in entries if _is_found(fed_disc, e))
        r_local = n_local / n_gt
        r_fed = n_fed / n_gt
        unlocked = (r_local == 0.0 and r_fed > 0.0)
        result[t] = {
            'n_gt': n_gt,
            'local_hits': n_local,
            'fed_hits': n_fed,
            'local_recall': round(r_local, 3),
            'fed_recall': round(r_fed, 3),
            'unlocked': unlocked,
        }

    if verbose:
        print('\n  Capability unlock analysis (types with local recall=0):')
        print(f'  {"Type":15s} {"GT":>4s} {"Loc":>5s} {"Fed":>5s} {"Status"}')
        print(f'  {"-"*50}')
        for t, info in sorted(result.items()):
            status = 'UNLOCKED' if info['unlocked'] else (
                'IMPROVED' if info['fed_recall'] > info['local_recall'] else
                ('REGRESSED' if info['fed_recall'] < info['local_recall'] else 'SAME')
            )
            print(f'  {t:15s} {info["n_gt"]:4d} {info["local_recall"]:5.3f}'
                  f' {info["fed_recall"]:5.3f} {status}')

    return result


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    fast: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load and prepare KEN (primary) + TZA/UGA (peers).

    Returns:
        df_ken_full: full KEN df for N-sweep
        df_ken_work: complete rows of KEN (primary engine input)
        dfs_peers:   {cc: df} for peer countries (may have NaN per row)
    """
    if verbose:
        print('Loading country data...')

    df_ken = load_country_data('KEN')
    avail_cols = [c for c in GT_COLS if c in df_ken.columns]
    df_ken_work = df_ken[avail_cols].dropna()

    if fast:
        df_ken_work = df_ken_work.head(min(15, len(df_ken_work)))

    if verbose:
        print(f'  KEN: {df_ken_work.shape} complete rows x cols, '
              f'years {df_ken_work.index.min()}-{df_ken_work.index.max()}')

    dfs_peers: dict[str, pd.DataFrame] = {}
    for cc in ('TZA', 'UGA'):
        try:
            df_peer = load_country_data(cc)
            peer_cols = [c for c in GT_COLS if c in df_peer.columns]
            df_peer_sub = df_peer[peer_cols]
            # Keep rows that have at least 8 non-NaN GT columns (partial ok for peers)
            row_ok = df_peer_sub.notna().sum(axis=1) >= 8
            df_peer_sub = df_peer_sub[row_ok]
            if len(df_peer_sub) > 0:
                dfs_peers[cc] = df_peer_sub
                if verbose:
                    print(f'  {cc}: {len(df_peer_sub)} usable rows '
                          f'(>=8 GT cols non-NaN), '
                          f'years {df_peer_sub.index.min()}-{df_peer_sub.index.max()}')
            else:
                if verbose:
                    print(f'  {cc}: no usable rows -- skipping as peer')
        except Exception as exc:
            if verbose:
                print(f'  {cc}: load failed ({exc}) -- skipping as peer')

    return df_ken[avail_cols], df_ken_work, dfs_peers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    verbose = not args.quiet
    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()

    print('K-Scarcity Federation Typed Validation')
    print('=' * 50)

    df_ken_full, df_ken_work, dfs_peers = prepare_data(
        fast=args.fast, verbose=verbose
    )

    if not dfs_peers:
        print('WARNING: No peer country data available. '
              'Federated run will equal local run.')

    print(f'\nRunning LOCAL K-Scarcity (KEN only)...')
    t0 = time.time()
    local_disc = run_kscarcity_local_typed(
        df_ken_work,
        buffer_size=args.buffer_size,
        min_conf=args.min_conf,
        use_causal=not args.no_causal,
        verbose=verbose,
    )
    print(f'  Done in {time.time()-t0:.1f}s')

    print(f'\nRunning FEDERATED K-Scarcity (KEN + peers: {list(dfs_peers)})...')
    t0 = time.time()
    fed_disc = run_kscarcity_federated_typed(
        df_ken_work,
        dfs_peers,
        buffer_size=args.buffer_size,
        min_conf=args.min_conf,
        peer_weight=args.peer_weight,
        use_causal=not args.no_causal,
        verbose=verbose,
    )
    print(f'  Done in {time.time()-t0:.1f}s')

    print('\nF1. Local vs Federated per-type comparison:')
    comparison = compare_local_vs_federated_typed(
        local_disc, fed_disc, gt, null_pairs, verbose=verbose
    )

    print('\nF2. Capability unlock analysis:')
    unlock = run_capability_unlock_analysis(
        local_disc, fed_disc, gt, verbose=verbose
    )

    print('\nF3. Confidence threshold sweep:')
    sweep = run_confidence_threshold_sweep(
        local_disc, fed_disc, gt,
        thresholds=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        verbose=verbose,
    )

    # Summary
    n_unlocked = sum(1 for v in unlock.values() if v['unlocked'])
    n_improved = sum(
        1 for v in unlock.values()
        if v['fed_recall'] > v['local_recall'] and not v['unlocked']
    )
    loc_f1 = comparison['overall']['local']['f1']
    fed_f1 = comparison['overall']['federated']['f1']
    print(f'\nSummary:')
    print(f'  Local F1:     {loc_f1:.3f}')
    print(f'  Federated F1: {fed_f1:.3f}  (delta={fed_f1-loc_f1:+.3f})')
    print(f'  Types unlocked by federation: {n_unlocked}')
    print(f'  Types improved by federation: {n_improved}')
    print(f'  Null FP: local={comparison["null_fp"]["local"]:.3f}  '
          f'fed={comparison["null_fp"]["federated"]:.3f}')

    # Save results
    out_dir = _ROOT / 'results' / 'typed_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    out_path = out_dir / 'federation_typed_results.json'

    save_data = {
        'local_n_discoveries': len(local_disc),
        'fed_n_discoveries': len(fed_disc),
        'local_f1': loc_f1,
        'fed_f1': fed_f1,
        'delta_f1': round(fed_f1 - loc_f1, 4),
        'n_unlocked': n_unlocked,
        'n_improved': n_improved,
        'peers': list(dfs_peers.keys()),
        'peer_weight': args.peer_weight,
        'delta_recall': comparison['delta_recall'],
        'unlock_analysis': unlock,
        'threshold_sweep': {
            str(k): {
                'local_f1': v['local']['f1'],
                'fed_f1': v['federated']['f1'],
            }
            for k, v in sweep.items()
        },
    }
    out_path.write_text(json.dumps(save_data, indent=2), encoding='utf-8')
    if verbose:
        print(f'\nResults saved to {out_path.relative_to(_ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-Scarcity federation typed discovery validation'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Use only first 15 KEN rows (quick smoke test)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--no-causal', action='store_true',
                        help='Disable CausalHypothesis seeding (faster)')
    parser.add_argument('--buffer-size', type=int, default=30,
                        help='Engine buffer size (default 30)')
    parser.add_argument('--min-conf', type=float, default=0.15,
                        help='Minimum confidence for hypothesis export (default 0.15)')
    parser.add_argument('--peer-weight', type=float, default=0.5,
                        help='Trust weight for peer country rows (default 0.5)')
    args = parser.parse_args()
    main(args)

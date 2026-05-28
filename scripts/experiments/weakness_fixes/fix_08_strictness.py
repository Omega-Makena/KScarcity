"""
Weakness Fix 8: Evaluate at multiple GT-matching strictness levels.

Current evaluation uses strict type matching throughout, which means a
TemporalHypothesis firing on (inflation_cpi → inflation_cpi) doesn't count
unless the discovery type is exactly 'temporal'. This is appropriate for
auditing type discrimination, but:
  - It masks whether the system identifies the correct PAIR even if the type
    is wrong (edge-only matching).
  - It hides whether a causal finding aligns with the broader causal family
    even if the sub-type differs (family matching).

Three levels:
  strict     — source, target, AND type must all match exactly.
  family     — source and target must match; type must be in the same family
               (temporal/causal/correlational → 'dependence';
                equilibrium/compositional/competitive → 'constraint';
                functional/synergistic/mediating/moderating → 'interaction').
  edge_only  — source and target must match (any type accepted).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Type family mapping
# ---------------------------------------------------------------------------

_TYPE_FAMILY: dict[str, str] = {
    # Temporal persistence / directed causation / co-movement
    'temporal': 'dependence',
    'causal': 'dependence',
    'correlational': 'dependence',
    # Mean-reversion / composition constraints
    'equilibrium': 'constraint',
    'compositional': 'constraint',
    'competitive': 'constraint',
    # Nonlinear / conditional / multi-variable interaction
    'functional': 'interaction',
    'synergistic': 'interaction',
    'mediating': 'interaction',
    'moderating': 'interaction',
    'structural': 'interaction',
    'probabilistic': 'interaction',
}


def _same_family(type_a: str, type_b: str) -> bool:
    fam_a = _TYPE_FAMILY.get(type_a, type_a)
    fam_b = _TYPE_FAMILY.get(type_b, type_b)
    return fam_a == fam_b


# ---------------------------------------------------------------------------
# Matching at each strictness level
# ---------------------------------------------------------------------------

from scripts.experiments.evaluation_typed import _SYMMETRIC_TYPES, _pair_match


def match_at_strictness(
    disc: dict,
    gt_entry: dict,
    level: str,
) -> bool:
    """
    Return True if disc matches gt_entry at the given strictness level.

    level: 'strict' | 'family' | 'edge_only'
    """
    g_src, g_tgt = gt_entry['source'], gt_entry['target']
    d_src, d_tgt = disc.get('source', ''), disc.get('target', '')
    g_type = gt_entry['type']
    d_type = disc.get('type', '')

    # Pair match always required
    if not _pair_match(d_src, d_tgt, g_src, g_tgt, g_type):
        return False

    if level == 'edge_only':
        return True

    if level == 'family':
        return _same_family(d_type, g_type)

    # strict
    if d_type != g_type:
        return False
    # Mediating requires mediator match
    if 'mediator' in gt_entry and disc.get('mediator') != gt_entry['mediator']:
        return False
    # Synergistic/moderating requires moderator match
    if 'moderator' in gt_entry and disc.get('moderator') != gt_entry['moderator']:
        return False
    return True


def evaluate_at_strictness(
    discoveries: list[dict],
    ground_truth: list[dict],
    level: str,
) -> dict:
    """
    Compute precision, recall, F1 at a given strictness level.

    Returns: {tp, fp, fn, precision, recall, f1, n_discoveries,
              unique_gt_matched, coverage_pct}
    """
    n_gt = len(ground_truth)
    matched_gt: set[int] = set()
    fp = 0

    for d in discoveries:
        found = False
        for i, gt_entry in enumerate(ground_truth):
            if match_at_strictness(d, gt_entry, level):
                matched_gt.add(i)
                found = True
                break
        if not found:
            fp += 1

    tp_unique = len(matched_gt)
    fn = n_gt - tp_unique
    prec = tp_unique / (tp_unique + fp) if (tp_unique + fp) > 0 else 0.0
    rec = tp_unique / n_gt if n_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'level': level,
        'n_discoveries': len(discoveries),
        'tp': tp_unique,
        'fp': fp,
        'fn': fn,
        'unique_gt_matched': tp_unique,
        'coverage_pct': round(100 * tp_unique / n_gt, 1) if n_gt else 0.0,
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
    }


def evaluate_at_multiple_strictness_levels(
    discoveries: list[dict],
    ground_truth: list[dict],
    levels: list[str] = ('strict', 'family', 'edge_only'),
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Evaluate discoveries at strict, family, and edge-only matching levels.

    Returns {level_name: metrics_dict}.
    """
    results: dict[str, dict] = {}
    for level in levels:
        m = evaluate_at_strictness(discoveries, ground_truth, level)
        results[level] = m

    if verbose:
        print(f"\n  Strictness-level breakdown  (N_disc={len(discoveries)}, N_GT={len(ground_truth)})")
        print(f"  {'Level':12s}  {'TP':>4s}  {'FP':>5s}  {'FN':>4s}  "
              f"{'P':>7s}  {'R':>7s}  {'F1':>7s}  {'Coverage':>9s}")
        print(f"  {'-'*62}")
        for level, m in results.items():
            print(f"  {level:12s}  {m['tp']:4d}  {m['fp']:5d}  {m['fn']:4d}  "
                  f"{m['precision']:7.4f}  {m['recall']:7.4f}  {m['f1']:7.4f}  "
                  f"{m['coverage_pct']:8.1f}%")

        # Explain the gap
        if 'strict' in results and 'edge_only' in results:
            s = results['strict']
            e = results['edge_only']
            delta_tp = e['tp'] - s['tp']
            if delta_tp > 0:
                print(f"\n  Type-discrimination gap: {delta_tp} GT pairs found "
                      f"at edge-only but NOT at strict type match.")
                print('  Interpretation: system identifies correct economic pairs '
                      'but assigns wrong relationship type.')
            else:
                print('\n  No type-discrimination gap: all GT pairs found at '
                      'strict level are also found at edge-only.')

    return results


def per_type_strictness_breakdown(
    discoveries: list[dict],
    ground_truth: list[dict],
) -> pd.DataFrame:
    """
    For each GT type, show recall at strict vs edge_only.
    Reveals which types the system finds but labels incorrectly.
    """
    from collections import defaultdict

    # Group GT by type
    gt_by_type: dict[str, list] = defaultdict(list)
    for i, entry in enumerate(ground_truth):
        gt_by_type[entry['type']].append((i, entry))

    rows = []
    for gt_type, gt_entries in sorted(gt_by_type.items()):
        n_gt = len(gt_entries)

        for level in ('strict', 'family', 'edge_only'):
            matched = 0
            for _, gt_entry in gt_entries:
                for d in discoveries:
                    if match_at_strictness(d, gt_entry, level):
                        matched += 1
                        break
            rows.append({
                'gt_type': gt_type,
                'level': level,
                'n_gt': n_gt,
                'n_matched': matched,
                'recall': round(matched / n_gt, 4) if n_gt else 0.0,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix8(
    fast: bool = False,
    verbose: bool = True,
) -> dict:
    """Run Weakness Fix 8: multi-strictness evaluation on KEN specialists."""
    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
    )
    from scripts.experiments.specialist_baselines import run_all_specialists
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

    disc_by_type = run_all_specialists(df, verbose=False)
    all_disc = [d for discs in disc_by_type.values() for d in discs]

    if verbose:
        print(f'  Total specialist discoveries: {len(all_disc)}')

    results = evaluate_at_multiple_strictness_levels(all_disc, gt, verbose=verbose)

    if verbose:
        print('\n  Per-type recall breakdown:')
        df_breakdown = per_type_strictness_breakdown(all_disc, gt)
        pivot = df_breakdown.pivot(index='gt_type', columns='level', values='recall')
        print(pivot.round(4).to_string())

    return {
        'strictness_levels': results,
        'n_discoveries': len(all_disc),
        'n_gt': len(gt),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 8: Strictness levels')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix8(fast=args.fast, verbose=not args.quiet)

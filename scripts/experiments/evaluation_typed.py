"""
Evaluation of typed relationship discovery against theory-grounded ground truth.

Four evaluation questions:
  Q1  Per-type recall   — for each GT type, what fraction of GT relationships
                          were discovered (type-strict matching)?
  Q2  Specialist comparison — precision / recall / F1 per specialist method
                              compared to the full GT.
  Q3  False positive cost — FP rate on known-null pairs; cost-weighted analysis.
  Q4  Scarcity curves   — per-type F1 as a function of observation count N.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

_SYMMETRIC_TYPES: frozenset[str] = frozenset({
    'correlational', 'competitive',
})

_SELF_LOOP_TYPES: frozenset[str] = frozenset({
    'temporal', 'equilibrium', 'structural',
})


def _pair_match(d_src: str, d_tgt: str,
                g_src: str, g_tgt: str,
                rel_type: str) -> bool:
    """Check source/target match, respecting symmetric types."""
    if rel_type in _SYMMETRIC_TYPES:
        return {d_src, d_tgt} == {g_src, g_tgt}
    return d_src == g_src and d_tgt == g_tgt


def match_discovery_to_gt(disc: dict, gt_entry: dict,
                           strict_type: bool = True) -> bool:
    """
    Return True if disc matches gt_entry.

    Matching rules:
      - pair match (symmetric for correlational/competitive)
      - type match (when strict_type=True)
      - mediator match (for mediating type)
      - moderator match (for synergistic type)
    Sign matching is soft: only penalised in cost analysis, not in recall.
    """
    g_type = gt_entry['type']

    if strict_type and disc.get('type') != g_type:
        return False

    g_src, g_tgt = gt_entry['source'], gt_entry['target']
    d_src, d_tgt = disc.get('source', ''), disc.get('target', '')

    if not _pair_match(d_src, d_tgt, g_src, g_tgt, g_type):
        return False

    if 'mediator' in gt_entry:
        if disc.get('mediator') != gt_entry['mediator']:
            return False

    if 'moderator' in gt_entry:
        if disc.get('moderator') != gt_entry['moderator']:
            return False

    return True


def _any_gt_match(disc: dict, gt: list[dict],
                  strict_type: bool = True) -> dict | None:
    """Return the first GT entry that disc matches, or None."""
    for entry in gt:
        if match_discovery_to_gt(disc, entry, strict_type=strict_type):
            return entry
    return None


def _sign_correct(disc: dict, gt_entry: dict) -> bool:
    """True if discovery sign matches expected_sign (0 = don't care)."""
    exp = gt_entry.get('expected_sign', 0)
    if exp == 0:
        return True
    return disc.get('sign', 0) == exp


# ---------------------------------------------------------------------------
# Q1: Per-type recall
# ---------------------------------------------------------------------------

def compute_per_type_recall(
    discoveries: dict[str, list[dict]],
    gt: list[dict],
    strict_type: bool = True,
) -> dict[str, dict]:
    """
    Q1: For each GT type, what fraction of GT relationships were discovered?

    Args:
        discoveries: {specialist_name: [discovery_dict, ...]}
        gt: ground truth list from get_typed_ground_truth()
        strict_type: if True, discovery type must match GT type

    Returns:
        {type_name: {n_gt, n_discovered, recall, missed, sign_correct_frac}}
    """
    all_discs = [d for dlist in discoveries.values() for d in dlist]

    # Group GT by type
    by_type: dict[str, list[dict]] = {}
    for entry in gt:
        t = entry['type']
        by_type.setdefault(t, []).append(entry)

    result: dict[str, dict] = {}
    for rel_type, entries in sorted(by_type.items()):
        discovered = []
        missed = []
        sign_ok = 0
        for entry in entries:
            matched_discs = [
                d for d in all_discs
                if match_discovery_to_gt(d, entry, strict_type=strict_type)
            ]
            if matched_discs:
                discovered.append(entry)
                best = max(matched_discs, key=lambda d: d.get('confidence', 0))
                if _sign_correct(best, entry):
                    sign_ok += 1
            else:
                missed.append(entry)

        n_gt = len(entries)
        n_disc = len(discovered)
        result[rel_type] = {
            'n_gt': n_gt,
            'n_discovered': n_disc,
            'recall': n_disc / n_gt if n_gt else 0.0,
            'sign_correct_frac': sign_ok / n_disc if n_disc else 0.0,
            'missed_pairs': [
                f"{e['source']} -> {e['target']}" for e in missed
            ],
        }

    return result


# ---------------------------------------------------------------------------
# Q2: Specialist comparison
# ---------------------------------------------------------------------------

def compare_specialists(
    discoveries: dict[str, list[dict]],
    gt: list[dict],
    strict_type: bool = True,
) -> dict[str, dict]:
    """
    Q2: Precision / recall / F1 per specialist vs full GT.

    Each specialist is compared against the complete GT (all types).
    A TP is any discovery that matches any GT entry.
    A FN is a GT entry matched by no discovery from that specialist.
    A FP is a discovery that matches no GT entry.

    Returns:
        {specialist_name: {tp, fp, fn, precision, recall, f1, n_discoveries,
                           sign_correct_frac}}
    """
    result: dict[str, dict] = {}
    n_gt = len(gt)

    for spec_name, disc_list in sorted(discoveries.items()):
        tp = 0
        fp = 0
        sign_ok = 0
        matched_gt: set[int] = set()

        for d in disc_list:
            match_entry = _any_gt_match(d, gt, strict_type=strict_type)
            if match_entry is not None:
                gt_idx = gt.index(match_entry)
                matched_gt.add(gt_idx)
                if _sign_correct(d, match_entry):
                    sign_ok += 1
                tp += 1
            else:
                fp += 1

        fn = n_gt - len(matched_gt)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = len(matched_gt) / n_gt if n_gt else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        result[spec_name] = {
            'n_discoveries': len(disc_list),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'unique_gt_matched': len(matched_gt),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'sign_correct_frac': round(sign_ok / tp, 4) if tp else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# Q3: False positive analysis on known-null pairs
# ---------------------------------------------------------------------------

_FP_COST: dict[str, float] = {
    'strong': 2.0,
    'moderate': 1.5,
    'weak': 1.0,
}


def false_positive_analysis(
    discoveries: dict[str, list[dict]],
    gt: list[dict],
    null_pairs: list[dict],
) -> dict:
    """
    Q3: How often do specialists fire on known-null pairs?

    null_pairs: from get_known_null_relationships() — pairs with no GT relationship.

    Returns:
        {
          null_fp_details: list of {pair, fired_by_specialists, n_fires},
          null_fp_rate: fraction of null pairs fired on by ≥1 specialist,
          total_fp_all: total FP count across all specialists (not null-pair specific),
          gt_fp_cost_by_strength: cost-weighted FP breakdown for GT pairs,
          sign_wrong_frac: fraction of GT-matched discoveries with wrong sign,
        }
    """
    all_discs = [d for dlist in discoveries.values() for d in dlist]

    # Null-pair FP analysis
    null_fp_details = []
    for null_entry in null_pairs:
        n_src, n_tgt = null_entry['source'], null_entry['target']
        fired_by = []
        for spec_name, disc_list in discoveries.items():
            for d in disc_list:
                d_src, d_tgt = d.get('source', ''), d.get('target', '')
                if ({d_src, d_tgt} == {n_src, n_tgt}
                        or (d_src == n_src and d_tgt == n_tgt)):
                    fired_by.append(spec_name)
                    break
        null_fp_details.append({
            'pair': f"{n_src} -- {n_tgt}",
            'fired_by': fired_by,
            'n_fires': len(fired_by),
        })

    null_fp_rate = (
        sum(1 for x in null_fp_details if x['n_fires'] > 0) / len(null_pairs)
        if null_pairs else 0.0
    )

    # Total FP across all specialists
    total_fp = sum(
        1 for d in all_discs
        if _any_gt_match(d, gt, strict_type=True) is None
    )

    # Sign-wrong fraction among GT-matched discoveries
    sign_wrong = 0
    gt_matched = 0
    for d in all_discs:
        entry = _any_gt_match(d, gt, strict_type=True)
        if entry is not None:
            gt_matched += 1
            if not _sign_correct(d, entry):
                sign_wrong += 1

    sign_wrong_frac = sign_wrong / gt_matched if gt_matched else 0.0

    # Cost-weighted FP by GT relationship strength
    fp_cost_by_strength: dict[str, float] = {'strong': 0.0, 'moderate': 0.0, 'weak': 0.0}
    for d in all_discs:
        entry = _any_gt_match(d, gt, strict_type=True)
        if entry is None:
            # FP: check if it's a wrong-type hit on any pair (type-relaxed)
            loose = _any_gt_match(d, gt, strict_type=False)
            if loose is not None:
                strength = loose.get('strength', 'weak')
                fp_cost_by_strength[strength] = (
                    fp_cost_by_strength.get(strength, 0.0)
                    + _FP_COST.get(strength, 1.0)
                )

    return {
        'null_fp_details': null_fp_details,
        'null_fp_rate': round(null_fp_rate, 4),
        'null_fp_count': sum(1 for x in null_fp_details if x['n_fires'] > 0),
        'total_fp_all': total_fp,
        'gt_matched_total': gt_matched,
        'sign_wrong_count': sign_wrong,
        'sign_wrong_frac': round(sign_wrong_frac, 4),
        'fp_cost_by_strength': {k: round(v, 2) for k, v in fp_cost_by_strength.items()},
    }


# ---------------------------------------------------------------------------
# Q4: N-sweep scarcity curves
# ---------------------------------------------------------------------------

def n_sweep_typed(
    df_full: pd.DataFrame,
    n_values: Sequence[int],
    gt: list[dict],
    null_pairs: list[dict],
    strict_type: bool = True,
) -> dict[int, dict]:
    """
    Q4: Run all specialists at each N and compute typed metrics.

    Args:
        df_full: full DataFrame (all years, all columns)
        n_values: list of row counts to evaluate (e.g. [8,12,15,20,25,30,34])
        gt: ground truth from get_typed_ground_truth()
        null_pairs: from get_known_null_relationships()
        strict_type: passed to matching functions

    Returns:
        {N: {
            'n_rows': N,
            'n_cols': n_cols,
            'per_specialist': compare_specialists() result,
            'per_type_recall': compute_per_type_recall() result,
            'fp_analysis': false_positive_analysis() result,
            'overall': {precision, recall, f1, null_fp_rate},
        }}
    """
    from scripts.experiments.specialist_baselines import run_all_specialists

    results: dict[int, dict] = {}
    df_complete = df_full.dropna()

    for n in sorted(n_values):
        df_n = df_complete.head(n) if len(df_complete) >= n else df_complete
        actual_n = len(df_n)

        try:
            disc = run_all_specialists(df_n, verbose=False)
        except Exception as exc:
            results[n] = {'error': str(exc), 'n_rows': actual_n}
            continue

        per_spec = compare_specialists(disc, gt, strict_type=strict_type)
        per_type = compute_per_type_recall(disc, gt, strict_type=strict_type)
        fp_info = false_positive_analysis(disc, gt, null_pairs)

        all_discs = [d for dlist in disc.values() for d in dlist]
        all_gt_matched: set[int] = set()
        total_fp_strict = 0
        for d in all_discs:
            entry = _any_gt_match(d, gt, strict_type=strict_type)
            if entry is not None:
                all_gt_matched.add(gt.index(entry))
            else:
                total_fp_strict += 1

        tp_unique = len(all_gt_matched)
        fp_total = total_fp_strict
        fn_total = len(gt) - tp_unique

        prec = tp_unique / (tp_unique + fp_total) if (tp_unique + fp_total) else 0.0
        rec = tp_unique / len(gt) if gt else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        results[n] = {
            'n_rows': actual_n,
            'n_cols': df_n.shape[1],
            'n_discoveries_total': len(all_discs),
            'per_specialist': per_spec,
            'per_type_recall': per_type,
            'fp_analysis': fp_info,
            'overall': {
                'tp_unique': tp_unique,
                'fp': fp_total,
                'fn': fn_total,
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1': round(f1, 4),
                'null_fp_rate': fp_info['null_fp_rate'],
            },
        }

    return results


# ---------------------------------------------------------------------------
# Aggregated summary helpers
# ---------------------------------------------------------------------------

def summarise_n_sweep(sweep: dict[int, dict]) -> pd.DataFrame:
    """Return a DataFrame summarising overall metrics at each N."""
    rows = []
    for n, info in sorted(sweep.items()):
        if 'error' in info:
            rows.append({'N': n, 'error': info['error']})
            continue
        ov = info['overall']
        rows.append({
            'N': n,
            'discoveries': info['n_discoveries_total'],
            'tp_unique': ov['tp_unique'],
            'fp': ov['fp'],
            'fn': ov['fn'],
            'precision': ov['precision'],
            'recall': ov['recall'],
            'f1': ov['f1'],
            'null_fp_rate': ov['null_fp_rate'],
        })
    return pd.DataFrame(rows).set_index('N')


def summarise_per_type_sweep(sweep: dict[int, dict]) -> pd.DataFrame:
    """
    Return a DataFrame: rows=N, cols=type_name, values=recall.
    """
    rows = []
    for n, info in sorted(sweep.items()):
        if 'error' in info:
            continue
        row = {'N': n}
        for t, tinfo in info['per_type_recall'].items():
            row[t] = tinfo['recall']
        rows.append(row)
    return pd.DataFrame(rows).set_index('N')


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    from scripts.experiments.data_loader import load_country_data
    from scripts.experiments.ground_truth_typed import (
        get_typed_ground_truth,
        get_known_null_relationships,
    )
    from scripts.experiments.specialist_baselines import run_all_specialists

    print('Loading Kenya data...')
    df_ken = load_country_data('KEN')
    gt_cols = [
        'gdp_growth', 'inflation_cpi', 'unemployment', 'real_interest_rate',
        'private_credit', 'govt_consumption', 'exports_gdp', 'imports_gdp',
        'current_account', 'gcf', 'electricity_access', 'internet_users',
        'school_enrollment', 'life_expectancy', 'broad_money',
    ]
    df = df_ken[[c for c in gt_cols if c in df_ken.columns]].dropna()
    print(f'  Working DataFrame: {df.shape}')

    gt = get_typed_ground_truth()
    null_pairs = get_known_null_relationships()
    print(f'  GT relationships: {len(gt)}')
    print(f'  Known null pairs: {len(null_pairs)}')

    print('\nRunning specialists...')
    disc = run_all_specialists(df, verbose=False)
    total_d = sum(len(v) for v in disc.values())
    print(f'  Total discoveries: {total_d}')

    print('\n--- Q1: Per-type recall ---')
    recall_info = compute_per_type_recall(disc, gt)
    for t, info in sorted(recall_info.items()):
        missed = ', '.join(info['missed_pairs']) if info['missed_pairs'] else 'none'
        print(f"  {t:15s}: {info['n_discovered']}/{info['n_gt']} "
              f"recall={info['recall']:.3f}  sign_ok={info['sign_correct_frac']:.2f}"
              f"  missed=[{missed}]")

    print('\n--- Q2: Specialist comparison ---')
    spec_cmp = compare_specialists(disc, gt)
    print(f"  {'Specialist':15s}  {'#disc':>6}  {'TP':>4}  {'FP':>5}  "
          f"{'FN':>4}  {'P':>6}  {'R':>6}  {'F1':>6}")
    for s, m in sorted(spec_cmp.items()):
        print(f"  {s:15s}  {m['n_discoveries']:6d}  {m['tp']:4d}  "
              f"{m['fp']:5d}  {m['fn']:4d}  "
              f"{m['precision']:6.3f}  {m['recall']:6.3f}  {m['f1']:6.3f}")

    print('\n--- Q3: False positive analysis ---')
    fp_info = false_positive_analysis(disc, gt, null_pairs)
    print(f"  Null-pair FP rate      : {fp_info['null_fp_rate']:.3f} "
          f"({fp_info['null_fp_count']}/{len(null_pairs)} null pairs fired)")
    for detail in fp_info['null_fp_details']:
        if detail['n_fires']:
            print(f"    {detail['pair']:45s}  fired by: {detail['fired_by']}")
    print(f"  Total FP (strict type) : {fp_info['total_fp_all']}")
    print(f"  Sign-wrong fraction    : {fp_info['sign_wrong_frac']:.3f} "
          f"({fp_info['sign_wrong_count']}/{fp_info['gt_matched_total']})")

    print('\n--- Q4: N-sweep scarcity curves ---')
    n_vals = [8, 12, 15, 20, 25, 30, len(df)]
    n_vals = sorted(set(n_vals))
    sweep = n_sweep_typed(df, n_vals, gt, null_pairs)

    summary = summarise_n_sweep(sweep)
    print(summary.to_string())

    print('\nPer-type recall across N:')
    type_sweep = summarise_per_type_sweep(sweep)
    print(type_sweep.round(3).to_string())

    print('\nSelf-test complete.')

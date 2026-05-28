"""
Weakness Fix 4: Ground truth sensitivity analysis.

Problem: The 27-entry GT was constructed from theoretical sources. If a few
controversial GT entries were wrong, reported recall could be inflated or
deflated. The system's performance should be robust to small GT perturbations.

Three analyses:
  1. Bootstrap GT: randomly sample 80% of GT entries N times; compute recall
     distribution. Stable systems have tight recall distributions.
  2. Leave-one-out GT (LOO): drop each GT entry once; measure recall change.
     If removing one entry causes large recall swings, that entry is
     disproportionately influential.
  3. Adversarial GT: add N fake GT entries (plausible-looking but wrong pairs
     that we know the system fires on). Measures how precision degrades.
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
# Bootstrap GT
# ---------------------------------------------------------------------------

def bootstrap_gt_sensitivity(
    discoveries: list[dict],
    ground_truth: list[dict],
    n_bootstrap: int = 200,
    sample_frac: float = 0.80,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Bootstrap the GT set. Each bootstrap samples 80% of GT entries (without
    replacement) and evaluates recall on that subset.

    Returns:
        {
          'recall_mean': float,
          'recall_std': float,
          'recall_ci_low': float,   # 5th percentile
          'recall_ci_high': float,  # 95th percentile
          'f1_mean': float,
          'f1_std': float,
          'samples': [{'recall': float, 'f1': float, 'n_gt': int}, ...],
        }
    """
    from scripts.experiments.evaluation_typed import compare_specialists

    rng = np.random.default_rng(seed)
    n_gt = len(ground_truth)
    k = max(1, int(round(n_gt * sample_frac)))

    recalls, f1s = [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_gt, size=k, replace=False)
        gt_sub = [ground_truth[i] for i in idx]
        m = compare_specialists({'sys': discoveries}, gt_sub).get('sys', {})
        recalls.append(m.get('recall', 0.0))
        f1s.append(m.get('f1', 0.0))

    result = {
        'recall_mean': float(np.mean(recalls)),
        'recall_std': float(np.std(recalls)),
        'recall_ci_low': float(np.percentile(recalls, 5)),
        'recall_ci_high': float(np.percentile(recalls, 95)),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'n_bootstrap': n_bootstrap,
        'sample_frac': sample_frac,
    }

    if verbose:
        print(f'  Bootstrap GT ({n_bootstrap}×{sample_frac:.0%} of {n_gt} entries):')
        print(f'    Recall: {result["recall_mean"]:.4f} ± {result["recall_std"]:.4f} '
              f'[{result["recall_ci_low"]:.4f}, {result["recall_ci_high"]:.4f}]')
        print(f'    F1:     {result["f1_mean"]:.4f} ± {result["f1_std"]:.4f}')
        cv = result['recall_std'] / result['recall_mean'] if result['recall_mean'] > 0 else 0
        print(f'    CV(recall): {cv:.3f} '
              f'{"(stable)" if cv < 0.15 else "(UNSTABLE)"}')

    return result


# ---------------------------------------------------------------------------
# Leave-one-out GT
# ---------------------------------------------------------------------------

def loo_gt_sensitivity(
    discoveries: list[dict],
    ground_truth: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Drop each GT entry once; measure recall change.

    Returns:
        {
          'baseline_recall': float,
          'loo_results': [{'removed': str, 'type': str, 'recall': float, 'delta': float}, ...],
          'max_delta': float,
          'min_delta': float,
          'most_influential': str,   # entry whose removal changes recall most
        }
    """
    from scripts.experiments.evaluation_typed import compare_specialists

    baseline_m = compare_specialists({'sys': discoveries}, ground_truth).get('sys', {})
    baseline_recall = baseline_m.get('recall', 0.0)

    loo_results = []
    for i, entry in enumerate(ground_truth):
        gt_loo = [e for j, e in enumerate(ground_truth) if j != i]
        m = compare_specialists({'sys': discoveries}, gt_loo).get('sys', {})
        rec = m.get('recall', 0.0)
        delta = rec - baseline_recall
        label = f"{entry['type']}({entry['source']}->{entry['target']})"
        loo_results.append({
            'removed': label,
            'type': entry['type'],
            'recall': round(rec, 4),
            'delta': round(delta, 4),
            'entry_idx': i,
        })

    # Sort by |delta| descending
    loo_results.sort(key=lambda x: abs(x['delta']), reverse=True)

    max_delta = max(abs(r['delta']) for r in loo_results)
    most_influential = loo_results[0]['removed'] if loo_results else ''

    if verbose:
        print(f'\n  LOO GT Sensitivity (baseline recall={baseline_recall:.4f}):')
        print(f"  {'Entry':55s}  {'Recall':>7s}  {'Delta':>7s}")
        print(f"  {'-'*72}")
        for r in loo_results[:10]:
            flag = ' <-- most influential' if r == loo_results[0] else ''
            print(f"  {r['removed']:55s}  {r['recall']:7.4f}  {r['delta']:+7.4f}{flag}")
        if len(loo_results) > 10:
            print(f'  ... ({len(loo_results)-10} more entries)')
        print(f'\n  Max |delta|={max_delta:.4f}  most influential: {most_influential}')
        if max_delta < 0.05:
            print('  ** GT is robust: no single entry shifts recall by >5pp **')
        else:
            print('  WARNING: some GT entries are highly influential')

    return {
        'baseline_recall': baseline_recall,
        'loo_results': loo_results,
        'max_delta': max_delta,
        'min_delta': min(r['delta'] for r in loo_results),
        'most_influential': most_influential,
    }


# ---------------------------------------------------------------------------
# Adversarial GT
# ---------------------------------------------------------------------------

def adversarial_gt_sensitivity(
    discoveries: list[dict],
    ground_truth: list[dict],
    n_adversarial: int = 5,
    seed: int = 123,
    verbose: bool = True,
) -> dict:
    """
    Add N fake GT entries (pairs the system fires on but are NOT true
    relationships). Measures how much precision degrades.

    The adversarial entries are chosen from the system's false-positive
    discoveries (discovered but not in GT).
    """
    from scripts.experiments.evaluation_typed import compare_specialists, _any_gt_match

    # Find false positives
    fps = [d for d in discoveries if _any_gt_match(d, ground_truth) is None]

    if not fps:
        if verbose:
            print('  No false positives found; adversarial test not applicable.')
        return {'applicable': False, 'n_fps': 0}

    rng = np.random.default_rng(seed)
    n_adv = min(n_adversarial, len(fps))
    adv_idx = rng.choice(len(fps), size=n_adv, replace=False)
    adv_fps = [fps[i] for i in adv_idx]

    # Convert FP discoveries to fake GT entries
    fake_gt_entries = []
    for d in adv_fps:
        fake = {
            'source': d.get('source', ''),
            'target': d.get('target', ''),
            'type': d.get('type', 'correlational'),
            'expected_sign': d.get('sign', 0),
            'strength': 'weak',
            'theoretical_basis': 'ADVERSARIAL (fake entry)',
        }
        if 'mediator' in d:
            fake['mediator'] = d['mediator']
        if 'moderator' in d:
            fake['moderator'] = d['moderator']
        fake_gt_entries.append(fake)

    gt_poisoned = ground_truth + fake_gt_entries

    # Evaluate on clean GT
    m_clean = compare_specialists({'sys': discoveries}, ground_truth).get('sys', {})
    # Evaluate on poisoned GT
    m_poisoned = compare_specialists({'sys': discoveries}, gt_poisoned).get('sys', {})

    if verbose:
        print(f'\n  Adversarial GT: {n_adv} fake entries added from FP pool')
        print(f'  Clean GT:    P={m_clean.get("precision",0):.4f}  '
              f'R={m_clean.get("recall",0):.4f}  F1={m_clean.get("f1",0):.4f}')
        print(f'  Poisoned GT: P={m_poisoned.get("precision",0):.4f}  '
              f'R={m_poisoned.get("recall",0):.4f}  F1={m_poisoned.get("f1",0):.4f}')
        delta_f1 = m_poisoned.get('f1', 0) - m_clean.get('f1', 0)
        print(f'  F1 delta: {delta_f1:+.4f}  '
              f'(inflated by {100*delta_f1/(m_clean.get("f1",1e-9)+1e-9):.1f}%)')

    return {
        'applicable': True,
        'n_adversarial': n_adv,
        'clean_metrics': {k: m_clean.get(k, 0) for k in ['precision','recall','f1']},
        'poisoned_metrics': {k: m_poisoned.get(k, 0) for k in ['precision','recall','f1']},
        'f1_inflation': round(
            m_poisoned.get('f1', 0) - m_clean.get('f1', 0), 4),
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix4(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 4: GT sensitivity analysis on KEN data."""
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
        print(f'  Total discoveries: {len(all_disc)}')

    n_boot = 50 if fast else 200
    bootstrap_res = bootstrap_gt_sensitivity(all_disc, gt, n_bootstrap=n_boot,
                                              verbose=verbose)
    loo_res = loo_gt_sensitivity(all_disc, gt, verbose=verbose)
    adv_res = adversarial_gt_sensitivity(all_disc, gt, verbose=verbose)

    return {
        'bootstrap': bootstrap_res,
        'loo': loo_res,
        'adversarial': adv_res,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 4: GT sensitivity')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix4(fast=args.fast, verbose=not args.quiet)

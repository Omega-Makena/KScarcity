"""
Weakness Fix 10: Economist baseline.

A competent economist with this dataset would first:
  1. Compute the full correlation matrix and flag all pairs |r| > 0.3 as
     'correlational' relationships.
  2. Run AR(1) on each variable and flag those with |rho| > 0.3 as
     'temporal' (autocorrelated/persistent).
  3. Mark any pair where |r| > 0.6 and the sign matches economic theory
     as a directional hint.

This is the simplest possible non-trivial benchmark. If K-Scarcity cannot
beat a 5-minute economist scan, the complexity is not justified.

Output: same discovery format as specialist_baselines.py, so it can be
fed directly to evaluation_typed.py compare_specialists().
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
from scipy import stats

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Correlation scan
# ---------------------------------------------------------------------------

def _corr_scan(df: pd.DataFrame, min_r: float = 0.30, alpha: float = 0.05) -> list[dict]:
    """
    Pairwise Pearson correlations: flag |r| >= min_r and p < alpha.
    Returns discoveries in evaluation format (type='correlational').
    """
    cols = df.columns.tolist()
    n = len(df)
    discoveries = []

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            x = df[c1].values
            y = df[c2].values
            try:
                r, p = stats.pearsonr(x, y)
            except Exception:
                continue
            if abs(r) >= min_r and p < alpha:
                discoveries.append({
                    'source': c1,
                    'target': c2,
                    'type': 'correlational',
                    'confidence': round(abs(r), 4),
                    'sign': int(np.sign(r)),
                    'statistic': round(r, 4),
                    'p_value': round(p, 6),
                    'method': 'economist_corr',
                })
    return discoveries


# ---------------------------------------------------------------------------
# Autocorrelation scan (temporal persistence)
# ---------------------------------------------------------------------------

def _ar1_scan(df: pd.DataFrame, min_rho: float = 0.30, alpha: float = 0.05) -> list[dict]:
    """
    AR(1) scan: flag variables with |rho_lag1| >= min_rho and p < alpha.
    Returns discoveries (type='temporal', source=target=varname).
    """
    cols = df.columns.tolist()
    n = len(df)
    discoveries = []

    for col in cols:
        x = df[col].values
        x_lag = x[:-1]
        x_curr = x[1:]
        try:
            r, p = stats.pearsonr(x_lag, x_curr)
        except Exception:
            continue
        if abs(r) >= min_rho and p < alpha:
            discoveries.append({
                'source': col,
                'target': col,
                'type': 'temporal',
                'confidence': round(abs(r), 4),
                'sign': int(np.sign(r)),
                'statistic': round(r, 4),
                'p_value': round(p, 6),
                'method': 'economist_ar1',
            })
    return discoveries


# ---------------------------------------------------------------------------
# Granger-style directional scan (for causal GT entries)
# ---------------------------------------------------------------------------

def _granger_scan(df: pd.DataFrame, min_r: float = 0.25, alpha: float = 0.05) -> list[dict]:
    """
    Lag-1 cross-correlation: X_{t-1} → Y_t.
    Treats significant lagged correlation as a 'causal' directional finding.
    """
    cols = df.columns.tolist()
    n = len(df)
    discoveries = []

    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                continue
            x_lag = df[c1].values[:-1]
            y_curr = df[c2].values[1:]
            try:
                r, p = stats.pearsonr(x_lag, y_curr)
            except Exception:
                continue
            if abs(r) >= min_r and p < alpha:
                discoveries.append({
                    'source': c1,
                    'target': c2,
                    'type': 'causal',
                    'confidence': round(abs(r), 4),
                    'sign': int(np.sign(r)),
                    'statistic': round(r, 4),
                    'p_value': round(p, 6),
                    'method': 'economist_granger_naive',
                })
    return discoveries


# ---------------------------------------------------------------------------
# Combined economist baseline
# ---------------------------------------------------------------------------

def economist_baseline(
    df: pd.DataFrame,
    min_corr: float = 0.30,
    min_ar1: float = 0.30,
    min_granger: float = 0.25,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """
    Run the full economist baseline: correlation + AR(1) + naive Granger.

    Returns:
        {
          'correlational': [...],
          'temporal':      [...],
          'causal':        [...],
        }
    where each value is a list of discovery dicts in evaluation format.
    """
    corr_disc = _corr_scan(df, min_r=min_corr, alpha=alpha)
    ar1_disc = _ar1_scan(df, min_rho=min_ar1, alpha=alpha)
    granger_disc = _granger_scan(df, min_r=min_granger, alpha=alpha)

    if verbose:
        print(f'  Economist baseline on N={len(df)} rows, {len(df.columns)} vars')
        print(f'    correlational (|r|>={min_corr}, p<{alpha}): {len(corr_disc)}')
        print(f'    temporal AR(1) (|rho|>={min_ar1}, p<{alpha}): {len(ar1_disc)}')
        print(f'    causal naive Granger (|r_lag|>={min_granger}): {len(granger_disc)}')
        print(f'    total: {len(corr_disc)+len(ar1_disc)+len(granger_disc)}')

    return {
        'correlational': corr_disc,
        'temporal': ar1_disc,
        'causal': granger_disc,
    }


def economist_baseline_flat(
    df: pd.DataFrame,
    min_corr: float = 0.30,
    min_ar1: float = 0.30,
    min_granger: float = 0.25,
    alpha: float = 0.05,
    verbose: bool = True,
) -> list[dict]:
    """Return flat list of all economist-baseline discoveries."""
    by_type = economist_baseline(df, min_corr, min_ar1, min_granger, alpha, verbose)
    return [d for discs in by_type.values() for d in discs]


def compare_economist_vs_specialist(
    df: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Run economist baseline and specialist baselines; compare on GT.
    Returns evaluation metrics for both.
    """
    from scripts.experiments.specialist_baselines import run_all_specialists
    from scripts.experiments.evaluation_typed import (
        compare_specialists,
        false_positive_analysis,
    )

    # Economist baseline
    econ_by_type = economist_baseline(df, verbose=verbose)
    econ_flat = [d for discs in econ_by_type.values() for d in discs]

    # Specialist baselines
    if verbose:
        print('  Running specialist baselines...')
    spec_by_type = run_all_specialists(df, verbose=False)
    spec_flat = [d for discs in spec_by_type.values() for d in discs]

    # Evaluate both
    combined = {
        'economist': econ_flat,
        'specialists': spec_flat,
    }
    metrics = compare_specialists(combined, ground_truth)
    fp_info = false_positive_analysis(combined, ground_truth, null_pairs)

    if verbose:
        print(f"\n  Comparison: Economist vs Specialists")
        print(f"  {'Method':12s}  {'#disc':>6s}  {'TP':>4s}  {'FP':>5s}  "
              f"{'P':>7s}  {'R':>7s}  {'F1':>7s}  {'NullFP':>7s}")
        print(f"  {'-'*60}")
        null_fps = {
            'economist': fp_info.get('null_fp_rate', 0.0),
            'specialists': fp_info.get('null_fp_rate', 0.0),
        }
        for method_key in ['economist', 'specialists']:
            m = metrics.get(method_key, {})
            nfp = null_fps.get(method_key, 0.0)
            print(f"  {method_key:12s}  {m.get('n_discoveries',0):6d}  "
                  f"{m.get('tp',0):4d}  {m.get('fp',0):5d}  "
                  f"{m.get('precision',0):7.4f}  {m.get('recall',0):7.4f}  "
                  f"{m.get('f1',0):7.4f}  {nfp:7.4f}")

    return {
        'economist_metrics': metrics.get('economist', {}),
        'specialist_metrics': metrics.get('specialists', {}),
        'economist_n_disc': len(econ_flat),
        'specialist_n_disc': len(spec_flat),
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix10(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 10: economist baseline comparison on KEN data."""
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

    results = compare_economist_vs_specialist(df, gt, null_pairs, verbose=verbose)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 10: Economist baseline')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix10(fast=args.fast, verbose=not args.quiet)

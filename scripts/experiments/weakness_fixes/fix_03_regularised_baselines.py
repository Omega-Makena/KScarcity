"""
Weakness Fix 3: Regularised statistical baselines.

Adds sklearn-based regularised baselines that are stronger than simple
Pearson correlation but much simpler than the full K-Scarcity engine.

Baselines:
  1. GraphicalLassoCV  — sparse inverse covariance; gold standard for
                         high-p-low-n multivariate relationship discovery.
  2. Lasso + interactions — linear Lasso with pairwise interaction terms.
  3. Elastic Net       — L1+L2 regression sweep (each variable as target).
  4. Pearson + Bonferroni — simple correlation with FWER correction.

All baselines output discoveries in the same dict format as specialist_baselines,
so they can be evaluated with compare_specialists() directly.
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
# Graphical Lasso
# ---------------------------------------------------------------------------

def graphical_lasso_baseline(
    df: pd.DataFrame,
    min_partial_corr: float = 0.15,
    verbose: bool = True,
) -> list[dict]:
    """
    Fit GraphicalLassoCV; extract non-zero partial correlations as
    'correlational' discoveries (the inverse covariance reveals direct
    connections after controlling for all other variables).
    """
    from sklearn.covariance import GraphicalLassoCV
    from sklearn.preprocessing import StandardScaler

    cols = df.columns.tolist()
    X = StandardScaler().fit_transform(df.values.astype(float))

    try:
        model = GraphicalLassoCV(cv=min(5, len(df) // 3), max_iter=200,
                                 tol=1e-3, enet_tol=1e-3)
        model.fit(X)
        prec = model.precision_  # inverse covariance matrix
    except Exception as exc:
        if verbose:
            print(f'  WARNING: GraphicalLassoCV failed: {exc}')
        return []

    # Convert to partial correlations: rho_ij = -prec_ij / sqrt(prec_ii * prec_jj)
    diag = np.sqrt(np.diag(prec))
    partial_corr = np.zeros_like(prec)
    n = len(cols)
    for i in range(n):
        for j in range(n):
            if i != j and diag[i] > 0 and diag[j] > 0:
                partial_corr[i, j] = -prec[i, j] / (diag[i] * diag[j])

    discoveries = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            pc = partial_corr[i, j]
            if abs(pc) >= min_partial_corr:
                discoveries.append({
                    'source': c1,
                    'target': c2,
                    'type': 'correlational',
                    'confidence': round(min(abs(pc), 1.0), 4),
                    'sign': int(np.sign(pc)),
                    'statistic': round(pc, 4),
                    'p_value': 0.0,  # GraphicalLasso doesn't provide p-values
                    'method': 'graphical_lasso',
                })

    if verbose:
        print(f'  GraphicalLassoCV: {len(discoveries)} edges '
              f'(|partial_corr|>={min_partial_corr})')
    return discoveries


# ---------------------------------------------------------------------------
# Lasso with pairwise interactions (targets each variable)
# ---------------------------------------------------------------------------

def lasso_interaction_baseline(
    df: pd.DataFrame,
    min_coef: float = 0.10,
    alpha: float = 0.01,
    verbose: bool = True,
) -> list[dict]:
    """
    For each variable Y, fit Lasso on X_j and X_j * X_k (pairwise interactions).
    Non-zero interaction coefficients → 'synergistic' discoveries.
    Non-zero main-effect coefficients → 'correlational' discoveries.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures

    cols = df.columns.tolist()
    X_raw = df.values.astype(float)
    n, p = X_raw.shape
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    # Build interaction features (degree=2, no bias, no powers)
    try:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_inter = poly.fit_transform(X_std)
        feat_names = poly.get_feature_names_out(cols)
    except Exception as exc:
        if verbose:
            print(f'  WARNING: PolynomialFeatures failed: {exc}')
        return []

    discoveries = []
    main_effect_set: set[frozenset] = set()
    interaction_set: set[frozenset] = set()

    for i, target_col in enumerate(cols):
        y = X_std[:, i]
        # Build feature matrix excluding target column
        keep_cols = [j for j, fn in enumerate(feat_names)
                     if target_col not in fn.split(' ')]
        if not keep_cols:
            continue
        X_sub = X_inter[:, keep_cols]
        feat_sub = [feat_names[j] for j in keep_cols]

        try:
            cv = min(5, max(2, n // 5))
            model = LassoCV(cv=cv, max_iter=1000, tol=1e-4)
            model.fit(X_sub, y)
        except Exception:
            continue

        for feat, coef in zip(feat_sub, model.coef_):
            if abs(coef) < min_coef:
                continue
            parts = feat.split(' ')
            if len(parts) == 1:
                # Main effect
                src = parts[0]
                pair = frozenset([src, target_col])
                if pair not in main_effect_set:
                    main_effect_set.add(pair)
                    discoveries.append({
                        'source': src,
                        'target': target_col,
                        'type': 'correlational',
                        'confidence': round(min(abs(coef), 1.0), 4),
                        'sign': int(np.sign(coef)),
                        'statistic': round(coef, 4),
                        'p_value': alpha,
                        'method': 'lasso_main',
                    })
            elif len(parts) == 2:
                # Interaction
                a, b = parts[0], parts[1]
                trio = frozenset([a, b, target_col])
                if trio not in interaction_set:
                    interaction_set.add(trio)
                    discoveries.append({
                        'source': a,
                        'target': target_col,
                        'moderator': b,
                        'type': 'synergistic',
                        'confidence': round(min(abs(coef), 1.0), 4),
                        'sign': int(np.sign(coef)),
                        'statistic': round(coef, 4),
                        'p_value': alpha,
                        'method': 'lasso_interaction',
                    })

    if verbose:
        n_main = sum(1 for d in discoveries if d['type'] == 'correlational')
        n_inter = sum(1 for d in discoveries if d['type'] == 'synergistic')
        print(f'  LassoCV interactions: {n_main} main + {n_inter} interaction '
              f'discoveries (min_coef={min_coef})')
    return discoveries


# ---------------------------------------------------------------------------
# Elastic Net sweep
# ---------------------------------------------------------------------------

def elastic_net_baseline(
    df: pd.DataFrame,
    min_coef: float = 0.10,
    verbose: bool = True,
) -> list[dict]:
    """
    For each variable Y, fit ElasticNetCV using all other variables as X.
    Non-zero coefficients → 'causal' (directed, Y as target) discoveries.
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    cols = df.columns.tolist()
    X_raw = df.values.astype(float)
    n, p = X_raw.shape
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    discoveries = []
    seen: set[tuple] = set()

    for i, target_col in enumerate(cols):
        y = X_std[:, i]
        X_sub = np.delete(X_std, i, axis=1)
        pred_cols = [c for j, c in enumerate(cols) if j != i]

        try:
            cv = min(5, max(2, n // 5))
            model = ElasticNetCV(cv=cv, max_iter=1000, l1_ratio=[0.1, 0.5, 0.9])
            model.fit(X_sub, y)
        except Exception:
            continue

        for src, coef in zip(pred_cols, model.coef_):
            if abs(coef) < min_coef:
                continue
            key = (src, target_col)
            if key not in seen:
                seen.add(key)
                discoveries.append({
                    'source': src,
                    'target': target_col,
                    'type': 'causal',
                    'confidence': round(min(abs(coef), 1.0), 4),
                    'sign': int(np.sign(coef)),
                    'statistic': round(coef, 4),
                    'p_value': 0.0,
                    'method': 'elastic_net',
                })

    if verbose:
        print(f'  ElasticNetCV: {len(discoveries)} directed discoveries '
              f'(min_coef={min_coef})')
    return discoveries


# ---------------------------------------------------------------------------
# Pearson + Bonferroni (stronger than economist baseline)
# ---------------------------------------------------------------------------

def pearson_bonferroni_baseline(
    df: pd.DataFrame,
    alpha: float = 0.05,
    verbose: bool = True,
) -> list[dict]:
    """
    Pearson correlation with Bonferroni correction across all pairs.
    Returns significant correlational discoveries.
    """
    cols = df.columns.tolist()
    p = len(cols)
    n_tests = p * (p - 1) // 2
    alpha_adj = alpha / max(n_tests, 1)
    discoveries = []

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            try:
                r, pval = stats.pearsonr(df[c1].values, df[c2].values)
            except Exception:
                continue
            if pval < alpha_adj:
                discoveries.append({
                    'source': c1,
                    'target': c2,
                    'type': 'correlational',
                    'confidence': round(abs(r), 4),
                    'sign': int(np.sign(r)),
                    'statistic': round(r, 4),
                    'p_value': round(pval, 8),
                    'method': 'pearson_bonferroni',
                })

    if verbose:
        print(f'  Pearson+Bonferroni: {len(discoveries)} discoveries '
              f'(n_tests={n_tests}, alpha_adj={alpha_adj:.6f})')
    return discoveries


# ---------------------------------------------------------------------------
# Combined comparison
# ---------------------------------------------------------------------------

def run_regularised_baselines(
    df: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Run all regularised baselines and compare against K-Scarcity and
    specialist baselines.

    Returns dict with per-method metrics.
    """
    from scripts.experiments.specialist_baselines import run_all_specialists
    from scripts.experiments.evaluation_typed import (
        compare_specialists,
        false_positive_analysis,
    )

    if verbose:
        print('\n  --- Regularised Baselines ---')

    glasso_disc = graphical_lasso_baseline(df, verbose=verbose)
    lasso_disc = lasso_interaction_baseline(df, verbose=verbose)
    enet_disc = elastic_net_baseline(df, verbose=verbose)
    bonf_disc = pearson_bonferroni_baseline(df, verbose=verbose)

    spec_by_type = run_all_specialists(df, verbose=False)
    spec_flat = [d for discs in spec_by_type.values() for d in discs]
    if verbose:
        print(f'  Specialist baselines: {len(spec_flat)} discoveries')

    combined = {
        'graphical_lasso': glasso_disc,
        'lasso_interactions': lasso_disc,
        'elastic_net': enet_disc,
        'pearson_bonferroni': bonf_disc,
        'specialists': spec_flat,
    }

    metrics = compare_specialists(combined, ground_truth)
    fp_info = false_positive_analysis(combined, ground_truth, null_pairs)

    if verbose:
        print(f"\n  {'Method':20s}  {'#disc':>6s}  {'TP':>4s}  {'FP':>5s}  "
              f"{'P':>7s}  {'R':>7s}  {'F1':>7s}")
        print(f"  {'-'*62}")
        for method_key, m in sorted(metrics.items()):
            print(f"  {method_key:20s}  {m.get('n_discoveries',0):6d}  "
                  f"{m.get('tp',0):4d}  {m.get('fp',0):5d}  "
                  f"{m.get('precision',0):7.4f}  {m.get('recall',0):7.4f}  "
                  f"{m.get('f1',0):7.4f}")

        # Highlight winner
        best = max(metrics.items(), key=lambda x: x[1].get('f1', 0.0))
        print(f"\n  Best F1: {best[0]} (F1={best[1].get('f1', 0.0):.4f})")

    return {
        'metrics': {k: dict(v) for k, v in metrics.items()},
        'n_disc': {k: len(v) for k, v in combined.items()},
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_fix3(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 3: regularised baselines on KEN data."""
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

    return run_regularised_baselines(df, gt, null_pairs, verbose=verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 3: Regularised baselines')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix3(fast=args.fast, verbose=not args.quiet)

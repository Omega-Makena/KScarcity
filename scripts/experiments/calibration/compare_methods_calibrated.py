"""
Apply the calibration pipeline to ALL methods for fair comparison.

Each method produces a ranked list of hypotheses. All are evaluated with
the same precision@k / recall@k / null FPR against the same GT.

Methods compared:
1. K-Scarcity (6-step calibrated)
2. Graphical Lasso (stability-selected)
3. Economist baseline (correlation + AR1, stability-selected)
4. Pearson + Bonferroni (no stability needed — already FWER-controlled)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from scripts.experiments.calibration.step5_stability_selection import (
    block_bootstrap_sample,
    _run_steps_1_to_4,
)
from scripts.experiments.calibration.step4_fdr_control import benjamini_hochberg
from scripts.experiments.calibration.evaluate_calibrated import (
    precision_recall_at_k_calibrated,
    null_fpr_calibrated,
    first_gt_rank,
)


# ---------------------------------------------------------------------------
# Graphical Lasso (stability-selected)
# ---------------------------------------------------------------------------

def calibrate_graphical_lasso(
    df: pd.DataFrame,
    B_boot: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    Graphical Lasso with stability selection.

    1. Fit GraphicalLassoCV on original data → precision matrix
    2. Non-zero off-diagonal entries = discovered partial correlations
    3. For stability: repeat on B_boot block-bootstrap resamples
    4. Selection frequency = fraction of resamples where edge appears
    5. Rank by |partial_corr| × selection_frequency
    """
    from collections import defaultdict

    def _fit_glasso(data: pd.DataFrame) -> dict[tuple, float]:
        """Returns {(src, tgt): partial_corr} for significant edges."""
        # Fill NaN with column means before fitting
        data_clean = data.copy()
        for col in data_clean.columns:
            if data_clean[col].isna().any():
                fill = data_clean[col].mean()
                data_clean[col] = data_clean[col].fillna(fill if np.isfinite(fill) else 0.0)

        mat = data_clean.values.astype(np.float64)
        try:
            from sklearn.covariance import GraphicalLassoCV
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                glasso = GraphicalLassoCV(cv=3, max_iter=300)
                glasso.fit(mat)
            prec = glasso.precision_
        except Exception:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(mat)
                prec = lw.precision_
            except Exception:
                return {}

        cols = list(data_clean.columns)
        k = len(cols)
        edges = {}
        for i in range(k):
            for j in range(i + 1, k):
                denom = np.sqrt(abs(prec[i, i] * prec[j, j]))
                if denom < 1e-12:
                    continue
                pcorr = -prec[i, j] / denom
                if abs(pcorr) > 0.05:
                    edges[(cols[i], cols[j])] = float(pcorr)
        return edges

    rng = np.random.default_rng(seed)

    # Original data
    orig_edges = _fit_glasso(df)
    edge_counts: dict[tuple, int] = {k: 0 for k in orig_edges}
    edge_pcorr: dict[tuple, float] = dict(orig_edges)

    # Bootstrap loop
    for _ in range(B_boot):
        boot_df = block_bootstrap_sample(df, block_size=4,
                                         rng=np.random.default_rng(int(rng.integers(0, 2**31))))
        boot_edges = _fit_glasso(boot_df)
        for k in boot_edges:
            if k not in edge_counts:
                edge_counts[k] = 0
            edge_counts[k] += 1

    results = []
    for (src, tgt), pcorr in orig_edges.items():
        freq = edge_counts.get((src, tgt), 0) / B_boot if B_boot > 0 else 0.0
        score = abs(pcorr) * freq
        # GLasso doesn't produce p-values. Use selection frequency directly:
        # passes if stable (pi >= 0.6) AND partial correlation exceeds 0.15.
        passes = freq >= 0.6 and abs(pcorr) > 0.15
        from scipy.stats import norm as _norm
        z = float(min(_norm.ppf(freq), 4.0)) if 0 < freq < 1 else (4.0 if freq >= 1 else 0.0)
        results.append({
            'source': src,
            'target': tgt,
            'pair': (src, tgt),
            'test_type': 'correlational',
            'T_obs': abs(pcorr),
            'p_value': max(1.0 - freq, 1e-6),  # monotone proxy for display
            'z_score': z,
            'fdr_adjusted_p': max(1.0 - freq, 1e-6),
            'fdr_significant': passes,
            'selection_frequency': freq,
            'score': score,
            'stable': freq >= 0.6,
            'passes_dual_threshold': passes,
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Economist baseline (stability-selected)
# ---------------------------------------------------------------------------

def calibrate_economist_baseline(
    df: pd.DataFrame,
    B_boot: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    Pearson |r| scan + AR(1) autocorrelation scan with stability selection.

    Pairwise: 'correlational' (Pearson |r|)
    Univariate: 'temporal' (lag-1 autocorrelation)
    """
    from collections import defaultdict

    def _compute_stats(data: pd.DataFrame) -> dict[tuple, tuple]:
        """Returns {(src, tgt): (statistic, type)}"""
        cols = list(data.columns)
        stats = {}
        vals = data.values.T  # K × N
        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if i == j:
                    # Temporal: lag-1 autocorrelation
                    x = vals[i]
                    if len(x) >= 4:
                        r = abs(np.corrcoef(x[:-1], x[1:])[0, 1])
                        if np.isfinite(r):
                            stats[(src, tgt)] = (r, 'temporal')
                elif j > i:
                    # Correlational: Pearson |r|
                    r = abs(np.corrcoef(vals[i], vals[j])[0, 1])
                    if np.isfinite(r):
                        stats[(src, tgt)] = (r, 'correlational')
        return stats

    rng = np.random.default_rng(seed)
    orig_stats = _compute_stats(df)
    n = len(df)
    cols = list(df.columns)

    # p-values via permutation (B=50 fast)
    p_values = {}
    for (src, tgt), (stat, htype) in orig_stats.items():
        null_stats = []
        for _ in range(50):
            perm_df = df.copy()
            if src == tgt:
                perm_df[src] = np.roll(df[src].values,
                                       int(rng.integers(1, max(n, 2))))
            else:
                perm_df[tgt] = rng.permutation(df[tgt].values)
            perm_stats = _compute_stats(perm_df)
            null_stats.append(perm_stats.get((src, tgt), (0.0, htype))[0])
        null_arr = np.array(null_stats)
        p = (1 + int(np.sum(null_arr >= stat))) / (1 + 50)
        p_values[(src, tgt)] = p

    # BH correction
    keys = list(p_values.keys())
    pvals_list = [p_values[k] for k in keys]
    reject = benjamini_hochberg(pvals_list, q=0.10)

    # Bootstrap selection frequency
    sel_counts: dict[tuple, int] = {k: 0 for k in orig_stats}
    for _ in range(B_boot):
        boot_df = block_bootstrap_sample(df, block_size=4,
                                         rng=np.random.default_rng(int(rng.integers(0, 2**31))))
        boot_stats = _compute_stats(boot_df)
        for key, (bstat, _) in boot_stats.items():
            # Count as selected if above median of original
            orig_stat = orig_stats.get(key, (0.0, ''))[0]
            if bstat >= orig_stat * 0.8:  # within 20% of original
                if key not in sel_counts:
                    sel_counts[key] = 0
                sel_counts[key] += 1

    results = []
    for i, (key, (stat, htype)) in enumerate(orig_stats.items()):
        src, tgt = key
        freq = sel_counts.get(key, 0) / B_boot if B_boot > 0 else 0.0
        from scipy.stats import norm
        z = min(norm.ppf(1 - p_values[key]), 4.0) if p_values[key] < 1.0 else 0.0
        score = z * freq

        results.append({
            'source': src,
            'target': tgt,
            'pair': key,
            'test_type': htype,
            'T_obs': stat,
            'p_value': p_values[key],
            'z_score': float(z),
            'fdr_adjusted_p': p_values[key],
            'fdr_significant': bool(reject[i]) if i < len(reject) else False,
            'selection_frequency': freq,
            'score': float(score),
            'stable': freq >= 0.6,
            'passes_dual_threshold': (
                bool(reject[i]) if i < len(reject) else False
            ) and freq >= 0.6,
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# K-Scarcity (full 6-step calibrated)
# ---------------------------------------------------------------------------

def calibrate_kscarcity(
    df: pd.DataFrame,
    B_boot: int = 50,
    B_perm: int = 100,
    fdr_q: float = 0.10,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """
    Full 6-step calibration pipeline for K-Scarcity hypothesis types.

    Chains Steps 1-6 in sequence. Returns ranked list.
    """
    from scripts.experiments.calibration.step5_stability_selection import stability_selection
    from scripts.experiments.calibration.step6_final_ranking import (
        apply_dual_threshold, evaluate_against_gt,
    )

    results = stability_selection(
        df,
        B_boot=B_boot,
        B_perm=B_perm,
        fdr_q=fdr_q,
        block_size=4,
        seed=seed,
        verbose=verbose,
    )
    ranked = apply_dual_threshold(results, fdr_q=fdr_q, stability_min=0.60, verbose=verbose)
    return ranked


# ---------------------------------------------------------------------------
# Pearson + Bonferroni (no stability needed)
# ---------------------------------------------------------------------------

def calibrate_pearson_bonferroni(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Pearson correlation with Bonferroni FWER correction.
    No stability selection needed — Bonferroni is already conservative.
    """
    from scipy.stats import pearsonr

    cols = list(df.columns)
    results = []
    all_p = []
    entries = []

    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            if j <= i:
                continue
            x = df[src].values
            y = df[tgt].values
            try:
                r, p = pearsonr(x, y)
                if not (np.isfinite(r) and np.isfinite(p)):
                    r, p = 0.0, 1.0
            except Exception:
                r, p = 0.0, 1.0
            all_p.append(p)
            entries.append({'source': src, 'target': tgt, 'r': r, 'p': p})

    m = len(all_p)
    alpha_bon = 0.05 / m if m > 0 else 0.05

    from scipy.stats import norm
    for entry, p_raw in zip(entries, all_p):
        sig = p_raw < alpha_bon
        z = min(norm.ppf(1 - p_raw), 4.0) if p_raw < 1.0 else 0.0
        results.append({
            'source': entry['source'],
            'target': entry['target'],
            'pair': (entry['source'], entry['target']),
            'test_type': 'correlational',
            'T_obs': abs(entry['r']),
            'p_value': float(p_raw),
            'z_score': float(z),
            'fdr_adjusted_p': float(min(p_raw * m, 1.0)),
            'fdr_significant': sig,
            'selection_frequency': 1.0 if sig else 0.0,
            'score': float(z) if sig else 0.0,
            'stable': True,
            'passes_dual_threshold': sig,
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------

def head_to_head_comparison(
    df: pd.DataFrame,
    ground_truth: list[dict],
    null_pairs: list[dict],
    B_boot: int = 50,
    B_perm: int = 100,
    verbose: bool = True,
) -> dict:
    """
    THE DEFINITIVE COMPARISON.

    Run all methods through calibration. Evaluate all with same metrics.
    """
    k_vals = [5, 10, 15, 20]

    if verbose:
        print('\n' + '=' * 70)
        print('HEAD-TO-HEAD CALIBRATED COMPARISON')
        print('=' * 70)

    method_results = {}

    # 1. K-Scarcity (full 6-step)
    if verbose:
        print('\n[1/4] K-Scarcity (6-step calibrated)...')
    ksc = calibrate_kscarcity(df, B_boot=B_boot, B_perm=B_perm, verbose=verbose)
    method_results['K-Scarcity calib.'] = ksc

    # 2. Graphical Lasso
    if verbose:
        print('\n[2/4] Graphical Lasso (stability-selected)...')
    gl = calibrate_graphical_lasso(df, B_boot=B_boot)
    method_results['Graphical Lasso'] = gl

    # 3. Economist baseline
    if verbose:
        print('\n[3/4] Economist baseline (stability-selected)...')
    econ = calibrate_economist_baseline(df, B_boot=B_boot)
    method_results['Economist baseline'] = econ

    # 4. Pearson + Bonferroni
    if verbose:
        print('\n[4/4] Pearson + Bonferroni...')
    bon = calibrate_pearson_bonferroni(df)
    method_results['Pearson+Bonferroni'] = bon

    # Evaluate all
    comparison: dict[str, dict] = {}
    for method_name, ranked in method_results.items():
        pr = precision_recall_at_k_calibrated(ranked, ground_truth, k_values=k_vals)
        fg_rank = first_gt_rank(ranked, ground_truth)
        nfpr = null_fpr_calibrated(ranked, null_pairs)
        n_sel = sum(1 for r in ranked if r.get('passes_dual_threshold', False))
        comparison[method_name] = {
            'precision': pr['precision'],
            'recall': pr['recall'],
            'first_gt_rank': fg_rank,
            'null_fpr': nfpr,
            'n_selected': n_sel,
        }

    if verbose:
        print('\n' + '=' * 70)
        print('COMPARISON TABLE')
        print('=' * 70)
        # Header
        k_cols = '  '.join([f'P@{k:2d}' for k in k_vals] + [f'R@{k:2d}' for k in k_vals])
        print(f"  {'Method':22s}  {k_cols}  {'1stGT':>6}  {'NulFPR':>7}  {'#Sel':>5}")
        print(f"  {'-' * 100}")
        for method_name, m in comparison.items():
            p_vals = '  '.join([f'{m["precision"].get(k, 0):>5.3f}' for k in k_vals])
            r_vals = '  '.join([f'{m["recall"].get(k, 0):>5.3f}' for k in k_vals])
            fg = m['first_gt_rank']
            fg_str = str(fg) if fg > 0 else 'N/A'
            print(f"  {method_name:22s}  {p_vals}  {r_vals}  {fg_str:>6}  "
                  f"{m['null_fpr']:>7.3f}  {m['n_selected']:>5d}")
        print()
        print('  If K-Scarcity calib. has higher P@k than Graphical Lasso: multi-type')
        print('  streaming design adds value even after proper calibration.')
        print('  If Graphical Lasso wins: contribution is the calibration pipeline')
        print('  and typed characterization, not discovery performance. Both publishable.')

    return {
        'comparison': comparison,
        'method_rankings': {k: v for k, v in method_results.items()},
    }

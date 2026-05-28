"""
Step 1: Replace Bayesian confidence with permutation-based p-values.

For each hypothesis, compute an empirically exact p-value by comparing the
observed test statistic against a null distribution built from permuted data.

Fixes:
- 41% FPR on noise (permutation p-values are uniform under null by construction)
- Distributional mismatch across tests (all converted to p-values)
- Asymptotic invalidity at N<15 (permutation tests are exact at any N)
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

# ---------------------------------------------------------------------------
# Supported test types and their arity
# ---------------------------------------------------------------------------

PAIRWISE_TYPES = ['correlational', 'causal', 'competitive', 'compositional', 'functional']
UNIVARIATE_TYPES = ['temporal', 'equilibrium', 'structural']
SKIP_TYPES = ['mediating', 'synergistic', 'moderating', 'graph', 'similarity', 'logical',
              'probabilistic']


def compute_native_statistic(
    x: np.ndarray,
    y: Optional[np.ndarray],
    test_type: str,
) -> float:
    """
    Compute the native test statistic for a given hypothesis type.

    Returns a float where higher = stronger evidence of relationship.
    Returns 0.0 when insufficient data or test not applicable.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # Fast NaN guard — any NaN in x or y → return 0 (avoids slow SVD failure path)
    if not np.isfinite(x).all():
        return 0.0
    if y is not None and not np.isfinite(np.asarray(y, dtype=np.float64)).all():
        return 0.0

    if test_type == 'temporal':
        if n < 4:
            return 0.0
        r = np.corrcoef(x[:-1], x[1:])[0, 1]
        return abs(r) if np.isfinite(r) else 0.0

    elif test_type == 'correlational':
        if y is None or n < 4:
            return 0.0
        y = np.asarray(y, dtype=np.float64)
        r = np.corrcoef(x, y)[0, 1]
        return abs(r) if np.isfinite(r) else 0.0

    elif test_type == 'causal':
        # Granger F-statistic: max of forward (x→y) and backward (y→x)
        if y is None or n < 10:
            return 0.0
        y = np.asarray(y, dtype=np.float64)

        def granger_f(cause: np.ndarray, effect: np.ndarray) -> float:
            try:
                # Restricted: effect_t = a + b*effect_{t-1}
                eff = effect[1:]
                eff_lag = effect[:-1]
                cau_lag = cause[:-1]
                m = len(eff)
                X_r = np.column_stack([np.ones(m), eff_lag])
                beta_r, _, _, _ = np.linalg.lstsq(X_r, eff, rcond=None)
                rss_r = float(np.sum((eff - X_r @ beta_r) ** 2))

                # Unrestricted: effect_t = a + b*effect_{t-1} + c*cause_{t-1}
                X_u = np.column_stack([np.ones(m), eff_lag, cau_lag])
                beta_u, _, _, _ = np.linalg.lstsq(X_u, eff, rcond=None)
                rss_u = float(np.sum((eff - X_u @ beta_u) ** 2))

                if rss_u < 1e-12 or m <= 3:
                    return 0.0
                f = ((rss_r - rss_u) / 1.0) / (rss_u / (m - 3))
                return max(f, 0.0)
            except Exception:
                return 0.0

        f_fwd = granger_f(x, y)
        f_bwd = granger_f(y, x)
        return max(f_fwd, f_bwd)

    elif test_type == 'competitive':
        if y is None or n < 4:
            return 0.0
        y = np.asarray(y, dtype=np.float64)
        r = np.corrcoef(x, y)[0, 1]
        return abs(r) if (np.isfinite(r) and r < 0) else 0.0

    elif test_type == 'compositional':
        # R² of linear regression y ~ x
        if y is None or n < 4:
            return 0.0
        y = np.asarray(y, dtype=np.float64)
        try:
            X = np.column_stack([np.ones(n), x])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot < 1e-12:
                return 0.0
            return float(max(1.0 - ss_res / ss_tot, 0.0))
        except Exception:
            return 0.0

    elif test_type == 'equilibrium':
        # |ADF-like statistic| — more negative = more mean-reverting
        if n < 5:
            return 0.0
        try:
            # Try statsmodels ADF
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(x, autolag='AIC', maxlag=min(2, (n - 1) // 3))
            return abs(adf_result[0])
        except Exception:
            pass
        # Fallback: |mean(diff) / std(diff)|
        try:
            dx = np.diff(x)
            if np.std(dx) < 1e-12:
                return 0.0
            return abs(np.mean(dx) / np.std(dx))
        except Exception:
            return 0.0

    elif test_type == 'functional':
        # R²_quadratic - R²_linear (nonlinearity gain)
        if y is None or n < 6:
            return 0.0
        y = np.asarray(y, dtype=np.float64)
        try:
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot < 1e-12:
                return 0.0

            # Linear fit
            X_lin = np.column_stack([np.ones(n), x])
            b_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
            r2_lin = max(1.0 - np.sum((y - X_lin @ b_lin) ** 2) / ss_tot, 0.0)

            # Quadratic fit
            X_quad = np.column_stack([np.ones(n), x, x ** 2])
            b_quad, _, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
            r2_quad = max(1.0 - np.sum((y - X_quad @ b_quad) ** 2) / ss_tot, 0.0)

            return float(max(r2_quad - r2_lin, 0.0))
        except Exception:
            return 0.0

    elif test_type == 'structural':
        # Max Chow F-statistic across candidate breakpoints
        if n < 12:
            return 0.0
        try:
            max_f = 0.0
            trend = np.arange(n, dtype=np.float64)
            for t in range(5, n - 5):
                try:
                    # Full model
                    X_full = np.column_stack([np.ones(n), trend])
                    b_full, _, _, _ = np.linalg.lstsq(X_full, x, rcond=None)
                    rss_full = float(np.sum((x - X_full @ b_full) ** 2))

                    # Sub-model 1
                    X1 = np.column_stack([np.ones(t), trend[:t]])
                    b1, _, _, _ = np.linalg.lstsq(X1, x[:t], rcond=None)
                    rss1 = float(np.sum((x[:t] - X1 @ b1) ** 2))

                    # Sub-model 2
                    X2 = np.column_stack([np.ones(n - t), trend[t:]])
                    b2, _, _, _ = np.linalg.lstsq(X2, x[t:], rcond=None)
                    rss2 = float(np.sum((x[t:] - X2 @ b2) ** 2))

                    denom = rss1 + rss2
                    if denom < 1e-12 or n <= 4:
                        continue
                    chow_f = ((rss_full - denom) / 2.0) / (denom / (n - 4))
                    max_f = max(max_f, chow_f)
                except Exception:
                    continue
            return max(max_f, 0.0)
        except Exception:
            return 0.0

    return 0.0


def permute_for_test(
    x: np.ndarray,
    y: Optional[np.ndarray],
    test_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate a single permutation appropriate for the test type.

    Cross-variable (correlational, competitive, compositional, functional):
        Shuffle y independently of x.

    Causal (Granger): circular shift y by random offset to preserve
        y's autocorrelation structure while breaking the x→y lag relationship.

    Univariate (temporal, equilibrium): phase randomization of x
        (preserves power spectrum, destroys temporal ordering).

    Structural: block permutation of x (blocks of 3-5) to preserve
        local structure while breaking global regime shifts.
    """
    n = len(x)

    if test_type == 'causal':
        # Circular shift y by random offset
        shift = rng.integers(1, max(n, 2))
        y_perm = np.roll(y, int(shift))
        return x.copy(), y_perm

    elif test_type in ('correlational', 'competitive', 'compositional', 'functional'):
        # Independent shuffle of y
        y_perm = y.copy()
        rng.shuffle(y_perm)
        return x.copy(), y_perm

    elif test_type in ('temporal', 'equilibrium'):
        # Phase randomization of x
        try:
            fft_x = np.fft.rfft(x)
            n_freqs = len(fft_x)
            # Randomize phases, keep amplitudes
            phases = rng.uniform(0, 2 * np.pi, n_freqs)
            fft_x_perm = np.abs(fft_x) * np.exp(1j * phases)
            x_perm = np.fft.irfft(fft_x_perm, n=n)
            return x_perm.astype(x.dtype), None
        except Exception:
            # Fallback: circular shift
            shift = rng.integers(1, max(n, 2))
            return np.roll(x, int(shift)), None

    elif test_type == 'structural':
        # Block permutation of x (blocks of 3)
        block_size = 3
        n_blocks = (n + block_size - 1) // block_size
        blocks = [x[i * block_size: (i + 1) * block_size] for i in range(n_blocks)]
        rng.shuffle(blocks)
        x_perm = np.concatenate(blocks)[:n]
        return x_perm, None

    # Default: shuffle x
    x_perm = x.copy()
    rng.shuffle(x_perm)
    return x_perm, None


def compute_permutation_pvalue(
    x: np.ndarray,
    y: Optional[np.ndarray],
    test_type: str,
    B: int = 200,
    seed: int = 42,
) -> dict:
    """
    Compute the permutation p-value for a single hypothesis.

    Uses the Phipson & Smyth 2010 correction: p = (1 + #{T_perm >= T_obs}) / (1 + B)
    This ensures p > 0 always and avoids the p=0 anti-conservatism.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)

    t_obs = compute_native_statistic(x, y, test_type)

    null_dist = np.zeros(B, dtype=np.float64)
    for b in range(B):
        x_p, y_p = permute_for_test(x, y, test_type, rng)
        null_dist[b] = compute_native_statistic(x_p, y_p, test_type)

    n_extreme = int(np.sum(null_dist >= t_obs))
    p_value = (1 + n_extreme) / (1 + B)

    return {
        'test_type': test_type,
        'T_obs': float(t_obs),
        'p_value': float(p_value),
        'null_mean': float(null_dist.mean()),
        'null_std': float(null_dist.std()),
        'B': B,
        'null_distribution': null_dist.tolist(),
    }


def _batch_multi_pvalues(
    mat: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> dict:
    """
    Vectorized permutation p-values for multiple pairwise types from one batch.

    For each permutation, compute the full shuffled correlation matrix once.
    Derive test statistics for correlational, competitive, and compositional
    simultaneously (all three are functions of Pearson r):
        correlational: |r|
        competitive:   |r| if r < 0 else 0
        compositional: r²  (R² of linear regression = r² for simple regression)

    Also computes univariate lag-1 autocorrelation (temporal).

    Returns dict:
        'p_corr':  (K, K) correlational p-values
        'p_comp':  (K, K) competitive p-values
        'p_compo': (K, K) compositional p-values
        'p_lag1':  (K,)   temporal p-values
        'obs_r':   (K, K) observed Pearson r matrix
    """
    n, k = mat.shape

    # Replace NaN with column means so correlation matrix doesn't explode.
    # NaN columns get neutralized (correlation → 0) but don't block computation.
    mat = mat.copy()
    for j in range(k):
        col = mat[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col_mean = np.nanmean(col)
            col[bad] = col_mean if np.isfinite(col_mean) else 0.0
            mat[:, j] = col

    # Observed statistics
    obs_r = np.corrcoef(mat.T)  # (K, K), keep signs
    obs_corr = np.abs(obs_r)    # |r|
    obs_comp = np.where(obs_r < 0, np.abs(obs_r), 0.0)  # competitive
    obs_compo = obs_r ** 2      # R²
    np.fill_diagonal(obs_corr, 0.0)
    np.fill_diagonal(obs_comp, 0.0)
    np.fill_diagonal(obs_compo, 0.0)

    # Lag-1 autocorrelations
    obs_lag1 = np.array([
        abs(float(np.corrcoef(mat[:-1, j], mat[1:, j])[0, 1]))
        if n >= 4 else 0.0
        for j in range(k)
    ])

    null_corr = np.zeros((k, k), dtype=np.int32)
    null_comp = np.zeros((k, k), dtype=np.int32)
    null_compo = np.zeros((k, k), dtype=np.int32)
    null_lag1 = np.zeros(k, dtype=np.int32)

    perm_mat = mat.copy()
    for _ in range(B):
        # Shuffle each column independently for cross-variable tests
        for j in range(k):
            rng.shuffle(perm_mat[:, j])
        pcorr = np.corrcoef(perm_mat.T)

        null_corr += (np.abs(pcorr) >= obs_corr).astype(np.int32)
        null_comp += (np.where(pcorr < 0, np.abs(pcorr), 0.0) >= obs_comp).astype(np.int32)
        null_compo += (pcorr ** 2 >= obs_compo).astype(np.int32)

        # Temporal: circular shift of each column
        for j in range(k):
            shift = int(rng.integers(1, max(n, 2)))
            col_p = np.roll(mat[:, j], shift)
            t = abs(float(np.corrcoef(col_p[:-1], col_p[1:])[0, 1]))
            if t >= obs_lag1[j]:
                null_lag1[j] += 1

    p_corr = (1 + null_corr) / (1 + B)
    p_comp = (1 + null_comp) / (1 + B)
    p_compo = (1 + null_compo) / (1 + B)
    p_lag1 = (1 + null_lag1) / (1 + B)

    np.fill_diagonal(p_corr, 1.0)
    np.fill_diagonal(p_comp, 1.0)
    np.fill_diagonal(p_compo, 1.0)

    return {
        'p_corr': p_corr,
        'p_comp': p_comp,
        'p_compo': p_compo,
        'p_lag1': p_lag1,
        'obs_r': obs_r,
    }


def _batch_corrcoef_pvalues(
    mat: np.ndarray,
    B: int,
    rng: np.random.Generator,
    lag1: bool = False,
) -> np.ndarray:
    """Kept for backward compatibility. Use _batch_multi_pvalues for new code."""
    res = _batch_multi_pvalues(mat, B, rng)
    if lag1:
        return res['p_lag1']
    return res['p_corr']


def compute_all_pvalues(
    df: pd.DataFrame,
    B: int = 200,
    seed: int = 42,
    include_types: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Compute permutation p-values for ALL variable pairs and ALL applicable types.

    Pairwise types: correlational, causal, competitive, compositional, functional
    Univariate types: temporal, equilibrium, structural
    Skip: mediating, synergistic (require triples)

    Performance: correlational and temporal use vectorized batch computation
    (all K² pairs in one permutation loop). Other types use pair-at-a-time.
    """
    cols = list(df.columns)
    K = len(cols)
    mat = df.values.astype(np.float64)
    n_pairs = K * (K - 1)
    n_uni = K

    pairwise = [t for t in PAIRWISE_TYPES if include_types is None or t in include_types]
    univariate = [t for t in UNIVARIATE_TYPES if include_types is None or t in include_types]

    total_tests = n_pairs * len(pairwise) + n_uni * len(univariate)
    if verbose:
        print(f'Computing p-values: {n_pairs} pairwise x {len(pairwise)} types '
              f'+ {n_uni} vars x {len(univariate)} types = {total_tests} tests')
        print(f'Each test: {B} permutations  Total computations: '
              f'{total_tests * B:,}')

    rng_master = np.random.default_rng(seed)
    results = []

    # -----------------------------------------------------------------------
    # BATCH: correlational + competitive + compositional + temporal — ONE loop
    # All four are derived from the Pearson correlation matrix in one pass.
    # -----------------------------------------------------------------------
    batch_types = {'correlational', 'competitive', 'compositional', 'temporal'}
    need_batch = (batch_types & (set(pairwise) | set(univariate)))
    if need_batch:
        batch_seed = int(rng_master.integers(0, 2**31))
        batch_rng = np.random.default_rng(batch_seed)
        if verbose:
            print(f'  Batch computing: {sorted(need_batch)} ...')
        batch = _batch_multi_pvalues(mat, B, batch_rng)
        obs_r = batch['obs_r']

        if 'correlational' in pairwise:
            obs_corr = np.abs(obs_r)
            np.fill_diagonal(obs_corr, 0.0)
            for i, src in enumerate(cols):
                for j, tgt in enumerate(cols):
                    if i == j:
                        continue
                    results.append({
                        'test_type': 'correlational',
                        'T_obs': float(obs_corr[i, j]),
                        'p_value': float(batch['p_corr'][i, j]),
                        'null_mean': 0.0, 'null_std': 0.0, 'B': B,
                        'null_distribution': [],
                        'source': src, 'target': tgt, 'pair': (src, tgt),
                    })

        if 'competitive' in pairwise:
            for i, src in enumerate(cols):
                for j, tgt in enumerate(cols):
                    if i == j:
                        continue
                    r_val = float(obs_r[i, j])
                    t_obs = abs(r_val) if r_val < 0 else 0.0
                    results.append({
                        'test_type': 'competitive',
                        'T_obs': t_obs,
                        'p_value': float(batch['p_comp'][i, j]),
                        'null_mean': 0.0, 'null_std': 0.0, 'B': B,
                        'null_distribution': [],
                        'source': src, 'target': tgt, 'pair': (src, tgt),
                    })

        if 'compositional' in pairwise:
            for i, src in enumerate(cols):
                for j, tgt in enumerate(cols):
                    if i == j:
                        continue
                    results.append({
                        'test_type': 'compositional',
                        'T_obs': float(obs_r[i, j] ** 2),
                        'p_value': float(batch['p_compo'][i, j]),
                        'null_mean': 0.0, 'null_std': 0.0, 'B': B,
                        'null_distribution': [],
                        'source': src, 'target': tgt, 'pair': (src, tgt),
                    })

        if 'temporal' in univariate:
            for j, var in enumerate(cols):
                x = mat[:, j]
                t_obs = abs(float(np.corrcoef(x[:-1], x[1:])[0, 1])) if len(x) >= 4 else 0.0
                results.append({
                    'test_type': 'temporal',
                    'T_obs': float(t_obs) if np.isfinite(t_obs) else 0.0,
                    'p_value': float(batch['p_lag1'][j]),
                    'null_mean': 0.0, 'null_std': 0.0, 'B': B,
                    'null_distribution': [],
                    'source': var, 'target': var, 'pair': (var, var),
                })

    # -----------------------------------------------------------------------
    # Per-pair: remaining slow pairwise types (causal, functional)
    # -----------------------------------------------------------------------
    slow_pairwise = [t for t in pairwise if t not in ('correlational', 'competitive', 'compositional')]
    if slow_pairwise:
        for i, src in enumerate(cols):
            x = mat[:, i]
            for j, tgt in enumerate(cols):
                if i == j:
                    continue
                y = mat[:, j]
                for test_type in slow_pairwise:
                    sub_seed = int(rng_master.integers(0, 2**31))
                    r = compute_permutation_pvalue(x, y, test_type, B=B, seed=sub_seed)
                    r['source'] = src
                    r['target'] = tgt
                    r['pair'] = (src, tgt)
                    results.append(r)
            if verbose and (i + 1) % max(1, K // 4) == 0:
                print(f'  Pairwise (causal/functional): variable {i + 1}/{K} done')

    # -----------------------------------------------------------------------
    # Per-var: remaining univariate types (equilibrium, structural)
    # -----------------------------------------------------------------------
    slow_univariate = [t for t in univariate if t != 'temporal']
    if slow_univariate:
        for var in cols:
            x = df[var].values.astype(np.float64)
            for test_type in slow_univariate:
                sub_seed = int(rng_master.integers(0, 2**31))
                r = compute_permutation_pvalue(x, None, test_type, B=B, seed=sub_seed)
                r['source'] = var
                r['target'] = var
                r['pair'] = (var, var)
                results.append(r)

    if verbose:
        print(f'  Done: {len(results)} p-values computed.')

    return results

"""
Per-type specialist baselines for typed relationship discovery.

Each specialist uses the canonical statistical test for one relationship type.
K-Scarcity's advantage is discovering ALL types in one streaming pass; each
specialist here can only discover its own type.

Output format is standardised across all specialists so evaluation can compare
K-Scarcity against the per-type expert on that expert's home turf.
"""
from __future__ import annotations

import sys
from itertools import combinations, permutations
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Standardised output record
# ---------------------------------------------------------------------------

def _discovery(source: str, target: str, rel_type: str,
               confidence: float, sign: int,
               statistic: float, p_value: float,
               method: str,
               mediator: str | None = None) -> dict:
    d = {
        'source': source,
        'target': target,
        'type': rel_type,
        'confidence': float(np.clip(confidence, 0.0, 1.0)),
        'sign': int(sign),
        'statistic': float(statistic),
        'p_value': float(p_value),
        'method': method,
    }
    if mediator is not None:
        d['mediator'] = mediator
    return d


def _clean_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either series is NaN. Return numpy arrays."""
    mask = x.notna() & y.notna()
    return x[mask].to_numpy(float), y[mask].to_numpy(float)


def _clean_triple(x: pd.Series, m: pd.Series,
                  y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = x.notna() & m.notna() & y.notna()
    return (x[mask].to_numpy(float),
            m[mask].to_numpy(float),
            y[mask].to_numpy(float))


# ---------------------------------------------------------------------------
# 1. TEMPORAL — Ljung-Box Q-test + lag-1 ACF
# ---------------------------------------------------------------------------

def discover_temporal(df: pd.DataFrame,
                      significance: float = 0.10) -> list[dict]:
    """
    Discover temporally persistent series using Ljung-Box Q-test at lag 1.

    For each variable:
        1. Drop NaN, compute lag-1 ACF.
        2. Run Ljung-Box Q-test at lag 1.
        3. Report if p < significance AND |acf_1| > 0.3.

    Confidence = |acf_1|. Sign = sign(acf_1).
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        acorr_ljungbox = None

    results = []
    for col in df.columns:
        x = df[col].dropna().to_numpy(float)
        if len(x) < 10:
            continue

        # Lag-1 autocorrelation
        acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        if np.isnan(acf1):
            continue

        # Ljung-Box test
        try:
            if acorr_ljungbox is not None:
                lb = acorr_ljungbox(x, lags=[1], return_df=True)
                p = float(lb['lb_pvalue'].iloc[0])
                stat = float(lb['lb_stat'].iloc[0])
            else:
                # Manual Q stat: Q = n*(n+2) * sum(rk^2/(n-k))
                n = len(x)
                stat = n * (n + 2) * acf1 ** 2 / (n - 1)
                p = float(1 - stats.chi2.cdf(stat, df=1))
        except Exception:
            continue

        if p < significance and abs(acf1) > 0.3:
            results.append(_discovery(
                source=col, target=col,
                rel_type='temporal',
                confidence=abs(acf1),
                sign=int(np.sign(acf1)),
                statistic=stat, p_value=p,
                method='ljung_box_acf',
            ))
    return results


# ---------------------------------------------------------------------------
# 2. CAUSAL — Granger causality test
# ---------------------------------------------------------------------------

def discover_causal(df: pd.DataFrame, max_lag: int = 2,
                    significance: float = 0.10) -> list[dict]:
    """
    Discover Granger-causal relationships between all ordered pairs (X, Y).

    For each pair: test X->Y and Y->X. Report the significant direction.
    If both are significant, report the lower-p direction.
    Confidence = 1 - p_value_min_across_lags.
    Sign estimated from OLS lag-1 coefficient.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print('  WARNING: statsmodels not available; Granger test skipped')
        return []

    cols = [c for c in df.columns]
    results = []
    skipped = 0

    for src, tgt in permutations(cols, 2):
        x_raw = df[src].dropna()
        y_raw = df[tgt].dropna()
        # Align on common index
        common = x_raw.index.intersection(y_raw.index)
        if len(common) < max_lag + 5:
            skipped += 1
            continue

        xy = pd.DataFrame({'x': x_raw[common], 'y': y_raw[common]})

        try:
            gc = grangercausalitytests(xy[['y', 'x']], maxlag=max_lag,
                                       verbose=False)
            # Collect p-values across lags
            p_vals = [gc[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            p_min = min(p_vals)
            best_lag = p_vals.index(p_min) + 1
            f_stat = gc[best_lag][0]['ssr_ftest'][0]
        except Exception:
            skipped += 1
            continue

        if p_min >= significance:
            continue

        # Estimate sign from OLS coefficient at best_lag
        try:
            y_arr = xy['y'].to_numpy(float)
            x_arr = xy['x'].to_numpy(float)
            y_dep = y_arr[best_lag:]
            x_lag = x_arr[:-best_lag] if best_lag > 0 else x_arr
            coef = float(np.polyfit(x_lag, y_dep, 1)[0])
            sign = int(np.sign(coef)) if not np.isnan(coef) else 0
        except Exception:
            sign = 0

        results.append(_discovery(
            source=src, target=tgt,
            rel_type='causal',
            confidence=1.0 - p_min,
            sign=sign,
            statistic=f_stat, p_value=p_min,
            method=f'granger_lag{best_lag}',
        ))

    if skipped:
        print(f'  Granger: skipped {skipped} pairs (insufficient data)')
    return results


# ---------------------------------------------------------------------------
# 3. CORRELATIONAL — Pearson + Spearman dual test
# ---------------------------------------------------------------------------

def discover_correlational(df: pd.DataFrame,
                           significance: float = 0.10,
                           min_abs_r: float = 0.3) -> list[dict]:
    """
    Discover significant correlations confirmed by both Pearson and Spearman.

    Confidence = |Pearson r|. Sign = sign(r).
    Reports each unordered pair once (canonical order by column name).
    """
    cols = df.columns.tolist()
    results = []

    for c1, c2 in combinations(cols, 2):
        xv, yv = _clean_pair(df[c1], df[c2])
        if len(xv) < 5:
            continue
        try:
            r, p_pear = stats.pearsonr(xv, yv)
            rho, p_spear = stats.spearmanr(xv, yv)
        except Exception:
            continue

        if (p_pear < significance and p_spear < significance
                and abs(r) >= min_abs_r):
            results.append(_discovery(
                source=c1, target=c2,
                rel_type='correlational',
                confidence=abs(r),
                sign=int(np.sign(r)),
                statistic=r, p_value=max(p_pear, p_spear),
                method='pearson_spearman',
            ))
    return results


# ---------------------------------------------------------------------------
# 4. COMPETITIVE — negative correlation + constant-sum test
# ---------------------------------------------------------------------------

def discover_competitive(df: pd.DataFrame,
                         significance: float = 0.10,
                         min_abs_r: float = 0.2) -> list[dict]:
    """
    Discover competitive (zero-sum / crowding-out) relationships.

    Two criteria (either triggers a report):
        A. Pearson r < -min_abs_r, significant.
        B. std(X+Y) < 0.5*(std(X)+std(Y)) — near-constant sum.

    Confidence = max(|r|, 1 - sum_cv) where sum_cv = std(X+Y)/mean(|X+Y|).
    Sign = -1 by construction (competitive).
    """
    cols = df.columns.tolist()
    results = []

    for c1, c2 in combinations(cols, 2):
        xv, yv = _clean_pair(df[c1], df[c2])
        if len(xv) < 5:
            continue

        try:
            r, p = stats.pearsonr(xv, yv)
        except Exception:
            r, p = 0.0, 1.0

        # Criterion A: significant negative correlation
        crit_a = (r < -min_abs_r) and (p < significance)

        # Criterion B: near-constant sum
        s = xv + yv
        std_sum = float(np.std(s))
        std_parts = float(np.std(xv) + np.std(yv))
        crit_b = (std_parts > 0) and (std_sum < 0.5 * std_parts)

        if crit_a or crit_b:
            conf = max(abs(r), 1.0 - std_sum / max(std_parts, 1e-9))
            results.append(_discovery(
                source=c1, target=c2,
                rel_type='competitive',
                confidence=conf,
                sign=-1,
                statistic=r, p_value=p,
                method='neg_corr_or_const_sum',
            ))
    return results


# ---------------------------------------------------------------------------
# 5. COMPOSITIONAL — sum-constraint test on pairs and triples
# ---------------------------------------------------------------------------

def discover_compositional(df: pd.DataFrame,
                           significance: float = 0.10,
                           residual_frac: float = 0.3) -> list[dict]:
    """
    Discover compositional (parts-of-whole) relationships.

    Strategy A — pair: X is a positive predictor of Y with high R^2 (both
    are components of the same aggregate).
        Criterion: Pearson r > 0.4, significant, and both are denominated as
        % of GDP (proxy: both are positive-valued and bounded [0, 200]).

    Strategy B — triple: X + Y ~ Z or X + Y + Z ~ constant.
        Criterion: std(X + Y - Z) / std(Z) < residual_frac.

    Confidence for A = R^2. For B = 1 - std(X+Y-Z)/std(Z).
    """
    cols = df.columns.tolist()
    results = []

    # Strategy A: GDP-component pairs
    for c1, c2 in combinations(cols, 2):
        xv, yv = _clean_pair(df[c1], df[c2])
        if len(xv) < 5:
            continue
        # Both must be plausibly GDP-component-scaled (0-200% range)
        if not (0 <= np.nanmin(xv) and np.nanmax(xv) <= 200 and
                0 <= np.nanmin(yv) and np.nanmax(yv) <= 200):
            continue
        try:
            r, p = stats.pearsonr(xv, yv)
        except Exception:
            continue
        if r > 0.4 and p < significance:
            results.append(_discovery(
                source=c1, target=c2,
                rel_type='compositional',
                confidence=r ** 2,
                sign=+1,
                statistic=r, p_value=p,
                method='gdp_component_pearson',
            ))

    # Strategy B: triples X + Y ~ Z
    seen = set()
    for c1, c2, c3 in permutations(cols, 3):
        key = tuple(sorted([c1, c2, c3]))
        if key in seen:
            continue
        seen.add(key)
        mask = df[c1].notna() & df[c2].notna() & df[c3].notna()
        if mask.sum() < 5:
            continue
        xv = df[c1][mask].to_numpy(float)
        yv = df[c2][mask].to_numpy(float)
        zv = df[c3][mask].to_numpy(float)
        resid = xv + yv - zv
        std_resid = float(np.std(resid))
        std_z = float(np.std(zv))
        if std_z < 1e-9:
            continue
        frac = std_resid / std_z
        if frac < residual_frac:
            conf = 1.0 - frac / residual_frac
            results.append(_discovery(
                source=c1, target=c3,
                rel_type='compositional',
                confidence=conf,
                sign=+1,
                statistic=frac, p_value=0.0,
                method=f'sum_constraint_{c2}',
            ))

    return results


# ---------------------------------------------------------------------------
# 6. EQUILIBRIUM — ADF + KPSS stationarity + Engle-Granger cointegration
# ---------------------------------------------------------------------------

def discover_equilibrium(df: pd.DataFrame,
                         significance: float = 0.10) -> list[dict]:
    """
    Discover mean-reverting series (unit-variable equilibrium) and
    cointegrated pairs (bivariate equilibrium).

    Single-variable: ADF rejects (p < significance) AND KPSS does not reject.
    Pair: Engle-Granger cointegration test p < significance.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss, coint
    except ImportError:
        print('  WARNING: statsmodels not available; equilibrium test skipped')
        return []

    cols = df.columns.tolist()
    results = []

    # Single-variable mean-reversion
    for col in cols:
        x = df[col].dropna().to_numpy(float)
        if len(x) < 10:
            continue
        try:
            adf_stat, adf_p, *_ = adfuller(x, autolag='AIC')
            kpss_stat, kpss_p, *_ = kpss(x, regression='c', nlags='auto')
        except Exception:
            continue

        if adf_p < significance and kpss_p > significance:
            crit_val = -2.86  # approx 5% ADF critical value
            conf = min(1.0, abs(adf_stat) / abs(crit_val))
            results.append(_discovery(
                source=col, target=col,
                rel_type='equilibrium',
                confidence=conf,
                sign=0,
                statistic=adf_stat, p_value=adf_p,
                method='adf_kpss',
            ))

    # Bivariate cointegration
    for c1, c2 in combinations(cols, 2):
        xv, yv = _clean_pair(df[c1], df[c2])
        if len(xv) < 12:
            continue
        try:
            score, p, _ = coint(xv, yv)
        except Exception:
            continue
        if p < significance:
            results.append(_discovery(
                source=c1, target=c2,
                rel_type='equilibrium',
                confidence=1.0 - p,
                sign=0,
                statistic=score, p_value=p,
                method='engle_granger_coint',
            ))

    return results


# ---------------------------------------------------------------------------
# 7. MEDIATING — Baron-Kenny + Sobel test
# ---------------------------------------------------------------------------

def _sobel_test(a: float, b: float, se_a: float, se_b: float) -> tuple[float, float]:
    """Sobel z-test for mediation. Returns (z, p_two_tailed)."""
    indirect = a * b
    se = float(np.sqrt(b ** 2 * se_a ** 2 + a ** 2 * se_b ** 2))
    if se < 1e-12:
        return 0.0, 1.0
    z = indirect / se
    p = 2 * float(stats.norm.sf(abs(z)))
    return z, p


def _ols_coef_se(x: np.ndarray, y: np.ndarray,
                 controls: np.ndarray | None = None) -> tuple[float, float]:
    """OLS coefficient and SE of x predicting y (with optional controls)."""
    if controls is not None:
        X = np.column_stack([np.ones(len(x)), x, controls])
    else:
        X = np.column_stack([np.ones(len(x)), x])
    try:
        betas, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        n, k = X.shape
        if n <= k:
            return float('nan'), float('nan')
        sigma2 = float(np.sum((y - X @ betas) ** 2)) / (n - k)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        coef = float(betas[1])
        se = float(np.sqrt(max(cov[1, 1], 0)))
        return coef, se
    except Exception:
        return float('nan'), float('nan')


def discover_mediating(df: pd.DataFrame,
                       significance: float = 0.05,
                       min_r_prefilter: float = 0.40,
                       min_indirect: float = 0.05) -> list[dict]:
    """
    Discover mediation chains X -> M -> Y using Baron-Kenny + Sobel test.

    Pre-filters (applied in order to reduce O(n^3) search):
        1. |r(X,M)| >= min_r_prefilter AND |r(M,Y)| >= min_r_prefilter
        2. |indirect effect a*b| >= min_indirect
    Bonferroni correction: alpha divided by number of qualifying triples.
    Confidence = |indirect_effect| / (|indirect| + |direct|).
    """
    cols = df.columns.tolist()
    results = []

    corr = df.corr(numeric_only=True)

    # Collect qualifying triples (pre-filter only)
    qualifying = []
    for src, med, tgt in permutations(cols, 3):
        try:
            r_xm = abs(corr.loc[src, med])
            r_my = abs(corr.loc[med, tgt])
        except KeyError:
            continue
        if r_xm >= min_r_prefilter and r_my >= min_r_prefilter:
            qualifying.append((src, med, tgt))

    n_tests = max(len(qualifying), 1)
    alpha_adj = min(significance, significance / n_tests * 10)  # mild Bonferroni

    for src, med, tgt in qualifying:
        xv, mv, yv = _clean_triple(df[src], df[med], df[tgt])
        if len(xv) < 15:
            continue

        a, se_a = _ols_coef_se(xv, mv)
        if np.isnan(a) or abs(a) < 0.01:
            continue

        b, se_b = _ols_coef_se(mv, yv, controls=xv)
        if np.isnan(b) or abs(b) < 0.01:
            continue

        indirect = a * b
        if abs(indirect) < min_indirect:
            continue

        c_prime, _ = _ols_coef_se(xv, yv, controls=mv)

        z, p = _sobel_test(a, b, se_a, se_b)
        if p >= alpha_adj:
            continue

        total_effect = abs(indirect) + abs(c_prime) if not np.isnan(c_prime) else abs(indirect)
        conf = abs(indirect) / max(total_effect, 1e-9)

        results.append(_discovery(
            source=src, target=tgt,
            rel_type='mediating',
            confidence=conf,
            sign=int(np.sign(indirect)),
            statistic=z, p_value=p,
            method='baron_kenny_sobel',
            mediator=med,
        ))

    return results


# ---------------------------------------------------------------------------
# 8. SYNERGISTIC — interaction regression X*Z -> Y
# ---------------------------------------------------------------------------

def discover_synergistic(df: pd.DataFrame,
                         significance: float = 0.05,
                         min_r_main: float = 0.25,
                         min_interaction_coef: float = 0.05) -> list[dict]:
    """
    Discover interaction / moderation effects (X*Z -> Y).

    Fits Y = b0 + b1*X + b2*Z + b3*(X*Z) + e.
    Pre-filters (both must hold before fitting):
        1. |r(X,Y)| >= min_r_main AND |r(Z,Y)| >= min_r_main (both main effects present)
        2. |b3| >= min_interaction_coef (non-trivial interaction)
    Bonferroni: alpha / n_qualifying_triples (mild).
    Confidence = |t_stat| / t_critical(alpha=0.05, df=n-4).
    """
    cols = df.columns.tolist()
    results = []
    corr = df.corr(numeric_only=True)

    qualifying = []
    for src, mod, tgt in permutations(cols, 3):
        try:
            r_xy = abs(corr.loc[src, tgt])
            r_zy = abs(corr.loc[mod, tgt])
        except KeyError:
            continue
        if r_xy >= min_r_main and r_zy >= min_r_main:
            qualifying.append((src, mod, tgt))

    n_tests = max(len(qualifying), 1)
    alpha_adj = min(significance, significance / n_tests * 10)
    t_crit = stats.t.ppf(1 - alpha_adj / 2, df=max(1, len(df) - 4))

    for src, mod, tgt in qualifying:
        mask = df[src].notna() & df[mod].notna() & df[tgt].notna()
        n = mask.sum()
        if n < 12:
            continue
        x = df[src][mask].to_numpy(float)
        z = df[mod][mask].to_numpy(float)
        y = df[tgt][mask].to_numpy(float)

        x_s = (x - x.mean()) / max(x.std(), 1e-9)
        z_s = (z - z.mean()) / max(z.std(), 1e-9)
        xz = x_s * z_s

        X = np.column_stack([np.ones(n), x_s, z_s, xz])
        try:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ betas
            sigma2 = float(np.sum(resid ** 2)) / max(n - 4, 1)
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            b3 = float(betas[3])
            se3 = float(np.sqrt(max(cov[3, 3], 0)))
            if se3 < 1e-12:
                continue
            t_stat = b3 / se3
            p = 2 * float(stats.t.sf(abs(t_stat), df=n - 4))
        except Exception:
            continue

        if p < alpha_adj and abs(b3) >= min_interaction_coef:
            conf = min(1.0, abs(t_stat) / max(t_crit, 1e-9))
            results.append(_discovery(
                source=src, target=tgt,
                rel_type='synergistic',
                confidence=conf,
                sign=int(np.sign(b3)),
                statistic=t_stat, p_value=p,
                method='interaction_regression',
                mediator=mod,
            ))

    return results


# ---------------------------------------------------------------------------
# 9. FUNCTIONAL — polynomial vs linear fit improvement
# ---------------------------------------------------------------------------

def discover_functional(df: pd.DataFrame,
                        significance: float = 0.05,
                        min_r2_gain: float = 0.15,
                        min_r2_abs: float = 0.35) -> list[dict]:
    """
    Discover nonlinear functional relationships f(X) ~ Y.

    For each pair (X, Y):
        1. Linear R^2.
        2. Quadratic R^2.
        3. Log-linear R^2 (if X > 0 everywhere).
    Report if best nonlinear R^2 > linear + min_r2_gain AND
    absolute nonlinear R^2 >= min_r2_abs AND
    F-test for the additional term is significant at `significance`.
    Confidence = R^2_gain (nonlinear improvement).
    """
    cols = df.columns.tolist()
    results = []

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / max(ss_tot, 1e-12)

    for src, tgt in permutations(cols, 2):
        xv, yv = _clean_pair(df[src], df[tgt])
        if len(xv) < 10:
            continue

        x = xv
        y = yv

        # Linear
        try:
            c_lin = np.polyfit(x, y, 1)
            r2_lin = _r2(y, np.polyval(c_lin, x))
        except Exception:
            continue

        best_r2 = r2_lin
        best_kind = 'linear'
        best_f = 0.0
        best_p = 1.0

        # Quadratic
        try:
            X_lin = np.column_stack([np.ones(len(x)), x])
            X_quad = np.column_stack([np.ones(len(x)), x, x ** 2])
            b_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
            b_quad, _, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
            r2_quad = _r2(y, X_quad @ b_quad)
            # F-test for added quadratic term
            n = len(y)
            ss_lin = np.sum((y - X_lin @ b_lin) ** 2)
            ss_quad = np.sum((y - X_quad @ b_quad) ** 2)
            if ss_quad < ss_lin and n > 3:
                f_quad = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
                p_quad = float(1 - stats.f.cdf(f_quad, 1, n - 3))
                if r2_quad > best_r2 + min_r2_gain and p_quad < significance:
                    best_r2 = r2_quad
                    best_kind = 'quadratic'
                    best_f = f_quad
                    best_p = p_quad
        except Exception:
            pass

        # Log-linear (only if X > 0)
        if np.all(x > 0):
            try:
                log_x = np.log(x)
                X_log = np.column_stack([np.ones(len(x)), log_x])
                b_log, _, _, _ = np.linalg.lstsq(X_log, y, rcond=None)
                r2_log = _r2(y, X_log @ b_log)
                n = len(y)
                X_lin = np.column_stack([np.ones(len(x)), x])
                b_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
                ss_lin = np.sum((y - X_lin @ b_lin) ** 2)
                ss_log = np.sum((y - X_log @ b_log) ** 2)
                if ss_log < ss_lin and n > 2:
                    f_log = ((ss_lin - ss_log) / 1) / (ss_log / (n - 2))
                    p_log = float(1 - stats.f.cdf(f_log, 1, n - 2))
                    if r2_log > best_r2 + min_r2_gain and p_log < significance:
                        best_r2 = r2_log
                        best_kind = 'log_linear'
                        best_f = f_log
                        best_p = p_log
            except Exception:
                pass

        if (best_kind != 'linear'
                and best_r2 > r2_lin + min_r2_gain
                and best_r2 >= min_r2_abs):
            results.append(_discovery(
                source=src, target=tgt,
                rel_type='functional',
                confidence=min(1.0, best_r2 - r2_lin),
                sign=+1,
                statistic=best_f, p_value=best_p,
                method=f'nonlinear_{best_kind}',
            ))

    return results


# ---------------------------------------------------------------------------
# 10. STRUCTURAL — Chow test for regime breaks
# ---------------------------------------------------------------------------

def discover_structural(df: pd.DataFrame,
                        significance: float = 0.10) -> list[dict]:
    """
    Discover structural regime changes using the Chow break-point test.

    For each variable X:
        For each candidate break year t (rows 5 through N-5):
            Fit two AR(1) models (before and after t).
            Chow F-statistic = ((RSS_pooled - RSS_1 - RSS_2) / k) / ((RSS_1 + RSS_2) / (n - 2k))
        Select the t with the highest F-statistic.
        Report if F_max > F_critical at significance.

    Confidence = F_max / F_critical.
    """
    results = []

    for col in df.columns:
        x = df[col].dropna().to_numpy(float)
        n = len(x)
        if n < 16:
            continue

        # Lag-1 AR setup
        y_all = x[1:]
        x_all = x[:-1]
        n_obs = len(y_all)

        # Pooled RSS
        try:
            X_all = np.column_stack([np.ones(n_obs), x_all])
            b_all, _, _, _ = np.linalg.lstsq(X_all, y_all, rcond=None)
            rss_pool = float(np.sum((y_all - X_all @ b_all) ** 2))
        except Exception:
            continue

        k = 2  # intercept + slope
        best_f = 0.0
        best_t = -1

        for t in range(5, n_obs - 5):
            y1, x1 = y_all[:t], x_all[:t]
            y2, x2 = y_all[t:], x_all[t:]
            if len(y1) < k + 1 or len(y2) < k + 1:
                continue
            try:
                X1 = np.column_stack([np.ones(len(y1)), x1])
                X2 = np.column_stack([np.ones(len(y2)), x2])
                b1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
                b2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
                rss1 = float(np.sum((y1 - X1 @ b1) ** 2))
                rss2 = float(np.sum((y2 - X2 @ b2) ** 2))
                denom = (rss1 + rss2) / max(n_obs - 2 * k, 1)
                if denom < 1e-12:
                    continue
                f = ((rss_pool - rss1 - rss2) / k) / denom
                if f > best_f:
                    best_f = f
                    best_t = t
            except Exception:
                continue

        if best_t < 0:
            continue

        f_crit = float(stats.f.ppf(1 - significance, k, n_obs - 2 * k))
        if best_f > f_crit:
            p = float(1 - stats.f.cdf(best_f, k, n_obs - 2 * k))
            results.append(_discovery(
                source=col, target=col,
                rel_type='structural',
                confidence=min(1.0, best_f / f_crit),
                sign=0,
                statistic=best_f, p_value=p,
                method=f'chow_break_at_t{best_t}',
            ))

    return results


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

def print_specialist_calibration_report(df: pd.DataFrame) -> None:
    """
    Print a calibration table showing discovery counts per specialist
    and per-type false-positive risk indicators.
    """
    n_rows = len(df.dropna())
    n_cols = len(df.columns)
    n_pairs = n_cols * (n_cols - 1)
    n_triples = n_cols * (n_cols - 1) * (n_cols - 2)
    print(f'\nSpecialist Calibration Report')
    print(f'  Dataset: {n_rows} complete rows, {n_cols} variables')
    print(f'  Search space: {n_pairs} ordered pairs, {n_triples} triples')
    print(f'  {"Specialist":15s} {"N_disc":>7s} {"Rate":>7s} {"Note"}')
    print(f'  {"-" * 60}')
    for name, fn in ALL_SPECIALISTS.items():
        disc = fn(df)  # type: ignore[call-arg]
        space = n_triples if name in ('mediating', 'synergistic') else n_pairs
        rate = len(disc) / max(space, 1)
        flag = ''
        if name == 'mediating' and len(disc) > 50:
            flag = ' WARN: high FP risk'
        elif name == 'synergistic' and len(disc) > 50:
            flag = ' WARN: high FP risk'
        elif name == 'functional' and len(disc) > 30:
            flag = ' WARN: check min_r2_gain'
        print(f'  {name:15s} {len(disc):7d} {rate:7.4f}{flag}')


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

ALL_SPECIALISTS: dict[str, object] = {
    'temporal':      discover_temporal,
    'causal':        discover_causal,
    'correlational': discover_correlational,
    'competitive':   discover_competitive,
    'compositional': discover_compositional,
    'equilibrium':   discover_equilibrium,
    'mediating':     discover_mediating,
    'synergistic':   discover_synergistic,
    'functional':    discover_functional,
    'structural':    discover_structural,
}


def run_all_specialists(df: pd.DataFrame,
                        verbose: bool = True) -> dict[str, list[dict]]:
    """Run all specialist baselines and return {type: [discoveries]}."""
    out: dict[str, list[dict]] = {}
    total = 0
    for name, fn in ALL_SPECIALISTS.items():
        disc = fn(df)  # type: ignore[call-arg]
        out[name] = disc
        total += len(disc)
        if verbose:
            print(f'  {name:15s}: {len(disc):4d} discoveries')
    if verbose:
        print(f'  {"---":15s}: {total:4d} total')
    return out


def run_specialists_n_sweep(
    df_full: pd.DataFrame,
    n_values: list[int] | None = None,
    verbose: bool = True,
) -> dict[int, dict[str, list[dict]]]:
    """
    Run all specialists at each N (truncating df_full to first N complete rows).

    Returns:
        {N -> {type -> [discoveries]}}
    """
    if n_values is None:
        n_values = [8, 12, 15, 20, 25, 30, 34]

    # Use rows with fewest NaN for truncation
    from scripts.experiments.data_loader import truncate_to_n

    results: dict[int, dict[str, list[dict]]] = {}
    for n in n_values:
        df_n = truncate_to_n(df_full, n)
        if verbose:
            print(f'\n  N={n} ({len(df_n)} rows available):')
        results[n] = run_all_specialists(df_n, verbose=verbose)
    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from scripts.experiments.data_loader import load_country_data

    print('Loading Kenya data...')
    df = load_country_data('KEN')
    # Use rows with complete data across most columns
    gt_cols = ['gdp_growth', 'inflation_cpi', 'unemployment', 'real_interest_rate',
               'private_credit', 'govt_consumption', 'exports_gdp', 'imports_gdp',
               'current_account', 'gcf', 'electricity_access', 'internet_users',
               'school_enrollment', 'life_expectancy', 'broad_money']
    df_work = df[gt_cols].dropna()
    print(f'Working DataFrame: {df_work.shape} (dropped NaN rows)')

    print('\nRunning all specialists on KEN data...')
    all_disc = run_all_specialists(df_work, verbose=True)

    print('\nTop 3 discoveries per type:')
    for name, disc in all_disc.items():
        disc_sorted = sorted(disc, key=lambda d: -d['confidence'])[:3]
        for d in disc_sorted:
            src, tgt = d['source'], d['target']
            if d.get('mediator'):
                pair = f'{src} -[{d["mediator"]}]-> {tgt}'
            else:
                pair = f'{src} ~ {tgt}'
            print(f'  {name:15s}: {pair:45s} conf={d["confidence"]:.3f} p={d["p_value"]:.3f}')

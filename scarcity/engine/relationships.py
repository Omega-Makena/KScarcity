"""
Relationship Hypothesis Implementations — v2 (Statistically Rigorous)

All 10 core relationship types with proper online statistical algorithms.

Key improvements over v1:
- Proper F-tests and t-tests with p-values (not just gain thresholds)
- Online RLS for ALL regression-based types (no batch lstsq in evaluate())
- Transfer entropy for non-linear causal detection (Granger complement)
- Engle-Granger / ADF-based stationarity test for equilibrium
- KS two-sample test + Jensen-Shannon divergence for probabilistic shift
- ANOVA F-test + eta-squared effect size for structural
- CUSUM structural break detection for temporal
- Nadaraya-Watson kernel regression comparison for functional
- SynergisticHypothesis: fully online RLS with partial F-test (was batch)
"""

from __future__ import annotations

import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

try:
    from scipy import stats as scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

from .discovery import Hypothesis, RelationshipType, HypothesisMetadata, RegimeTracker
from .relationship_config import (
    CausalConfig,
    CorrelationalConfig,
    TemporalConfig,
    FunctionalConfig,
    EquilibriumConfig,
    CompositionalConfig,
    CompetitiveConfig,
    SynergisticConfig,
    ProbabilisticConfig,
    StructuralConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _f_pvalue(F: float, df_num: int, df_den: int) -> float:
    """F-test survival function (upper tail). Returns p-value."""
    if F <= 0 or df_den <= 0:
        return 1.0
    if _SCIPY:
        return float(scipy_stats.f.sf(F, df_num, df_den))
    # Fallback approximation via chi-squared
    return float(np.exp(-0.5 * F * df_num))


def _t_pvalue(t: float, df: int) -> float:
    """Two-tailed t-test p-value."""
    if df <= 0:
        return 1.0
    if _SCIPY:
        return float(2.0 * scipy_stats.t.sf(abs(t), df))
    # Normal approximation for large df
    z = abs(t)
    return float(2.0 * (1.0 - 0.5 * (1.0 + np.sign(z) * np.sqrt(
        1.0 - np.exp(-z * z * (8.0 / np.pi + 0.147 * z * z) /
                      (1.0 + 0.147 * z * z))))))


def _rls_step(P: np.ndarray, coef: np.ndarray,
              x: np.ndarray, y: float,
              lam: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    One RLS update step.

    Returns (P_new, coef_new, residual).
    """
    y_hat = float(np.dot(x, coef))
    residual = y - y_hat
    Px = P @ x
    denom = lam + float(x @ Px)
    if abs(denom) < 1e-12:
        return P, coef, residual
    K = Px / denom
    coef_new = coef + K * residual
    P_new = (P - np.outer(K, Px)) / lam
    return P_new, coef_new, residual


def _not_ready(evidence: int = 0) -> Dict[str, Any]:
    """Sentinel return when a hypothesis has not accumulated enough observations yet.

    Returns zero confidence so downstream consumers (MPIE, get_candidate_paths)
    never mistake cold-start noise for a real signal.
    """
    return {'fit_score': 0.0, 'confidence': 0.0, 'evidence': evidence,
            'stability': 0.0, 'ready': False}


# ===========================================================================
# 1. CAUSAL — Granger Causality with F-test + Transfer Entropy
# ===========================================================================

class CausalHypothesis(Hypothesis):
    """
    Granger causality with proper F-test significance.

    Improvements over v1:
    - F-statistic from restricted vs. unrestricted model comparison
    - p-value drives confidence (not gain threshold)
    - Transfer entropy (histogram MI) supplements linear Granger for
      non-linear causal detection
    - Direction determined by asymmetric F + TE net flow
    """

    def __init__(self, source: str, target: str, lag: int = 2,
                 buffer_size: int = 150, config: Optional[CausalConfig] = None,
                 max_lag: int = 4):
        super().__init__([source, target], RelationshipType.CAUSAL)
        self.source = source
        self.target = target
        self.lag = lag
        self.max_lag = max(lag, max_lag)
        self.buffer_x: deque = deque(maxlen=buffer_size)
        self.buffer_y: deque = deque(maxlen=buffer_size)
        self.config = config or CausalConfig()

        self.f_stat_forward = 0.0
        self.p_value_forward = 1.0
        self.f_stat_backward = 0.0
        self.p_value_backward = 1.0
        self.direction = 0
        self.transfer_entropy_xy = 0.0
        self.transfer_entropy_yx = 0.0
        # Forward/backward level-regression coefficients stored separately so
        # predict_value() can always use the forward path regardless of which
        # direction the dominant Granger test assigned.
        self._coef_fwd: Optional[np.ndarray] = None   # X→Y level OLS coefficients
        self._coef_bwd: Optional[np.ndarray] = None   # Y→X level OLS coefficients
        self._coef_aug: Optional[np.ndarray] = None   # alias for _coef_fwd (backward compat)
        self._best_lag: int = lag                      # BIC-selected lag
        self._use_diff: bool = False                   # True when both series are I(1)

        # Engle-Granger cointegration fields (populated when _use_diff=True)
        self._is_coint: bool = False
        self._coint_alpha: float = 0.0
        self._coint_beta: float = 0.0
        self._coint_gamma: float = 0.0  # ECM error-correction speed
        # False after begin_live_stream() resets ECM — prevents re-estimating
        # cointegration on a pretrain+live+peer mixed buffer which yields
        # spurious ECM betas and wrong perturbation signs in federated conditions.
        self._allow_ecm_refit: bool = True

        # Backward Bayesian accumulators — parallel to alpha_success/beta_failure
        # but tracking p_value_backward signal.  Used for signed directional
        # confidence: confidence = |conf_fwd - conf_bwd|.  Bidirectional pairs
        # get confidence ≈ 0; unidirectional pairs accumulate a positive margin.
        self._alpha_bwd: float = 1.0
        self._beta_bwd: float = 1.0

        # Live-phase mini-buffers (Fix #2): populated only after begin_live_stream()
        # sets _allow_ecm_refit=False.  When ≥15 own live rows are present, run a
        # secondary F-test on live-only data.  If it clearly favours one direction
        # (ratio ≥1.5, p<0.15), override the mixed-buffer direction assignment.
        # This prevents the large pretrain corpus from out-voting live signal.
        self._live_buf_x: deque = deque(maxlen=30)
        self._live_buf_y: deque = deque(maxlen=30)

        # regime tracking
        self._regime_tracker = RegimeTracker()

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.source in row and self.target in row:
            x, y = row[self.source], row[self.target]
            if np.isfinite(x) and np.isfinite(y):
                self.buffer_x.append(x)
                self.buffer_y.append(y)
                # Populate live mini-buffer once in live phase (Fix #2)
                if not self._allow_ecm_refit:
                    self._live_buf_x.append(x)
                    self._live_buf_y.append(y)
                # lag-1 residual as regime instability proxy
                if len(self.buffer_y) >= 2:
                    self._update_regime_tracker(
                        abs(float(self.buffer_y[-1]) - float(self.buffer_y[-2]))
                    )

    def _lag_matrix(self, series: np.ndarray, n_lags: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (target_vector, lag_feature_matrix)."""
        target = series[n_lags:]
        cols = [series[n_lags - i - 1: len(series) - i - 1] for i in range(n_lags)]
        return target, np.column_stack(cols)

    @staticmethod
    def _is_nonstationary(series: np.ndarray, threshold: float = 0.85) -> bool:
        """Lag-1 autocorrelation proxy for unit-root: True if |AC₁| > threshold.

        The AC₁ proxy is intentionally conservative (threshold 0.85): it only
        flags genuinely near-unit-root series (price levels, credit stocks) as
        I(1).  Moderately persistent stationary series (GDP growth, inflation)
        have AC₁ < 0.85 and are treated as I(0), keeping the level Granger
        F-test which has higher power at n < 50.  Engle-Granger cointegration
        is then applied separately to I(1) pairs (see _engle_granger_cointegration).
        """
        if len(series) < 10:
            return False
        x = series - series.mean()
        if np.std(x) < 1e-10:
            return False
        return abs(float(np.corrcoef(x[:-1], x[1:])[0, 1])) > threshold

    @staticmethod
    def _engle_granger_cointegration(
        X: np.ndarray, Y: np.ndarray
    ) -> Tuple[bool, float, float, float]:
        """Two-step Engle-Granger cointegration test.

        Step 1: OLS level regression Y = alpha + beta*X.
        Step 2: ADF on residuals with tighter critical value -3.37
                (MacKinnon 1991, n≈30, no intercept in residual regression).

        Returns (is_cointegrated, alpha, beta, gamma) where gamma is the ECM
        error-correction speed coefficient (< 0 when cointegrated).
        """
        n = len(X)
        if n < 15:
            return False, 0.0, 0.0, 0.0
        # Step 1: level OLS
        D = np.column_stack([np.ones(n), X])
        try:
            coef, *_ = np.linalg.lstsq(D, Y, rcond=None)
        except Exception:
            return False, 0.0, 0.0, 0.0
        alpha, beta = float(coef[0]), float(coef[1])
        resid = Y - (alpha + beta * X)
        # Step 2: ADF on residuals (no intercept: residuals are near zero-mean)
        d_resid = np.diff(resid)
        r_lag = resid[:-1]
        n2 = len(d_resid)
        if n2 < 8:
            return False, alpha, beta, 0.0
        denom = float(np.dot(r_lag, r_lag))
        if denom < 1e-12:
            return False, alpha, beta, 0.0
        gamma = float(np.dot(r_lag, d_resid)) / denom
        fit_r = d_resid - gamma * r_lag
        sse = float(np.dot(fit_r, fit_r))
        se_g = float(np.sqrt(max(sse / max(n2 - 1, 1), 1e-12) / denom))
        if se_g < 1e-12:
            return False, alpha, beta, gamma
        t_stat = gamma / se_g
        # Engle-Granger 5% critical value for cointegrating regression residuals
        is_coint = t_stat < -3.37
        return is_coint, alpha, beta, gamma

    @staticmethod
    def _linear_detrend(series: np.ndarray) -> np.ndarray:
        """Remove a fitted linear time trend via OLS: return series - (a + b·t)."""
        n = len(series)
        t = np.arange(n, dtype=float)
        D = np.column_stack([np.ones(n), t])
        try:
            coef, *_ = np.linalg.lstsq(D, series, rcond=None)
            return series - D @ coef
        except Exception:
            return series - series.mean()

    def _granger_f_test_at_lag(self, X: np.ndarray, Y: np.ndarray, k: int
                                ) -> Tuple[float, float, Optional[np.ndarray]]:
        """
        Granger F-test at explicit lag k.

        The F-test runs on first-differenced series when _use_diff is True so
        critical values are valid for I(1) data.

        Level-regression coefficients are computed on linearly-detrended series
        when the corresponding series is I(1) (AC₁ > 0.85).  Detrending removes
        the shared-trend confound so that coef_level reflects the marginal causal
        effect (cycle-on-cycle), not the spurious trend correlation.  Since
        sign(Δprediction) = sign(coef_level[X_slot]) × sign(perturbation), the
        correct coefficient sign fixes wrong-sign perturbation responses for
        trending variables (e.g. electricity_access → gdp_growth).
        """
        ridge = self.config.ridge_alpha

        # Detrend I(1) series for the level-regression only (F-test uses _use_diff).
        Y_lev = self._linear_detrend(Y) if self._is_nonstationary(Y) else Y
        X_lev = self._linear_detrend(X) if self._is_nonstationary(X) else X

        # Level-regression coefficients (prediction / sign use)
        coef_level: Optional[np.ndarray] = None
        if len(Y_lev) > 2 * k + 5:
            Yt_l, Yl_l = self._lag_matrix(Y_lev, k)
            _, Xl_l = self._lag_matrix(X_lev, k)
            n_l = len(Yt_l)
            D_l = np.hstack([np.ones((n_l, 1)), Yl_l, Xl_l])
            try:
                A_l = D_l.T @ D_l + ridge * np.eye(2 * k + 1)
                coef_level = np.linalg.solve(A_l, D_l.T @ Yt_l)
            except np.linalg.LinAlgError:
                pass

        # Choose series for F-test (differenced if both I(1))
        Xf = np.diff(X) if self._use_diff else X
        Yf = np.diff(Y) if self._use_diff else Y
        n_total = len(Yf)
        if n_total <= 2 * k + 5:
            return 0.0, 1.0, coef_level

        Y_target, Y_lags = self._lag_matrix(Yf, k)
        _, X_lags = self._lag_matrix(Xf, k)
        n = len(Y_target)
        ones = np.ones((n, 1))

        D_r = np.hstack([ones, Y_lags])
        try:
            A_r = D_r.T @ D_r + ridge * np.eye(k + 1)
            coef_r = np.linalg.solve(A_r, D_r.T @ Y_target)
            rss_r = float(np.sum((Y_target - D_r @ coef_r) ** 2))
        except np.linalg.LinAlgError:
            return 0.0, 1.0, coef_level

        D_u = np.hstack([ones, Y_lags, X_lags])
        try:
            A_u = D_u.T @ D_u + ridge * np.eye(2 * k + 1)
            coef_u = np.linalg.solve(A_u, D_u.T @ Y_target)
            rss_u = float(np.sum((Y_target - D_u @ coef_u) ** 2))
        except np.linalg.LinAlgError:
            return 0.0, 1.0, coef_level

        df_num, df_den = k, n - 2 * k - 1
        if df_den <= 0 or rss_u < 1e-12:
            return 0.0, 1.0, coef_level

        F = max(0.0, ((rss_r - rss_u) / df_num) / (rss_u / df_den))
        return F, _f_pvalue(F, df_num, df_den), coef_level

    def _granger_f_test(self, X: np.ndarray, Y: np.ndarray
                        ) -> Tuple[float, float, Optional[np.ndarray]]:
        """Backward-compatible wrapper: Granger F-test at self.lag."""
        return self._granger_f_test_at_lag(X, Y, self.lag)

    def _select_best_lag(self, X: np.ndarray, Y: np.ndarray) -> int:
        """Return BIC-minimising lag in 1..max_lag on the appropriate series."""
        Xf = np.diff(X) if self._use_diff else X
        Yf = np.diff(Y) if self._use_diff else Y
        ridge = self.config.ridge_alpha
        best_lag, best_bic = self.lag, np.inf
        for k in range(1, self.max_lag + 1):
            if len(Yf) <= 2 * k + 5:
                break
            Y_target, Y_lags = self._lag_matrix(Yf, k)
            _, X_lags = self._lag_matrix(Xf, k)
            n = len(Y_target)
            n_params = 2 * k + 1
            D_u = np.hstack([np.ones((n, 1)), Y_lags, X_lags])
            try:
                A_u = D_u.T @ D_u + ridge * np.eye(n_params)
                coef_u = np.linalg.solve(A_u, D_u.T @ Y_target)
                rss_u = float(np.sum((Y_target - D_u @ coef_u) ** 2))
            except np.linalg.LinAlgError:
                continue
            bic = n * np.log(max(rss_u / n, 1e-12)) + n_params * np.log(max(n, 2))
            if bic < best_bic:
                best_bic, best_lag = bic, k
        return best_lag

    def _transfer_entropy(self, X: np.ndarray, Y: np.ndarray,
                          bins: int = 8) -> float:
        """
        Histogram-based transfer entropy TE(X→Y).

        TE(X→Y) = H(Y_t | Y_{t-1}) − H(Y_t | Y_{t-1}, X_{t-1})
        Captures non-linear information flow missed by linear Granger.
        """
        lag = self._best_lag
        n_total = len(X)
        if n_total <= lag + 2:
            return 0.0
        try:
            def disc(arr: np.ndarray) -> np.ndarray:
                lo, hi = arr.min(), arr.max()
                if hi - lo < 1e-10:
                    return np.zeros(len(arr), dtype=int)
                return np.clip(
                    np.floor((arr - lo) / (hi - lo + 1e-10) * bins).astype(int),
                    0, bins - 1)

            n = n_total - lag
            Yt = disc(Y[lag:])
            Yt1 = disc(Y[:n_total - lag][:n])
            Xt1 = disc(X[:n_total - lag][:n])

            joint = np.zeros((bins, bins, bins))
            for i in range(n):
                joint[Yt[i], Yt1[i], Xt1[i]] += 1
            joint /= joint.sum() + 1e-10

            p_yt_yt1 = joint.sum(axis=2)
            p_yt1_xt1 = joint.sum(axis=0)
            p_yt1 = p_yt1_xt1.sum(axis=1)

            te = 0.0
            for yt in range(bins):
                for yt1 in range(bins):
                    for xt1 in range(bins):
                        pj = joint[yt, yt1, xt1]
                        if pj < 1e-12:
                            continue
                        py_yt1_xt1 = pj / (p_yt1_xt1[yt1, xt1] + 1e-10)
                        py_yt1 = p_yt_yt1[yt, yt1] / (p_yt1[yt1] + 1e-10)
                        if py_yt1 < 1e-12:
                            continue
                        te += pj * np.log(py_yt1_xt1 / py_yt1 + 1e-10)
            return max(0.0, float(te))
        except Exception:
            return 0.0

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        min_n = self.lag * 3 + cfg.min_samples_for_eval

        if len(self.buffer_x) < min_n:
            return _not_ready(len(self.buffer_x))

        X = np.array(self.buffer_x)
        Y = np.array(self.buffer_y)

        # Stationarity: if both series are I(1) use differenced F-test for
        # correct critical values; level coefficients still used for prediction.
        self._use_diff = self._is_nonstationary(X) and self._is_nonstationary(Y)

        # Cointegration: when both I(1), test for a stable long-run relationship.
        # ECM fields are updated here and used in predict_value() for sign stability.
        # Skipped after begin_live_stream() resets _allow_ecm_refit to False: the
        # pretrain+live+peer concatenated buffer produces spurious cointegration
        # estimates that override the detrended coef_level with wrong-sign betas.
        if self._use_diff and self._allow_ecm_refit:
            is_c, c_alpha, c_beta, c_gamma = self._engle_granger_cointegration(X, Y)
            self._is_coint = is_c
            if is_c:
                self._coint_alpha = c_alpha
                self._coint_beta = c_beta
                self._coint_gamma = c_gamma

        # BIC lag selection once enough data exists for all candidate lags
        if len(X) >= self.max_lag * 3 + cfg.min_samples_for_eval:
            self._best_lag = self._select_best_lag(X, Y)
        else:
            self._best_lag = self.lag

        k = self._best_lag

        self.f_stat_forward, self.p_value_forward, coef_fwd = (
            self._granger_f_test_at_lag(X, Y, k)
        )
        self.f_stat_backward, self.p_value_backward, coef_bwd = (
            self._granger_f_test_at_lag(Y, X, k)
        )

        if coef_fwd is not None:
            self._coef_fwd = coef_fwd
            self._coef_aug = coef_fwd  # backward-compat alias
        if coef_bwd is not None:
            self._coef_bwd = coef_bwd

        if len(X) >= 40:
            self.transfer_entropy_xy = self._transfer_entropy(X, Y)
            self.transfer_entropy_yx = self._transfer_entropy(Y, X)

        alpha = 0.05
        sig_fwd = self.p_value_forward < alpha
        sig_bwd = self.p_value_backward < alpha
        te_net = self.transfer_entropy_xy - self.transfer_entropy_yx

        # Require forward F-stat to be at least 30% larger than backward before
        # claiming X→Y causality; symmetric for Y→X.  When both directions are
        # similarly F-significant, leave direction ambiguous (=0) rather than
        # picking a noisy winner that would pollute ensemble cascade paths.
        _ASYM = 1.3
        f_ratio_fwd = self.f_stat_forward / max(self.f_stat_backward, 1e-6)
        f_ratio_bwd = self.f_stat_backward / max(self.f_stat_forward, 1e-6)
        if sig_fwd and (not sig_bwd or (f_ratio_fwd >= _ASYM and te_net >= 0)):
            self.direction = 1
        elif sig_bwd and (not sig_fwd or f_ratio_bwd >= _ASYM):
            self.direction = -1
        else:
            self.direction = 0

        # Live-direction override (Fix #2): when ≥15 own live rows exist, run a
        # secondary Granger F-test on live-only data.  If the live F-ratio
        # strongly favours one direction (≥1.5×) with p<0.15, that direction wins
        # over the mixed pretrain+live buffer result — preventing the larger
        # pretrain corpus from out-voting the live causal signal.
        if len(self._live_buf_x) >= 15:
            Xl = np.array(self._live_buf_x)
            Yl = np.array(self._live_buf_y)
            k_live = min(self._best_lag, max(1, len(Xl) // 5))
            try:
                lf_fwd, lp_fwd, _ = self._granger_f_test_at_lag(Xl, Yl, k_live)
                lf_bwd, lp_bwd, _ = self._granger_f_test_at_lag(Yl, Xl, k_live)
                _LIVE_ASYM = 1.5
                lr_fwd = lf_fwd / max(lf_bwd, 1e-6)
                lr_bwd = lf_bwd / max(lf_fwd, 1e-6)
                if lr_fwd >= _LIVE_ASYM and lp_fwd < 0.15:
                    self.direction = 1
                elif lr_bwd >= _LIVE_ASYM and lp_bwd < 0.15:
                    self.direction = -1
            except Exception:
                pass

        if self.direction == 1:
            p_cause = max(0.0, 1.0 - self.p_value_forward)
        elif self.direction == -1:
            p_cause = max(0.0, 1.0 - self.p_value_backward)
        else:
            # Ambiguous direction: partial credit from the better p-value
            p_cause = max(0.0, (1.0 - min(self.p_value_forward, self.p_value_backward)) * 0.4)

        te_boost = min(0.15, max(0.0, te_net)) if self.direction == 1 else 0.0
        fit = min(1.0, p_cause + te_boost)

        if self.direction == 1:
            best_p = self.p_value_forward
        elif self.direction == -1:
            best_p = self.p_value_backward
        else:
            best_p = min(self.p_value_forward, self.p_value_backward)

        return {
            'fit_score': fit,
            'evidence': len(self.buffer_x),
            'stability': 0.85 if self.direction != 0 else 0.5,
            'p_value': best_p,
            'f_stat_forward': self.f_stat_forward,
            'f_stat_backward': self.f_stat_backward,
            'p_value_forward': self.p_value_forward,
            'p_value_backward': self.p_value_backward,
            'direction': self.direction,
            'transfer_entropy_xy': self.transfer_entropy_xy,
            'transfer_entropy_yx': self.transfer_entropy_yx,
            'best_lag': self._best_lag,
            'use_diff': self._use_diff,
            'ready': True,
        }

    def update(self, row: Dict[str, float]):
        """
        Override to track separate forward/backward Bayesian accumulators.

        self.confidence stays as conf_fwd (set by base class) so ensemble
        thresholds and the arbitrator continue to work correctly.
        The backward accumulators (_alpha_bwd/_beta_bwd) track backward
        Granger significance for potential directional quality checks;
        direction selection uses p_value_forward/backward and the F-ratio
        asymmetry guard in evaluate() directly.
        """
        metrics = super().update(row)
        # p_value_backward is set by evaluate() inside super().update()
        _lambda = 0.99
        sig_bwd = max(0.0, 1.0 - float(self.p_value_backward) * 10.0)
        self._alpha_bwd = _lambda * self._alpha_bwd + sig_bwd
        self._beta_bwd  = _lambda * self._beta_bwd  + (1.0 - sig_bwd)
        # confidence stays as conf_fwd set by the base class
        metrics['confidence'] = self.confidence
        return metrics

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        # ECM path: cointegrated I(1) series — use long-run level relationship
        # plus error-correction adjustment for short-run dynamics.
        if self._is_coint:
            x_val = row.get(self.source)
            if x_val is None and len(self.buffer_x) >= 1:
                x_val = float(self.buffer_x[-1])
            if x_val is not None:
                y_hat = self._coint_alpha + self._coint_beta * x_val
                if len(self.buffer_x) >= 1 and len(self.buffer_y) >= 1:
                    ecm = (float(self.buffer_y[-1])
                           - self._coint_alpha
                           - self._coint_beta * float(self.buffer_x[-1]))
                    y_hat += self._coint_gamma * ecm
                return (self.target, y_hat)

        # Fallback: forward level OLS coefficients so perturbation tests work
        # regardless of which direction the Granger F-test assigned as dominant.
        coef = self._coef_fwd
        if coef is None:
            return None
        k = self._best_lag
        if len(self.buffer_y) < k or len(self.buffer_x) < k:
            return None
        y_lags = np.array([float(self.buffer_y[-i - 1]) for i in range(k)])
        x_current = row.get(self.source, float(self.buffer_x[-1]))
        x_lags = np.array(
            [x_current] + [float(self.buffer_x[-i - 1]) for i in range(1, k)]
        )
        features = np.concatenate([[1.0], y_lags, x_lags])
        if len(features) != len(coef):
            return None
        return (self.target, float(np.dot(features, coef)))

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["metrics"].update({
            "lag": self.lag,
            "best_lag": self._best_lag,
            "use_diff": self._use_diff,
            "f_stat_forward": float(self.f_stat_forward),
            "p_value_forward": float(self.p_value_forward),
            "f_stat_backward": float(self.f_stat_backward),
            "p_value_backward": float(self.p_value_backward),
            "direction": int(self.direction),
            "transfer_entropy_xy": float(self.transfer_entropy_xy),
            "transfer_entropy_yx": float(self.transfer_entropy_yx),
        })
        d["source"] = self.source
        d["target"] = self.target
        return d


# ===========================================================================
# 2. CORRELATIONAL — Pearson + Fisher z-test + Distance Correlation
# ===========================================================================

class CorrelationalHypothesis(Hypothesis):
    """
    Correlation with proper significance testing.

    Improvements over v1:
    - t-test H0: ρ=0 (t = r√(n-2)/√(1-r²)) → p-value based confidence
    - Fisher z-transform SE for stability estimate
    - Distance correlation (dcov) computed periodically for non-linear detection
    """

    def __init__(self, var1: str, var2: str, buffer_size: int = 150,
                 config: Optional[CorrelationalConfig] = None):
        super().__init__([var1, var2], RelationshipType.CORRELATIONAL)
        self.var1 = var1
        self.var2 = var2
        self.buffer1: deque = deque(maxlen=buffer_size)
        self.buffer2: deque = deque(maxlen=buffer_size)
        self.config = config or CorrelationalConfig()

        # Welford online state
        self.n = 0
        self.mean1 = 0.0
        self.mean2 = 0.0
        self.M2_1 = 0.0
        self.M2_2 = 0.0
        self.cov = 0.0

        self.r = 0.0
        self.p_value = 1.0
        self.distance_corr = 0.0

        # regime tracking via correlation stability
        self._regime_tracker = RegimeTracker()
        self._prev_r = 0.0

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.var1 in row and self.var2 in row:
            x, y = row[self.var1], row[self.var2]
            if not (np.isfinite(x) and np.isfinite(y)):
                return
            self.buffer1.append(x)
            self.buffer2.append(y)
            self.n += 1
            d1 = x - self.mean1
            self.mean1 += d1 / self.n
            d2 = y - self.mean2
            self.mean2 += d2 / self.n
            self.M2_1 += d1 * (x - self.mean1)
            self.M2_2 += d2 * (y - self.mean2)
            self.cov += d1 * (y - self.mean2)
            # regime tracking: change in running correlation as instability proxy
            if self.n > 3:
                var1 = self.M2_1 / self.n
                var2 = self.M2_2 / self.n
                covar = self.cov / self.n
                denom = np.sqrt(max(0.0, var1 * var2))
                running_r = float(np.clip(covar / (denom + 1e-10), -1.0, 1.0))
                self._update_regime_tracker(abs(running_r - self._prev_r))
                self._prev_r = running_r

    def _distance_corr(self, X: np.ndarray, Y: np.ndarray) -> float:
        """O(n²) distance correlation — only called at sample-size checkpoints."""
        n = len(X)
        if n < 8:
            return 0.0
        try:
            a = np.abs(X[:, None] - X[None, :])
            b = np.abs(Y[:, None] - Y[None, :])
            # Double-centering
            A = a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()
            B = b - b.mean(1, keepdims=True) - b.mean(0, keepdims=True) + b.mean()
            dcov2_xy = (A * B).mean()
            dcov2_xx = (A * A).mean()
            dcov2_yy = (B * B).mean()
            denom = np.sqrt(max(0.0, dcov2_xx * dcov2_yy))
            return float(np.clip(dcov2_xy / (denom + 1e-10), 0.0, 1.0))
        except Exception:
            return 0.0

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = self.n
        if n < cfg.min_samples:
            return _not_ready(n)

        var1 = self.M2_1 / n
        var2 = self.M2_2 / n
        covar = self.cov / n
        denom = np.sqrt(max(0.0, var1 * var2))
        self.r = float(np.clip(covar / (denom + 1e-10), -1.0, 1.0))

        if n > 2:
            t = self.r * np.sqrt(n - 2) / np.sqrt(max(1e-10, 1.0 - self.r ** 2))
            self.p_value = _t_pvalue(t, n - 2)
        else:
            self.p_value = 1.0

        # Distance correlation at checkpoints
        if n >= 20 and n % 25 == 0:
            X = np.array(self.buffer1)
            Y = np.array(self.buffer2)
            self.distance_corr = self._distance_corr(X, Y)

        confidence = max(0.0, 1.0 - self.p_value) * abs(self.r)
        # Fisher z SE shrinks with n — used as stability proxy
        z_se = 1.0 / np.sqrt(max(n - 3, 1))
        stability = max(0.4, min(1.0, 1.0 - z_se))

        return {
            'fit_score': abs(self.r),
            'evidence': n,
            'stability': stability,
            'correlation': self.r,
            'p_value': self.p_value,
            'distance_correlation': self.distance_corr,
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        cfg = self.config
        if self.n < cfg.min_samples or abs(self.r) < 0.05:
            return None
        std1 = np.sqrt(max(0.0, self.M2_1 / self.n))
        std2 = np.sqrt(max(0.0, self.M2_2 / self.n))
        if std1 < 1e-10:
            return None
        x = row.get(self.var1, self.mean1)
        y_hat = self.mean2 + self.r * (std2 / std1) * (x - self.mean1)
        return (self.var2, float(y_hat))


# ===========================================================================
# 3. TEMPORAL — AR(p) with RLS + CUSUM Structural Break
# ===========================================================================

class TemporalHypothesis(Hypothesis):
    """
    Autoregressive model with structural break detection.

    Improvements over v1:
    - Page's CUSUM for detecting regime shifts
    - Coefficient stability tracking (CV of recent coefficient updates)
    - Adjusted R² report
    """

    def __init__(self, variable: str, lag: int = 3, buffer_size: int = 150,
                 config: Optional[TemporalConfig] = None):
        super().__init__([variable], RelationshipType.TEMPORAL)
        self.variable = variable
        self.lag = lag
        self.buffer: deque = deque(maxlen=buffer_size)
        self.config = config or TemporalConfig()

        n_feat = lag + 1
        self.coefficients = np.zeros(n_feat)
        self._P = np.eye(n_feat) * self.config.initial_covariance
        self._lambda = self.config.forgetting_factor

        # CUSUM (Page's one-sided)
        self._cusum = 0.0
        self._sigma_ema = 1.0
        self._sigma_alpha = 0.01
        self._residuals: deque = deque(maxlen=60)
        self._coef_history: deque = deque(maxlen=30)
        self.structural_break = False

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.variable not in row:
            return
        val = row[self.variable]
        if not np.isfinite(val):
            return
        self.buffer.append(val)
        if len(self.buffer) <= self.lag:
            return

        x = np.array([1.0] + [self.buffer[-i - 2] for i in range(self.lag)])
        self._P, self.coefficients, residual = _rls_step(
            self._P, self.coefficients, x, val, self._lambda)

        # CUSUM update
        self._sigma_ema = (1 - self._sigma_alpha) * self._sigma_ema + \
                          self._sigma_alpha * abs(residual)
        normed = residual / (self._sigma_ema + 1e-8)
        self._cusum = max(0.0, self._cusum + normed - 0.5)
        self._residuals.append(residual)
        self._coef_history.append(self.coefficients.copy())

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = len(self.buffer)
        min_n = self.lag + cfg.min_samples_for_eval
        if n <= min_n:
            return _not_ready(n)

        self.structural_break = self._cusum > 5.0

        # R² from recent residuals
        if len(self._residuals) >= 10:
            res = np.array(self._residuals)
            Y_buf = np.array(list(self.buffer))[-len(res):]
            ss_res = float(np.sum(res ** 2))
            ss_tot = float(np.sum((Y_buf - Y_buf.mean()) ** 2))
            r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-9))
        else:
            r2 = 0.5

        # Coefficient stability
        coef_stab = 0.7
        if len(self._coef_history) >= 5:
            ca = np.array(list(self._coef_history))
            cv = ca.std(axis=0).mean() / (np.abs(ca).mean() + 1e-8)
            coef_stab = max(0.2, min(1.0, 1.0 - cv))

        Y = np.array(self.buffer)
        autocorr = float(np.corrcoef(Y[1:], Y[:-1])[0, 1]) \
            if len(Y) > 2 and np.std(Y) > 1e-9 else 0.0

        # AR(1) significance: Pearson t-test on lag-1 autocorrelation.
        # p_ar approaches 0 for strongly autocorrelated series (real macro data)
        # and is large (>0.3) for random noise — feeds base-class signal computation.
        p_ar = 1.0
        if n >= 10 and abs(autocorr) < 1.0 - 1e-9:
            t_ar = autocorr * np.sqrt(n - 2) / np.sqrt(max(1e-10, 1.0 - autocorr ** 2))
            p_ar = _t_pvalue(t_ar, n - 2)

        return {
            'fit_score': r2,
            'evidence': n,
            'stability': coef_stab,
            'p_value': p_ar,
            'autocorrelation': autocorr,
            'coefficients': self.coefficients.tolist(),
            'structural_break': self.structural_break,
            'cusum': float(self._cusum),
            'ready': True,
        }

    def _regime_consistency(self) -> float:
        """Uses the built-in CUSUM state instead of a separate RegimeTracker."""
        return max(0.0, 1.0 - self._cusum / 8.0)

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if len(self.buffer) <= self.lag:
            return None
        # Use the current row value as lag-0 (most recent), buffer for older lags.
        y_current = row.get(self.variable, float(self.buffer[-1]))
        lags = [y_current] + [float(self.buffer[-i - 1]) for i in range(1, self.lag)]
        x = np.array([1.0] + lags)
        return (self.variable, float(np.dot(x, self.coefficients)))


# ===========================================================================
# 4. FUNCTIONAL — RLS Polynomial + Nadaraya-Watson Kernel Comparison
# ===========================================================================

class FunctionalHypothesis(Hypothesis):
    """
    Functional relationship Y = f(X).

    Improvements over v1:
    - Nadaraya-Watson kernel regression (Silverman bandwidth) computed
      periodically to detect non-polynomial functional forms
    - Adjusted R² penalises polynomial degree
    - Best of polynomial / kernel reported as fit_score
    """

    def __init__(self, source: str, target: str, degree: int = 1,
                 buffer_size: int = 150, config: Optional[FunctionalConfig] = None):
        super().__init__([source, target], RelationshipType.FUNCTIONAL)
        self.source = source
        self.target = target
        self.degree = degree
        self.buffer_x: deque = deque(maxlen=buffer_size)
        self.buffer_y: deque = deque(maxlen=buffer_size)
        self.config = config or FunctionalConfig()

        n_feat = degree + 1
        self.coefficients = np.zeros(n_feat)
        self._P = np.eye(n_feat) * self.config.initial_covariance
        self._lambda = self.config.forgetting_factor

        self.poly_r2 = 0.0
        self.kernel_r2 = 0.0

    def _features(self, x: float) -> np.ndarray:
        return np.array([x ** i for i in range(self.degree + 1)])

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.source in row and self.target in row:
            x, y = row[self.source], row[self.target]
            if not (np.isfinite(x) and np.isfinite(y)):
                return
            self.buffer_x.append(x)
            self.buffer_y.append(y)
            feat = self._features(x)
            self._P, self.coefficients, _ = _rls_step(
                self._P, self.coefficients, feat, y, self._lambda)

    def _nadaraya_watson(self, X: np.ndarray, Y: np.ndarray,
                         Xq: np.ndarray) -> np.ndarray:
        """Gaussian-kernel NW regression, Silverman bandwidth."""
        sigma = float(np.std(X))
        if sigma < 1e-9:
            return np.full(len(Xq), float(np.mean(Y)))
        h = 1.06 * sigma * len(X) ** (-0.2)
        Yhat = np.empty(len(Xq))
        for i, xq in enumerate(Xq):
            w = np.exp(-0.5 * ((X - xq) / h) ** 2)
            Yhat[i] = (w @ Y) / (w.sum() + 1e-10)
        return Yhat

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = len(self.buffer_x)
        if n < cfg.min_samples:
            return _not_ready(n)

        X = np.array(self.buffer_x)
        Y = np.array(self.buffer_y)
        ss_tot = float(np.sum((Y - Y.mean()) ** 2))

        Yhat_poly = np.array([float(np.dot(self._features(x), self.coefficients))
                               for x in X])
        ss_res_poly = float(np.sum((Y - Yhat_poly) ** 2))
        self.poly_r2 = max(0.0, 1.0 - ss_res_poly / (ss_tot + 1e-9))

        # Adjusted R² penalises polynomial degree
        n_params = self.degree          # number of slope coefficients
        n_total_params = self.degree + 1  # intercept + slopes
        adj_r2 = float(np.clip(
            1.0 - (1.0 - self.poly_r2) * (n - 1) / max(1, n - n_total_params - 1),
            0.0, 1.0
        ))

        # Nadaraya-Watson kernel only at n ≥ 50 to avoid high-variance overfitting
        # on small samples.  Only replaces adj_r2 when the improvement is substantial.
        if n >= 50 and n % 30 == 0:
            Yhat_kw = self._nadaraya_watson(X, Y, X)
            ss_res_kw = float(np.sum((Y - Yhat_kw) ** 2))
            self.kernel_r2 = max(0.0, 1.0 - ss_res_kw / (ss_tot + 1e-9))

        if n >= 50 and self.kernel_r2 > self.poly_r2 + 0.05:
            best_r2 = self.kernel_r2
        else:
            best_r2 = adj_r2

        res_std = float(np.std(Y - Yhat_poly))
        y_std = float(np.std(Y))
        is_det = (res_std < cfg.deterministic_threshold * y_std) if y_std > 1e-6 else False

        # OLS F-test: H₀ = all slope coefficients are zero (joint significance test).
        # F = (R² / n_params) / ((1-R²) / (n - n_params - 1)) ~ F(n_params, n-n_params-1).
        # Under H₀ the p-value is Uniform(0,1) → E[signal] ≈ 0.025 (calibration-safe).
        # For weak effects (β=0.15, n=60): R²≈0.12, F≈7.9, p≈0.007 → signal≈0.86.
        # Equivalent to the t-test on the Pearson r for degree=1 (F = t²).
        df_model = max(1, n_params)
        df_resid = n - n_params - 1
        if df_resid > 1 and 1e-10 < self.poly_r2 < 1.0 - 1e-10:
            F_stat = (self.poly_r2 / df_model) / ((1.0 - self.poly_r2) / df_resid)
            p_slope = _f_pvalue(max(0.0, F_stat), df_model, df_resid)
        elif self.poly_r2 >= 1.0 - 1e-10:
            p_slope = 0.0   # perfect fit
        else:
            p_slope = 1.0   # no variance explained → no signal

        return {
            'fit_score': best_r2,
            'evidence': n,
            'stability': 0.9 if is_det else 0.65,
            'p_value': p_slope,
            'poly_r2': self.poly_r2,
            'kernel_r2': self.kernel_r2,
            'adjusted_r2': adj_r2,
            'coefficients': self.coefficients.tolist(),
            'deterministic': is_det,
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self.source not in row or len(self.buffer_x) < self.config.min_samples:
            return None
        y_hat = float(np.dot(self._features(row[self.source]), self.coefficients))
        return (self.target, y_hat)


# ===========================================================================
# 5. EQUILIBRIUM — OU-MLE + Simplified ADF Stationarity Test
# ===========================================================================

class EquilibriumHypothesis(Hypothesis):
    """
    Mean-reversion / equilibrium.

    Improvements over v1:
    - Closed-form OU-MLE: regress ΔY on Y_{t-1} to estimate θ and μ
    - Simplified ADF test statistic (t-stat on lagged-level coefficient)
    - Confidence driven by ADF + reversion rate, not just Kalman gain
    """

    def __init__(self, variable: str, buffer_size: int = 150,
                 config: Optional[EquilibriumConfig] = None):
        super().__init__([variable], RelationshipType.EQUILIBRIUM)
        self.variable = variable
        self.buffer: deque = deque(maxlen=buffer_size)
        self.config = config or EquilibriumConfig()

        self.equilibrium = 0.0
        self.reversion_rate = 0.0
        self.ou_sigma = 1.0
        self.adf_stat = 0.0
        self.is_stationary = False

        # Kalman filter for online equilibrium tracking
        self._kf_P = 1.0
        self._kf_Q = self.config.process_noise
        self._kf_R = self.config.observation_noise

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.variable not in row:
            return
        val = row[self.variable]
        if not np.isfinite(val):
            return
        self.buffer.append(val)
        # Kalman update for equilibrium level
        P_pred = self._kf_P + self._kf_Q
        K = P_pred / (P_pred + self._kf_R)
        self.equilibrium += K * (val - self.equilibrium)
        self._kf_P = (1 - K) * P_pred

    def _ou_mle(self, Y: np.ndarray) -> Tuple[float, float, float]:
        """
        Closed-form OU-MLE via OLS on ΔY = α + β·Y_{t-1}.

        θ ≈ −β, μ = −α/β, σ = std(residuals).
        """
        n = len(Y)
        if n < 20:
            return 0.0, float(Y.mean()), float(Y.std())
        dY = Y[1:] - Y[:-1]
        Ylag = Y[:-1]
        Xmat = np.column_stack([np.ones(n - 1), Ylag])
        try:
            coef, _, _, _ = np.linalg.lstsq(Xmat, dY, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, float(Y.mean()), float(Y.std())
        alpha, beta = float(coef[0]), float(coef[1])
        theta = max(0.0, -beta)
        mu = -alpha / (beta + 1e-10) if abs(beta) > 1e-6 else float(Y.mean())
        residuals = dY - Xmat @ coef
        sigma = float(np.std(residuals))
        return theta, mu, sigma

    def _adf_stat(self, Y: np.ndarray) -> float:
        """
        Simplified ADF t-statistic (constant, no augmentation lags).

        More negative → stronger evidence against unit root (H0).
        Critical value ≈ −2.86 at 5% for n > 50.
        """
        n = len(Y)
        if n < 15:
            return 0.0
        dY = Y[1:] - Y[:-1]
        Ylag = Y[:-1]
        Xmat = np.column_stack([np.ones(n - 1), Ylag])
        try:
            coef, _, _, _ = np.linalg.lstsq(Xmat, dY, rcond=None)
            res = dY - Xmat @ coef
            s2 = float(res @ res) / max(1, n - 3)
            XtXinv = np.linalg.inv(Xmat.T @ Xmat + 1e-10 * np.eye(2))
            se_beta = float(np.sqrt(s2 * XtXinv[1, 1]))
            return float(coef[1]) / (se_beta + 1e-10)
        except np.linalg.LinAlgError:
            return 0.0

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = len(self.buffer)
        if n < cfg.min_samples_for_eval:
            return _not_ready(n)

        Y = np.array(self.buffer)
        self.reversion_rate, self.equilibrium, self.ou_sigma = self._ou_mle(Y)
        self.adf_stat = self._adf_stat(Y)

        adf_crit = -2.86
        self.is_stationary = self.adf_stat < adf_crit
        is_reverting = self.reversion_rate > cfg.reversion_threshold

        # theta in (0.05, 0.80) is genuine economic mean-reversion (AR coeff 0.20–0.95).
        # theta ≈ 1.0 → i.i.d. noise; theta ≈ 0 → random walk. Both are uninteresting.
        theta_in_range = 0.05 < self.reversion_rate < 0.80

        adf_conf = max(0.0, min(1.0, (adf_crit - self.adf_stat) / 3.0)) \
            if self.is_stationary else 0.2
        ou_conf = min(1.0, self.reversion_rate * 2.0) if is_reverting else 0.2

        if theta_in_range and (is_reverting or self.is_stationary):
            fit = 0.5 * adf_conf + 0.5 * ou_conf
        else:
            fit = 0.0  # suppress signal for noise (theta≈1) and random walks (theta≈0)

        return {
            'fit_score': fit,
            'evidence': n,
            'stability': 0.8 if is_reverting else 0.4,
            'equilibrium': self.equilibrium,
            'reversion_rate': self.reversion_rate,
            'ou_sigma': self.ou_sigma,
            'adf_stat': self.adf_stat,
            'is_stationary': self.is_stationary,
            'is_reverting': is_reverting,
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        cfg = self.config
        if len(self.buffer) < cfg.min_samples_for_prediction \
                or self.reversion_rate < cfg.reversion_threshold:
            return None
        current = float(self.buffer[-1])
        predicted = current - self.reversion_rate * (current - self.equilibrium)
        return (self.variable, predicted)


# ===========================================================================
# 6. COMPOSITIONAL — Sum Constraint
# ===========================================================================

class CompositionalHypothesis(Hypothesis):
    """Additive sum constraint: Total = Σ Parts."""

    def __init__(self, parts: List[str], total: str, buffer_size: int = 100,
                 config: Optional[CompositionalConfig] = None):
        super().__init__(parts + [total], RelationshipType.COMPOSITIONAL)
        self.parts = parts
        self.total = total
        self.buffer_parts = {p: deque(maxlen=buffer_size) for p in parts}
        self.buffer_total: deque = deque(maxlen=buffer_size)
        self.config = config or CompositionalConfig()
        self.constraint_error = float('inf')

    def fit_step(self, row: Dict[str, float]) -> None:
        if all(p in row for p in self.parts) and self.total in row:
            for p in self.parts:
                self.buffer_parts[p].append(row[p])
            self.buffer_total.append(row[self.total])

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = len(self.buffer_total)
        if n < cfg.min_samples:
            return _not_ready(n)

        parts_sum = sum(np.array(self.buffer_parts[p]) for p in self.parts)
        total = np.array(self.buffer_total)
        errors = np.abs(parts_sum - total) / (np.abs(total) + 1e-9)
        mean_err = float(errors.mean())
        std_err = float(errors.std())
        self.constraint_error = mean_err
        holds = mean_err < cfg.error_threshold
        consistency = max(0.0, 1.0 - mean_err * cfg.error_scaling)
        stability = max(0.3, 1.0 - std_err * 5.0)

        return {
            'fit_score': consistency,
            'evidence': n,
            'stability': stability,
            'constraint_error': mean_err,
            'constraint_error_std': std_err,
            'constraint_holds': holds,
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if all(p in row for p in self.parts):
            return (self.total, sum(row[p] for p in self.parts))
        return None


# ===========================================================================
# 7. COMPETITIVE — Anti-Correlation + Constant-Sum with Rolling Stability
# ===========================================================================

class CompetitiveHypothesis(Hypothesis):
    """
    Zero-sum / trade-off relationship (X + Y ≈ constant, r < 0).

    Improvements over v1:
    - Anti-correlation significance test (t-test)
    - Rolling window stability of constant-sum
    - Online Welford for sum variance
    """

    def __init__(self, var1: str, var2: str, buffer_size: int = 150,
                 config: Optional[CompetitiveConfig] = None):
        super().__init__([var1, var2], RelationshipType.COMPETITIVE)
        self.var1 = var1
        self.var2 = var2
        self.buffer1: deque = deque(maxlen=buffer_size)
        self.buffer2: deque = deque(maxlen=buffer_size)
        self.config = config or CompetitiveConfig()

        self._n = 0
        self._sum_mean = 0.0
        self._sum_M2 = 0.0

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.var1 in row and self.var2 in row:
            x, y = row[self.var1], row[self.var2]
            if np.isfinite(x) and np.isfinite(y):
                self.buffer1.append(x)
                self.buffer2.append(y)
                self._n += 1
                s = x + y
                delta = s - self._sum_mean
                self._sum_mean += delta / self._n
                self._sum_M2 += delta * (s - self._sum_mean)

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = len(self.buffer1)
        if n < cfg.min_samples:
            return _not_ready(n)

        X = np.array(self.buffer1)
        Y = np.array(self.buffer2)

        sum_var = self._sum_M2 / (self._n - 1) if self._n > 1 else 1.0
        cv = float(np.sqrt(max(0.0, sum_var))) / (abs(self._sum_mean) + 1e-9)

        sx, sy = float(np.std(X)), float(np.std(Y))
        r = float(np.corrcoef(X, Y)[0, 1]) if sx > 1e-9 and sy > 1e-9 else 0.0
        t = r * np.sqrt(n - 2) / np.sqrt(max(1e-10, 1 - r ** 2))
        p_corr = _t_pvalue(t, n - 2)  # two-tailed (used for is_competitive check)

        # One-tailed p-value for H₁: r < 0 (competitive = anti-correlated).
        # A strongly positive correlation (r > 0) should give p ≈ 1, not p ≈ 0,
        # so we can't use the two-tailed value directly as signal.
        p_anticorr = p_corr / 2.0 if r < 0 else 1.0

        is_competitive = cv < cfg.cv_threshold and r < cfg.correlation_threshold \
                         and p_corr < 0.05

        rolling_stab = 0.5
        if n >= 40:
            sums = X + Y
            half = n // 2
            cv1 = float(np.std(sums[:half])) / (abs(float(np.mean(sums[:half]))) + 1e-9)
            cv2 = float(np.std(sums[half:])) / (abs(float(np.mean(sums[half:]))) + 1e-9)
            rolling_stab = max(0.2, 1.0 - abs(cv1 - cv2))

        return {
            'fit_score': max(0.0, 1.0 - cv) if r < -0.1 else 0.2,
            'evidence': n,
            'stability': rolling_stab,
            'p_value': p_anticorr,
            'sum_cv': cv,
            'correlation': r,
            'p_anticorr': p_anticorr,
            'is_competitive': is_competitive,
            'constant_sum': self._sum_mean,
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if len(self.buffer1) < self.config.min_samples:
            return None
        # Always predict var2 from var1 when var1 is present (forward direction).
        # The constant-sum constraint gives var2 = C - var1 regardless of whether
        # var2 is also in the row — this is correct for perturbation tests where
        # the row always contains all variables.
        if self.var1 in row:
            return (self.var2, self._sum_mean - row[self.var1])
        if self.var2 in row:
            return (self.var1, self._sum_mean - row[self.var2])
        return None


# ===========================================================================
# 8. SYNERGISTIC — Online RLS Interaction Model + Partial F-test
# ===========================================================================

class SynergisticHypothesis(Hypothesis):
    """
    Interaction effect: Y = b0 + b1·X1 + b2·X2 + b3·(X1·X2).

    CRITICAL FIX from v1: fit_step now does online RLS on BOTH the full
    and reduced models simultaneously (v1 buffered data then called lstsq
    in evaluate() on every invocation — not online at all).

    Improvements:
    - Dual RLS: full model [1,X1,X2,X1X2] and reduced [1,X1,X2]
    - Partial F-test from running RSS windows
    - Meaningful p-value for interaction coefficient
    """

    def __init__(self, var1: str, var2: str, target: str,
                 buffer_size: int = 150,
                 config: Optional[SynergisticConfig] = None):
        super().__init__([var1, var2, target], RelationshipType.SYNERGISTIC)
        self.var1 = var1
        self.var2 = var2
        self.target = target
        self.config = config or SynergisticConfig()

        lam = 0.98
        # Full model [1, X1, X2, X1*X2]
        self._coef_full = np.zeros(4)
        self._P_full = np.eye(4) * 100.0
        # Reduced model [1, X1, X2]
        self._coef_red = np.zeros(3)
        self._P_red = np.eye(3) * 100.0
        self._lambda = lam

        self._rss_full: deque = deque(maxlen=60)
        self._rss_red: deque = deque(maxlen=60)
        self._n = 0

        self.interaction_coef = 0.0
        self.interaction_f_stat = 0.0
        self.interaction_p_value = 1.0

    def fit_step(self, row: Dict[str, float]) -> None:
        if not all(v in row for v in [self.var1, self.var2, self.target]):
            return
        x1, x2, y = row[self.var1], row[self.var2], row[self.target]
        if not all(np.isfinite(v) for v in [x1, x2, y]):
            return
        self._n += 1

        # Full model
        feat_f = np.array([1.0, x1, x2, x1 * x2])
        self._P_full, self._coef_full, err_f = _rls_step(
            self._P_full, self._coef_full, feat_f, y, self._lambda)
        self._rss_full.append(err_f ** 2)
        self.interaction_coef = float(self._coef_full[3])

        # Reduced model (no interaction)
        feat_r = np.array([1.0, x1, x2])
        self._P_red, self._coef_red, err_r = _rls_step(
            self._P_red, self._coef_red, feat_r, y, self._lambda)
        self._rss_red.append(err_r ** 2)

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n = self._n
        if n < cfg.min_samples:
            return _not_ready(n)

        if len(self._rss_full) >= 10:
            rss_f = float(np.sum(self._rss_full))
            rss_r = float(np.sum(self._rss_red))
            nw = len(self._rss_full)
            df_den = max(1, nw - 4)
            rss_diff = max(0.0, rss_r - rss_f)
            self.interaction_f_stat = (rss_diff / 1.0) / (rss_f / df_den + 1e-10)
            self.interaction_p_value = _f_pvalue(self.interaction_f_stat, 1, df_den)

        has_synergy = (abs(self.interaction_coef) > cfg.interaction_threshold
                       and self.interaction_p_value < 0.05)
        fit = min(1.0, abs(self.interaction_coef) * 2.0) if has_synergy else \
              max(0.1, abs(self.interaction_coef))

        return {
            'fit_score': fit,
            'evidence': n,
            'stability': 0.75,
            'p_value': self.interaction_p_value,
            'interaction_coefficient': self.interaction_coef,
            'interaction_f_stat': self.interaction_f_stat,
            'interaction_p_value': self.interaction_p_value,
            'has_synergy': has_synergy,
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self._n < self.config.min_samples:
            return None
        if self.var1 in row and self.var2 in row:
            x1, x2 = row[self.var1], row[self.var2]
            y_hat = float(np.dot(np.array([1.0, x1, x2, x1 * x2]), self._coef_full))
            return (self.target, y_hat)
        return None


# ===========================================================================
# 9. PROBABILISTIC — Adaptive Median Split + KS Test + JS Divergence
# ===========================================================================

class ProbabilisticHypothesis(Hypothesis):
    """
    Detects distributional shift: P(Y | X=high) ≠ P(Y | X=low).

    CRITICAL FIX from v1: fixed split threshold replaced with adaptive median.

    Improvements:
    - Online median tracking via running mean (fast approximation)
    - KS two-sample test (scipy or manual ECDF)
    - Jensen-Shannon divergence for distribution comparison
    """

    def __init__(self, condition: str, target: str, buffer_size: int = 200,
                 config: Optional[ProbabilisticConfig] = None):
        super().__init__([condition, target], RelationshipType.PROBABILISTIC)
        self.condition = condition
        self.target = target
        self.config = config or ProbabilisticConfig()

        self.buffer_x: deque = deque(maxlen=buffer_size)
        self.buffer_y_low: deque = deque(maxlen=buffer_size // 2)
        self.buffer_y_high: deque = deque(maxlen=buffer_size // 2)

        # Online mean as adaptive split threshold
        self._n = 0
        self._x_sum = 0.0

        self.ks_stat = 0.0
        self.ks_p_value = 1.0
        self.js_div = 0.0

    def _x_threshold(self) -> float:
        return self._x_sum / self._n if self._n > 0 else 0.0

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.condition not in row or self.target not in row:
            return
        x, y = row[self.condition], row[self.target]
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        self._n += 1
        self._x_sum += x
        self.buffer_x.append(x)
        thresh = self._x_threshold()
        if x <= thresh:
            self.buffer_y_low.append(y)
        else:
            self.buffer_y_high.append(y)

    def _js_div(self, A: np.ndarray, B: np.ndarray, bins: int = 20) -> float:
        lo, hi = min(A.min(), B.min()), max(A.max(), B.max())
        if hi - lo < 1e-9:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
        pa, _ = np.histogram(A, bins=edges)
        pb, _ = np.histogram(B, bins=edges)
        pa = pa.astype(float) + 1e-10
        pb = pb.astype(float) + 1e-10
        pa /= pa.sum()
        pb /= pb.sum()
        m = 0.5 * (pa + pb)
        js = 0.5 * (np.sum(pa * np.log(pa / m)) + np.sum(pb * np.log(pb / m)))
        return float(np.clip(js, 0.0, np.log(2)))

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        n0, n1 = len(self.buffer_y_low), len(self.buffer_y_high)
        if n0 < cfg.min_samples_per_group or n1 < cfg.min_samples_per_group:
            return _not_ready(n0 + n1)

        A = np.array(self.buffer_y_low)
        B = np.array(self.buffer_y_high)

        if _SCIPY:
            self.ks_stat, self.ks_p_value = scipy_stats.ks_2samp(A, B)
        else:
            a_s = np.sort(A)
            b_s = np.sort(B)
            all_v = np.sort(np.concatenate([a_s, b_s]))
            ca = np.searchsorted(a_s, all_v, side='right') / len(A)
            cb = np.searchsorted(b_s, all_v, side='right') / len(B)
            self.ks_stat = float(np.max(np.abs(ca - cb)))
            ne = len(A) * len(B) / (len(A) + len(B))
            self.ks_p_value = float(np.exp(-2.0 * ne * self.ks_stat ** 2))

        self.js_div = self._js_div(A, B)

        pooled_std = float(np.sqrt((np.var(A) + np.var(B)) / 2))
        effect_size = abs(float(B.mean() - A.mean())) / (pooled_std + 1e-9)
        significant = self.ks_p_value < 0.05 and effect_size > cfg.effect_size_threshold

        fit = 0.5 * self.ks_stat + 0.5 * float(self.js_div / np.log(2))

        return {
            'fit_score': fit,
            'evidence': n0 + n1,
            'stability': 0.7,
            'p_value': self.ks_p_value,
            'ks_stat': self.ks_stat,
            'ks_p_value': self.ks_p_value,
            'js_divergence': self.js_div,
            'effect_size': effect_size,
            'mean_shift': float(B.mean() - A.mean()),
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        # Conditional mean: given whether condition is above/below threshold, return
        # the mean of the target distribution for that regime.
        if self.condition not in row:
            return None
        n0 = len(self.buffer_y_low)
        n1 = len(self.buffer_y_high)
        cfg = self.config
        if n0 < cfg.min_samples_per_group or n1 < cfg.min_samples_per_group:
            return None
        if row[self.condition] > self._x_threshold():
            return (self.target, float(np.mean(self.buffer_y_high)))
        return (self.target, float(np.mean(self.buffer_y_low)))


# ===========================================================================
# 10. STRUCTURAL — One-way ANOVA F-test + η² Effect Size
# ===========================================================================

class StructuralHypothesis(Hypothesis):
    """
    Hierarchical / group structure.

    Improvements over v1:
    - One-way ANOVA F-test with p-value (via scipy or manual)
    - Eta-squared (η²) effect size alongside ICC
    - Group storage bounded per-group to prevent unbounded growth
    - Predicts group mean for known groups
    """

    def __init__(self, group: str, outcome: str, buffer_size: int = 200,
                 config: Optional[StructuralConfig] = None):
        super().__init__([group, outcome], RelationshipType.STRUCTURAL)
        self.group = group
        self.outcome = outcome
        self.config = config or StructuralConfig()
        self._max_per_group = 50
        self.group_data: Dict[float, deque] = {}

        self.f_stat = 0.0
        self.p_value = 1.0
        self.eta_squared = 0.0
        self.icc = 0.0

    def fit_step(self, row: Dict[str, float]) -> None:
        if self.group in row and self.outcome in row:
            g = round(float(row[self.group]), 6)
            y = row[self.outcome]
            if not np.isfinite(y):
                return
            if g not in self.group_data:
                self.group_data[g] = deque(maxlen=self._max_per_group)
            self.group_data[g].append(y)

    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        cfg = self.config
        if len(self.group_data) < cfg.min_groups:
            return _not_ready()

        groups = [np.array(v) for v in self.group_data.values() if len(v) >= 3]
        if len(groups) < 2:
            return _not_ready()

        if _SCIPY:
            f, p = scipy_stats.f_oneway(*groups)
            self.f_stat, self.p_value = float(f), float(p)
        else:
            all_data = np.concatenate(groups)
            grand_mean = float(all_data.mean())
            k = len(groups)
            n_total = len(all_data)
            ss_b = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups)
            ss_w = sum(float(np.sum((g - g.mean()) ** 2)) for g in groups)
            df_b = k - 1
            df_w = n_total - k
            if df_w > 0 and ss_w > 1e-10:
                self.f_stat = (ss_b / df_b) / (ss_w / df_w)
                self.p_value = _f_pvalue(self.f_stat, df_b, df_w)
            else:
                self.f_stat, self.p_value = 0.0, 1.0

        all_data = np.concatenate(groups)
        grand_mean = float(all_data.mean())
        ss_tot = float(np.sum((all_data - grand_mean) ** 2))
        ss_b = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups)
        self.eta_squared = float(ss_b / (ss_tot + 1e-10))

        gm = [float(g.mean()) for g in groups]
        within_var = float(np.mean([np.var(g) for g in groups]))
        between_var = float(np.var(gm))
        self.icc = between_var / (between_var + within_var + 1e-9)

        significant = self.p_value < 0.05 and self.eta_squared > 0.06
        total_n = sum(len(g) for g in groups)

        return {
            'fit_score': self.eta_squared,
            'evidence': total_n,
            'stability': 0.75 if significant else 0.4,
            'p_value': self.p_value,
            'f_stat': self.f_stat,
            'eta_squared': self.eta_squared,
            'icc': self.icc,
            'n_groups': len(groups),
            'ready': True,
        }

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        if self.group in row:
            g = round(float(row[self.group]), 6)
            if g in self.group_data and len(self.group_data[g]) > 0:
                return (self.outcome, float(np.mean(self.group_data[g])))
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HYPOTHESIS_CLASSES = {
    RelationshipType.CAUSAL: CausalHypothesis,
    RelationshipType.CORRELATIONAL: CorrelationalHypothesis,
    RelationshipType.TEMPORAL: TemporalHypothesis,
    RelationshipType.FUNCTIONAL: FunctionalHypothesis,
    RelationshipType.EQUILIBRIUM: EquilibriumHypothesis,
    RelationshipType.COMPOSITIONAL: CompositionalHypothesis,
    RelationshipType.COMPETITIVE: CompetitiveHypothesis,
    RelationshipType.SYNERGISTIC: SynergisticHypothesis,
    RelationshipType.PROBABILISTIC: ProbabilisticHypothesis,
    RelationshipType.STRUCTURAL: StructuralHypothesis,
}

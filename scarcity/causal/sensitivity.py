"""
Sensitivity analysis for unobserved confounding.

The refutation suite (``validation.py``) asks whether an estimate survives
perturbations of the data or specification. It cannot address the defining threat
to any observational effect: a confounder you never measured. Sensitivity
analysis answers the different, sharper question —

    *How strong would an unmeasured confounder have to be, in its association with
    both treatment and outcome, to explain away this effect?*

For a linear backdoor-adjusted effect (the estimand the engine reports) the exact
tool is the omitted-variable-bias framework of Cinelli & Hazlett (2020). From the
treatment coefficient's t-statistic and the residual degrees of freedom it yields
three interpretable, unit-free quantities:

- **partial R^2 of treatment with outcome** — how much of the outcome's residual
  variation the treatment explains; also the bias an unmeasured confounder as
  strong as the treatment itself would produce.
- **Robustness Value (RV)** — the minimum strength (partial R^2 with *both*
  treatment and outcome) an unmeasured confounder would need to reduce the
  estimate to zero. RV = 0.10 means a confounder explaining 10% of the residual
  variation in both would suffice; larger RV means a more robust finding.
- **RV to insignificance** — the same, but only strong enough to push the effect
  to the edge of statistical significance at level ``alpha`` (always <= RV).

Formulas verified against the canonical sensemakr Darfur benchmark (t = 4.18,
df = 783 -> partial R^2 = 0.022, RV = 0.139, RV_alpha=0.05 = 0.076).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def partial_r2(t_stat: float, dof: int) -> float:
    """Partial R^2 implied by a coefficient t-statistic. In [0, 1]."""
    t2 = float(t_stat) ** 2
    denom = t2 + float(dof)
    return float(t2 / denom) if denom > 0 else 0.0


def _t_critical(alpha: float, dof: int) -> float:
    """Two-sided critical t value; normal approximation if scipy is absent."""
    if _HAS_SCIPY:
        return float(abs(_stats.t.ppf(alpha / 2.0, max(dof, 1))))
    # Wilson–Hilferty-free normal fallback (dof large): z for two-sided alpha.
    # Coarse inverse-normal via Acklam-style approximation is overkill here;
    # scipy is a core dependency, so this path is a safety net only.
    return 1.959963985 if abs(alpha - 0.05) < 1e-9 else 2.575829304


def robustness_value(t_stat: float, dof: int, q: float = 1.0, alpha: float = 1.0) -> float:
    """Cinelli–Hazlett robustness value.

    The minimum partial R^2 an unmeasured confounder would need with *both*
    treatment and outcome to reduce the estimate by a proportion ``q`` (q=1 =>
    to zero). With ``alpha`` < 1 it instead measures the strength needed to reach
    the boundary of significance at level ``alpha`` (a weaker bar, so smaller).
    Returns a value in [0, 1]; 0 means even a null confounder would overturn it.
    """
    dof = int(dof)
    if dof <= 1:
        return 0.0
    f_q = q * abs(float(t_stat)) / math.sqrt(dof)
    if alpha < 1.0:
        f_crit = _t_critical(alpha, dof - 1) / math.sqrt(dof - 1)
        f_q = f_q - f_crit
    if f_q <= 0.0:
        return 0.0
    f2 = f_q * f_q
    rv = 0.5 * (math.sqrt(f2 * f2 + 4.0 * f2) - f2)
    return float(min(max(rv, 0.0), 1.0))


def _adjusted_t_stat(
    df: pd.DataFrame, treatment: str, outcome: str, confounders: Sequence[str]
) -> Optional[Dict[str, float]]:
    """Backdoor-adjusted treatment effect, its SE, t-stat, and residual dof.

    Frisch–Waugh–Lovell: residualize treatment and outcome on the confounders
    (with intercept), regress the residuals, and read the effect and its standard
    error from the residual variance. Matches the linear_regression backdoor ATE.
    """
    cols = [treatment, outcome, *confounders]
    d = df[cols].dropna()
    n = len(d)
    k = len(confounders)
    if n <= k + 3:
        return None
    t = d[treatment].to_numpy(float)
    y = d[outcome].to_numpy(float)
    if k:
        C = np.column_stack([np.ones(n), d[list(confounders)].to_numpy(float)])
    else:
        C = np.ones((n, 1))
    beta_c, *_ = np.linalg.lstsq(C, np.column_stack([t, y]), rcond=None)
    rt = t - C @ beta_c[:, 0]
    ry = y - C @ beta_c[:, 1]
    rtt = float(rt @ rt)
    if rtt <= 1e-12:
        return None
    beta = float(rt @ ry / rtt)
    resid = ry - beta * rt
    dof = n - (k + 2)  # intercept + k confounders + treatment
    if dof <= 1:
        return None
    sigma2 = float(resid @ resid) / dof
    se = math.sqrt(max(sigma2 / rtt, 1e-300))
    t_stat = beta / se if se > 0 else 0.0
    return {"estimate": beta, "se": se, "t_stat": t_stat, "dof": dof, "n": float(n)}


def sensitivity_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Sequence[str],
    *,
    q: float = 1.0,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Omitted-variable-bias sensitivity of the backdoor-adjusted effect.

    Returns the point estimate, the partial R^2 of treatment with outcome, the
    robustness value (to nullify) and the robustness value to insignificance at
    ``alpha``, plus a one-line interpretation. ``status`` is ``"ok"`` or
    ``"skipped"`` when there are too few rows to fit the adjusted model.
    """
    fit = _adjusted_t_stat(df, treatment, outcome, list(confounders))
    if fit is None:
        return {"status": "skipped", "reason": "insufficient rows for adjusted fit"}
    t_stat, dof = fit["t_stat"], int(fit["dof"])
    pr2 = partial_r2(t_stat, dof)
    rv = robustness_value(t_stat, dof, q=q, alpha=1.0)
    rv_a = robustness_value(t_stat, dof, q=q, alpha=alpha)
    return {
        "status": "ok",
        "estimate": fit["estimate"],
        "t_stat": t_stat,
        "dof": dof,
        "partial_r2_treatment_outcome": pr2,
        "robustness_value": rv,
        "robustness_value_alpha": rv_a,
        "alpha": alpha,
        "q": q,
        "summary": (
            f"RV(q={q:g})={rv:.3f}: a confounder explaining {rv*100:.1f}% of the "
            f"residual variation in both treatment and outcome would nullify the "
            f"effect; {rv_a*100:.1f}% would render it insignificant at alpha={alpha:g}. "
            f"Treatment's own partial R^2 with the outcome is {pr2:.3f}."
        ),
    }

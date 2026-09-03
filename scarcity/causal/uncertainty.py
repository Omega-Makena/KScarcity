"""Propagate discovery + estimation uncertainty into a prediction interval.

Downstream prediction usually treats a discovered edge as binary: a parent is
either in the model or not. But discovery returns a *confidence* per edge, and
regression returns *parameter* uncertainty. Collapsing both to a point estimate
throws that away and yields over-confident forecasts.

``predict_with_uncertainty`` carries both sources through to a prediction
interval on the target:

  point      = sum_i  c_i * beta_i * x_i        (+ intercept)
  Var(point) = x' diag(c) Cov(beta) diag(c) x   (parameter uncertainty)
             + sum_i x_i^2 beta_i^2 c_i(1-c_i)   (structure uncertainty:
                                                  Bernoulli edge inclusion)
             + sigma^2                            (irreducible noise)

With every confidence = 1 this reduces to the ordinary OLS prediction interval.
As a parent's confidence drops, its contribution to the point estimate shrinks
and the interval widens — an edge the discovery layer is unsure about makes the
forecast less certain, exactly as it should.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class PredictionInterval:
    point: float
    lower: float
    upper: float
    level: float
    param_var: float        # variance from coefficient uncertainty
    structure_var: float    # variance from edge-inclusion (confidence) uncertainty
    noise_var: float        # residual variance


def _z(level: float) -> float:
    # inverse standard-normal CDF at (1+level)/2, via the rational approximation
    # (avoids a scipy dependency for a single quantile)
    from math import sqrt, log
    p = (1.0 + level) / 2.0
    # Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = sqrt(-2 * log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def predict_with_uncertainty(
    X: np.ndarray,
    y: np.ndarray,
    x_new: Sequence[float],
    confidences: Optional[Sequence[float]] = None,
    level: float = 0.90,
    ridge: float = 1e-8,
) -> PredictionInterval:
    """Fit y ~ [1, X] by OLS and predict at x_new, propagating both the parameter
    covariance and the per-parent inclusion confidence into a prediction interval.

    Parameters
    ----------
    X : (n, p) parent (feature) matrix.
    y : (n,) target.
    x_new : (p,) feature values to predict at.
    confidences : (p,) discovery confidence per parent in [0, 1] (default all 1).
    level : central interval mass (e.g. 0.90).
    ridge : tiny diagonal load to keep X'X invertible.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    n, p = X.shape
    if confidences is None:
        c = np.ones(p)
    else:
        c = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)

    Xa = np.column_stack([np.ones(n), X])          # intercept + parents
    xa = np.concatenate([[1.0], x_new])
    ca = np.concatenate([[1.0], c])                # intercept always included

    XtX = Xa.T @ Xa + ridge * np.eye(p + 1)
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xa.T @ y)

    resid = y - Xa @ beta
    dof = max(n - (p + 1), 1)
    sigma2 = float(resid @ resid) / dof
    cov_beta = sigma2 * XtX_inv

    cx = ca * xa                                   # confidence-weighted design point
    point = float(cx @ beta)
    param_var = float(cx @ cov_beta @ cx)
    # structure uncertainty: parent i is "really" a parent with prob c_i, so its
    # contribution beta_i*x_i is a Bernoulli(c_i)-scaled term with variance
    # x_i^2 beta_i^2 c_i(1-c_i). Intercept (c=1) contributes nothing here.
    structure_var = float(np.sum((xa[1:] ** 2) * (beta[1:] ** 2) * c * (1.0 - c)))

    pred_var = param_var + structure_var + sigma2   # predictive: include noise
    half = _z(level) * float(np.sqrt(max(pred_var, 0.0)))
    return PredictionInterval(
        point=point, lower=point - half, upper=point + half, level=level,
        param_var=param_var, structure_var=structure_var, noise_var=sigma2,
    )

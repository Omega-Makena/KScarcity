"""Tests for discovery+estimation uncertainty propagation into prediction intervals."""
import numpy as np
import pytest

from scarcity.causal.uncertainty import predict_with_uncertainty


def _fit_data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    beta = np.array([1.5, -0.8])
    X = rng.normal(size=(n, 2))
    y = 2.0 + X @ beta + rng.normal(size=n)
    return X, y, beta


def test_full_confidence_interval_is_calibrated():
    """With confidence=1 the interval reduces to an OLS prediction interval and
    should cover the truth at ~the nominal rate."""
    rng = np.random.default_rng(1)
    beta = np.array([1.5, -0.8])
    n, level, hits, M = 120, 0.90, 0, 2500
    for _ in range(M):
        X = rng.normal(size=(n, 2))
        y = 2.0 + X @ beta + rng.normal(size=n)
        xn = rng.normal(size=2)
        yn = 2.0 + xn @ beta + rng.normal()
        pi = predict_with_uncertainty(X, y, xn, confidences=[1, 1], level=level)
        hits += int(pi.lower <= yn <= pi.upper)
    cov = hits / M
    assert 0.86 <= cov <= 0.93, f"coverage {cov:.3f} off nominal {level}"


def test_structure_variance_zero_at_full_confidence():
    X, y, _ = _fit_data()
    pi = predict_with_uncertainty(X, y, [1.0, 1.0], confidences=[1.0, 1.0])
    assert pi.structure_var == pytest.approx(0.0)


def test_uncertain_edge_widens_the_interval():
    """A coin-flip edge (c=0.5) is more uncertain than a certain one (c=1),
    so the interval must be wider."""
    X, y, _ = _fit_data()
    xn = [1.0, 1.0]
    w_sure = _width(predict_with_uncertainty(X, y, xn, confidences=[1.0, 1.0]))
    w_unsure = _width(predict_with_uncertainty(X, y, xn, confidences=[0.5, 0.5]))
    assert w_unsure > w_sure
    assert predict_with_uncertainty(X, y, xn, confidences=[0.5, 0.5]).structure_var > 0


def test_zero_confidence_drops_the_parent_to_baseline():
    """A parent with confidence 0 contributes nothing, so the point estimate is
    the intercept-only baseline — invariant to the feature values at x_new."""
    X, y, _ = _fit_data()
    p_a = predict_with_uncertainty(X, y, [5.0, 5.0], confidences=[0.0, 0.0])
    p_b = predict_with_uncertainty(X, y, [-3.0, 10.0], confidences=[0.0, 0.0])
    assert p_a.point == pytest.approx(p_b.point)      # x_new no longer matters
    assert p_a.structure_var == pytest.approx(0.0)


def test_point_estimate_shrinks_monotonically_toward_baseline():
    X, y, _ = _fit_data()
    xn = [2.0, 0.0]                       # only the first parent is active
    baseline = float(np.mean(y))
    pts = [predict_with_uncertainty(X, y, xn, confidences=[c, 1.0]).point
           for c in (1.0, 0.75, 0.5, 0.25, 0.0)]
    dist = [abs(pt - baseline) for pt in pts]
    assert all(dist[i] >= dist[i + 1] - 1e-9 for i in range(len(dist) - 1)), dist


def _width(pi):
    return pi.upper - pi.lower

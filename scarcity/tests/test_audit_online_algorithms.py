import numpy as np

from scarcity.engine.algorithms_online import KalmanFilter1D, RecursiveLeastSquares


def test_rls_kalman_updates_finite():
    rls = RecursiveLeastSquares(2)
    rls.update(np.array([1.0, 2.0], dtype=np.float32), 1.0)
    kf = KalmanFilter1D()
    kf.update(0.5)
    assert np.isfinite(rls.w).all() and np.isfinite(kf.x)

import numpy as np

from scarcity.engine.operators.evaluation_ops import granger_step


def test_granger_step_bounds():
    x = np.arange(20, dtype=np.float32)
    y = x * 2 + 1
    score = granger_step(x, y, lag=1)
    assert 0.0 <= score <= 1.0 and np.isfinite(score)

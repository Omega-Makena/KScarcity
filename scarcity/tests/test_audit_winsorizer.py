from scarcity.engine.algorithms_online import FunctionalLinearHypothesis


def test_evaluate_does_not_update_winsorizer():
    hyp = FunctionalLinearHypothesis("x", "y")
    for i in range(30):
        hyp.fit_step({"x": i, "y": i + 1})
    before = len(hyp.win_x.window)
    hyp.evaluate({"x": 999, "y": 0})
    assert len(hyp.win_x.window) == before

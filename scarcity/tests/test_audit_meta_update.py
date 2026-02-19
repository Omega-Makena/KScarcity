import asyncio

from scarcity.engine.engine import MPIEOrchestrator


def test_meta_prior_update_applies_policy():
    orch = MPIEOrchestrator()
    payload = {
        "prior": {
            "controller": {"tau": 0.25, "gamma_diversity": 1.1},
            "evaluator": {"g_min": 0.5, "lambda_ci": 2.0},
        }
    }
    asyncio.run(orch._handle_meta_policy_update("meta_prior_update", payload))
    assert orch.controller.config.epsilon == 0.25
    assert 0.0 <= orch.evaluator.gain_min <= 0.1

from __future__ import annotations

from scarcity.simulation.parameters import AllParams
from scarcity.simulation.sfc_engine import find_steady_state, step
from scarcity.simulation.types import PolicyState, SECTORS, ShockVector


def test_steady_state_solver_converges_and_hits_targets() -> None:
    params = AllParams.default_kenya()
    ss = find_steady_state(params)

    target_gdp = params.national_accounts.gdp_real_2023
    assert ss.gdp_real > 0.0
    assert abs(ss.gdp_real - target_gdp) / target_gdp < 0.80

    result = step(ss, PolicyState.default(), ShockVector.neutral(), None, params)
    assert max(abs(v) for v in result.accounting_errors.values()) < 1e4


def test_steady_state_stability_under_small_perturbation() -> None:
    params = AllParams.default_kenya()
    ss = find_steady_state(params)

    perturbed = ss
    perturbed_y = dict(perturbed.Y)
    for s in SECTORS:
        perturbed_y[s] *= 1.001

    perturbed = type(ss)(**{**ss.__dict__, "Y": perturbed_y})

    state = perturbed
    for _ in range(40):
        state = step(state, PolicyState.default(), ShockVector.neutral(), None, params).state

    rel_gap = abs(state.gdp_real - ss.gdp_real) / max(ss.gdp_real, 1e-12)
    assert rel_gap < 0.25

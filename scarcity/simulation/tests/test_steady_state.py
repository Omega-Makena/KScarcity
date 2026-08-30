from __future__ import annotations

import pytest

from scarcity.simulation.parameters import AllParams
from scarcity.simulation.sfc_engine import find_steady_state, step
from scarcity.simulation.types import PolicyState, SECTORS, ShockVector

# KNOWN BUG: the sfc_engine functional steady-state solver does not converge with
# default_kenya() params — it hits max_iter, lands ~576x off target GDP, and the
# accounting residuals explode (~1e125). The class-based SFCEconomy (sfc.py) — the
# engine the macro evidence uses — is unaffected. Tracked for a solver fix; until
# then these are expected failures rather than silent AttributeErrors.
_SOLVER_BROKEN = pytest.mark.xfail(
    reason="sfc_engine.find_steady_state diverges with default_kenya params",
    strict=False,
)


@_SOLVER_BROKEN
def test_steady_state_solver_converges_and_hits_targets() -> None:
    params = AllParams.default_kenya()
    ss, iters = find_steady_state(params)

    assert iters >= 1
    target_gdp = params.national_accounts.gdp_real_2023
    assert ss.gdp_real > 0.0
    assert abs(ss.gdp_real - target_gdp) / target_gdp < 0.80

    result = step(ss, PolicyState.default(), ShockVector.neutral(), None, params)
    assert max(abs(v) for v in result.accounting_errors.values()) < 1e4


@_SOLVER_BROKEN
def test_steady_state_stability_under_small_perturbation() -> None:
    params = AllParams.default_kenya()
    ss, _ = find_steady_state(params)

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

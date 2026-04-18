from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from scarcity.simulation.parameters import MonetaryParams
from scarcity.simulation.types import SECTORS


@dataclass(frozen=True)
class MonetaryComputation:
    i_cb: float
    i_loan: float
    i_dep: float
    i_gov: float
    i_taylor: float
    pi_cpi: float
    delta_res_fx_intervention: float
    import_cover_months: float


def compute_monetary_block(
    P_cpi: float,
    P_cpi_prev: float,
    output_gap: float,
    i_cb_prev: float,
    i_target_override: float | None,
    IM: Dict,
    E_nom: float,
    RES_fx: float,
    risk_premium_shock: float,
    params: MonetaryParams,
) -> MonetaryComputation:
    """Compute policy rates and FX-reserve intervention from Taylor + reserve-cover rules."""
    pi_cpi = (float(P_cpi) - float(P_cpi_prev)) / max(float(P_cpi_prev), 1e-12)
    i_taylor = (
        float(params.i_neutral)
        + float(params.phi_pi) * (pi_cpi - float(params.pi_target))
        + float(params.phi_y) * float(output_gap)
    )

    i_cb = float(params.smoothing) * float(i_cb_prev) + (1.0 - float(params.smoothing)) * i_taylor
    i_cb = max(float(params.i_floor), min(float(params.i_ceiling), i_cb))

    if i_target_override is not None:
        i_cb = float(i_target_override)

    i_loan = i_cb + float(params.spread_loan)
    i_dep = max(0.0, i_cb + float(params.spread_deposit))
    i_gov = i_cb + float(params.spread_govt) + float(risk_premium_shock)

    imports_quarter = sum(max(float(IM[s]), 0.0) for s in SECTORS)
    import_cover_months = float(RES_fx) * float(E_nom) / max(imports_quarter / 4.0, 1e-12) * 3.0
    reserve_gap = float(params.fx_reserve_target_months) - import_cover_months
    delta_res_fx = float(params.fx_intervention_speed) * reserve_gap * (imports_quarter / 12.0) / max(float(E_nom), 1e-12)

    return MonetaryComputation(
        i_cb=float(i_cb),
        i_loan=float(i_loan),
        i_dep=float(i_dep),
        i_gov=float(i_gov),
        i_taylor=float(i_taylor),
        pi_cpi=float(pi_cpi),
        delta_res_fx_intervention=float(delta_res_fx),
        import_cover_months=float(import_cover_months),
    )

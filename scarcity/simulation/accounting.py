from __future__ import annotations

from typing import Dict

from scarcity.simulation.types import EconomyState, SECTORS, Sector


def _safe_float(value: float) -> float:
    return float(value)


def run_accounting_checks(
    prev_state: EconomyState,
    state: EconomyState,
    flows: Dict[str, float | dict[Sector, float]],
) -> dict[str, float]:
    """Run the required stock-flow consistency residual checks."""

    residuals: dict[str, float] = {}

    # 1. National income identity.
    y_gdp_identity = _safe_float(state.C) + _safe_float(flows.get("I_total", 0.0)) + _safe_float(state.G_exp) + (
        sum(_safe_float(state.EX[s]) for s in SECTORS) - sum(_safe_float(state.IM[s]) for s in SECTORS)
    )
    residuals["residual_1"] = y_gdp_identity - sum(_safe_float(state.Y[s]) for s in SECTORS)

    # 2. Household budget constraint.
    delta_d_h = _safe_float(flows.get("delta_D_h", _safe_float(state.D_h) - _safe_float(prev_state.D_h)))
    delta_l_h = _safe_float(flows.get("delta_L_h", _safe_float(state.L_h) - _safe_float(prev_state.L_h)))
    residuals["residual_2"] = delta_d_h - (_safe_float(state.S_h) + delta_l_h)

    # 3. Government budget constraint.
    delta_b_gov = _safe_float(flows.get("delta_B_gov", _safe_float(state.B_gov) - _safe_float(prev_state.B_gov)))
    residuals["residual_3"] = delta_b_gov - _safe_float(state.DEFICIT)

    # 4. Bank balance sheet.
    reserves_at_cb = _safe_float(flows.get("bank_reserves_at_cb", 0.0))
    bank_assets = (
        sum(_safe_float(state.L_f[s]) for s in SECTORS)
        + _safe_float(state.L_h)
        + _safe_float(state.B_bank)
        + reserves_at_cb
    )
    bank_liabilities = _safe_float(state.D_h) + sum(_safe_float(state.D_f[s]) for s in SECTORS) + _safe_float(state.BANK_EQUITY)
    residuals["residual_4"] = bank_assets - bank_liabilities

    # 5. Bond market clearing (B_household omitted in this closure).
    residuals["residual_5"] = _safe_float(state.B_gov) - (
        _safe_float(state.B_bank) + _safe_float(state.B_cb) + _safe_float(state.B_foreign)
    )

    # 6. Current account / capital account / reserves identity.
    delta_res_fx = _safe_float(flows.get("delta_res_fx", _safe_float(state.RES_fx) - _safe_float(prev_state.RES_fx)))
    residuals["residual_6"] = _safe_float(state.CA) + _safe_float(state.KA) - delta_res_fx * _safe_float(state.E_nom)

    # 7. Walras law redundant equation check.
    excess_demand = flows.get("excess_demand", 0.0)
    if isinstance(excess_demand, dict):
        residuals["residual_7"] = sum(_safe_float(excess_demand[s]) for s in SECTORS)
    else:
        residuals["residual_7"] = _safe_float(excess_demand)

    # 8. Capital stock evolution by sector.
    investment_by_sector = flows.get("investment_by_sector", {})
    if isinstance(investment_by_sector, dict):
        for sector in SECTORS:
            i_s = _safe_float(investment_by_sector.get(sector, 0.0))
            delta_s = _safe_float(flows.get(f"delta_{sector.value}", 0.0))
            destruction_s = _safe_float(state.capital_destruction[sector])
            rhs = (1.0 - delta_s - destruction_s) * _safe_float(prev_state.K[sector]) + i_s
            residuals[f"residual_8_{sector.value}"] = _safe_float(state.K[sector]) - rhs
    else:
        for sector in SECTORS:
            residuals[f"residual_8_{sector.value}"] = 0.0

    return residuals


def accounting_warnings(
    residuals: dict[str, float],
    gdp: float,
    tolerance_factor: float = 1e-6,
) -> list[str]:
    """Generate warning messages when residual magnitudes exceed tolerance."""

    tol = tolerance_factor * max(abs(gdp), 1.0)
    warnings: list[str] = []
    for key, value in residuals.items():
        if abs(value) > tol:
            warnings.append(f"{key} exceeds tolerance: {value:.6e} (tol={tol:.6e})")
    return warnings

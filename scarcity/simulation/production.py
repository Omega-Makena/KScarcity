from __future__ import annotations

from typing import Dict, Tuple

from scarcity.simulation.parameters import InputOutputParams, ProductionParams
from scarcity.simulation.types import SECTORS, Sector, SectorFeedback, ShockVector

_EPS = 1e-12
_COBB_DOUGLAS_TOL = 0.01


def _rho_from_sigma(sigma: float) -> float:
    sigma_safe = max(sigma, _EPS)
    return (sigma_safe - 1.0) / sigma_safe


def _destruction_by_sector(feedback: SectorFeedback | None) -> Dict[Sector, float]:
    if feedback is None or feedback.capital_destruction is None:
        return {s: 0.0 for s in SECTORS}
    return {s: min(max(float(feedback.capital_destruction[s]), 0.0), 1.0) for s in SECTORS}


def _productivity_multiplier(
    sector: Sector,
    shocks: ShockVector,
    feedback: SectorFeedback | None,
) -> float:
    supply_shock = float(shocks.supply_shock[sector])
    productivity_shock = 1.0
    if feedback is not None and feedback.labor_productivity_factor is not None:
        productivity_shock = float(feedback.labor_productivity_factor[sector])

    tfp_shock = productivity_shock * supply_shock
    if sector == Sector.AGRICULTURE:
        tfp_shock *= float(shocks.rainfall_shock)
        if feedback is not None:
            tfp_shock *= float(feedback.yield_factor)
    return max(tfp_shock, 0.0)


def _ces_or_cd_output(
    A_eff: float,
    alpha: float,
    sigma: float,
    K_eff: float,
    hN: float,
) -> float:
    if A_eff <= 0.0 or K_eff <= 0.0 or hN <= 0.0:
        return 0.0

    if abs(sigma - 1.0) <= _COBB_DOUGLAS_TOL:
        # Cobb-Douglas limit for sigma -> 1 (rho -> 0).
        return A_eff * (K_eff ** alpha) * (hN ** (1.0 - alpha))

    rho = _rho_from_sigma(sigma)
    inside = alpha * (K_eff ** rho) + (1.0 - alpha) * (hN ** rho)
    inside = max(inside, _EPS)
    return A_eff * (inside ** (1.0 / max(abs(rho), _EPS) * (1.0 if rho >= 0.0 else -1.0)))


def compute_gross_output(
    K: Dict[Sector, float],
    N: Dict[Sector, float],
    params: ProductionParams,
    shocks: ShockVector,
    feedback: SectorFeedback | None,
) -> Dict[Sector, float]:
    """Compute gross output for all sectors using the CES formula and CD limit."""
    destruction = _destruction_by_sector(feedback)
    outputs: Dict[Sector, float] = {}

    for sector in SECTORS:
        alpha_s = float(params.alpha[sector])
        sigma_s = float(params.sigma[sector])
        A_s = float(params.A[sector])
        h_s = float(params.h[sector])

        K_eff_s = max(float(K[sector]), 0.0) * (1.0 - destruction[sector])
        tfp_shock_s = _productivity_multiplier(sector, shocks, feedback)
        A_eff_s = A_s * tfp_shock_s
        hN_s = max(h_s * max(float(N[sector]), 0.0), _EPS)

        outputs[sector] = _ces_or_cd_output(
            A_eff=A_eff_s,
            alpha=alpha_s,
            sigma=sigma_s,
            K_eff=max(K_eff_s, _EPS),
            hN=hN_s,
        )

    return outputs


def compute_value_added(
    Y_gross: Dict[Sector, float],
    io_params: InputOutputParams,
) -> Dict[Sector, float]:
    """Compute value added by sector: VA_s = Y_gross_s * (1 - sum_j io[s][j])."""
    va: Dict[Sector, float] = {}
    for s in SECTORS:
        io_sum = float(sum(io_params.io_matrix[s][j] for j in SECTORS))
        va[s] = max(float(Y_gross[s]) * (1.0 - io_sum), 0.0)
    return va


def compute_marginal_products(
    Y_gross: Dict[Sector, float],
    K: Dict[Sector, float],
    N: Dict[Sector, float],
    params: ProductionParams,
    feedback: SectorFeedback | None,
) -> Tuple[Dict[Sector, float], Dict[Sector, float]]:
    """Compute MPK and MPL from CES first-order conditions."""
    destruction = _destruction_by_sector(feedback)
    mpk: Dict[Sector, float] = {}
    mpl: Dict[Sector, float] = {}

    for s in SECTORS:
        sigma_s = max(float(params.sigma[s]), _EPS)
        rho_s = _rho_from_sigma(sigma_s)
        A_s = max(float(params.A[s]), _EPS)
        alpha_s = float(params.alpha[s])
        h_s = max(float(params.h[s]), _EPS)

        Y_s = max(float(Y_gross[s]), _EPS)
        K_eff_s = max(float(K[s]) * (1.0 - destruction[s]), _EPS)
        hN_s = max(h_s * max(float(N[s]), _EPS), _EPS)

        mpk[s] = (A_s ** rho_s) * alpha_s * ((Y_s / K_eff_s) ** (1.0 / sigma_s))
        mpl[s] = (A_s ** rho_s) * (1.0 - alpha_s) * (h_s ** rho_s) * ((Y_s / hN_s) ** (1.0 / sigma_s))

    return mpk, mpl


def compute_potential_output(
    K: Dict[Sector, float],
    N_pot: Dict[Sector, float],
    params: ProductionParams,
    io_params: InputOutputParams,
) -> Dict[Sector, float]:
    """Potential output at NAIRU employment, no shocks/destruction."""
    y_pot: Dict[Sector, float] = {}
    for s in SECTORS:
        alpha_s = float(params.alpha[s])
        sigma_s = float(params.sigma[s])
        A_s = float(params.A[s])
        h_s = float(params.h[s])
        K_s = max(float(K[s]), _EPS)
        hN_s = max(h_s * max(float(N_pot[s]), 0.0), _EPS)

        gross = _ces_or_cd_output(
            A_eff=A_s,
            alpha=alpha_s,
            sigma=sigma_s,
            K_eff=K_s,
            hN=hN_s,
        )
        io_sum = float(sum(io_params.io_matrix[s][j] for j in SECTORS))
        y_pot[s] = max(gross * (1.0 - io_sum), 0.0)
    return y_pot


def compute_intermediate_demand(
    Y_gross: Dict[Sector, float],
    io_params: InputOutputParams,
) -> Dict[Sector, Dict[Sector, float]]:
    """Return intermediate matrix: using sector rows, supplying sector columns."""
    result: Dict[Sector, Dict[Sector, float]] = {}
    for using_sector in SECTORS:
        row: Dict[Sector, float] = {}
        for supplying_sector in SECTORS:
            row[supplying_sector] = max(
                float(io_params.io_matrix[using_sector][supplying_sector]) * float(Y_gross[using_sector]),
                0.0,
            )
        result[using_sector] = row
    return result

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from scarcity.simulation.parameters import InputOutputParams, ProductionParams
from scarcity.simulation.parameters import HouseholdParams
from scarcity.simulation.types import SECTORS, Sector


@dataclass(frozen=True)
class PriceComputation:
    P_new: Dict[Sector, float]
    P_cpi: float
    profits: Dict[Sector, float]
    unit_cost: Dict[Sector, float]
    pi_cpi: float


def compute_prices_and_profits(
    P_prev: Dict[Sector, float],
    w: Dict[Sector, float],
    N: Dict[Sector, float],
    Y_gross: Dict[Sector, float],
    E_nom: float,
    P_world: float,
    io_params: InputOutputParams,
    household_params: HouseholdParams,
    production_params: ProductionParams,
    L_f: Dict[Sector, float],
    K: Dict[Sector, float],
    i_loan: float,
    price_adjustment_speed: float = 0.4,
) -> PriceComputation:
    """Compute sticky sector prices, CPI, and firm profits from cost-plus equations."""
    markups = {
        Sector.AGRICULTURE: 0.10,
        Sector.MANUFACTURING: 0.15,
        Sector.SERVICES: 0.20,
        Sector.INFORMAL: 0.05,
    }

    P_target: Dict[Sector, float] = {}
    P_new: Dict[Sector, float] = {}
    unit_cost: Dict[Sector, float] = {}
    profits: Dict[Sector, float] = {}

    for s in SECTORS:
        int_cost = sum(float(io_params.io_matrix[s][j]) * float(P_prev[j]) * float(Y_gross[s]) for j in SECTORS)
        INT_s = sum(float(io_params.io_matrix[s][j]) * float(Y_gross[s]) for j in SECTORS)
        import_cost = float(io_params.import_content[s]) * float(E_nom) * float(P_world) * float(INT_s)
        numerator = float(w[s]) * float(N[s]) + int_cost + import_cost
        denom = max(float(Y_gross[s]), 1e-12)
        uc = numerator / denom
        unit_cost[s] = float(uc)

        target_P_s = (1.0 + float(markups[s])) * float(uc)
        P_target[s] = target_P_s
        P_new[s] = max(float(P_prev[s]) + float(price_adjustment_speed) * (target_P_s - float(P_prev[s])), 1e-9)

    P_cpi = (
        sum(float(household_params.consumption_shares[s]) * float(P_new[s]) for s in SECTORS)
        + float(household_params.import_share_consumption) * float(E_nom) * float(P_world)
    )
    P_cpi_prev = (
        sum(float(household_params.consumption_shares[s]) * float(P_prev[s]) for s in SECTORS)
        + float(household_params.import_share_consumption) * float(E_nom) * float(P_world)
    )
    pi_cpi = (float(P_cpi) - float(P_cpi_prev)) / max(float(P_cpi_prev), 1e-12)

    P_mfg = max(float(P_new[Sector.MANUFACTURING]), 1e-12)
    for s in SECTORS:
        import_cost_s = float(io_params.import_content[s]) * float(E_nom) * float(P_world) * float(Y_gross[s])
        inter_cost_s = sum(float(P_new[j]) * float(io_params.io_matrix[s][j]) * float(Y_gross[s]) for j in SECTORS)
        profits[s] = (
            float(P_new[s]) * float(Y_gross[s])
            - float(w[s]) * float(N[s])
            - inter_cost_s
            - import_cost_s
            - float(i_loan) * float(L_f[s])
            - float(production_params.delta[s]) * P_mfg * float(K[s])
        )

    return PriceComputation(
        P_new=P_new,
        P_cpi=float(P_cpi),
        profits=profits,
        unit_cost=unit_cost,
        pi_cpi=float(pi_cpi),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from scarcity.simulation.types import SECTORS, Sector


@dataclass(frozen=True)
class LaborMarketComputation:
    N: Dict[Sector, float]
    N_s_total: float
    U: float
    w: Dict[Sector, float]


def compute_labor_market(
    Y: Dict[Sector, float],
    Y_prev: Dict[Sector, float],
    N_prev: Dict[Sector, float],
    w_prev: Dict[Sector, float],
    P: Dict[Sector, float],
    labor_force_prev: float,
    labor_force_growth_rate: float,
    labor_supply_shock: float,
    sigma: Dict[Sector, float],
    pi_cpi_prev: float,
    U_nairu: float = 0.05,
    phillips_slope: float = 0.3,
    okun_elasticity: float = 0.5,
    adjustment_speed: float = 0.3,
) -> LaborMarketComputation:
    """Compute labor demand/supply, unemployment, and wages with Phillips relation.

    The capital-labour substitution term (wage_term) compares last period's real wage
    with the expected current real wage.  Because sector prices are not yet known at
    the labour-market step, we use the lagged CPI inflation (pi_cpi_prev) as a proxy
    for current-period price growth.

        w_real_prev = w_prev / P                           (last period real wage)
        w_real      = w_prev / (P * (1 + pi_cpi_prev))    (expected current real wage)
        wage_term   = (w_real_prev / w_real) ^ sigma
                    = (1 + pi_cpi_prev) ^ sigma

    When pi_cpi_prev > 0 (inflation), real wages fall → wage_term > 1 → firms
    demand more labour (capital-labour substitution).  The original code used the same
    P for both terms, making wage_term identically 1.0.
    """
    substitution_elasticity = sigma
    N_target: Dict[Sector, float] = {}
    N_new: Dict[Sector, float] = {}

    # Expected quarterly price-level multiplier from lagged CPI inflation.
    price_growth = max(1.0 + float(pi_cpi_prev), 1e-12)

    for s in SECTORS:
        # Real wage last period: deflate previous nominal wage by current sector price.
        w_real_prev = float(w_prev[s]) / max(float(P[s]), 1e-12)
        # Expected current real wage: same nominal wage, but prices have risen by pi_cpi_prev.
        w_real = w_real_prev / price_growth
        growth_term = (max(float(Y[s]), 1e-12) / max(float(Y_prev[s]), 1e-12)) ** float(okun_elasticity)
        wage_term = (max(float(w_real_prev), 1e-12) / max(float(w_real), 1e-12)) ** float(substitution_elasticity[s])
        N_target[s] = float(N_prev[s]) * growth_term * wage_term
        N_new[s] = max(
            (float(N_prev[s]) + float(adjustment_speed) * (N_target[s] - float(N_prev[s]))) * float(labor_supply_shock),
            0.0,
        )

    N_s_total = max(float(labor_force_prev) * (1.0 + float(labor_force_growth_rate) / 4.0) * float(labor_supply_shock), 1e-12)
    total_employment = sum(float(N_new[s]) for s in SECTORS)
    U = 1.0 - total_employment / N_s_total
    U = max(0.01, min(0.50, U))

    sector_premium = {
        Sector.AGRICULTURE: -0.02,
        Sector.MANUFACTURING: 0.01,
        Sector.SERVICES: 0.02,
        Sector.INFORMAL: -0.03,
    }

    w_new: Dict[Sector, float] = {}
    for s in SECTORS:
        wage_growth = float(pi_cpi_prev) + float(phillips_slope) * (float(U_nairu) - float(U)) + float(sector_premium[s])
        w_new[s] = max(float(w_prev[s]) * (1.0 + wage_growth), 1e-9)

    return LaborMarketComputation(
        N=N_new,
        N_s_total=float(N_s_total),
        U=float(U),
        w=w_new,
    )

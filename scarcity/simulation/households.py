from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from scarcity.simulation.parameters import BankingParams, HouseholdParams
from scarcity.simulation.types import SECTORS, Sector

_EPS = 1e-12


@dataclass(frozen=True)
class HouseholdComputation:
    Y_disp: float
    C: float
    C_by_sector: Dict[Sector, float]
    S_h: float
    D_h_new: float
    L_h_new: float
    delta_D_h: float
    delta_L_h: float
    GINI: float
    POVERTY: float


def compute_disposable_income(
    w: Dict[Sector, float],
    N: Dict[Sector, float],
    dividends: float,
    rem_h: float,
    transfers_gov: float,
    tax_rate_income: float,
    tax_rate_vat: float,
    C_guess: float,
) -> float:
    """Y_disp = wage_income + DIV + REM_h + TRANS - TAX_income - VAT(C)."""
    wage_income = sum(float(w[s]) * float(N[s]) for s in SECTORS)
    tax_income = float(tax_rate_income) * wage_income
    tax_vat_on_consumption = float(tax_rate_vat) * max(float(C_guess), 0.0) / (1.0 + float(tax_rate_vat))
    return wage_income + float(dividends) + float(rem_h) + float(transfers_gov) - tax_income - tax_vat_on_consumption


def compute_consumption(
    Y_disp: float,
    D_h_prev: float,
    L_h_prev: float,
    params: HouseholdParams,
) -> float:
    """C = c_1 * Y_disp + c_2 * W_h_prev with W_h_prev = D_h_prev - L_h_prev."""
    W_h_prev = float(D_h_prev) - float(L_h_prev)
    return max(params.c_1 * float(Y_disp) + params.c_2 * W_h_prev, 0.0)


def allocate_consumption_by_sector(
    C: float,
    demand_shift: Dict[Sector, float],
    params: HouseholdParams,
) -> Dict[Sector, float]:
    """C_s = consumption_shares[s] * C * demand_shift[s]."""
    return {
        s: max(float(params.consumption_shares[s]) * float(C) * float(demand_shift[s]), 0.0)
        for s in SECTORS
    }


def compute_inequality_and_poverty(
    Y_disp: float,
    U: float,
    U_nairu: float,
    poverty_base: float,
) -> tuple[float, float]:
    """Compute Gini from unemployment-adjusted quintile incomes and poverty elasticity rule."""
    unemployment_gap = max(0.0, float(U) - float(U_nairu))
    effects = [
        1.0 - 2.0 * unemployment_gap,
        1.0 - 1.5 * unemployment_gap,
        1.0 - 1.0 * unemployment_gap,
        1.0 - 0.5 * unemployment_gap,
        1.0 - 0.2 * unemployment_gap,
    ]
    effects = [max(e, 0.01) for e in effects]

    base_shares = [0.047, 0.081, 0.121, 0.190, 0.561]
    incomes = [base_shares[q] * float(Y_disp) * effects[q] for q in range(5)]
    total_income = max(sum(incomes), _EPS)
    shares = [inc / total_income for inc in incomes]

    cumulative_prev = 0.0
    gini_sum = 0.0
    for share in shares:
        gini_sum += 2.0 * cumulative_prev + share
        cumulative_prev += share
    gini = 1.0 - (1.0 / 5.0) * gini_sum
    gini = min(max(gini, 0.0), 1.0)

    mean_income_index_change = (float(Y_disp) / max(total_income, _EPS)) - 1.0
    poverty_elasticity = -2.5
    poverty = float(poverty_base) * (1.0 + poverty_elasticity * mean_income_index_change)
    poverty = min(max(poverty, 0.0), 1.0)

    return gini, poverty


def compute_households(
    w: Dict[Sector, float],
    N: Dict[Sector, float],
    dividends: float,
    rem_h: float,
    transfers_gov: float,
    tax_rate_income: float,
    tax_rate_vat: float,
    demand_shift: Dict[Sector, float],
    D_h_prev: float,
    L_h_prev: float,
    U: float,
    U_nairu: float,
    poverty_base: float,
    params: HouseholdParams,
    banking_params: BankingParams,
    credit_multiplier: float,
    loan_to_income_ratio: float = 0.40,
    adjustment_speed: float = 0.30,
) -> HouseholdComputation:
    """Compute the full household block with endogenous borrowing and distributional metrics."""
    C_first_pass = compute_consumption(
        Y_disp=sum(float(w[s]) * float(N[s]) for s in SECTORS) + float(dividends) + float(rem_h) + float(transfers_gov),
        D_h_prev=D_h_prev,
        L_h_prev=L_h_prev,
        params=params,
    )
    Y_disp = compute_disposable_income(
        w=w,
        N=N,
        dividends=dividends,
        rem_h=rem_h,
        transfers_gov=transfers_gov,
        tax_rate_income=tax_rate_income,
        tax_rate_vat=tax_rate_vat,
        C_guess=C_first_pass,
    )

    C = compute_consumption(Y_disp=Y_disp, D_h_prev=D_h_prev, L_h_prev=L_h_prev, params=params)
    C_by_sector = allocate_consumption_by_sector(C=C, demand_shift=demand_shift, params=params)
    S_h = float(Y_disp) - float(C)

    wage_income = sum(float(w[s]) * float(N[s]) for s in SECTORS)
    L_h_desired = float(loan_to_income_ratio) * wage_income
    raw_delta_L_h = float(adjustment_speed) * (L_h_desired - float(L_h_prev))
    credit_multiplier = min(max(float(credit_multiplier), 0.0), 1.0)
    delta_L_h = raw_delta_L_h * credit_multiplier

    # Respect a simple solvency envelope from banking leverage assumptions.
    max_household_loan = max(0.0, banking_params.max_leverage_ratio * max(D_h_prev, 0.0))
    L_h_new = min(max(float(L_h_prev) + delta_L_h, 0.0), max_household_loan)
    delta_L_h = L_h_new - float(L_h_prev)

    delta_D_h = float(S_h) + float(delta_L_h)
    D_h_new = max(float(D_h_prev) + float(delta_D_h), 0.0)

    gini, poverty = compute_inequality_and_poverty(
        Y_disp=Y_disp,
        U=U,
        U_nairu=U_nairu,
        poverty_base=poverty_base,
    )

    return HouseholdComputation(
        Y_disp=float(Y_disp),
        C=float(C),
        C_by_sector=C_by_sector,
        S_h=float(S_h),
        D_h_new=float(D_h_new),
        L_h_new=float(L_h_new),
        delta_D_h=float(delta_D_h),
        delta_L_h=float(delta_L_h),
        GINI=float(gini),
        POVERTY=float(poverty),
    )

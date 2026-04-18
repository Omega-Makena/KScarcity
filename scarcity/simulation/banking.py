from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from scarcity.simulation.parameters import BankingParams
from scarcity.simulation.types import SECTORS, Sector


@dataclass(frozen=True)
class BankingComputation:
    L_f_new: dict[Sector, float]
    D_f_new: dict[Sector, float]
    L_h_new: float
    D_h_new: float
    BANK_EQUITY_new: float
    BANK_CAR: float
    NPL_ratio_new: float
    credit_multiplier: float
    bank_profit: float
    reserves_at_cb: float
    rwa: float
    delta_L_f: dict[Sector, float]


def compute_banking_block(
    L_f_prev: Dict[Sector, float],
    D_f_prev: Dict[Sector, float],
    L_h_prev: float,
    D_h_prev: float,
    B_bank: float,
    BANK_EQUITY_prev: float,
    BANK_CAR_prev: float,
    NPL_ratio_prev: float,
    U: float,
    U_nairu: float,
    output_gap_by_sector: Dict[Sector, float],
    i_loan: float,
    i_dep: float,
    i_gov: float,
    delta_D_h: float,
    delta_L_h: float,
    params: BankingParams,
) -> BankingComputation:
    """Compute banking sector credit, risk, and balance-sheet dynamics."""
    max_loans = float(BANK_EQUITY_prev) * float(params.max_leverage_ratio)
    L_total_prev = sum(float(L_f_prev[s]) for s in SECTORS) + float(L_h_prev)
    available_credit = max(max_loans - L_total_prev, 0.0)

    if float(BANK_CAR_prev) < float(params.credit_rationing_threshold):
        denom = max(float(params.credit_rationing_threshold) - float(params.min_capital_adequacy), 1e-12)
        credit_multiplier = (float(BANK_CAR_prev) - float(params.min_capital_adequacy)) / denom
        credit_multiplier = max(0.0, min(1.0, credit_multiplier))
    else:
        credit_multiplier = 1.0

    L_f_new: Dict[Sector, float] = {}
    delta_L_f: Dict[Sector, float] = {}
    per_sector_credit_cap = available_credit / 4.0

    for s in SECTORS:
        L_desired_s = float(L_f_prev[s]) * (
            1.0 + float(params.credit_growth_sensitivity_to_output_gap) * float(output_gap_by_sector[s])
        )
        requested = L_desired_s - float(L_f_prev[s])
        delta = min(credit_multiplier * requested, per_sector_credit_cap)
        L_f_new[s] = max(float(L_f_prev[s]) + delta, 0.0)
        delta_L_f[s] = float(L_f_new[s] - float(L_f_prev[s]))

    npl_new = float(NPL_ratio_prev) + float(params.npl_sensitivity_to_unemployment) * (float(U) - float(U_nairu)) * (1.0 - float(NPL_ratio_prev))
    npl_new = max(0.02, min(0.40, npl_new))

    L_h_new = max(float(L_h_prev) + float(delta_L_h), 0.0)
    D_h_new = max(float(D_h_prev) + float(delta_D_h), 0.0)
    D_f_new = {s: max(float(D_f_prev[s]) + 0.20 * max(delta_L_f[s], 0.0), 0.0) for s in SECTORS}

    L_f_sum = sum(float(L_f_new[s]) for s in SECTORS)
    D_total = float(D_h_new) + sum(float(D_f_new[s]) for s in SECTORS)

    interest_income = float(i_loan) * L_f_sum + float(i_loan) * float(L_h_new) + float(i_gov) * float(B_bank)
    interest_expense = float(i_dep) * D_total
    existing_provisions = float(params.provision_rate) * float(NPL_ratio_prev) * (sum(float(L_f_prev[s]) for s in SECTORS) + float(L_h_prev))
    provisions = float(params.provision_rate) * (npl_new * (L_f_sum + float(L_h_new)) - existing_provisions)
    operating_costs = 0.04 * (L_f_sum + float(L_h_new) + float(B_bank))

    bank_profit = interest_income - interest_expense - provisions - operating_costs
    BANK_EQUITY_new = max(float(BANK_EQUITY_prev) + bank_profit * (1.0 - float(params.dividend_payout_ratio)), 0.0)

    other_assets = max(0.0, D_total - (L_f_sum + float(L_h_new) + float(B_bank)))
    rwa = 1.0 * L_f_sum + 0.75 * float(L_h_new) + 0.0 * float(B_bank) + 0.2 * other_assets
    BANK_CAR = BANK_EQUITY_new / max(rwa, 1e-12)
    BANK_CAR = min(max(BANK_CAR, 0.0), 1.0)

    reserves_at_cb = float(params.capital_adequacy_ratio) * D_total

    return BankingComputation(
        L_f_new=L_f_new,
        D_f_new=D_f_new,
        L_h_new=float(L_h_new),
        D_h_new=float(D_h_new),
        BANK_EQUITY_new=float(BANK_EQUITY_new),
        BANK_CAR=float(BANK_CAR),
        NPL_ratio_new=float(npl_new),
        credit_multiplier=float(credit_multiplier),
        bank_profit=float(bank_profit),
        reserves_at_cb=float(reserves_at_cb),
        rwa=float(rwa),
        delta_L_f=delta_L_f,
    )

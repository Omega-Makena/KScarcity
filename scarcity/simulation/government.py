from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from scarcity.simulation.parameters import GovernmentParams
from scarcity.simulation.types import SECTORS, Sector


@dataclass(frozen=True)
class GovernmentComputation:
    T_rev: float
    T_income: float
    T_corporate: float
    T_vat: float
    T_trade: float
    G_exp: float
    G_inv: float
    G_total: float
    DEFICIT: float
    delta_B_gov: float
    B_gov_new: float
    B_bank_new: float
    B_cb_new: float
    B_foreign_new: float
    K_pub_new: float


def compute_government_block(
    w: Dict[Sector, float],
    N: Dict[Sector, float],
    profits: Dict[Sector, float],
    C: float,
    IM: Dict[Sector, float],
    NGDP: float,
    i_gov: float,
    B_gov_prev: float,
    B_bank_prev: float,
    B_cb_prev: float,
    B_foreign_prev: float,
    fiscal_pressure: float,
    tax_rate_income: float,
    tax_rate_corporate: float,
    tax_rate_vat: float,
    trade_tax_rate: float,
    gov_consumption_ratio: float,
    gov_investment_ratio: float,
    K_pub_prev: float,
    P_mfg: float,
    params: GovernmentParams,
    cb_absorption_share: float = 0.25,
) -> GovernmentComputation:
    """Compute fiscal flows/stocks with the budget and debt accumulation equations."""
    wage_bill = sum(float(w[s]) * float(N[s]) for s in SECTORS)
    T_income = float(tax_rate_income) * wage_bill
    T_corporate = float(tax_rate_corporate) * sum(max(float(profits[s]), 0.0) for s in SECTORS)
    T_vat = float(tax_rate_vat) * max(float(C), 0.0) / (1.0 + float(tax_rate_vat))
    T_trade = float(trade_tax_rate) * sum(max(float(IM[s]), 0.0) for s in SECTORS)
    T_rev = T_income + T_corporate + T_vat + T_trade

    G_planned = float(gov_consumption_ratio) * float(NGDP) + float(gov_investment_ratio) * float(NGDP)
    G_wages = float(params.wage_bill_share) * G_planned
    G_transfers = float(params.transfers_share) * G_planned
    G_interest = float(i_gov) * float(B_gov_prev)
    G_investment = float(params.investment_share) * G_planned
    G_other = float(params.other_recurrent_share) * G_planned

    G_total = G_wages + G_transfers + G_interest + G_investment + G_other + float(fiscal_pressure)
    DEFICIT = G_total - T_rev

    amortization_payments = 0.0
    delta_B_gov = DEFICIT + amortization_payments
    B_gov_new = max(float(B_gov_prev) + delta_B_gov, 0.0)

    delta_B_domestic = float(params.domestic_share_of_debt) * delta_B_gov
    delta_B_external = (1.0 - float(params.domestic_share_of_debt)) * delta_B_gov

    bank_absorption_share = 1.0 - cb_absorption_share
    B_bank_new = max(float(B_bank_prev) + bank_absorption_share * delta_B_domestic, 0.0)
    B_cb_new = max(float(B_cb_prev) + cb_absorption_share * delta_B_domestic, 0.0)
    B_foreign_new = max(float(B_foreign_prev) + delta_B_external, 0.0)

    delta_pub = 0.0125
    K_pub_new = max((1.0 - delta_pub) * float(K_pub_prev) + G_investment / max(float(P_mfg), 1e-12), 0.0)

    return GovernmentComputation(
        T_rev=float(T_rev),
        T_income=float(T_income),
        T_corporate=float(T_corporate),
        T_vat=float(T_vat),
        T_trade=float(T_trade),
        G_exp=float(G_total - G_investment),
        G_inv=float(G_investment),
        G_total=float(G_total),
        DEFICIT=float(DEFICIT),
        delta_B_gov=float(delta_B_gov),
        B_gov_new=float(B_gov_new),
        B_bank_new=float(B_bank_new),
        B_cb_new=float(B_cb_new),
        B_foreign_new=float(B_foreign_new),
        K_pub_new=float(K_pub_new),
    )

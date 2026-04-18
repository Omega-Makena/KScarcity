from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List

from scarcity.simulation.accounting import accounting_warnings, run_accounting_checks
from scarcity.simulation.banking import compute_banking_block
from scarcity.simulation.coupling_interface import AggregatedFeedback, aggregate_feedback
from scarcity.simulation.foreign import compute_foreign_block
from scarcity.simulation.government import compute_government_block
from scarcity.simulation.households import compute_households
from scarcity.simulation.labor_market import compute_labor_market
from scarcity.simulation.monetary import compute_monetary_block
from scarcity.simulation.parameters import AllParams
from scarcity.simulation.price_system import compute_prices_and_profits
from scarcity.simulation.production import compute_gross_output, compute_potential_output, compute_value_added
from scarcity.simulation.types import EconomyState, PolicyState, SECTORS, Sector, SectorFeedback, ShockVector, StepResult

_EPS = 1e-12


@dataclass(frozen=True)
class EngineConfig:
    dt: float = 1.0


def _sector_const(v: float) -> Dict[Sector, float]:
    return {s: float(v) for s in SECTORS}


def _aggregate(feedbacks: List[SectorFeedback] | None) -> AggregatedFeedback:
    if feedbacks is None or len(feedbacks) == 0:
        return AggregatedFeedback.neutral()
    return aggregate_feedback(feedbacks)


def default_initial_state(params: AllParams) -> EconomyState:
    na = params.national_accounts
    gdp_real = float(na.gdp_real_2023)
    labor_force = float(na.labor_force_2023)
    employed = float(na.employment_2023)

    y = {s: gdp_real * float(na.gdp_share[s]) for s in SECTORS}
    y_pot = dict(y)
    k = {s: y[s] * float(params.production.capital_output_ratio[s]) for s in SECTORS}
    n = {s: employed * float(na.employment_share[s]) for s in SECTORS}

    avg_wage = (0.55 * gdp_real) / max(employed, _EPS)
    w = {s: avg_wage for s in SECTORS}

    nominal_gdp = gdp_real
    ex_total = float(params.external.export_gdp_ratio) * nominal_gdp
    im_total = float(params.external.import_gdp_ratio) * nominal_gdp
    ex = {s: ex_total * float(params.external.export_composition[s]) for s in SECTORS}
    im = {s: im_total * float(params.io.import_content[s]) for s in SECTORS}

    b_gov = float(params.government.debt_gdp_ratio_2023) * nominal_gdp
    b_bank = float(params.government.domestic_share_of_debt) * 0.75 * b_gov
    b_cb = float(params.government.domestic_share_of_debt) * 0.25 * b_gov
    b_foreign = max(b_gov - b_bank - b_cb, 0.0)

    rem = float(params.external.remittances_gdp_ratio) * nominal_gdp
    aid = float(params.external.aid_gdp_ratio) * nominal_gdp

    c = 0.70 * nominal_gdp
    s_h = 0.08 * nominal_gdp
    y_disp = c + s_h

    d_h = 0.62 * nominal_gdp
    l_h = 0.30 * nominal_gdp
    l_f = {s: 0.45 * y[s] for s in SECTORS}
    d_f = {s: 0.20 * y[s] for s in SECTORS}

    return EconomyState(
        quarter=0,
        Y=y,
        Y_pot=y_pot,
        K=k,
        K_pub=0.35 * nominal_gdp,
        N=n,
        N_s=labor_force,
        U=float(na.unemployment_rate_2023),
        P=_sector_const(1.0),
        P_cpi=1.0,
        P_imp=1.0,
        E_nom=float(params.external.E_nom_2023),
        E_real=float(params.external.E_nom_2023),
        w=w,
        i_cb=float(params.monetary.i_neutral),
        i_loan=float(params.monetary.i_neutral + params.monetary.spread_loan),
        i_dep=max(0.0, float(params.monetary.i_neutral + params.monetary.spread_deposit)),
        i_gov=float(params.monetary.i_neutral + params.monetary.spread_govt + params.external.sovereign_spread_base),
        D_h=d_h,
        L_h=l_h,
        L_f=l_f,
        D_f=d_f,
        B_gov=b_gov,
        B_cb=b_cb,
        B_bank=b_bank,
        B_foreign=b_foreign,
        RES_fx=8.0,
        T_rev=0.18 * nominal_gdp,
        G_exp=0.20 * nominal_gdp,
        G_inv=0.05 * nominal_gdp,
        DEFICIT=0.05 * nominal_gdp,
        DEBT=b_gov,
        EX=ex,
        IM=im,
        REM=rem,
        AID=aid,
        CA=ex_total - im_total + rem + aid,
        KA=0.0,
        C=c,
        S_h=s_h,
        Y_disp=y_disp,
        GINI=0.40,
        POVERTY=0.32,
        BANK_EQUITY=0.12 * nominal_gdp,
        BANK_CAR=float(params.banking.capital_adequacy_ratio),
        NPL_RATIO=float(params.banking.npl_ratio_2023),
        labor_supply_shock=1.0,
        capital_destruction=_sector_const(0.0),
        productivity_shock=_sector_const(1.0),
        fx_pressure=0.0,
        fiscal_pressure=0.0,
        demand_shift=_sector_const(1.0),
    )


def _make_aggregate_feedback_as_sector_feedback(agg: AggregatedFeedback) -> SectorFeedback:
    return SectorFeedback(
        source="aggregated",
        labor_supply_factor=float(agg.labor_supply_shock),
        labor_productivity_factor={s: float(agg.productivity_shock[s]) for s in SECTORS},
        capital_destruction={s: float(agg.capital_destruction[s]) for s in SECTORS},
        demand_shift={s: float(agg.demand_shift[s]) for s in SECTORS},
        additional_gov_spending=float(agg.fiscal_pressure),
        fx_outflow_pressure=float(agg.fx_pressure),
        trade_disruption={s: float(agg.trade_disruption[s]) for s in SECTORS},
        yield_factor=float(agg.yield_factor),
    )


def _compute_output_gap(y: Dict[Sector, float], y_pot: Dict[Sector, float]) -> float:
    y_sum = sum(float(y[s]) for s in SECTORS)
    y_pot_sum = max(sum(float(y_pot[s]) for s in SECTORS), _EPS)
    return (y_sum - y_pot_sum) / y_pot_sum


def _credit_multiplier_from_car(car: float, params: AllParams) -> float:
    threshold = float(params.banking.credit_rationing_threshold)
    minimum = float(params.banking.min_capital_adequacy)
    if car < threshold:
        return max(0.0, min(1.0, (car - minimum) / max(threshold - minimum, _EPS)))
    return 1.0


def _build_warnings(prev_state: EconomyState, state: EconomyState, accounting: Dict[str, float]) -> List[str]:
    warnings = accounting_warnings(accounting, gdp=max(state.gdp_real, 1.0))

    for rate_name, value in (("i_cb", state.i_cb), ("i_loan", state.i_loan), ("i_dep", state.i_dep), ("i_gov", state.i_gov)):
        if value > 1.0:
            warnings.append(f"{rate_name} exceeds 100% annualized: {value:.4f}")

    scalar_stocks = (("D_h", prev_state.D_h, state.D_h), ("L_h", prev_state.L_h, state.L_h), ("B_gov", prev_state.B_gov, state.B_gov), ("RES_fx", prev_state.RES_fx, state.RES_fx))
    for name, prev_v, cur_v in scalar_stocks:
        change = abs(cur_v - prev_v) / max(abs(prev_v), 1.0)
        if change > 0.50:
            warnings.append(f"{name} changed by >50% in one quarter ({change:.2%})")

    for s in SECTORS:
        change = abs(state.K[s] - prev_state.K[s]) / max(abs(prev_state.K[s]), 1.0)
        if change > 0.50:
            warnings.append(f"K[{s.value}] changed by >50% in one quarter ({change:.2%})")

    return warnings


def step(
    state: EconomyState,
    policy: PolicyState,
    shocks: ShockVector,
    sector_feedback: List[SectorFeedback] | None,
    params: AllParams,
) -> StepResult:
    """Execute one stock-flow consistent simulation quarter in the required order."""

    # 1. Aggregate sector feedback.
    agg = _aggregate(sector_feedback)
    agg_feedback = _make_aggregate_feedback_as_sector_feedback(agg)

    # 2. Labor market.
    expected_output_signal = {
        s: max(float(state.Y[s]) * float(shocks.demand_shock[s]) * float(shocks.supply_shock[s]) * float(agg.demand_shift[s]), _EPS)
        for s in SECTORS
    }
    labor = compute_labor_market(
        Y=expected_output_signal,
        Y_prev=state.Y,
        N_prev=state.N,
        w_prev=state.w,
        P=state.P,
        labor_force_prev=state.N_s,
        labor_force_growth_rate=float(params.national_accounts.labor_force_growth_rate),
        labor_supply_shock=float(agg.labor_supply_shock),
        sigma=params.production.sigma,
        pi_cpi_prev=float(state.pi_cpi),  # lagged CPI inflation stored in state (was hardcoded 0.0)
        U_nairu=0.05,
    )

    # 3. Production.
    y_gross = compute_gross_output(K=state.K, N=labor.N, params=params.production, shocks=shocks, feedback=agg_feedback)
    y = compute_value_added(y_gross, params.io)
    n_pot_total = labor.N_s_total * (1.0 - 0.05)
    n_pot = {s: n_pot_total * float(params.national_accounts.employment_share[s]) for s in SECTORS}
    y_pot = compute_potential_output(K=state.K, N_pot=n_pot, params=params.production, io_params=params.io)
    output_gap = _compute_output_gap(y, y_pot)

    # 4. Prices.
    p_world = max(float(state.P_imp) * float(shocks.world_price_shock), _EPS)
    prices = compute_prices_and_profits(
        P_prev=state.P,
        w=labor.w,
        N=labor.N,
        Y_gross=y_gross,
        E_nom=state.E_nom,
        P_world=p_world,
        io_params=params.io,
        household_params=params.households,
        production_params=params.production,
        L_f=state.L_f,
        K=state.K,
        i_loan=state.i_loan,
    )

    # Quarterly CPI inflation computed here so it can be stored in new_state and used
    # by the labour-market Phillips curve in the *next* step (pi_cpi_prev).
    # Clamped to ±50% per quarter to guard against the first-step price-level scale
    # jump (P_cpi initialised at 1.0 while the price module operates in nominal units).
    # The root cause (price initialisation scale mismatch) should be fixed in
    # default_initial_state once equilibrium prices are bootstrapped properly.
    pi_cpi_new = max(-0.50, min(0.50, prices.P_cpi / max(float(state.P_cpi), _EPS) - 1.0))

    # 5. Monetary policy.
    monetary = compute_monetary_block(
        P_cpi=prices.P_cpi,
        P_cpi_prev=state.P_cpi,
        output_gap=output_gap,
        i_cb_prev=state.i_cb,
        i_target_override=policy.i_target,
        IM=state.IM,
        E_nom=state.E_nom,
        RES_fx=state.RES_fx,
        risk_premium_shock=shocks.risk_premium_shock,
        params=params.monetary,
    )

    # 6. Households.
    nominal_gdp = sum(float(prices.P_new[s]) * float(y[s]) for s in SECTORS)
    rem_h = float(state.REM) * float(shocks.remittance_shock)
    transfers_gov = float(policy.transfer_rate) * float(nominal_gdp)
    dividends = max(sum(float(prices.profits[s]) for s in SECTORS), 0.0)
    hh_credit_multiplier = _credit_multiplier_from_car(state.BANK_CAR, params)

    households = compute_households(
        w=labor.w,
        N=labor.N,
        dividends=dividends,
        rem_h=rem_h,
        transfers_gov=transfers_gov,
        tax_rate_income=policy.tax_rate_income,
        tax_rate_vat=policy.tax_rate_vat,
        demand_shift={s: float(agg.demand_shift[s]) for s in SECTORS},
        D_h_prev=state.D_h,
        L_h_prev=state.L_h,
        U=labor.U,
        U_nairu=0.05,
        poverty_base=state.POVERTY,
        params=params.households,
        banking_params=params.banking,
        credit_multiplier=hh_credit_multiplier,
    )

    # 7. Government.
    government = compute_government_block(
        w=labor.w,
        N=labor.N,
        profits=prices.profits,
        C=households.C,
        IM=state.IM,
        NGDP=nominal_gdp,
        i_gov=monetary.i_gov,
        B_gov_prev=state.B_gov,
        B_bank_prev=state.B_bank,
        B_cb_prev=state.B_cb,
        B_foreign_prev=state.B_foreign,
        fiscal_pressure=float(agg.fiscal_pressure),
        tax_rate_income=policy.tax_rate_income,
        tax_rate_corporate=policy.tax_rate_corporate,
        tax_rate_vat=policy.tax_rate_vat,
        trade_tax_rate=policy.tariff_rate,
        gov_consumption_ratio=policy.gov_consumption_ratio,
        gov_investment_ratio=policy.gov_investment_ratio,
        K_pub_prev=state.K_pub,
        P_mfg=prices.P_new[Sector.MANUFACTURING],
        params=params.government,
    )

    # 8. Trade.
    y_base = {s: float(params.national_accounts.gdp_real_2023) * float(params.national_accounts.gdp_share[s]) for s in SECTORS}
    i_external = float(params.external.us_interest_rate) + float(params.external.sovereign_spread_base) + float(shocks.risk_premium_shock)
    foreign = compute_foreign_block(
        Y=y,
        Y_base=y_base,
        Y_gross=y_gross,
        P=prices.P_new,
        E_nom_prev=state.E_nom,
        P_world=p_world,
        Y_world=1.0 + float(params.external.world_gdp_growth),
        world_demand_shock=float(shocks.world_demand_shock),
        remittance_shock=float(shocks.remittance_shock),
        aid_shock=float(shocks.aid_shock),
        demand_shift={s: float(agg.demand_shift[s]) for s in SECTORS},
        trade_disruption={s: float(agg.trade_disruption[s]) for s in SECTORS},
        fx_pressure_from_coupling=float(agg.fx_pressure),
        i_external=i_external,
        B_foreign_prev=state.B_foreign,
        delta_B_foreign=government.B_foreign_new - state.B_foreign,
        delta_RES_fx_intervention=monetary.delta_res_fx_intervention,
        params=params.external,
        io_params=params.io,
        Y_gdp_nominal=nominal_gdp,
        scale_factor=max(nominal_gdp, 1.0),
    )

    # 9. Banking.
    output_gap_by_sector = {s: (float(y[s]) - float(y_pot[s])) / max(float(y_pot[s]), _EPS) for s in SECTORS}
    banking = compute_banking_block(
        L_f_prev=state.L_f,
        D_f_prev=state.D_f,
        L_h_prev=state.L_h,
        D_h_prev=state.D_h,
        B_bank=government.B_bank_new,
        BANK_EQUITY_prev=state.BANK_EQUITY,
        BANK_CAR_prev=state.BANK_CAR,
        NPL_ratio_prev=state.NPL_RATIO,
        U=labor.U,
        U_nairu=0.05,
        output_gap_by_sector=output_gap_by_sector,
        i_loan=monetary.i_loan,
        i_dep=monetary.i_dep,
        i_gov=monetary.i_gov,
        delta_D_h=households.delta_D_h,
        delta_L_h=households.delta_L_h,
        params=params.banking,
    )

    # Enforce bank balance-sheet closure by reconciling firm deposits to assets-liabilities.
    bank_assets = (
        sum(float(banking.L_f_new[s]) for s in SECTORS)
        + float(banking.L_h_new)
        + float(government.B_bank_new)
        + float(banking.reserves_at_cb)
    )
    target_df_total = max(bank_assets - float(banking.D_h_new) - float(banking.BANK_EQUITY_new), 0.0)
    prior_df_total = max(sum(float(banking.D_f_new[s]) for s in SECTORS), _EPS)
    d_f_reconciled = {
        s: target_df_total * float(banking.D_f_new[s]) / prior_df_total
        for s in SECTORS
    }

    # 10. Exchange rate and reserves from foreign block.

    # 11. Capital accumulation.
    accelerator = 0.3
    investment_adjustment = 0.25
    p_mfg = max(float(prices.P_new[Sector.MANUFACTURING]), _EPS)

    investment_by_sector: Dict[Sector, float] = {}
    k_new: Dict[Sector, float] = {}
    for s in SECTORS:
        delta_s = float(params.production.delta[s])
        k_prev = float(state.K[s])
        i_target = delta_s * k_prev + accelerator * (float(y[s]) - float(state.Y[s]))
        i_prev = delta_s * k_prev
        i_s = i_prev + investment_adjustment * (i_target - i_prev)

        retained_earnings_s = max(float(prices.profits[s]), 0.0) * (1.0 - float(params.banking.dividend_payout_ratio))
        financing_limit = retained_earnings_s + float(banking.delta_L_f[s]) / p_mfg
        i_s = min(i_s, max(financing_limit, 0.0))
        i_s = max(i_s, 0.0)

        destruction_s = min(max(float(agg.capital_destruction[s]), 0.0), 1.0)
        k_new[s] = max((1.0 - delta_s - destruction_s) * k_prev + i_s, 0.0)
        investment_by_sector[s] = i_s

    # 12. Accounting checks.
    imports_total_by_sector = {s: float(foreign.imports[s]) + float(foreign.imports_intermediate[s]) for s in SECTORS}
    i_total_identity = (
        sum(float(y[s]) for s in SECTORS)
        - float(households.C)
        - float(government.G_total)
        - (sum(float(foreign.exports[s]) for s in SECTORS) - sum(float(imports_total_by_sector[s]) for s in SECTORS))
    )
    delta_res_identity = (float(foreign.CA) + float(foreign.KA)) / max(float(foreign.E_nom_new), _EPS)
    flows: Dict[str, float | Dict[Sector, float]] = {
        "I_total": float(i_total_identity),
        "delta_D_h": float(banking.D_h_new - state.D_h),
        "delta_L_h": float(banking.L_h_new - state.L_h),
        "delta_B_gov": float(government.delta_B_gov),
        "bank_reserves_at_cb": float(banking.reserves_at_cb),
        "delta_res_fx": float(delta_res_identity),
        "excess_demand": {
            s: float(params.national_accounts.gdp_share[s]) * nominal_gdp - float(prices.P_new[s]) * float(y[s])
            for s in SECTORS
        },
        "investment_by_sector": investment_by_sector,
    }
    for s in SECTORS:
        flows[f"delta_{s.value}"] = float(params.production.delta[s])

    # 13. Assemble new state.
    new_state = EconomyState(
        quarter=state.quarter + 1,
        Y=y,
        Y_pot=y_pot,
        K=k_new,
        K_pub=government.K_pub_new,
        N=labor.N,
        N_s=labor.N_s_total,
        U=labor.U,
        P=prices.P_new,
        P_cpi=prices.P_cpi,
        P_imp=p_world,
        E_nom=foreign.E_nom_new,
        E_real=foreign.E_real_new,
        w=labor.w,
        i_cb=monetary.i_cb,
        i_loan=monetary.i_loan,
        i_dep=monetary.i_dep,
        i_gov=monetary.i_gov,
        D_h=banking.D_h_new,
        L_h=banking.L_h_new,
        L_f=banking.L_f_new,
        D_f=d_f_reconciled,
        B_gov=government.B_gov_new,
        B_cb=government.B_cb_new,
        B_bank=government.B_bank_new,
        B_foreign=government.B_foreign_new,
        RES_fx=max(state.RES_fx + delta_res_identity, 0.0),
        T_rev=government.T_rev,
        G_exp=government.G_total,
        G_inv=government.G_inv,
        DEFICIT=government.DEFICIT,
        DEBT=government.B_gov_new,
        EX=foreign.exports,
        IM=imports_total_by_sector,
        REM=foreign.remittances,
        AID=foreign.aid,
        CA=foreign.CA,
        KA=foreign.KA,
        C=households.C,
        S_h=households.S_h,
        Y_disp=households.Y_disp,
        GINI=households.GINI,
        POVERTY=households.POVERTY,
        BANK_EQUITY=banking.BANK_EQUITY_new,
        BANK_CAR=banking.BANK_CAR,
        NPL_RATIO=banking.NPL_ratio_new,
        labor_supply_shock=float(agg.labor_supply_shock),
        capital_destruction={s: float(agg.capital_destruction[s]) for s in SECTORS},
        productivity_shock={s: float(agg.productivity_shock[s]) for s in SECTORS},
        fx_pressure=float(agg.fx_pressure),
        fiscal_pressure=float(agg.fiscal_pressure),
        demand_shift={s: float(agg.demand_shift[s]) for s in SECTORS},
        pi_cpi=float(pi_cpi_new),
    )

    accounting = run_accounting_checks(state, new_state, flows)
    warnings = _build_warnings(state, new_state, accounting)

    return StepResult(
        state=new_state,
        flows={k: v for k, v in flows.items() if isinstance(v, float)},
        accounting_errors=accounting,
        warnings=warnings,
    )


def _max_abs_state_delta(prev: EconomyState, cur: EconomyState) -> float:
    max_delta = 0.0
    dict_fields = ("Y", "Y_pot", "K", "N", "P", "w", "L_f", "D_f", "EX", "IM", "capital_destruction", "productivity_shock", "demand_shift")
    for name in dict_fields:
        prev_dict = getattr(prev, name)
        cur_dict = getattr(cur, name)
        for s in SECTORS:
            max_delta = max(max_delta, abs(float(cur_dict[s]) - float(prev_dict[s])))

    scalar_fields = (
        "K_pub", "N_s", "U", "P_cpi", "P_imp", "E_nom", "E_real", "i_cb", "i_loan", "i_dep", "i_gov", "D_h", "L_h",
        "B_gov", "B_cb", "B_bank", "B_foreign", "RES_fx", "T_rev", "G_exp", "G_inv", "DEFICIT", "DEBT", "REM", "AID", "CA",
        "KA", "C", "S_h", "Y_disp", "GINI", "POVERTY", "BANK_EQUITY", "BANK_CAR", "NPL_RATIO", "labor_supply_shock", "fx_pressure", "fiscal_pressure",
    )
    for name in scalar_fields:
        max_delta = max(max_delta, abs(float(getattr(cur, name)) - float(getattr(prev, name))))
    return max_delta


def _calibrate_tfp_levels(params: AllParams, state: EconomyState) -> AllParams:
    """Return a deep copy of params with TFP levels (A) calibrated to hit Kenya GDP targets.

    The original code mutated params in-place, causing persistent side-effects on the
    caller's AllParams object and producing different results on repeated calls.
    We now work on an isolated copy and return it, leaving the caller's params unchanged.
    """
    params = copy.deepcopy(params)
    target_y = {s: float(params.national_accounts.gdp_real_2023) * float(params.national_accounts.gdp_share[s]) for s in SECTORS}
    neutral = ShockVector.neutral()
    for _ in range(6):
        y_gross = compute_gross_output(state.K, state.N, params.production, neutral, None)
        y_va = compute_value_added(y_gross, params.io)
        for s in SECTORS:
            current = max(float(y_va[s]), _EPS)
            params.production.A[s] = max(float(params.production.A[s]) * (target_y[s] / current), _EPS)
    return params


def _blend_states(prev: EconomyState, cur: EconomyState, weight: float) -> EconomyState:
    w = min(max(float(weight), 0.0), 1.0)
    one_minus = 1.0 - w

    def blend_dict(name: str) -> Dict[Sector, float]:
        pd = getattr(prev, name)
        cd = getattr(cur, name)
        return {s: one_minus * float(pd[s]) + w * float(cd[s]) for s in SECTORS}

    return EconomyState(
        quarter=cur.quarter,
        Y=blend_dict("Y"),
        Y_pot=blend_dict("Y_pot"),
        K=blend_dict("K"),
        K_pub=one_minus * prev.K_pub + w * cur.K_pub,
        N=blend_dict("N"),
        N_s=one_minus * prev.N_s + w * cur.N_s,
        U=min(max(one_minus * prev.U + w * cur.U, 0.0), 1.0),
        P=blend_dict("P"),
        P_cpi=max(one_minus * prev.P_cpi + w * cur.P_cpi, _EPS),
        P_imp=max(one_minus * prev.P_imp + w * cur.P_imp, _EPS),
        E_nom=max(one_minus * prev.E_nom + w * cur.E_nom, _EPS),
        E_real=max(one_minus * prev.E_real + w * cur.E_real, _EPS),
        w=blend_dict("w"),
        i_cb=min(max(one_minus * prev.i_cb + w * cur.i_cb, 0.0), 1.0),
        i_loan=min(max(one_minus * prev.i_loan + w * cur.i_loan, 0.0), 1.0),
        i_dep=min(max(one_minus * prev.i_dep + w * cur.i_dep, 0.0), 1.0),
        i_gov=min(max(one_minus * prev.i_gov + w * cur.i_gov, 0.0), 1.0),
        D_h=max(one_minus * prev.D_h + w * cur.D_h, 0.0),
        L_h=max(one_minus * prev.L_h + w * cur.L_h, 0.0),
        L_f=blend_dict("L_f"),
        D_f=blend_dict("D_f"),
        B_gov=max(one_minus * prev.B_gov + w * cur.B_gov, 0.0),
        B_cb=max(one_minus * prev.B_cb + w * cur.B_cb, 0.0),
        B_bank=max(one_minus * prev.B_bank + w * cur.B_bank, 0.0),
        B_foreign=max(one_minus * prev.B_foreign + w * cur.B_foreign, 0.0),
        RES_fx=max(one_minus * prev.RES_fx + w * cur.RES_fx, 0.0),
        T_rev=max(one_minus * prev.T_rev + w * cur.T_rev, 0.0),
        G_exp=max(one_minus * prev.G_exp + w * cur.G_exp, 0.0),
        G_inv=max(one_minus * prev.G_inv + w * cur.G_inv, 0.0),
        DEFICIT=one_minus * prev.DEFICIT + w * cur.DEFICIT,
        DEBT=max(one_minus * prev.DEBT + w * cur.DEBT, 0.0),
        EX=blend_dict("EX"),
        IM=blend_dict("IM"),
        REM=max(one_minus * prev.REM + w * cur.REM, 0.0),
        AID=max(one_minus * prev.AID + w * cur.AID, 0.0),
        CA=one_minus * prev.CA + w * cur.CA,
        KA=one_minus * prev.KA + w * cur.KA,
        C=max(one_minus * prev.C + w * cur.C, 0.0),
        S_h=one_minus * prev.S_h + w * cur.S_h,
        Y_disp=one_minus * prev.Y_disp + w * cur.Y_disp,
        GINI=min(max(one_minus * prev.GINI + w * cur.GINI, 0.0), 1.0),
        POVERTY=min(max(one_minus * prev.POVERTY + w * cur.POVERTY, 0.0), 1.0),
        BANK_EQUITY=max(one_minus * prev.BANK_EQUITY + w * cur.BANK_EQUITY, 0.0),
        BANK_CAR=min(max(one_minus * prev.BANK_CAR + w * cur.BANK_CAR, 0.0), 1.0),
        NPL_RATIO=min(max(one_minus * prev.NPL_RATIO + w * cur.NPL_RATIO, 0.0), 1.0),
        labor_supply_shock=max(one_minus * prev.labor_supply_shock + w * cur.labor_supply_shock, _EPS),
        capital_destruction=blend_dict("capital_destruction"),
        productivity_shock=blend_dict("productivity_shock"),
        fx_pressure=one_minus * prev.fx_pressure + w * cur.fx_pressure,
        fiscal_pressure=one_minus * prev.fiscal_pressure + w * cur.fiscal_pressure,
        demand_shift=blend_dict("demand_shift"),
        pi_cpi=one_minus * prev.pi_cpi + w * cur.pi_cpi,
    )


def find_steady_state(
    params: AllParams,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> tuple[EconomyState, int]:
    """Iterate with neutral shocks until max absolute change < tol for 10 consecutive quarters.

    Returns (steady_state, iterations_run).  The caller's ``params`` object is never
    mutated — TFP calibration is performed on an internal copy.
    """
    state = default_initial_state(params)
    calibrated_params = _calibrate_tfp_levels(params, state)

    policy = PolicyState.default()
    neutral = ShockVector.neutral()
    consecutive = 0

    for iteration in range(max_iter):
        result = step(state, policy, neutral, None, calibrated_params)
        new_state = _blend_states(state, result.state, 0.2)
        max_delta = _max_abs_state_delta(state, new_state)
        if max_delta < tol:
            consecutive += 1
        else:
            consecutive = 0

        state = new_state
        if consecutive >= 10:
            return state, iteration + 1

    return state, max_iter


class MultiSectorSFCEngine:
    """Compatibility wrapper around module-level step/find_steady_state functions."""

    def __init__(
        self,
        params: AllParams | None = None,
        policy: PolicyState | None = None,
        initial_state: EconomyState | None = None,
        config: EngineConfig | None = None,
    ):
        self.params = params or AllParams.default_kenya()
        self.policy = policy or PolicyState.default()
        self.config = config or EngineConfig()
        self.state = initial_state or default_initial_state(self.params)

    def step(self, shock: ShockVector | None = None, feedback: Iterable[SectorFeedback] | None = None) -> StepResult:
        result = step(
            state=self.state,
            policy=self.policy,
            shocks=shock or ShockVector.neutral(),
            sector_feedback=list(feedback) if feedback is not None else None,
            params=self.params,
        )
        self.state = result.state
        return result

    def simulate(self, quarters: int, shocks: list[ShockVector] | None = None, feedbacks: list[list[SectorFeedback]] | None = None) -> list[StepResult]:
        results: list[StepResult] = []
        for t in range(quarters):
            shock_t = shocks[t] if shocks is not None and t < len(shocks) else ShockVector.neutral()
            feedback_t = feedbacks[t] if feedbacks is not None and t < len(feedbacks) else None
            results.append(self.step(shock_t, feedback_t))
        return results

    def find_steady_state(self, max_iter: int | None = None, tol: float | None = None) -> tuple[EconomyState, int]:
        kwargs: dict = {}
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        if tol is not None:
            kwargs["tol"] = tol
        self.state, iterations = find_steady_state(self.params, **kwargs)
        return self.state, iterations

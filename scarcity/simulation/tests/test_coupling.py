from __future__ import annotations

from scarcity.simulation.coupling_interface import MacroExposure, aggregate_feedback
from scarcity.simulation.parameters import AllParams
from scarcity.simulation.sfc_engine import default_initial_state, step
from scarcity.simulation.types import EconomyState, PolicyState, SECTORS, Sector, SectorFeedback, ShockVector


def _sector(v: float) -> dict[Sector, float]:
    return {s: v for s in SECTORS}


def _make_state() -> EconomyState:
    y = _sector(100.0)
    return EconomyState(
        quarter=0,
        Y=y,
        Y_pot=y,
        K=_sector(200.0),
        K_pub=500.0,
        N=_sector(5.0),
        N_s=20.0,
        U=0.05,
        P=_sector(1.0),
        P_cpi=1.0,
        P_imp=1.0,
        E_nom=140.0,
        E_real=1.0,
        w=_sector(10.0),
        i_cb=0.02,
        i_loan=0.03,
        i_dep=0.01,
        i_gov=0.025,
        D_h=100.0,
        L_h=50.0,
        L_f=_sector(30.0),
        D_f=_sector(30.0),
        B_gov=300.0,
        B_cb=100.0,
        B_bank=120.0,
        B_foreign=80.0,
        RES_fx=8.0,
        T_rev=100.0,
        G_exp=110.0,
        G_inv=20.0,
        DEFICIT=10.0,
        DEBT=500.0,
        EX=_sector(20.0),
        IM=_sector(15.0),
        REM=10.0,
        AID=5.0,
        CA=2.0,
        KA=1.0,
        C=200.0,
        S_h=20.0,
        Y_disp=220.0,
        GINI=0.40,
        POVERTY=0.28,
        BANK_EQUITY=50.0,
        BANK_CAR=0.17,
        NPL_RATIO=0.12,
        labor_supply_shock=1.0,
        capital_destruction=_sector(0.0),
        productivity_shock=_sector(1.0),
        fx_pressure=0.0,
        fiscal_pressure=0.0,
        demand_shift=_sector(1.0),
    )


def test_neutral_feedback_aggregation_is_identity() -> None:
    agg = aggregate_feedback([SectorFeedback(source="neutral")])

    assert agg.labor_supply_shock == 1.0
    assert all(agg.productivity_shock[s] == 1.0 for s in SECTORS)
    assert all(agg.capital_destruction[s] == 0.0 for s in SECTORS)
    assert all(agg.demand_shift[s] == 1.0 for s in SECTORS)
    assert agg.fx_pressure == 0.0
    assert agg.fiscal_pressure == 0.0


def test_neutral_feedback_produces_no_change_vs_none() -> None:
    params = AllParams.default_kenya()
    state0 = default_initial_state(params)

    result_none = step(state0, PolicyState.default(), ShockVector.neutral(), None, params)
    result_neutral = step(
        state0,
        PolicyState.default(),
        ShockVector.neutral(),
        [SectorFeedback(source="neutral")],
        params,
    )

    assert abs(result_none.state.gdp_real - result_neutral.state.gdp_real) < 1e-9


def test_multiplicative_and_additive_feedback_combination() -> None:
    f1 = SectorFeedback(
        source="health",
        labor_supply_factor=0.95,
        demand_shift={
            Sector.AGRICULTURE: 0.90,
            Sector.MANUFACTURING: 1.00,
            Sector.SERVICES: 0.95,
            Sector.INFORMAL: 0.92,
        },
        additional_gov_spending=10.0,
    )
    f2 = SectorFeedback(
        source="security",
        labor_supply_factor=0.97,
        demand_shift={
            Sector.AGRICULTURE: 0.80,
            Sector.MANUFACTURING: 0.98,
            Sector.SERVICES: 0.99,
            Sector.INFORMAL: 0.97,
        },
        additional_gov_spending=15.0,
    )

    agg = aggregate_feedback([f1, f2])

    assert abs(agg.labor_supply_shock - (0.95 * 0.97)) < 1e-12
    assert abs(agg.demand_shift[Sector.AGRICULTURE] - (0.90 * 0.80)) < 1e-12
    assert abs(agg.fiscal_pressure - 25.0) < 1e-12


def test_labor_supply_shock_reduces_gdp() -> None:
    params = AllParams.default_kenya()
    state0 = default_initial_state(params)

    baseline = step(state0, PolicyState.default(), ShockVector.neutral(), None, params).state.gdp_real
    shocked = step(
        state0,
        PolicyState.default(),
        ShockVector.neutral(),
        [SectorFeedback(source="health", labor_supply_factor=0.9)],
        params,
    ).state.gdp_real

    assert shocked < baseline


def test_capital_destruction_combines_with_survival_product_rule() -> None:
    f1 = SectorFeedback(source="flood", capital_destruction=_sector(0.10))
    f2 = SectorFeedback(source="conflict", capital_destruction=_sector(0.05))

    agg = aggregate_feedback([f1, f2])
    expected = 1.0 - (1.0 - 0.10) * (1.0 - 0.05)

    for s in SECTORS:
        assert abs(agg.capital_destruction[s] - expected) < 1e-12


def test_macro_exposure_extracts_expected_fields() -> None:
    state = _make_state()
    exposure = MacroExposure.from_state(state)

    assert exposure.gdp_real == 400.0
    assert exposure.output_gap == 0.0
    assert exposure.unemployment_rate == state.U
    assert exposure.food_price_index == state.P[Sector.AGRICULTURE]
    assert exposure.exchange_rate == state.E_nom
    assert exposure.interest_rate == state.i_cb
    assert exposure.sector_output == state.Y
    assert exposure.sector_employment == state.N

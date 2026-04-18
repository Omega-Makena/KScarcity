from __future__ import annotations

from scarcity.simulation.accounting import accounting_warnings, run_accounting_checks
from scarcity.simulation.types import EconomyState, SECTORS, Sector


def _sector(v: float) -> dict[Sector, float]:
    return {s: v for s in SECTORS}


def _build_states() -> tuple[EconomyState, EconomyState, dict[str, float | dict[Sector, float]]]:
    d_f = {
        Sector.AGRICULTURE: 4.0,
        Sector.MANUFACTURING: 4.0,
        Sector.SERVICES: 4.0,
        Sector.INFORMAL: 4.0,
    }
    prev = EconomyState(
        quarter=0,
        Y=_sector(80.0),
        Y_pot=_sector(80.0),
        K=_sector(100.0),
        K_pub=400.0,
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
        L_h=20.0,
        L_f=_sector(20.0),
        D_f=d_f,
        B_gov=100.0,
        B_cb=20.0,
        B_bank=50.0,
        B_foreign=30.0,
        RES_fx=5.0,
        T_rev=75.0,
        G_exp=80.0,
        G_inv=15.0,
        DEFICIT=5.0,
        DEBT=100.0,
        EX={
            Sector.AGRICULTURE: 20.0,
            Sector.MANUFACTURING: 20.0,
            Sector.SERVICES: 20.0,
            Sector.INFORMAL: 20.0,
        },
        IM={
            Sector.AGRICULTURE: 10.0,
            Sector.MANUFACTURING: 10.0,
            Sector.SERVICES: 10.0,
            Sector.INFORMAL: 10.0,
        },
        REM=5.0,
        AID=0.0,
        CA=10.0,
        KA=130.0,
        C=100.0,
        S_h=8.0,
        Y_disp=108.0,
        GINI=0.40,
        POVERTY=0.28,
        BANK_EQUITY=42.5,
        BANK_CAR=0.17,
        NPL_RATIO=0.12,
        labor_supply_shock=1.0,
        capital_destruction=_sector(0.0),
        productivity_shock=_sector(1.0),
        fx_pressure=0.0,
        fiscal_pressure=0.0,
        demand_shift=_sector(1.0),
    )

    state = EconomyState(
        quarter=1,
        Y=_sector(80.0),
        Y_pot=_sector(80.0),
        K=_sector(100.0),
        K_pub=405.0,
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
        D_h=108.0,
        L_h=20.0,
        L_f=_sector(20.0),
        D_f=d_f,
        B_gov=105.0,
        B_cb=22.5,
        B_bank=52.5,
        B_foreign=30.0,
        RES_fx=6.0,
        T_rev=75.0,
        G_exp=80.0,
        G_inv=15.0,
        DEFICIT=5.0,
        DEBT=105.0,
        EX={
            Sector.AGRICULTURE: 20.0,
            Sector.MANUFACTURING: 20.0,
            Sector.SERVICES: 20.0,
            Sector.INFORMAL: 20.0,
        },
        IM={
            Sector.AGRICULTURE: 10.0,
            Sector.MANUFACTURING: 10.0,
            Sector.SERVICES: 10.0,
            Sector.INFORMAL: 10.0,
        },
        REM=5.0,
        AID=0.0,
        CA=10.0,
        KA=130.0,
        C=100.0,
        S_h=8.0,
        Y_disp=108.0,
        GINI=0.40,
        POVERTY=0.28,
        BANK_EQUITY=42.5,
        BANK_CAR=0.17,
        NPL_RATIO=0.12,
        labor_supply_shock=1.0,
        capital_destruction=_sector(0.0),
        productivity_shock=_sector(1.0),
        fx_pressure=0.0,
        fiscal_pressure=0.0,
        demand_shift=_sector(1.0),
    )

    flows: dict[str, float | dict[Sector, float]] = {
        "I_total": 100.0,
        "delta_D_h": 8.0,
        "delta_L_h": 0.0,
        "delta_B_gov": 5.0,
        "bank_reserves_at_cb": 14.0,
        "delta_res_fx": 1.0,
        "excess_demand": _sector(0.0),
        "investment_by_sector": _sector(1.5),
        "delta_agri": 0.015,
        "delta_mfg": 0.015,
        "delta_svc": 0.015,
        "delta_inf": 0.015,
    }
    return prev, state, flows


def test_accounting_checks_residuals_near_zero_for_consistent_state() -> None:
    prev, state, flows = _build_states()
    residuals = run_accounting_checks(prev, state, flows)

    assert all(abs(v) < 1e-10 for v in residuals.values())
    warnings = accounting_warnings(residuals, gdp=state.gdp_real)
    assert warnings == []


def test_accounting_checks_detect_inconsistency() -> None:
    prev, state, flows = _build_states()
    flows["delta_B_gov"] = 7.5

    residuals = run_accounting_checks(prev, state, flows)
    assert abs(residuals["residual_3"]) > 0.0

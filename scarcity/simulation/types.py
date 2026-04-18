from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Sector(Enum):
    """Production sectors for the multi-sector macro block."""

    AGRICULTURE = "agri"
    MANUFACTURING = "mfg"
    SERVICES = "svc"
    INFORMAL = "inf"


SECTORS: tuple[Sector, ...] = (
    Sector.AGRICULTURE,
    Sector.MANUFACTURING,
    Sector.SERVICES,
    Sector.INFORMAL,
)


SectorFloatDict = Dict[Sector, float]


@dataclass(frozen=True)
class EconomyState:
    # Time
    quarter: int

    # Output (real)
    Y: SectorFloatDict
    Y_pot: SectorFloatDict

    # Capital stocks (real)
    K: SectorFloatDict
    K_pub: float

    # Labor
    N: SectorFloatDict
    N_s: float
    U: float

    # Prices
    P: SectorFloatDict
    P_cpi: float
    P_imp: float
    E_nom: float
    E_real: float
    w: SectorFloatDict

    # Interest rates
    i_cb: float
    i_loan: float
    i_dep: float
    i_gov: float

    # Financial stocks
    D_h: float
    L_h: float
    L_f: SectorFloatDict
    D_f: SectorFloatDict
    B_gov: float
    B_cb: float
    B_bank: float
    B_foreign: float
    RES_fx: float

    # Government
    T_rev: float
    G_exp: float
    G_inv: float
    DEFICIT: float
    DEBT: float

    # External
    EX: SectorFloatDict
    IM: SectorFloatDict
    REM: float
    AID: float
    CA: float
    KA: float

    # Household
    C: float
    S_h: float
    Y_disp: float
    GINI: float
    POVERTY: float

    # Banking
    BANK_EQUITY: float
    BANK_CAR: float
    NPL_RATIO: float

    # Coupling interface fields
    labor_supply_shock: float
    capital_destruction: SectorFloatDict
    productivity_shock: SectorFloatDict
    fx_pressure: float
    fiscal_pressure: float
    demand_shift: SectorFloatDict

    # Lagged inflation rate (quarterly, annualized fraction) used by the
    # labour-market Phillips curve in the *next* step.  Default 0.0 so
    # existing callsites that omit it keep working.
    pi_cpi: float = 0.0

    def __post_init__(self) -> None:
        self._validate_sector_dict(self.Y, "Y")
        self._validate_sector_dict(self.Y_pot, "Y_pot")
        self._validate_sector_dict(self.K, "K")
        self._validate_sector_dict(self.N, "N")
        self._validate_sector_dict(self.P, "P")
        self._validate_sector_dict(self.w, "w")
        self._validate_sector_dict(self.L_f, "L_f")
        self._validate_sector_dict(self.D_f, "D_f")
        self._validate_sector_dict(self.EX, "EX")
        self._validate_sector_dict(self.IM, "IM")
        self._validate_sector_dict(self.capital_destruction, "capital_destruction")
        self._validate_sector_dict(self.productivity_shock, "productivity_shock")
        self._validate_sector_dict(self.demand_shift, "demand_shift")

        if self.quarter < 0:
            raise ValueError("quarter must be non-negative")
        if not (0.0 <= self.U <= 1.0):
            raise ValueError("U must be in [0, 1]")
        if not (0.0 <= self.GINI <= 1.0):
            raise ValueError("GINI must be in [0, 1]")
        if not (0.0 <= self.POVERTY <= 1.0):
            raise ValueError("POVERTY must be in [0, 1]")
        if not (0.0 <= self.BANK_CAR <= 1.0):
            raise ValueError("BANK_CAR must be in [0, 1]")
        if not (0.0 <= self.NPL_RATIO <= 1.0):
            raise ValueError("NPL_RATIO must be in [0, 1]")

        if self.N_s < 0.0:
            raise ValueError("N_s must be non-negative")
        if self.P_cpi <= 0.0:
            raise ValueError("P_cpi must be positive")
        if self.P_imp <= 0.0:
            raise ValueError("P_imp must be positive")
        if self.E_nom <= 0.0:
            raise ValueError("E_nom must be positive")
        if self.E_real <= 0.0:
            raise ValueError("E_real must be positive")

    @staticmethod
    def _validate_sector_dict(data: SectorFloatDict, name: str) -> None:
        missing = [s for s in SECTORS if s not in data]
        if missing:
            raise ValueError(f"{name} missing sectors: {missing}")

    @property
    def gdp_real(self) -> float:
        return float(sum(self.Y[s] for s in SECTORS))

    @property
    def gdp_potential(self) -> float:
        return float(sum(self.Y_pot[s] for s in SECTORS))


@dataclass(frozen=True)
class PolicyState:
    # Monetary
    i_target: Optional[float]
    reserve_requirement: float

    # Fiscal
    tax_rate_vat: float
    tax_rate_income: float
    tax_rate_corporate: float
    gov_consumption_ratio: float
    gov_investment_ratio: float
    transfer_rate: float

    # Trade
    tariff_rate: float
    export_subsidy: float
    capital_controls: float

    def __post_init__(self) -> None:
        for field_name in (
            "reserve_requirement",
            "tax_rate_vat",
            "tax_rate_income",
            "tax_rate_corporate",
            "gov_consumption_ratio",
            "gov_investment_ratio",
            "transfer_rate",
            "tariff_rate",
            "export_subsidy",
            "capital_controls",
        ):
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1]")

    @staticmethod
    def default() -> PolicyState:
        return PolicyState(
            i_target=None,
            reserve_requirement=0.045,
            tax_rate_vat=0.16,
            tax_rate_income=0.12,
            tax_rate_corporate=0.30,
            gov_consumption_ratio=0.13,
            gov_investment_ratio=0.05,
            transfer_rate=0.02,
            tariff_rate=0.12,
            export_subsidy=0.0,
            capital_controls=0.1,
        )


@dataclass(frozen=True)
class ShockVector:
    """Exogenous shocks applied for one simulation quarter."""

    demand_shock: SectorFloatDict
    supply_shock: SectorFloatDict
    world_price_shock: float
    world_demand_shock: float
    remittance_shock: float
    aid_shock: float
    risk_premium_shock: float
    rainfall_shock: float

    def __post_init__(self) -> None:
        EconomyState._validate_sector_dict(self.demand_shock, "demand_shock")
        EconomyState._validate_sector_dict(self.supply_shock, "supply_shock")
        if self.world_price_shock <= 0.0:
            raise ValueError("world_price_shock must be positive")
        if self.world_demand_shock <= 0.0:
            raise ValueError("world_demand_shock must be positive")
        if self.remittance_shock <= 0.0:
            raise ValueError("remittance_shock must be positive")
        if self.aid_shock <= 0.0:
            raise ValueError("aid_shock must be positive")
        if self.rainfall_shock <= 0.0:
            raise ValueError("rainfall_shock must be positive")

    @staticmethod
    def neutral() -> ShockVector:
        sectors = {s: 1.0 for s in SECTORS}
        return ShockVector(
            demand_shock=sectors,
            supply_shock=sectors,
            world_price_shock=1.0,
            world_demand_shock=1.0,
            remittance_shock=1.0,
            aid_shock=1.0,
            risk_premium_shock=0.0,
            rainfall_shock=1.0,
        )


@dataclass(frozen=True)
class StepResult:
    state: EconomyState
    flows: Dict[str, float]
    accounting_errors: Dict[str, float]
    warnings: List[str]


@dataclass(frozen=True)
class SectorFeedback:
    """Feedback emitted by one external sector model."""

    source: str

    # Labor market effects
    labor_supply_factor: float = 1.0
    labor_productivity_factor: Optional[SectorFloatDict] = None

    # Capital stock effects
    capital_destruction: Optional[SectorFloatDict] = None

    # Demand effects
    demand_shift: Optional[SectorFloatDict] = None

    # Fiscal effects
    additional_gov_spending: float = 0.0

    # External effects
    fx_outflow_pressure: float = 0.0
    trade_disruption: Optional[SectorFloatDict] = None

    # Agricultural effects
    yield_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.labor_supply_factor <= 0.0:
            raise ValueError("labor_supply_factor must be positive")
        if self.yield_factor <= 0.0:
            raise ValueError("yield_factor must be positive")

        if self.labor_productivity_factor is not None:
            EconomyState._validate_sector_dict(
                self.labor_productivity_factor,
                "labor_productivity_factor",
            )

        if self.capital_destruction is not None:
            EconomyState._validate_sector_dict(
                self.capital_destruction,
                "capital_destruction",
            )
            for value in self.capital_destruction.values():
                if not (0.0 <= value <= 1.0):
                    raise ValueError("capital_destruction values must be in [0, 1]")

        if self.demand_shift is not None:
            EconomyState._validate_sector_dict(self.demand_shift, "demand_shift")
            for value in self.demand_shift.values():
                if value <= 0.0:
                    raise ValueError("demand_shift values must be positive")

        if self.trade_disruption is not None:
            EconomyState._validate_sector_dict(self.trade_disruption, "trade_disruption")
            for value in self.trade_disruption.values():
                if value <= 0.0:
                    raise ValueError("trade_disruption values must be positive")

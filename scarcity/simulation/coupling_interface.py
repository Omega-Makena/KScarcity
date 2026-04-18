from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scarcity.simulation.types import EconomyState, SECTORS, Sector, SectorFeedback


@dataclass(frozen=True)
class AggregatedFeedback:
    labor_supply_shock: float
    productivity_shock: dict[Sector, float]
    capital_destruction: dict[Sector, float]
    demand_shift: dict[Sector, float]
    trade_disruption: dict[Sector, float]
    fx_pressure: float
    fiscal_pressure: float
    yield_factor: float

    @staticmethod
    def neutral() -> AggregatedFeedback:
        unit = {s: 1.0 for s in SECTORS}
        zero = {s: 0.0 for s in SECTORS}
        return AggregatedFeedback(
            labor_supply_shock=1.0,
            productivity_shock=unit,
            capital_destruction=zero,
            demand_shift=unit,
            trade_disruption=unit,
            fx_pressure=0.0,
            fiscal_pressure=0.0,
            yield_factor=1.0,
        )


def aggregate_feedback(feedbacks: list[SectorFeedback]) -> AggregatedFeedback:
    """Aggregate feedback from multiple sector models.

    Rules:
    - Multiplicative factors are multiplied across sources.
    - Additive factors are summed across sources.
    - Capital destruction combines as 1 - product(1 - d_i).
    """

    agg = AggregatedFeedback.neutral()

    labor_supply = agg.labor_supply_shock
    productivity = dict(agg.productivity_shock)
    destruction_survival = {s: 1.0 for s in SECTORS}
    demand_shift = dict(agg.demand_shift)
    trade_disruption = dict(agg.trade_disruption)
    fx_pressure = agg.fx_pressure
    fiscal_pressure = agg.fiscal_pressure
    yield_factor = agg.yield_factor

    for fb in feedbacks:
        labor_supply *= fb.labor_supply_factor

        if fb.labor_productivity_factor is not None:
            for sector in SECTORS:
                productivity[sector] *= fb.labor_productivity_factor[sector]

        if fb.capital_destruction is not None:
            for sector in SECTORS:
                destruction_survival[sector] *= 1.0 - fb.capital_destruction[sector]

        if fb.demand_shift is not None:
            for sector in SECTORS:
                demand_shift[sector] *= fb.demand_shift[sector]

        if fb.trade_disruption is not None:
            for sector in SECTORS:
                trade_disruption[sector] *= fb.trade_disruption[sector]

        fiscal_pressure += fb.additional_gov_spending
        fx_pressure += fb.fx_outflow_pressure
        yield_factor *= fb.yield_factor

    destruction = {s: 1.0 - destruction_survival[s] for s in SECTORS}

    # Apply agricultural yield shock directly into productivity channel.
    productivity[Sector.AGRICULTURE] *= yield_factor

    # Clamp destruction to [0, 1].
    for sector in SECTORS:
        destruction[sector] = max(0.0, min(1.0, destruction[sector]))

    return AggregatedFeedback(
        labor_supply_shock=labor_supply,
        productivity_shock=productivity,
        capital_destruction=destruction,
        demand_shift=demand_shift,
        trade_disruption=trade_disruption,
        fx_pressure=fx_pressure,
        fiscal_pressure=fiscal_pressure,
        yield_factor=yield_factor,
    )


@dataclass(frozen=True)
class MacroExposure:
    """Read-only macro view passed to external sector models."""

    gdp_real: float
    gdp_growth: float
    output_gap: float
    inflation_rate: float
    unemployment_rate: float
    real_wage_index: float
    food_price_index: float
    exchange_rate: float
    interest_rate: float
    govt_health_spending: float
    govt_spending_total: float
    poverty_rate: float
    gini: float
    sector_output: dict[Sector, float]
    sector_employment: dict[Sector, float]

    @staticmethod
    def from_state(state: EconomyState) -> MacroExposure:
        gdp_real = state.gdp_real
        gdp_potential = max(state.gdp_potential, 1e-12)
        output_gap = (gdp_real - gdp_potential) / gdp_potential

        avg_nominal_wage = sum(state.w[s] for s in SECTORS) / len(SECTORS)
        real_wage_index = avg_nominal_wage / max(state.P_cpi, 1e-12)

        return MacroExposure(
            gdp_real=gdp_real,
            gdp_growth=0.0,
            output_gap=output_gap,
            inflation_rate=0.0,
            unemployment_rate=state.U,
            real_wage_index=real_wage_index,
            food_price_index=state.P[Sector.AGRICULTURE],
            exchange_rate=state.E_nom,
            interest_rate=state.i_cb,
            govt_health_spending=0.0,
            govt_spending_total=state.G_exp,
            poverty_rate=state.POVERTY,
            gini=state.GINI,
            sector_output={s: state.Y[s] for s in SECTORS},
            sector_employment={s: state.N[s] for s in SECTORS},
        )


class SectorModelProtocol(Protocol):
    def initialize(self, macro: MacroExposure) -> None:
        """Set initial conditions from macro state."""

    def step(self, macro: MacroExposure, dt: float) -> SectorFeedback:
        """Advance one timestep and return macro feedback."""

    def get_state(self) -> dict[str, float]:
        """Return current sector model state for logging."""

    def get_indicators(self) -> dict[str, float]:
        """Return output indicators for reporting."""

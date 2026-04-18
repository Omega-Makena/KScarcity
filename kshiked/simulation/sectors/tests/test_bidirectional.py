from __future__ import annotations

import numpy as np

from scarcity.simulation.coupling_interface import MacroExposure
from scarcity.simulation.sfc_engine import MultiSectorSFCEngine
from scarcity.simulation.types import SECTORS, Sector, ShockVector

from kshiked.simulation.sectors.base import CompartmentalSectorModel
from kshiked.simulation.sectors.config import SectorConfig
from kshiked.simulation.sectors.configs.reference_health import make_health_config
from kshiked.simulation.sectors.cross_sector import CrossSectorResolver


def _macro(poverty_rate: float = 0.36, gdp_growth: float = 0.03) -> MacroExposure:
    sectors = {sector: 1.0 for sector in Sector}
    return MacroExposure(
        gdp_real=200.0,
        gdp_growth=gdp_growth,
        output_gap=0.0,
        inflation_rate=0.06,
        unemployment_rate=0.09,
        real_wage_index=1.0,
        food_price_index=1.0,
        exchange_rate=130.0,
        interest_rate=0.12,
        govt_health_spending=3.0,
        govt_spending_total=25.0,
        poverty_rate=poverty_rate,
        gini=0.38,
        sector_output=sectors,
        sector_employment=sectors,
    )


def _health_config(**overrides: float) -> SectorConfig:
    base = dict(
        transmission_rate=3.0,
        incubation_rate=4.0,
        recovery_rate=2.0,
        hospitalization_rate=0.4,
        hospital_recovery_rate=2.5,
        hospital_mortality_rate=0.02,
        untreated_mortality_rate=0.5,
        immunity_waning_rate=0.0,
    )
    base.update(overrides)
    return make_health_config(**base)


def _neutral_shock() -> ShockVector:
    return ShockVector(
        demand_shock={s: 1.0 for s in SECTORS},
        supply_shock={s: 1.0 for s in SECTORS},
        world_price_shock=1.0,
        world_demand_shock=1.0,
        remittance_shock=1.0,
        aid_shock=1.0,
        risk_premium_shock=0.0,
        rainfall_shock=1.0,
    )


def _agri_supply_shock(factor: float) -> ShockVector:
    supply = {s: 1.0 for s in SECTORS}
    supply[Sector.AGRICULTURE] = factor
    return ShockVector(
        demand_shock={s: 1.0 for s in SECTORS},
        supply_shock=supply,
        world_price_shock=1.0,
        world_demand_shock=1.0,
        remittance_shock=1.0,
        aid_shock=1.0,
        risk_premium_shock=0.0,
        rainfall_shock=1.0,
    )


def _run_macro(
    quarters: int,
    models: list[CompartmentalSectorModel] | None = None,
    shock: ShockVector | None = None,
) -> list[float]:
    engine = MultiSectorSFCEngine()
    selected_shock = shock or _neutral_shock()

    if models:
        init_macro = MacroExposure.from_state(engine.state)
        for model in models:
            model.initialize(init_macro)

    gdp: list[float] = []
    for _ in range(quarters):
        feedbacks = None
        if models:
            feedbacks = []
            for model in models:
                macro = MacroExposure.from_state(engine.state)
                feedbacks.append(model.step(macro, dt=1.0))
        result = engine.step(shock=selected_shock, feedback=feedbacks)
        gdp.append(result.state.gdp_real)

    return gdp


def test_sector_model_changes_macro() -> None:
    baseline = _run_macro(quarters=20, models=None)

    active_model = CompartmentalSectorModel(_health_config(initial_infected_fraction=0.0))
    active_model.transfer("susceptible", "exposed", 0.8)
    with_model = _run_macro(quarters=20, models=[active_model])

    assert not np.allclose(baseline, with_model)


def test_macro_changes_sector() -> None:
    good_model = CompartmentalSectorModel(_health_config(initial_infected_fraction=0.001))
    bad_model = CompartmentalSectorModel(_health_config(initial_infected_fraction=0.001))

    good_model.initialize(_macro(poverty_rate=0.30, gdp_growth=0.03))
    bad_model.initialize(_macro(poverty_rate=0.60, gdp_growth=-0.05))

    good_peak = 0.0
    bad_peak = 0.0
    for _ in range(30):
        good_model.step(_macro(poverty_rate=0.30, gdp_growth=0.03), dt=1.0)
        bad_model.step(_macro(poverty_rate=0.60, gdp_growth=-0.05), dt=1.0)
        good_peak = max(good_peak, good_model.get_state()["infected"])
        bad_peak = max(bad_peak, bad_model.get_state()["infected"])

    assert bad_peak > good_peak


def test_feedback_loop_amplification() -> None:
    shock = _agri_supply_shock(0.92)
    no_sector = _run_macro(quarters=20, models=None, shock=shock)

    active_model = CompartmentalSectorModel(_health_config(initial_infected_fraction=0.0))
    active_model.transfer("susceptible", "exposed", 0.8)
    with_sector = _run_macro(quarters=20, models=[active_model], shock=shock)

    no_sector_drop = no_sector[0] - no_sector[-1]
    with_sector_drop = with_sector[0] - with_sector[-1]
    assert with_sector_drop > no_sector_drop


def test_neutral_model_no_effect() -> None:
    baseline = _run_macro(quarters=16, models=None)

    neutral_model = CompartmentalSectorModel(_health_config(initial_infected_fraction=0.0))
    with_model = _run_macro(quarters=16, models=[neutral_model])

    assert np.allclose(baseline, with_model)


def test_simultaneous_coupling_converges() -> None:
    engine = MultiSectorSFCEngine()
    model = CompartmentalSectorModel(_health_config(initial_infected_fraction=0.002))
    model.initialize(MacroExposure.from_state(engine.state))

    max_iterations = 20
    previous_factor = None
    converged_at = None

    for i in range(max_iterations):
        macro = MacroExposure.from_state(engine.state)
        feedback = model.step(macro, dt=0.25)
        engine.step(feedback=[feedback])

        if previous_factor is not None and abs(feedback.labor_supply_factor - previous_factor) < 1e-3:
            converged_at = i
            break
        previous_factor = feedback.labor_supply_factor

    assert converged_at is not None
    assert converged_at < max_iterations


def test_multiple_sector_models() -> None:
    primary = CompartmentalSectorModel(_health_config(name="primary", initial_infected_fraction=0.002))

    secondary_cfg = _health_config(name="secondary", initial_infected_fraction=0.001)
    secondary_cfg.transitions[0].cross_sector_modifiers = [("primary.infected", 0.25)]
    secondary = CompartmentalSectorModel(secondary_cfg)

    class Registry:
        def __init__(self, models: dict[str, CompartmentalSectorModel]) -> None:
            self.models = models

        def get_model(self, name: str) -> CompartmentalSectorModel | None:
            return self.models.get(name)

    registry = Registry({"primary": primary, "secondary": secondary})
    secondary.set_cross_sector_resolver(CrossSectorResolver(registry))

    initial_macro = _macro()
    primary.initialize(initial_macro)
    secondary.initialize(initial_macro)
    first_primary = primary.step(initial_macro, dt=1.0)
    first_secondary = secondary.step(initial_macro, dt=1.0)

    single_model_path = _run_macro(quarters=18, models=[primary])
    dual_model_path = _run_macro(quarters=18, models=[primary, secondary])

    assert first_primary.source == "primary"
    assert first_secondary.source == "secondary"
    assert not np.allclose(single_model_path, dual_model_path)

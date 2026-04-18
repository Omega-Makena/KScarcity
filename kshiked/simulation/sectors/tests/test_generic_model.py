from __future__ import annotations

from scarcity.simulation.coupling_interface import MacroExposure
from scarcity.simulation.types import Sector

from kshiked.simulation.sectors.base import CompartmentalSectorModel
from kshiked.simulation.sectors.configs.reference_health import make_health_config


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


def _config(**kwargs: float) -> object:
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
    base.update(kwargs)
    return make_health_config(**base)


def test_model_from_config() -> None:
    model = CompartmentalSectorModel(_config())
    state = model.get_state()
    assert "susceptible" in state
    assert "infected" in state


def test_step_returns_feedback() -> None:
    model = CompartmentalSectorModel(_config())
    model.initialize(_macro())
    feedback = model.step(_macro(), dt=1.0)
    assert feedback.source == "health"


def test_no_infection_stable() -> None:
    cfg = _config()
    model = CompartmentalSectorModel(cfg)
    model.initialize(_macro())

    initial = model.get_state()
    for _ in range(50):
        model.step(_macro(), dt=1.0)
    final = model.get_state()

    for key in ("susceptible", "exposed", "infected", "hospitalized", "recovered", "dead"):
        assert abs(final[key] - initial[key]) < 1e-9


def test_outbreak_dynamics() -> None:
    cfg = _config(initial_infected_fraction=0.005)
    model = CompartmentalSectorModel(cfg)
    model.initialize(_macro())

    infected_path: list[float] = []
    dead_path: list[float] = []
    susceptible_path: list[float] = []

    for _ in range(100):
        model.step(_macro(), dt=1.0)
        state = model.get_state()
        infected_path.append(state["infected"])
        dead_path.append(state["dead"])
        susceptible_path.append(state["susceptible"])

    assert max(infected_path) > infected_path[0]
    assert infected_path[-1] < max(infected_path)
    assert all(x <= y for x, y in zip(dead_path[:-1], dead_path[1:]))
    assert susceptible_path[-1] < susceptible_path[0]


def test_capacity_affects_mortality() -> None:
    high_capacity = CompartmentalSectorModel(
        _config(
            initial_infected_fraction=0.01,
            hospital_beds_per_1000=8.0,
            untreated_mortality_rate=0.8,
            hospitalization_rate=0.9,
        )
    )
    low_capacity = CompartmentalSectorModel(
        _config(
            initial_infected_fraction=0.01,
            hospital_beds_per_1000=0.01,
            untreated_mortality_rate=0.8,
            hospitalization_rate=0.9,
        )
    )

    macro = _macro()
    high_capacity.initialize(macro)
    low_capacity.initialize(macro)

    for _ in range(80):
        high_capacity.step(macro, dt=1.0)
        low_capacity.step(macro, dt=1.0)

    high_dead = high_capacity.get_state()["dead"]
    low_dead = low_capacity.get_state()["dead"]
    assert low_dead > high_dead * 1.2


def test_macro_driver_modifies_transmission() -> None:
    cfg = _config(initial_infected_fraction=0.001)
    baseline_model = CompartmentalSectorModel(cfg)
    high_poverty_model = CompartmentalSectorModel(cfg)

    baseline_model.initialize(_macro(poverty_rate=0.36))
    high_poverty_model.initialize(_macro(poverty_rate=0.60))

    baseline_peak = 0.0
    high_poverty_peak = 0.0
    for _ in range(30):
        baseline_model.step(_macro(poverty_rate=0.36), dt=1.0)
        high_poverty_model.step(_macro(poverty_rate=0.60), dt=1.0)
        baseline_peak = max(baseline_peak, baseline_model.get_state()["infected"])
        high_poverty_peak = max(high_poverty_peak, high_poverty_model.get_state()["infected"])

    assert high_poverty_peak > baseline_peak


def test_feedback_labor_supply() -> None:
    model = CompartmentalSectorModel(_config())
    model.initialize(_macro())
    model.inject_shock("infected", 2.6)
    feedback = model.step(_macro(), dt=1.0)
    assert feedback.labor_supply_factor < 1.0


def test_feedback_scales_with_severity() -> None:
    mild = CompartmentalSectorModel(_config())
    severe = CompartmentalSectorModel(_config())

    macro = _macro()
    mild.initialize(macro)
    severe.initialize(macro)
    mild.inject_shock("infected", 0.52)
    severe.inject_shock("infected", 5.2)

    mild_feedback = mild.step(macro, dt=1.0)
    severe_feedback = severe.step(macro, dt=1.0)
    assert severe_feedback.labor_supply_factor < mild_feedback.labor_supply_factor


def test_inject_shock() -> None:
    model = CompartmentalSectorModel(_config())
    before = model.get_state()["exposed"]
    model.inject_shock("exposed", 1000.0)
    after = model.get_state()["exposed"]
    assert abs(after - before - 1000.0) < 1e-9


def test_seasonal_modulation() -> None:
    model = CompartmentalSectorModel(_config(initial_infected_fraction=0.001, seasonal_amplitude=0.5))
    model.initialize(_macro())

    seasonal_values: list[float] = []
    for _ in range(8):
        model.step(_macro(), dt=1.0)
        seasonal_values.append(model.get_indicators()["seasonal_factor_transmission"])

    assert max(seasonal_values) - min(seasonal_values) > 0.5


def test_immunity_waning() -> None:
    model = CompartmentalSectorModel(
        _config(
            transmission_rate=0.0,
            initial_infected_fraction=0.0,
            initial_recovered_fraction=0.30,
            immunity_waning_rate=0.3,
        )
    )
    model.initialize(_macro())

    recovered_path: list[float] = []
    susceptible_path: list[float] = []
    for _ in range(60):
        model.step(_macro(), dt=1.0)
        state = model.get_state()
        recovered_path.append(state["recovered"])
        susceptible_path.append(state["susceptible"])

    assert recovered_path[-1] < max(recovered_path[:15])
    assert susceptible_path[-1] > susceptible_path[0]


def test_different_configs_different_behavior() -> None:
    low_transmission = CompartmentalSectorModel(_config(transmission_rate=1.2, initial_infected_fraction=0.001))
    high_transmission = CompartmentalSectorModel(_config(transmission_rate=4.0, initial_infected_fraction=0.001))

    macro = _macro()
    low_transmission.initialize(macro)
    high_transmission.initialize(macro)

    for _ in range(20):
        low_transmission.step(macro, dt=1.0)
        high_transmission.step(macro, dt=1.0)

    assert high_transmission.get_state()["infected"] != low_transmission.get_state()["infected"]

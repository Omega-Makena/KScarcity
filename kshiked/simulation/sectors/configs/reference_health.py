from __future__ import annotations

from kshiked.simulation.sectors.config import (
    CapacityConstraint,
    Compartment,
    FeedbackChannel,
    MacroDriver,
    SectorConfig,
    Transition,
)

PARAMETER_SOURCES: dict[str, str] = {
    "population": "World Bank Population, total (SP.POP.TOTL), Kenya",
    "hospital_beds_per_1000": "WHO Global Health Observatory: Hospital beds (per 1,000 people), Kenya",
    "baseline_bed_occupancy": "Kenya Health Sector Strategic and Investment Plan 2018-2023",
    "poverty_baseline": "World Bank Poverty headcount ratio at national poverty line, Kenya",
    "poverty_transmission_sensitivity": "Fung et al. (2014), cholera transmission dynamic models under socioeconomic stress",
    "absenteeism_parameters": "Bloom et al. (2018), macroeconomic burden of epidemic disease",
    "hospital_cost_scale": "Kenya Ministry of Health costing references and WHO CHOICE order-of-magnitude guidance",
}


def make_health_config(
    *,
    transmission_rate: float,
    incubation_rate: float,
    recovery_rate: float,
    hospitalization_rate: float,
    hospital_recovery_rate: float,
    hospital_mortality_rate: float,
    untreated_mortality_rate: float,
    immunity_waning_rate: float,
    name: str = "health",
    population: float = 52.0,
    seasonal_amplitude: float = 0.0,
    seasonal_peak_quarter: int = 0,
    hospital_beds_per_1000: float = 1.4,
    baseline_bed_occupancy: float = 0.65,
    initial_infected_fraction: float = 0.0,
    initial_recovered_fraction: float = 0.0,
    poverty_transmission_sensitivity: float = 0.5,
    absenteeism_rate_infected: float = 0.5,
    absenteeism_rate_hospitalized: float = 1.0,
    caregiving_ratio: float = 0.15,
) -> SectorConfig:
    """Build a reference health sector config using generic framework components.

    The engine code remains domain-agnostic; only this configuration contains
    disease and health-system semantics.
    """

    if initial_infected_fraction < 0.0 or initial_recovered_fraction < 0.0:
        raise ValueError("Initial fractions must be non-negative")
    if initial_infected_fraction + initial_recovered_fraction > 1.0:
        raise ValueError("Initial infected and recovered fractions must sum to <= 1")

    beds_total_million = hospital_beds_per_1000 * population / 1000.0
    beds_available_million = beds_total_million * (1.0 - baseline_bed_occupancy)

    susceptible_init = population * (1.0 - initial_infected_fraction - initial_recovered_fraction)
    exposed_init = 0.0
    infected_init = population * initial_infected_fraction
    hospitalized_init = 0.0
    recovered_init = population * initial_recovered_fraction
    dead_init = 0.0

    return SectorConfig(
        name=name,
        description="SEIR-style compartmental model with constrained treatment capacity",
        compartments=[
            Compartment(name="susceptible", initial_value=susceptible_init, unit="millions"),
            Compartment(name="exposed", initial_value=exposed_init, unit="millions"),
            Compartment(name="infected", initial_value=infected_init, unit="millions"),
            Compartment(name="hospitalized", initial_value=hospitalized_init, unit="millions"),
            Compartment(name="recovered", initial_value=recovered_init, unit="millions"),
            Compartment(name="dead", initial_value=dead_init, unit="millions"),
        ],
        transitions=[
            Transition(
                name="transmission",
                source="susceptible",
                target="exposed",
                base_rate=transmission_rate,
                interaction_with="infected",
                normalization="total",
                macro_modifiers=[("poverty_rate", poverty_transmission_sensitivity)],
                seasonal_amplitude=seasonal_amplitude,
                seasonal_peak_quarter=seasonal_peak_quarter,
            ),
            Transition(
                name="incubation",
                source="exposed",
                target="infected",
                base_rate=incubation_rate,
            ),
            Transition(
                name="community_recovery",
                source="infected",
                target="recovered",
                base_rate=recovery_rate,
            ),
            Transition(
                name="hospitalization",
                source="infected",
                target="hospitalized",
                base_rate=hospitalization_rate,
                capacity_constraint="treatment_capacity",
                overflow_target="infected",
                overflow_rate=0.0,
            ),
            Transition(
                name="hospital_recovery",
                source="hospitalized",
                target="recovered",
                base_rate=hospital_recovery_rate,
            ),
            Transition(
                name="hospital_death",
                source="hospitalized",
                target="dead",
                base_rate=hospital_mortality_rate,
            ),
            Transition(
                name="community_death",
                source="infected",
                target="dead",
                base_rate=untreated_mortality_rate * 0.01,
            ),
            Transition(
                name="waning",
                source="recovered",
                target="susceptible",
                base_rate=immunity_waning_rate,
            ),
        ],
        capacity_constraints=[
            CapacityConstraint(
                name="treatment_capacity",
                demand_compartments=["hospitalized"],
                base_capacity=beds_available_million,
                max_surge_factor=1.3,
                surge_trigger=0.85,
                surge_ramp_quarters=0.5,
                midpoint=0.85,
                steepness=12.0,
                baseline_effectiveness=0.85,
                stressed_effectiveness=0.10,
                capacity_decay_rate=0.05,
                capacity_recovery_rate=0.02,
            )
        ],
        feedback_channels=[
            FeedbackChannel(
                name="labor_supply",
                source_compartments=["infected", "hospitalized"],
                weights=[absenteeism_rate_infected, absenteeism_rate_hospitalized],
                normalization="population",
                target_field="labor_supply_factor",
                transform="multiplicative_reduction",
                sensitivity=1.0 + caregiving_ratio,
                min_value=0.5,
                max_value=1.0,
            ),
            FeedbackChannel(
                name="fiscal_cost",
                source_compartments=["hospitalized"],
                target_field="additional_gov_spending",
                transform="additive",
                sensitivity=0.005,
                min_value=0.0,
            ),
            FeedbackChannel(
                name="supply_drag",
                source_compartments=["infected"],
                normalization="population",
                target_field="demand_shift",
                target_sector="agri",
                transform="multiplicative_reduction",
                sensitivity=0.2,
                min_value=0.7,
                max_value=1.0,
            ),
        ],
        macro_drivers=[
            MacroDriver(
                name="poverty_transmission",
                macro_field="poverty_rate",
                baseline=0.36,
                target_transitions=["transmission"],
                sensitivity=poverty_transmission_sensitivity,
                mode="linear",
            )
        ],
        population_compartments=[
            "susceptible",
            "exposed",
            "infected",
            "hospitalized",
            "recovered",
        ],
        substeps_per_quarter=13,
        metadata={"parameter_sources": dict(PARAMETER_SOURCES)},
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Compartment:
    """A single state variable in a sector model."""

    name: str
    initial_value: float
    min_value: float = 0.0
    max_value: float = float("inf")
    unit: str = ""
    description: str = ""
    group: str | None = None


@dataclass
class Transition:
    """A flow between two compartments, optionally with interaction terms."""

    name: str
    source: str
    target: str
    base_rate: float
    interaction_with: str | None = None
    normalization: str | None = None
    macro_modifiers: list[tuple[str, float]] = field(default_factory=list)
    cross_sector_modifiers: list[tuple[str, float]] = field(default_factory=list)
    seasonal_amplitude: float = 0.0
    seasonal_peak_quarter: int = 0
    capacity_constraint: str | None = None
    overflow_target: str | None = None
    overflow_rate: float = 0.0


@dataclass
class ExternalInflow:
    """Flow into a compartment from outside the modeled system."""

    target: str
    base_rate: float
    macro_modifiers: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ExternalOutflow:
    """Flow out of a compartment to outside the modeled system."""

    source: str
    base_rate: float
    macro_modifiers: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class CapacityConstraint:
    """A non-linear capacity limit on one or more transitions."""

    name: str
    demand_compartments: list[str]
    base_capacity: float
    capacity_per_1000_pop: float | None = None
    max_surge_factor: float = 1.0
    surge_trigger: float = 0.85
    surge_ramp_quarters: float = 0.5
    midpoint: float = 0.85
    steepness: float = 12.0
    baseline_effectiveness: float = 1.0
    stressed_effectiveness: float = 0.1
    capacity_decay_rate: float = 0.0
    capacity_recovery_rate: float = 0.0


@dataclass
class FeedbackChannel:
    """Declarative mapping from sector state to macro feedback fields."""

    name: str
    source_compartments: list[str]
    weights: list[float] | None = None
    normalization: str | None = None
    target_field: str = "labor_supply_factor"
    target_sector: str | None = None
    transform: str = "multiplicative_reduction"
    sensitivity: float = 1.0
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class MacroDriver:
    """Specifies how a macro variable modulates one or more transitions."""

    name: str
    macro_field: str
    baseline: float
    target_transitions: list[str]
    sensitivity: float
    mode: str = "linear"
    threshold: float | None = None


@dataclass
class SectorConfig:
    """Complete configuration for a generic sector model."""

    name: str
    description: str = ""
    compartments: list[Compartment] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    external_inflows: list[ExternalInflow] = field(default_factory=list)
    external_outflows: list[ExternalOutflow] = field(default_factory=list)
    capacity_constraints: list[CapacityConstraint] = field(default_factory=list)
    feedback_channels: list[FeedbackChannel] = field(default_factory=list)
    macro_drivers: list[MacroDriver] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    substeps_per_quarter: int = 13
    groups: list[str] | None = None
    group_interaction_matrix: dict[str, dict[str, float]] | None = None
    population_compartments: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.substeps_per_quarter <= 0:
            raise ValueError("substeps_per_quarter must be positive")

        comp_names = {c.name for c in self.compartments}
        if len(comp_names) != len(self.compartments):
            raise ValueError("compartment names must be unique")

        cc_names = {c.name for c in self.capacity_constraints}

        for transition in self.transitions:
            if transition.source not in comp_names:
                raise ValueError(
                    f"Transition {transition.name}: source '{transition.source}' not in compartments"
                )
            if transition.target not in comp_names:
                raise ValueError(
                    f"Transition {transition.name}: target '{transition.target}' not in compartments"
                )
            if transition.interaction_with and transition.interaction_with not in comp_names:
                raise ValueError(
                    f"Transition {transition.name}: interaction_with '{transition.interaction_with}' not in compartments"
                )
            if transition.capacity_constraint and transition.capacity_constraint not in cc_names:
                raise ValueError(
                    f"Transition {transition.name}: capacity_constraint '{transition.capacity_constraint}' not found"
                )
            if transition.normalization and transition.normalization not in comp_names and transition.normalization != "total":
                raise ValueError(
                    f"Transition {transition.name}: normalization '{transition.normalization}' must be a compartment or 'total'"
                )

        for inflow in self.external_inflows:
            if inflow.target not in comp_names:
                raise ValueError(f"ExternalInflow target '{inflow.target}' not in compartments")

        for outflow in self.external_outflows:
            if outflow.source not in comp_names:
                raise ValueError(f"ExternalOutflow source '{outflow.source}' not in compartments")

        for channel in self.feedback_channels:
            for src in channel.source_compartments:
                if src not in comp_names:
                    raise ValueError(
                        f"FeedbackChannel {channel.name}: source '{src}' not in compartments"
                    )
            if channel.weights is not None and len(channel.weights) != len(channel.source_compartments):
                raise ValueError(
                    f"FeedbackChannel {channel.name}: weights must match source_compartments length"
                )

        transition_names = {t.name for t in self.transitions}
        for driver in self.macro_drivers:
            if driver.mode not in {"linear", "threshold"}:
                raise ValueError(f"MacroDriver {driver.name}: mode must be 'linear' or 'threshold'")
            for target in driver.target_transitions:
                if target not in transition_names:
                    raise ValueError(
                        f"MacroDriver {driver.name}: target transition '{target}' not found"
                    )

        if self.groups is not None:
            if len(self.groups) < 2:
                raise ValueError("groups must contain at least two groups when provided")
            if len(set(self.groups)) != len(self.groups):
                raise ValueError("groups must be unique")
            if self.group_interaction_matrix is not None:
                for group_i in self.groups:
                    if group_i not in self.group_interaction_matrix:
                        raise ValueError(
                            f"group_interaction_matrix missing row for group '{group_i}'"
                        )
                    for group_j in self.groups:
                        if group_j not in self.group_interaction_matrix[group_i]:
                            raise ValueError(
                                f"group_interaction_matrix missing value for '{group_i}' -> '{group_j}'"
                            )

        if self.population_compartments is not None:
            for name in self.population_compartments:
                if name not in comp_names:
                    raise ValueError(
                        f"population_compartments contains unknown compartment '{name}'"
                    )

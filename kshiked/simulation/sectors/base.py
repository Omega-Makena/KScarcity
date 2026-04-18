from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from kshiked.simulation.sectors.capacity import CapacitySystem
from kshiked.simulation.sectors.config import ExternalInflow, ExternalOutflow, SectorConfig, Transition
from kshiked.simulation.sectors.cross_sector import CrossSectorResolver
from kshiked.simulation.sectors.feedback_map import FeedbackMapper
from kshiked.simulation.sectors.integrator import CompartmentalIntegrator
from kshiked.simulation.sectors.macro_drivers import MacroDriverSystem

try:
    from kshiked.simulation.coupling import MacroExposure, SectorFeedback, SectorModelProtocol  # type: ignore
except ImportError:
    from scarcity.simulation.coupling_interface import MacroExposure, SectorModelProtocol
    from scarcity.simulation.types import SectorFeedback


class CompartmentalSectorModel:
    """Generic sector model driven entirely by SectorConfig."""

    def __init__(self, config: SectorConfig) -> None:
        self._config = config
        self._integrator = CompartmentalIntegrator()
        self._capacity = CapacitySystem(config.capacity_constraints)
        self._feedback_mapper = FeedbackMapper(config.feedback_channels, config.name)
        self._macro_drivers = MacroDriverSystem(config.macro_drivers)
        self._cross_resolver: CrossSectorResolver | None = None

        self._compartment_bounds: dict[str, tuple[float, float]] = {}
        self._base_compartments = [compartment.name for compartment in config.compartments]
        self._state = self._build_initial_state()

        self._indicators: dict[str, float] = {}
        self._current_macro: MacroExposure | None = None
        self._quarter = 0
        self._compute_indicators(self._state, self._quarter)

    @property
    def name(self) -> str:
        return self._config.name

    def set_cross_sector_resolver(self, resolver: CrossSectorResolver) -> None:
        self._cross_resolver = resolver

    def initialize(self, macro: MacroExposure) -> None:
        self._current_macro = macro

        if self._config.population_compartments:
            total_now = self._population_total(self._state)
            target_population = float(sum(getattr(macro, "sector_employment", {}).values()))
            if target_population > 0.0 and total_now > 0.0:
                scale = target_population / total_now
                for compartment_name in self._config.population_compartments:
                    for key in self._resolve_state_keys(compartment_name, self._state):
                        self._state[key] *= scale

        self._compute_indicators(self._state, self._quarter)

    def step(self, macro: MacroExposure, dt: float = 1.0) -> SectorFeedback:
        self._current_macro = macro

        rate_modifiers = self._macro_drivers.compute_rate_modifiers(macro)
        cross_modifiers = {
            transition.name: (
                self._cross_resolver.compute_cross_modifiers(transition)
                if self._cross_resolver is not None
                else 1.0
            )
            for transition in self._config.transitions
        }

        derivative_fn = self._build_derivative_fn(rate_modifiers, cross_modifiers, self._quarter)
        bounds = dict(self._compartment_bounds)
        conservation_groups = self._conservation_groups(self._state)

        self._state = self._integrator.integrate(
            state=self._state,
            derivative_fn=derivative_fn,
            dt=dt,
            substeps=self._config.substeps_per_quarter,
            clamp_bounds=bounds,
            conservation_groups=conservation_groups,
        )

        capacity_state = self._state_with_totals(self._state)
        capacity_state["population"] = self._population_total(self._state)
        self._capacity.update_dynamic_capacity(dt=dt, state=capacity_state)

        self._quarter += 1
        self._compute_indicators(self._state, self._quarter)

        feedback = self._feedback_mapper.compute_feedback(
            state=self._state_with_totals(self._state),
            population=self._population_total(self._state),
        )
        return feedback

    def _build_derivative_fn(
        self,
        rate_modifiers: dict[str, float],
        cross_modifiers: dict[str, float],
        quarter: int,
    ) -> Callable[[dict[str, float]], dict[str, float]]:
        def derivative(local_state: dict[str, float]) -> dict[str, float]:
            totals = self._state_with_totals(local_state)
            population_total = self._population_total(local_state)
            derivs = {key: 0.0 for key in local_state}

            for transition in self._config.transitions:
                base_rate = transition.base_rate
                base_rate *= rate_modifiers.get(transition.name, 1.0)
                base_rate *= cross_modifiers.get(transition.name, 1.0)
                base_rate *= self._inline_macro_modifier(transition, self._current_macro)
                base_rate *= self._seasonal_multiplier(transition, quarter)

                flow_specs = self._transition_flows(
                    local_state=local_state,
                    totals=totals,
                    transition=transition,
                    rate=base_rate,
                    population_total=population_total,
                )

                for source_key, target_key, effective_flow, overflow_key, overflow_flow in flow_specs:
                    derivs[source_key] = derivs.get(source_key, 0.0) - effective_flow
                    derivs[target_key] = derivs.get(target_key, 0.0) + effective_flow
                    if overflow_key is not None and overflow_flow > 0.0:
                        derivs[overflow_key] = derivs.get(overflow_key, 0.0) + overflow_flow

            self._apply_external_inflows(derivs, local_state, self._config.external_inflows)
            self._apply_external_outflows(derivs, local_state, self._config.external_outflows)
            return derivs

        return derivative

    def get_state(self) -> dict[str, float]:
        result = self._state_with_totals(self._state)
        result.update(self._capacity.get_state())
        return result

    def get_indicators(self) -> dict[str, float]:
        return dict(self._indicators)

    def inject_shock(self, compartment: str, amount: float) -> None:
        keys = self._resolve_state_keys(compartment, self._state)
        if not keys:
            raise KeyError(f"Unknown compartment '{compartment}'")

        delta_per_key = amount / float(len(keys))
        for key in keys:
            lo, hi = self._compartment_bounds.get(key, (0.0, float("inf")))
            self._state[key] = min(max(self._state.get(key, 0.0) + delta_per_key, lo), hi)

        self._compute_indicators(self._state, self._quarter)

    def transfer(self, source: str, target: str, amount: float) -> None:
        if amount <= 0.0:
            return

        source_keys = self._resolve_state_keys(source, self._state)
        target_keys = self._resolve_state_keys(target, self._state)
        if not source_keys or not target_keys:
            raise KeyError("Unknown source or target compartment")

        source_total = sum(max(self._state.get(key, 0.0), 0.0) for key in source_keys)
        moved_total = min(amount, source_total)
        if moved_total <= 0.0:
            return

        if len(source_keys) == len(target_keys):
            pairs = list(zip(source_keys, target_keys))
        else:
            pairs = [(source_key, target_keys[idx % len(target_keys)]) for idx, source_key in enumerate(source_keys)]

        if source_total > 0.0:
            for source_key, target_key in pairs:
                source_value = max(self._state.get(source_key, 0.0), 0.0)
                share = source_value / source_total
                moved = moved_total * share
                self._state[source_key] = max(self._state.get(source_key, 0.0) - moved, 0.0)
                self._state[target_key] = self._state.get(target_key, 0.0) + moved

        self._compute_indicators(self._state, self._quarter)

    def _build_initial_state(self) -> dict[str, float]:
        state: dict[str, float] = {}
        if self._config.groups:
            group_count = float(len(self._config.groups))
            for compartment in self._config.compartments:
                for group in self._config.groups:
                    key = f"{compartment.name}_{group}"
                    state[key] = compartment.initial_value / group_count
                    self._compartment_bounds[key] = (compartment.min_value, compartment.max_value)
        else:
            for compartment in self._config.compartments:
                state[compartment.name] = compartment.initial_value
                self._compartment_bounds[compartment.name] = (
                    compartment.min_value,
                    compartment.max_value,
                )
        return state

    def _resolve_state_keys(self, compartment: str, state: dict[str, float]) -> list[str]:
        if compartment in state:
            return [compartment]
        if self._config.groups and compartment in self._base_compartments:
            return [
                f"{compartment}_{group}"
                for group in self._config.groups
                if f"{compartment}_{group}" in state
            ]
        return []

    def _state_with_totals(self, state: dict[str, float]) -> dict[str, float]:
        if not self._config.groups:
            return dict(state)

        result = dict(state)
        for name in self._base_compartments:
            total = sum(state.get(f"{name}_{group}", 0.0) for group in self._config.groups or [])
            result[name] = total
        return result

    def _population_total(self, state: dict[str, float]) -> float:
        if self._config.population_compartments:
            total = 0.0
            for name in self._config.population_compartments:
                for key in self._resolve_state_keys(name, state):
                    total += max(state.get(key, 0.0), 0.0)
            return total
        return sum(max(value, 0.0) for value in state.values())

    def _conservation_groups(self, state: dict[str, float]) -> list[list[str]] | None:
        if not self._config.population_compartments:
            return None

        if not self._config.groups:
            return [list(self._config.population_compartments)]

        grouped_sets: list[list[str]] = []
        for group in self._config.groups:
            grouped_sets.append([
                f"{name}_{group}"
                for name in self._config.population_compartments
                if f"{name}_{group}" in state
            ])
        return grouped_sets

    def _inline_macro_modifier(self, transition: Transition, macro: MacroExposure | None) -> float:
        if macro is None:
            return 1.0

        modifier = 1.0
        for field_path, sensitivity in transition.macro_modifiers:
            value = self._resolve_macro_field(macro, field_path)
            modifier *= min(max(1.0 + sensitivity * value, 0.1), 5.0)
        return modifier

    def _resolve_macro_field(self, macro: MacroExposure, field_path: str) -> float:
        field = field_path
        if field.startswith("macro."):
            field = field.split(".", 1)[1]

        value = getattr(macro, field, 0.0)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _seasonal_multiplier(self, transition: Transition, quarter: int) -> float:
        if transition.seasonal_amplitude == 0.0:
            return 1.0
        phase = (2.0 * math.pi * (quarter - transition.seasonal_peak_quarter)) / 4.0
        multiplier = 1.0 + transition.seasonal_amplitude * float(np.cos(phase))
        return max(multiplier, 0.0)

    def _transition_flows(
        self,
        local_state: dict[str, float],
        totals: dict[str, float],
        transition: Transition,
        rate: float,
        population_total: float,
    ) -> list[tuple[str, str, float, str | None, float]]:
        source_keys = self._resolve_state_keys(transition.source, local_state)
        target_keys = self._resolve_state_keys(transition.target, local_state)
        if not source_keys or not target_keys:
            return []

        if len(source_keys) == len(target_keys):
            source_target_pairs = list(zip(source_keys, target_keys))
        else:
            source_target_pairs = [
                (source_key, target_keys[idx % len(target_keys)])
                for idx, source_key in enumerate(source_keys)
            ]

        specs: list[tuple[str, str, float, str | None, float]] = []
        for source_key, target_key in source_target_pairs:
            source_value = max(local_state.get(source_key, 0.0), 0.0)
            if source_value <= 0.0:
                continue

            flow = self._interaction_flow(
                local_state=local_state,
                totals=totals,
                transition=transition,
                rate=rate,
                source_key=source_key,
                source_value=source_value,
                population_total=population_total,
            )

            flow = max(flow, 0.0)
            effective_flow = flow
            overflow = 0.0
            if transition.capacity_constraint:
                demand = self._constraint_demand(transition.capacity_constraint, totals)
                effective_flow, overflow = self._capacity.compute_effective_flow(
                    transition.capacity_constraint,
                    demand,
                    flow,
                )

            overflow_key: str | None = None
            overflow_flow = 0.0
            if transition.overflow_target and overflow > 0.0 and transition.overflow_rate > 0.0:
                candidate_keys = self._resolve_state_keys(transition.overflow_target, local_state)
                overflow_key = self._pick_overflow_key(source_key, candidate_keys)
                if overflow_key is not None:
                    overflow_flow = overflow * transition.overflow_rate

            specs.append((source_key, target_key, effective_flow, overflow_key, overflow_flow))

        return specs

    def _pick_overflow_key(self, source_key: str, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        if source_key in candidates:
            return source_key

        suffix = source_key.split("_", 1)[1] if "_" in source_key else None
        if suffix is not None:
            for candidate in candidates:
                if candidate.endswith(f"_{suffix}"):
                    return candidate
        return candidates[0]

    def _interaction_flow(
        self,
        local_state: dict[str, float],
        totals: dict[str, float],
        transition: Transition,
        rate: float,
        source_key: str,
        source_value: float,
        population_total: float,
    ) -> float:
        if transition.interaction_with is None:
            return rate * source_value

        if self._config.groups and "_" in source_key:
            source_group = source_key.split("_", 1)[1]
            matrix = self._config.group_interaction_matrix
            interaction_sum = 0.0
            for target_group in self._config.groups:
                interaction_key = f"{transition.interaction_with}_{target_group}"
                interaction_value = max(local_state.get(interaction_key, 0.0), 0.0)
                if transition.normalization in (None, "total"):
                    norm = max(population_total, 1e-12)
                else:
                    norm_key = f"{transition.normalization}_{target_group}"
                    norm = max(local_state.get(norm_key, totals.get(transition.normalization, 0.0)), 1e-12)
                weight = 1.0
                if matrix is not None:
                    weight = matrix.get(source_group, {}).get(target_group, 0.0)
                interaction_sum += weight * interaction_value / norm
            return rate * source_value * interaction_sum

        interaction_value = max(totals.get(transition.interaction_with, 0.0), 0.0)
        if transition.normalization in (None, "total"):
            norm = max(population_total, 1e-12)
        else:
            norm = max(totals.get(transition.normalization, 0.0), 1e-12)
        return rate * source_value * interaction_value / norm

    def _constraint_demand(self, constraint_name: str, totals: dict[str, float]) -> float:
        constraint = next(
            (item for item in self._config.capacity_constraints if item.name == constraint_name),
            None,
        )
        if constraint is None:
            return 0.0
        return float(sum(max(totals.get(name, 0.0), 0.0) for name in constraint.demand_compartments))

    def _apply_external_inflows(
        self,
        derivs: dict[str, float],
        local_state: dict[str, float],
        inflows: list[ExternalInflow],
    ) -> None:
        for inflow in inflows:
            target_keys = self._resolve_state_keys(inflow.target, local_state)
            if not target_keys:
                continue

            modifier = self._external_macro_modifier(inflow.macro_modifiers)
            flow_total = inflow.base_rate * modifier
            per_target = flow_total / float(len(target_keys))
            for target_key in target_keys:
                derivs[target_key] = derivs.get(target_key, 0.0) + per_target

    def _apply_external_outflows(
        self,
        derivs: dict[str, float],
        local_state: dict[str, float],
        outflows: list[ExternalOutflow],
    ) -> None:
        for outflow in outflows:
            source_keys = self._resolve_state_keys(outflow.source, local_state)
            if not source_keys:
                continue

            modifier = self._external_macro_modifier(outflow.macro_modifiers)
            for source_key in source_keys:
                source_value = max(local_state.get(source_key, 0.0), 0.0)
                derivs[source_key] = derivs.get(source_key, 0.0) - outflow.base_rate * modifier * source_value

    def _external_macro_modifier(self, modifiers: list[tuple[str, float]]) -> float:
        if self._current_macro is None:
            return 1.0

        value = 1.0
        for field_path, sensitivity in modifiers:
            macro_value = self._resolve_macro_field(self._current_macro, field_path)
            value *= min(max(1.0 + sensitivity * macro_value, 0.1), 5.0)
        return value

    def _compute_indicators(self, state: dict[str, float], quarter: int) -> None:
        totals = self._state_with_totals(state)
        population = self._population_total(state)

        indicators: dict[str, float] = {"quarter": float(quarter), "population": float(population)}
        for name in self._base_compartments:
            indicators[name] = float(totals.get(name, 0.0))
            indicators[f"share_{name}"] = float(totals.get(name, 0.0) / max(population, 1e-12))

        for transition in self._config.transitions:
            indicators[f"seasonal_factor_{transition.name}"] = self._seasonal_multiplier(
                transition,
                quarter,
            )

        self._indicators = indicators

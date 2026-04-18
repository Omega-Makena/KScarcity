from __future__ import annotations

import math

from kshiked.simulation.sectors.config import CapacityConstraint


class CapacitySystem:
    """Manages non-linear capacity constraints with surge and degradation."""

    def __init__(self, constraints: list[CapacityConstraint]) -> None:
        self._constraints: dict[str, CapacityConstraint] = {c.name: c for c in constraints}
        self._current_capacity: dict[str, float] = {}
        self._surge_progress: dict[str, float] = {}
        self._time_above_surge_trigger: dict[str, float] = {}
        self._last_utilization: dict[str, float] = {}
        self._last_stress: dict[str, float] = {}

        for constraint in constraints:
            self._current_capacity[constraint.name] = max(constraint.base_capacity, 1e-12)
            self._surge_progress[constraint.name] = 0.0
            self._time_above_surge_trigger[constraint.name] = 0.0
            self._last_utilization[constraint.name] = 0.0
            self._last_stress[constraint.name] = 0.0

    def compute_stress(self, name: str, demand: float) -> float:
        constraint = self._constraints[name]
        capacity = max(self._effective_capacity(name), 1e-12)
        utilization = max(demand, 0.0) / capacity
        stress = 1.0 / (1.0 + math.exp(-constraint.steepness * (utilization - constraint.midpoint)))
        stress = min(max(stress, 0.0), 1.0)
        self._last_utilization[name] = utilization
        self._last_stress[name] = stress
        return stress

    def compute_effective_flow(
        self,
        constraint_name: str,
        demand: float,
        base_flow: float,
    ) -> tuple[float, float]:
        constraint = self._constraints[constraint_name]
        stress = self.compute_stress(constraint_name, demand)

        baseline_eff = max(constraint.baseline_effectiveness, 1e-12)
        stressed_eff = min(max(constraint.stressed_effectiveness, 0.0), baseline_eff)
        ratio = stressed_eff / baseline_eff
        effective_factor = 1.0 - stress * (1.0 - ratio)
        effective_factor = min(max(effective_factor, 0.0), 1.0)

        effective_flow = max(base_flow, 0.0) * effective_factor
        # Overflow is expressed in baseline-equivalent units so that
        # effective_flow + overflow * ratio ~= base_flow.
        overflow = (max(base_flow, 0.0) - effective_flow) / max(ratio, 1e-12)
        return effective_flow, overflow

    def update_dynamic_capacity(self, dt: float, state: dict[str, float]) -> None:
        for name, constraint in self._constraints.items():
            current_base_capacity = self._current_capacity[name]
            baseline_from_population = self._baseline_capacity_from_population(constraint, state)
            target_baseline = baseline_from_population if baseline_from_population is not None else constraint.base_capacity
            target_baseline = max(target_baseline, 1e-12)

            demand = self._constraint_demand(constraint, state)
            stress = self.compute_stress(name, demand)
            utilization = self._last_utilization[name]

            if utilization > constraint.surge_trigger:
                self._time_above_surge_trigger[name] += dt
                ramp = max(constraint.surge_ramp_quarters, 1e-12)
                self._surge_progress[name] = min(1.0, self._surge_progress[name] + dt / ramp)
            else:
                self._time_above_surge_trigger[name] = max(0.0, self._time_above_surge_trigger[name] - dt)
                ramp = max(constraint.surge_ramp_quarters, 1e-12)
                self._surge_progress[name] = max(0.0, self._surge_progress[name] - dt / ramp)

            decayed = current_base_capacity * (1.0 - constraint.capacity_decay_rate * stress * dt)
            recovered = decayed + (target_baseline - decayed) * constraint.capacity_recovery_rate * (1.0 - stress) * dt
            self._current_capacity[name] = max(recovered, 1e-12)

    def get_utilization(self, name: str, state: dict[str, float]) -> float:
        constraint = self._constraints[name]
        demand = self._constraint_demand(constraint, state)
        capacity = max(self._effective_capacity(name), 1e-12)
        utilization = demand / capacity
        self._last_utilization[name] = utilization
        self._last_stress[name] = self.compute_stress(name, demand)
        return utilization

    def get_state(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in self._constraints:
            out[f"{name}_capacity"] = self._effective_capacity(name)
            out[f"{name}_utilization"] = self._last_utilization.get(name, 0.0)
            out[f"{name}_stress"] = self._last_stress.get(name, 0.0)
            out[f"{name}_surge_active"] = 1.0 if self._surge_progress.get(name, 0.0) > 1e-9 else 0.0
        return out

    def _effective_capacity(self, name: str) -> float:
        constraint = self._constraints[name]
        base_capacity = max(self._current_capacity[name], 1e-12)
        surge_multiplier = 1.0 + (constraint.max_surge_factor - 1.0) * self._surge_progress[name]
        return base_capacity * surge_multiplier

    def _constraint_demand(self, constraint: CapacityConstraint, state: dict[str, float]) -> float:
        return float(sum(max(state.get(compartment, 0.0), 0.0) for compartment in constraint.demand_compartments))

    def _baseline_capacity_from_population(
        self,
        constraint: CapacityConstraint,
        state: dict[str, float],
    ) -> float | None:
        if constraint.capacity_per_1000_pop is None:
            return None
        population = state.get("population", 0.0)
        if population <= 0.0:
            population = sum(max(value, 0.0) for value in state.values())
        return constraint.capacity_per_1000_pop * population / 1000.0

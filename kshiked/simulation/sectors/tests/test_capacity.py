from __future__ import annotations

import numpy as np

from kshiked.simulation.sectors.capacity import CapacitySystem
from kshiked.simulation.sectors.config import CapacityConstraint


def _constraint() -> CapacityConstraint:
    return CapacityConstraint(
        name="capacity",
        demand_compartments=["demand"],
        base_capacity=100.0,
        max_surge_factor=1.3,
        surge_trigger=0.85,
        surge_ramp_quarters=0.5,
        midpoint=0.85,
        steepness=12.0,
        baseline_effectiveness=0.85,
        stressed_effectiveness=0.1,
        capacity_decay_rate=0.05,
        capacity_recovery_rate=0.05,
    )


def test_no_stress_below_capacity() -> None:
    system = CapacitySystem([_constraint()])
    stress = system.compute_stress("capacity", demand=50.0)
    assert stress < 0.05


def test_full_stress_above_capacity() -> None:
    system = CapacitySystem([_constraint()])
    stress = system.compute_stress("capacity", demand=200.0)
    assert stress > 0.95


def test_logistic_shape() -> None:
    system = CapacitySystem([_constraint()])

    midpoint_stress = system.compute_stress("capacity", demand=85.0)
    assert abs(midpoint_stress - 0.5) < 0.05

    demands = np.linspace(40.0, 150.0, 21)
    stresses = [system.compute_stress("capacity", float(d)) for d in demands]
    assert all(x <= y for x, y in zip(stresses[:-1], stresses[1:]))

    second_diff = np.diff(np.diff(np.array(stresses)))
    assert np.any(second_diff > 0.0)
    assert np.any(second_diff < 0.0)


def test_overflow_computation() -> None:
    system = CapacitySystem([_constraint()])
    effective, overflow = system.compute_effective_flow(
        constraint_name="capacity",
        demand=150.0,
        base_flow=10.0,
    )

    assert effective < 10.0
    assert overflow > 0.0

    c = _constraint()
    adjusted = effective + overflow * (c.stressed_effectiveness / c.baseline_effectiveness)
    assert abs(adjusted - 10.0) < 1e-6


def test_surge_activation() -> None:
    system = CapacitySystem([_constraint()])
    base_capacity = system.get_state()["capacity_capacity"]

    for _ in range(6):
        system.update_dynamic_capacity(dt=0.1, state={"demand": 140.0})

    surged_capacity = system.get_state()["capacity_capacity"]
    assert surged_capacity > base_capacity


def test_capacity_degradation() -> None:
    system = CapacitySystem([_constraint()])

    initial_capacity = system.get_state()["capacity_capacity"]
    for _ in range(10):
        system.update_dynamic_capacity(dt=1.0, state={"demand": 200.0})

    degraded_capacity = system.get_state()["capacity_capacity"]
    assert degraded_capacity < initial_capacity


def test_capacity_recovery() -> None:
    system = CapacitySystem([_constraint()])

    for _ in range(8):
        system.update_dynamic_capacity(dt=1.0, state={"demand": 180.0})
    degraded_capacity = system.get_state()["capacity_capacity"]

    for _ in range(40):
        system.update_dynamic_capacity(dt=1.0, state={"demand": 0.0})
    recovered_capacity = system.get_state()["capacity_capacity"]

    assert recovered_capacity > degraded_capacity

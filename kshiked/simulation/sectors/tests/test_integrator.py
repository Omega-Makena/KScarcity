from __future__ import annotations

import math

from kshiked.simulation.sectors.integrator import CompartmentalIntegrator


def test_exponential_decay() -> None:
    integrator = CompartmentalIntegrator()
    initial = {"x": 100.0}

    def derivative(state: dict[str, float]) -> dict[str, float]:
        return {"x": -0.1 * state["x"]}

    result = integrator.integrate(
        state=initial,
        derivative_fn=derivative,
        dt=10.0,
        substeps=130,
    )

    expected = 100.0 * math.exp(-1.0)
    assert abs(result["x"] - expected) / expected < 0.01


def test_conservation() -> None:
    integrator = CompartmentalIntegrator()

    def derivative(state: dict[str, float]) -> dict[str, float]:
        rate = 0.2
        flow = rate * state["a"]
        return {"a": -flow, "b": flow}

    state = {"a": 70.0, "b": 30.0}
    for _ in range(100):
        state = integrator.integrate(
            state=state,
            derivative_fn=derivative,
            dt=0.01,
            substeps=1,
            conservation_groups=[["a", "b"]],
        )
        assert abs((state["a"] + state["b"]) - 100.0) < 1e-2


def test_non_negative_clamping() -> None:
    integrator = CompartmentalIntegrator()

    def derivative(_: dict[str, float]) -> dict[str, float]:
        return {"x": -100.0}

    result = integrator.integrate(
        state={"x": 1.0},
        derivative_fn=derivative,
        dt=1.0,
        substeps=1,
        clamp_bounds={"x": (0.0, float("inf"))},
    )
    assert result["x"] >= 0.0


def test_rk4_more_accurate_than_euler() -> None:
    integrator = CompartmentalIntegrator()

    def derivative(state: dict[str, float]) -> dict[str, float]:
        return {"x": -0.4 * state["x"]}

    expected = 100.0 * math.exp(-0.8)
    rk4 = integrator.integrate({"x": 100.0}, derivative, dt=2.0, substeps=13)["x"]
    coarse = integrator.integrate({"x": 100.0}, derivative, dt=2.0, substeps=1)["x"]

    rk4_error = abs(rk4 - expected)
    coarse_error = abs(coarse - expected)
    assert rk4_error < coarse_error


def test_sir_dynamics() -> None:
    integrator = CompartmentalIntegrator()

    beta = 0.3
    gamma = 0.1

    def derivative(state: dict[str, float]) -> dict[str, float]:
        s = state["S"]
        i = state["I"]
        r = state["R"]
        n = max(s + i + r, 1e-12)
        infection = beta * s * i / n
        recovery = gamma * i
        return {
            "S": -infection,
            "I": infection - recovery,
            "R": recovery,
        }

    state = {"S": 999.0, "I": 1.0, "R": 0.0}
    infected_series: list[float] = []
    for _ in range(600):
        state = integrator.integrate(
            state=state,
            derivative_fn=derivative,
            dt=0.1,
            substeps=1,
            conservation_groups=[["S", "I", "R"]],
        )
        infected_series.append(state["I"])

    peak = max(infected_series)
    assert peak > infected_series[0]
    assert infected_series[-1] < peak
    assert abs((state["S"] + state["I"] + state["R"]) - 1000.0) < 1e-2


def test_multiple_compartments() -> None:
    integrator = CompartmentalIntegrator()

    def derivative(state: dict[str, float]) -> dict[str, float]:
        return {
            "a": -0.05 * state["a"],
            "b": 0.05 * state["a"] - 0.02 * state["b"],
            "c": 0.02 * state["b"] - 0.01 * state["c"],
            "d": 0.01 * state["c"],
            "e": -0.03 * state["e"],
            "f": 0.03 * state["e"],
        }

    initial = {"a": 100.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 20.0, "f": 0.0}
    result = integrator.integrate(initial, derivative_fn=derivative, dt=4.0, substeps=52)

    assert set(result.keys()) == set(initial.keys())
    assert all(value >= 0.0 for value in result.values())

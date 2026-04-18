from __future__ import annotations

from collections.abc import Callable


class CompartmentalIntegrator:
    """Fourth-order Runge-Kutta integrator for compartmental systems."""

    def integrate(
        self,
        state: dict[str, float],
        derivative_fn: Callable[[dict[str, float]], dict[str, float]],
        dt: float,
        substeps: int,
        clamp_bounds: dict[str, tuple[float, float]] | None = None,
        conservation_groups: list[list[str]] | None = None,
    ) -> dict[str, float]:
        if substeps <= 0:
            raise ValueError("substeps must be positive")

        h = dt / float(substeps)
        current = dict(state)
        bounds = clamp_bounds or {name: (0.0, float("inf")) for name in current}

        for _ in range(substeps):
            baseline_group_totals = self._group_totals(current, conservation_groups)

            k1 = derivative_fn(current)
            k2_state = self._add_scaled(current, k1, 0.5 * h)
            k2 = derivative_fn(k2_state)
            k3_state = self._add_scaled(current, k2, 0.5 * h)
            k3 = derivative_fn(k3_state)
            k4_state = self._add_scaled(current, k3, h)
            k4 = derivative_fn(k4_state)

            updated: dict[str, float] = {}
            all_keys = set(current) | set(k1) | set(k2) | set(k3) | set(k4)
            for key in all_keys:
                x = current.get(key, 0.0)
                d1 = k1.get(key, 0.0)
                d2 = k2.get(key, 0.0)
                d3 = k3.get(key, 0.0)
                d4 = k4.get(key, 0.0)
                updated[key] = x + (h / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)

            self._clamp_state(updated, bounds)
            self._enforce_conservation(updated, baseline_group_totals, conservation_groups, bounds)
            self._clamp_state(updated, bounds)
            current = updated

        return current

    def _add_scaled(
        self,
        state: dict[str, float],
        deriv: dict[str, float],
        scale: float,
    ) -> dict[str, float]:
        out = dict(state)
        for key, value in deriv.items():
            out[key] = out.get(key, 0.0) + scale * value
        return out

    def _clamp_state(
        self,
        state: dict[str, float],
        bounds: dict[str, tuple[float, float]],
    ) -> None:
        for key, value in list(state.items()):
            lo, hi = bounds.get(key, (0.0, float("inf")))
            if value < lo:
                state[key] = lo
            elif value > hi:
                state[key] = hi

    def _group_totals(
        self,
        state: dict[str, float],
        groups: list[list[str]] | None,
    ) -> dict[int, float]:
        if not groups:
            return {}
        totals: dict[int, float] = {}
        for idx, group in enumerate(groups):
            totals[idx] = sum(state.get(name, 0.0) for name in group)
        return totals

    def _enforce_conservation(
        self,
        state: dict[str, float],
        baseline_group_totals: dict[int, float],
        groups: list[list[str]] | None,
        bounds: dict[str, tuple[float, float]],
    ) -> None:
        if not groups:
            return

        for idx, group in enumerate(groups):
            target_total = baseline_group_totals.get(idx)
            if target_total is None:
                continue
            current_total = sum(state.get(name, 0.0) for name in group)
            error = target_total - current_total
            if abs(error) <= 1e-12:
                continue

            weights = [max(state.get(name, 0.0), 0.0) for name in group]
            weight_sum = sum(weights)
            if weight_sum <= 1e-12:
                weights = [1.0 for _ in group]
                weight_sum = float(len(group))

            for name, weight in zip(group, weights):
                lo, hi = bounds.get(name, (0.0, float("inf")))
                candidate = state.get(name, 0.0) + error * (weight / weight_sum)
                state[name] = min(max(candidate, lo), hi)

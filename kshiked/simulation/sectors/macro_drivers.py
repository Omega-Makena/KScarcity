from __future__ import annotations

from functools import reduce
from operator import mul

from kshiked.simulation.sectors.config import MacroDriver

try:
    from kshiked.simulation.coupling import MacroExposure  # type: ignore
except ImportError:
    from scarcity.simulation.coupling_interface import MacroExposure


class MacroDriverSystem:
    """Applies macro conditions to transition rates."""

    def __init__(self, drivers: list[MacroDriver]) -> None:
        self._drivers = drivers
        self._transition_modifiers: dict[str, list[MacroDriver]] = {}
        for driver in drivers:
            for transition_name in driver.target_transitions:
                self._transition_modifiers.setdefault(transition_name, []).append(driver)

    def compute_rate_modifiers(self, macro: MacroExposure) -> dict[str, float]:
        modifiers: dict[str, float] = {}
        for transition_name, drivers in self._transition_modifiers.items():
            transition_effects: list[float] = []
            for driver in drivers:
                macro_value = self._resolve_macro_value(macro, driver.macro_field)
                if driver.mode == "linear":
                    value = 1.0 + driver.sensitivity * (macro_value - driver.baseline)
                elif driver.mode == "threshold":
                    threshold = driver.threshold if driver.threshold is not None else driver.baseline
                    value = 1.0 + driver.sensitivity * max(0.0, macro_value - threshold)
                else:
                    raise ValueError(f"Unsupported MacroDriver mode: {driver.mode}")
                transition_effects.append(min(max(value, 0.1), 5.0))
            modifiers[transition_name] = reduce(mul, transition_effects, 1.0)
        return modifiers

    def _resolve_macro_value(self, macro: MacroExposure, field: str) -> float:
        value = getattr(macro, field, None)
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

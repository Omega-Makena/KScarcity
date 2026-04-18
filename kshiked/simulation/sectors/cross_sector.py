from __future__ import annotations

import logging
from typing import Any

from kshiked.simulation.sectors.config import Transition

try:
    from kshiked.simulation.coupling import SectorModelRegistry  # type: ignore
except ImportError:
    SectorModelRegistry = object  # type: ignore[misc,assignment]


_LOG = logging.getLogger(__name__)


class CrossSectorResolver:
    """Resolves cross-sector transition modifiers from a registry."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def compute_cross_modifiers(self, transition: Transition) -> float:
        modifier = 1.0
        for path, sensitivity in transition.cross_sector_modifiers:
            sector_name, variable_name = self._parse_path(path)
            if sector_name is None or variable_name is None:
                _LOG.warning("Invalid cross-sector path '%s'", path)
                continue

            value = self._lookup_value(sector_name, variable_name)
            if value is None:
                _LOG.warning(
                    "Missing cross-sector reference for '%s' in transition '%s'",
                    path,
                    transition.name,
                )
                continue

            local_modifier = 1.0 + sensitivity * value
            modifier *= min(max(local_modifier, 0.1), 5.0)

        return modifier

    def _parse_path(self, path: str) -> tuple[str | None, str | None]:
        pieces = path.split(".", 1)
        if len(pieces) != 2:
            return None, None
        return pieces[0].strip(), pieces[1].strip()

    def _lookup_value(self, sector_name: str, variable_name: str) -> float | None:
        model = self._get_model(sector_name)
        if model is None:
            return None

        state_getter = getattr(model, "get_state", None)
        if callable(state_getter):
            state = state_getter()
            if isinstance(state, dict) and variable_name in state:
                return self._to_float(state[variable_name])

        indicator_getter = getattr(model, "get_indicators", None)
        if callable(indicator_getter):
            indicators = indicator_getter()
            if isinstance(indicators, dict) and variable_name in indicators:
                return self._to_float(indicators[variable_name])

        return None

    def _get_model(self, sector_name: str) -> object | None:
        getter = getattr(self._registry, "get_model", None)
        if callable(getter):
            model = getter(sector_name)
            if model is not None:
                return model

        for attr in ("models", "_models", "sector_models", "_sector_models"):
            mapping = getattr(self._registry, attr, None)
            if isinstance(mapping, dict) and sector_name in mapping:
                return mapping[sector_name]

        return None

    def _to_float(self, value: object) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

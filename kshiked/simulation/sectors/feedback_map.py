from __future__ import annotations

from dataclasses import asdict

from kshiked.simulation.sectors.config import FeedbackChannel

try:
    from kshiked.simulation.coupling import SectorFeedback  # type: ignore
except ImportError:
    from scarcity.simulation.types import Sector, SectorFeedback
else:
    from scarcity.simulation.types import Sector


class FeedbackMapper:
    """Translates sector state into SectorFeedback using declarative channels."""

    def __init__(self, channels: list[FeedbackChannel], sector_name: str) -> None:
        self._channels = channels
        self._sector_name = sector_name

    def compute_feedback(
        self,
        state: dict[str, float],
        population: float | None = None,
    ) -> SectorFeedback:
        scalar_values: dict[str, float] = {
            "labor_supply_factor": 1.0,
            "yield_factor": 1.0,
            "additional_gov_spending": 0.0,
            "fx_outflow_pressure": 0.0,
        }
        dict_values: dict[str, dict[Sector, float]] = {}

        for channel in self._channels:
            weighted = self._weighted_sum(channel, state)
            normalized = self._normalize(weighted, channel.normalization, state, population)
            transformed = self._apply_transform(channel, normalized)
            transformed = self._apply_bounds(transformed, channel.min_value, channel.max_value)

            target_field = self._canonical_field(channel.target_field)
            if target_field in scalar_values:
                scalar_values[target_field] = self._combine_scalar(
                    target_field,
                    scalar_values[target_field],
                    transformed,
                )
                continue

            if target_field in {
                "labor_productivity_factor",
                "demand_shift",
                "trade_disruption",
                "capital_destruction",
            }:
                key = self._resolve_sector_key(channel.target_sector)
                if key is None:
                    continue
                if target_field not in dict_values:
                    dict_values[target_field] = self._neutral_sector_dict(target_field)
                dict_values[target_field][key] = self._combine_dict(
                    target_field,
                    dict_values[target_field][key],
                    transformed,
                )

        kwargs = {
            "source": self._sector_name,
            "labor_supply_factor": scalar_values["labor_supply_factor"],
            "yield_factor": scalar_values["yield_factor"],
            "additional_gov_spending": scalar_values["additional_gov_spending"],
            "fx_outflow_pressure": scalar_values["fx_outflow_pressure"],
        }
        kwargs.update(dict_values)
        return SectorFeedback(**kwargs)

    def _weighted_sum(self, channel: FeedbackChannel, state: dict[str, float]) -> float:
        weights = channel.weights if channel.weights is not None else [1.0] * len(channel.source_compartments)
        return float(
            sum(
                state.get(compartment, 0.0) * weight
                for compartment, weight in zip(channel.source_compartments, weights)
            )
        )

    def _normalize(
        self,
        value: float,
        normalization: str | None,
        state: dict[str, float],
        population: float | None,
    ) -> float:
        if normalization is None:
            return value
        if normalization == "population":
            denom = population if population is not None else sum(max(v, 0.0) for v in state.values())
            return value / max(denom, 1e-12)
        return value / max(state.get(normalization, 0.0), 1e-12)

    def _apply_transform(self, channel: FeedbackChannel, value: float) -> float:
        if channel.transform == "multiplicative_reduction":
            return 1.0 - value * channel.sensitivity
        if channel.transform == "additive":
            return value * channel.sensitivity
        if channel.transform == "multiplicative_factor":
            return value
        raise ValueError(f"Unsupported transform '{channel.transform}'")

    def _apply_bounds(
        self,
        value: float,
        min_value: float | None,
        max_value: float | None,
    ) -> float:
        bounded = value
        if min_value is not None:
            bounded = max(min_value, bounded)
        if max_value is not None:
            bounded = min(max_value, bounded)
        return bounded

    def _canonical_field(self, field: str) -> str:
        aliases = {
            "fx_pressure": "fx_outflow_pressure",
            "fiscal_pressure": "additional_gov_spending",
        }
        return aliases.get(field, field)

    def _combine_scalar(self, field: str, current: float, incoming: float) -> float:
        additive_fields = {"additional_gov_spending", "fx_outflow_pressure"}
        if field in additive_fields:
            return current + incoming
        return current * incoming

    def _combine_dict(self, field: str, current: float, incoming: float) -> float:
        if field == "capital_destruction":
            x = min(max(current, 0.0), 1.0)
            y = min(max(incoming, 0.0), 1.0)
            return 1.0 - (1.0 - x) * (1.0 - y)
        return current * incoming

    def _neutral_sector_dict(self, field: str) -> dict[Sector, float]:
        if field == "capital_destruction":
            neutral_value = 0.0
        else:
            neutral_value = 1.0
        return {sector: neutral_value for sector in Sector}

    def _resolve_sector_key(self, target_sector: str | None) -> Sector | None:
        if target_sector is None:
            return None

        for sector in Sector:
            if target_sector == sector.name or target_sector == sector.name.lower():
                return sector
            if target_sector == sector.value:
                return sector
        return None

    def as_dict(self, feedback: SectorFeedback) -> dict[str, object]:
        return dict(asdict(feedback))

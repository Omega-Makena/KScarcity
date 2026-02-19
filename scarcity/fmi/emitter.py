"""
Emitter for FMI aggregated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from scarcity.runtime import EventBus, get_bus

from .aggregator import AggregationResult
from .contracts import MetaPolicyHint, MetaPriorUpdate, WarmStartProfile


@dataclass
class EmitterConfig:
    prior_broadcast_interval_windows: int = 10

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EmitterConfig":
        return cls(
            prior_broadcast_interval_windows=int(data.get("prior_broadcast_interval_windows", 10))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prior_broadcast_interval_windows": self.prior_broadcast_interval_windows,
        }


class FMIEmitter:
    """
    Publishes aggregated meta signals to the runtime bus.
    """

    META_PRIOR_TOPIC = "fmi.meta_prior_update"
    WARM_START_TOPIC = "fmi.warm_start_profile"
    POLICY_HINT_TOPIC = "fmi.meta_policy_hint"
    TELEMETRY_TOPIC = "fmi.telemetry"
    LEGACY_PRIOR_TOPIC = "meta_prior_update"
    LEGACY_POLICY_TOPIC = "meta_policy_update"

    def __init__(
        self,
        config: EmitterConfig,
        bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self.bus = bus or get_bus()
        self._last_window_sent: Optional[int] = None

    async def emit_prior_update(self, update: MetaPriorUpdate, window: Optional[int] = None) -> None:
        if not self._should_emit(window):
            return
        payload = update.as_dict()
        await self.bus.publish(self.META_PRIOR_TOPIC, payload)
        # Bridge FMI output to legacy meta topic for engine integration.
        await self.bus.publish(self.LEGACY_PRIOR_TOPIC, {"prior": update.prior, "meta": {
            "rev": update.rev,
            "cohorts": update.cohorts,
            "confidence": update.confidence,
            "contexts": update.contexts,
        }})
        if window is not None:
            self._last_window_sent = window

    async def emit_warm_start(self, profile: WarmStartProfile) -> None:
        await self.bus.publish(self.WARM_START_TOPIC, profile.as_dict())

    async def emit_policy_hint(self, hint: MetaPolicyHint) -> None:
        payload = hint.as_dict()
        await self.bus.publish(self.POLICY_HINT_TOPIC, payload)
        # Hint bundles often align with controller/evaluator/ops configs.
        if isinstance(hint.bundle, Mapping):
            await self.bus.publish(self.LEGACY_POLICY_TOPIC, hint.bundle)

    async def emit_result(self, result: AggregationResult, window: Optional[int] = None) -> None:
        if result.prior_update:
            await self.emit_prior_update(result.prior_update, window=window)
        if result.warm_start:
            await self.emit_warm_start(result.warm_start)
        if result.policy_hint:
            await self.emit_policy_hint(result.policy_hint)
        if result.telemetry:
            await self.bus.publish(self.TELEMETRY_TOPIC, result.telemetry)

    def _should_emit(self, window: Optional[int]) -> bool:
        if window is None or self._last_window_sent is None:
            return True
        delta = window - self._last_window_sent
        return delta >= self.config.prior_broadcast_interval_windows


__all__ = [
    "EmitterConfig",
    "FMIEmitter",
]



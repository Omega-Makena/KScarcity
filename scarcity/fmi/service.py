"""
High-level orchestrator for the FMI pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

from .aggregator import AggregationResult, FMIAggregator
from .contracts import FMIContractRegistry, PacketBase, PacketType
from .encoder import FMIEncoder, Precision
from .emitter import EmitterConfig, FMIEmitter
from .router import FMIRouter, RouterConfig
from .telemetry import FMITelemetry
from .validator import FMIValidator, ValidatorConfig, ValidationResult

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from . import FMIConfig


@dataclass
class ProcessOutcome:
    accepted: bool
    reason: Optional[str] = None
    cohort: Optional[str] = None
    aggregation: List[AggregationResult] = field(default_factory=list)
    quarantined: bool = False


class FMIService:
    """
    Ties together validation, routing, aggregation, and emission.
    """

    def __init__(
        self,
        config: Optional["FMIConfig"] = None,
        registry: FMIContractRegistry | None = None,
    ) -> None:
        if config is None:
            from . import load_config as _load_config

            self.config = _load_config()
        else:
            self.config = config
        self.registry = registry or FMIContractRegistry()
        self.validator = FMIValidator(self.registry, self.config.validator)
        self.router = FMIRouter(self.config.router, self.registry)
        self.encoder = FMIEncoder(self.registry, self.config.codec)
        self.aggregator = FMIAggregator(self.config.aggregation, self.registry)
        self.emitter = FMIEmitter(self.config.emitter)
        self.telemetry = FMITelemetry()

        self._adaptation_enabled = bool(self.config.drg_hooks.get("enable_adaptation", False))
        self._suspend_pop = False
        self._defer_aggregation = False

    async def ingest(self, payload: Mapping[str, Any]) -> ProcessOutcome:
        size_kb = len(json.dumps(payload, separators=(",", ":")).encode("utf-8")) / 1024
        trust = self.validator.extract_trust(payload)

        validation = self.validator.validate(payload)
        packet = validation.payload or self.registry.coerce(payload)

        self.telemetry.record_ingress(
            packet,
            size_kb=size_kb,
            trust=trust,
            dropped=validation.dropped,
            quarantined=validation.quarantined,
        )

        if not validation.ok:
            return ProcessOutcome(
                accepted=False,
                reason=validation.reason,
                quarantined=validation.quarantined,
            )

        if self._suspend_pop and packet.type in {PacketType.POP, PacketType.CCS}:
            return ProcessOutcome(
                accepted=False,
                reason="suspended_by_drg",
            )

        cohort = self.router.route(packet)
        self.telemetry.update_active_cohorts(self.router.cohort_count())

        if self._defer_aggregation:
            last_prior = self.aggregator._last_prior  # pragma: no cover - only under DRG pressure
            if last_prior:
                await self.emitter.emit_prior_update(last_prior, window=None)
            return ProcessOutcome(accepted=True, cohort=cohort, aggregation=[])

        ready = self.router.ready()
        outcomes: List[AggregationResult] = []
        for ready_cohort, cohort_packets in ready.items():
            result = self.aggregator.aggregate(ready_cohort, cohort_packets)
            outcomes.append(result)
            window_end = self._latest_window(cohort_packets)
            await self.emitter.emit_result(result, window=window_end)
            self.telemetry.record_emit({"meta_gain_delta": result.telemetry.get("meta_gain_delta", 0.0)})

        return ProcessOutcome(
            accepted=True,
            cohort=cohort,
            aggregation=outcomes,
        )

    def apply_drg_signal(self, signal: str) -> None:
        if not self._adaptation_enabled:
            return

        signal = signal.lower()
        if signal == "bandwidth_low":
            self.encoder.config.precision = Precision.Q8
        elif signal == "latency_high":
            self._suspend_pop = True
        elif signal == "vram_high":
            self._defer_aggregation = True
        elif signal == "util_low":
            self.encoder.config.precision = Precision.FP16
            self._suspend_pop = False
            self._defer_aggregation = False

    def snapshot(self) -> Dict[str, Any]:
        return self.telemetry.snapshot()

    @staticmethod
    def _latest_window(packets: Sequence[PacketBase]) -> Optional[int]:
        window_end = 0
        for packet in packets:
            span = getattr(packet, "window_span", None)
            if span:
                window_end = max(window_end, span[1])
        return window_end or None


__all__ = [
    "FMIService",
    "ProcessOutcome",
]



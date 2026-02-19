"""
Routing and cohorting logic for FMI packets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .contracts import FMIContractRegistry, PacketBase, PacketType


@dataclass
class RouterConfig:
    cohort_key: Sequence[str]
    cold_cohort_merge_interval_windows: int = 20

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RouterConfig":
        cohort_key = data.get("cohort_key", ("schema_hash", "profile_class"))
        return cls(
            cohort_key=tuple(cohort_key),
            cold_cohort_merge_interval_windows=int(
                data.get("cold_cohort_merge_interval_windows", 20)
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cohort_key": list(self.cohort_key),
            "cold_cohort_merge_interval_windows": self.cold_cohort_merge_interval_windows,
        }


class FMIRouter:
    """
    Groups packets into cohorts for aggregation.
    """

    def __init__(
        self,
        config: RouterConfig,
        registry: FMIContractRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or FMIContractRegistry()
        self._buffers: Dict[str, List[PacketBase]] = defaultdict(list)
        self._window_bounds: Dict[str, Tuple[int, int]] = {}

    def route(self, packet: Mapping[str, Any] | PacketBase) -> str:
        packet_obj = self.registry.coerce(packet)
        cohort_key = self._build_key(packet_obj)
        self._buffers[cohort_key].append(packet_obj)
        self._update_bounds(cohort_key, packet_obj)
        return cohort_key

    def _build_key(self, packet: PacketBase) -> str:
        payload = packet.as_dict()
        components = []
        for field in self.config.cohort_key:
            value = self._extract_field(payload, field)
            components.append(str(value))
        return "/".join(components)

    @staticmethod
    def _extract_field(payload: Mapping[str, Any], field: str) -> Any:
        if field in payload:
            return payload[field]
        provenance = payload.get("provenance", {})
        if isinstance(provenance, Mapping) and field in provenance:
            return provenance[field]
        return "na"

    def _update_bounds(self, cohort: str, packet: PacketBase) -> None:
        window_span = getattr(packet, "window_span", None)
        if not window_span:
            return
        start, end = window_span
        current = self._window_bounds.get(cohort)
        if current is None:
            self._window_bounds[cohort] = (start, end)
        else:
            min_start = min(current[0], start)
            max_end = max(current[1], end)
            self._window_bounds[cohort] = (min_start, max_end)

    def ready(self) -> Dict[str, List[PacketBase]]:
        ready: Dict[str, List[PacketBase]] = {}
        for cohort, packets in list(self._buffers.items()):
            if not packets:
                continue
            if any(packet.type != PacketType.MSP for packet in packets):
                ready[cohort] = packets.copy()
            elif self._span_exceeded(cohort):
                ready[cohort] = packets.copy()
            elif len(packets) >= self.config.cold_cohort_merge_interval_windows:
                ready[cohort] = packets.copy()

            if cohort in ready:
                self.clear(cohort)
        return ready

    def clear(self, cohort: str) -> None:
        self._buffers.pop(cohort, None)
        self._window_bounds.pop(cohort, None)

    def flush_all(self) -> Dict[str, List[PacketBase]]:
        ready = {cohort: packets.copy() for cohort, packets in self._buffers.items() if packets}
        self._buffers.clear()
        self._window_bounds.clear()
        return ready

    def cohort_count(self) -> int:
        return sum(1 for packets in self._buffers.values() if packets)

    def _span_exceeded(self, cohort: str) -> bool:
        if cohort not in self._window_bounds:
            return False
        start, end = self._window_bounds[cohort]
        span = end - start
        return span >= self.config.cold_cohort_merge_interval_windows


__all__ = [
    "FMIRouter",
    "RouterConfig",
]



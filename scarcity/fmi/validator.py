"""
Validation logic for FMI packets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple
import time
import math

from .contracts import FMIContractRegistry, PacketBase


@dataclass
class ValidatorConfig:
    dp_required: bool = False
    trust_min: float = 0.2
    max_packet_kb: int = 256
    max_age_seconds: Optional[float] = None
    max_value_norm: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ValidatorConfig":
        return cls(
            dp_required=bool(data.get("dp_required", False)),
            trust_min=float(data.get("trust_min", 0.2)),
            max_packet_kb=int(data.get("max_packet_kb", 256)),
            max_age_seconds=data.get("max_age_seconds"),
            max_value_norm=data.get("max_value_norm"),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dp_required": self.dp_required,
            "trust_min": self.trust_min,
            "max_packet_kb": self.max_packet_kb,
            "max_age_seconds": self.max_age_seconds,
            "max_value_norm": self.max_value_norm,
        }


@dataclass
class ValidationResult:
    ok: bool
    reason: Optional[str] = None
    quarantined: bool = False
    dropped: bool = False
    warnings: list[str] = field(default_factory=list)
    payload: Optional[PacketBase] = None

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class FMIValidator:
    """
    Applies schema, privacy, trust, and size checks.
    """

    def __init__(
        self,
        registry: FMIContractRegistry | None = None,
        config: ValidatorConfig | None = None,
    ) -> None:
        self.registry = registry or FMIContractRegistry()
        self.config = config or ValidatorConfig()

    def validate(self, payload: Mapping[str, Any]) -> ValidationResult:
        ok, issues = self.registry.validate(payload)
        if not ok:
            reason = f"schema:{'|'.join(issues)}"
            return ValidationResult(ok=False, reason=reason, dropped=True)

        packet = self.registry.coerce(payload)

        if not self._check_size(payload):
            return ValidationResult(ok=False, reason="size_limit", dropped=True)

        if not self._check_staleness(payload):
            return ValidationResult(ok=False, reason="stale_packet", dropped=True)

        if not self._check_value_norm(payload):
            return ValidationResult(ok=False, reason="norm_exceeded", dropped=True)

        trust_score = self.extract_trust(payload)
        if trust_score < self.config.trust_min:
            return ValidationResult(
                ok=False,
                reason=f"trust<{self.config.trust_min}",
                quarantined=True,
                payload=packet,
            )

        if self.config.dp_required and not self._has_dp_flag(payload):
            return ValidationResult(
                ok=False,
                reason="dp_flag_missing",
                dropped=True,
                payload=packet,
            )
        if self.config.dp_required and not self._has_dp_params(payload):
            return ValidationResult(
                ok=False,
                reason="dp_params_missing",
                dropped=True,
                payload=packet,
            )

        result = ValidationResult(ok=True, payload=packet)
        if trust_score < 0.5:
            result.add_warning("low_trust")

        return result

    def _check_size(self, payload: Mapping[str, Any]) -> bool:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return len(raw) / 1024 <= self.config.max_packet_kb

    def _check_staleness(self, payload: Mapping[str, Any]) -> bool:
        if self.config.max_age_seconds is None:
            return True
        ts = payload.get("timestamp")
        if ts is None:
            return True
        try:
            ts_val = float(ts)
        except (TypeError, ValueError):
            return False
        # Support milliseconds timestamps
        if ts_val > 1e12:
            ts_val = ts_val / 1000.0
        age = time.time() - ts_val
        return age <= float(self.config.max_age_seconds)

    def _check_value_norm(self, payload: Mapping[str, Any]) -> bool:
        limit = self.config.max_value_norm
        if limit is None:
            return True
        values = []
        for key in ("metrics", "controller", "evaluator", "operators", "bundle", "before", "after", "evidence"):
            if key in payload:
                values.extend(self._collect_numeric(payload.get(key)))
        if not values:
            return True
        norm = math.sqrt(sum(v * v for v in values))
        return norm <= float(limit)

    def _collect_numeric(self, obj: Any) -> list[float]:
        out: list[float] = []
        if isinstance(obj, Mapping):
            for value in obj.values():
                out.extend(self._collect_numeric(value))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                out.extend(self._collect_numeric(item))
        elif isinstance(obj, (int, float)) and math.isfinite(obj):
            out.append(float(obj))
        return out

    @staticmethod
    def extract_trust(payload: Mapping[str, Any]) -> float:
        provenance = payload.get("provenance", {})
        if isinstance(provenance, Mapping):
            value = provenance.get("trust") or provenance.get("trust_score")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    return 0.0
        if "trust" in payload:
            try:
                return float(payload["trust"])
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return 0.0
        return 1.0

    @staticmethod
    def _has_dp_flag(payload: Mapping[str, Any]) -> bool:
        if "dp_flag" in payload:
            return bool(payload["dp_flag"])
        privacy = payload.get("privacy")
        if isinstance(privacy, Mapping):
            dp_info = privacy.get("dp")
            if isinstance(dp_info, Mapping):
                return bool(dp_info.get("enabled", False))
            return bool(privacy.get("dp_enabled", False))
        return False

    @staticmethod
    def _has_dp_params(payload: Mapping[str, Any]) -> bool:
        def _positive(value: Any) -> bool:
            try:
                return float(value) > 0
            except (TypeError, ValueError):
                return False

        privacy = payload.get("privacy")
        if isinstance(privacy, Mapping):
            dp_info = privacy.get("dp")
            if isinstance(dp_info, Mapping):
                epsilon = dp_info.get("epsilon")
                delta = dp_info.get("delta")
                if epsilon is not None and delta is not None:
                    return _positive(epsilon) and _positive(delta)
            epsilon = privacy.get("dp_epsilon")
            delta = privacy.get("dp_delta")
            if epsilon is not None and delta is not None:
                return _positive(epsilon) and _positive(delta)
        epsilon = payload.get("dp_epsilon")
        delta = payload.get("dp_delta")
        if epsilon is not None and delta is not None:
            return _positive(epsilon) and _positive(delta)
        return False


__all__ = [
    "FMIValidator",
    "ValidationResult",
    "ValidatorConfig",
]



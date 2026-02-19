"""
Federation–Meta Interface (FMI).

The FMI package bridges the federation layer and the meta-learning layer by
standardising packet contracts, routing, validation, aggregation, and emission
of meta priors. It operates strictly on knowledge-level signals (no raw data)
and is aware of Dynamic Resource Governor (DRG) adaptation cues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .contracts import (
    FMIContractRegistry,
    MetaPolicyHint,
    MetaPriorUpdate,
    MetaSignalPack,
    PolicyOutcomePack,
    ConceptCausalSummary,
    WarmStartProfile,
)
from .encoder import CodecConfig, FMIEncoder, Precision
from .validator import FMIValidator, ValidatorConfig, ValidationResult
from .router import FMIRouter, RouterConfig
from .aggregator import FMIAggregator, AggregationConfig, AggregationResult
from .emitter import FMIEmitter, EmitterConfig
from .telemetry import FMITelemetry
from .service import FMIService, ProcessOutcome


DEFAULT_CONFIG_PATH = Path("config") / "fmi.yaml"


class FMIConfig:
    """
    Container for all FMI configuration sections.
    """

    def __init__(
        self,
        router: RouterConfig,
        validator: ValidatorConfig,
        codec: CodecConfig,
        aggregation: AggregationConfig,
        emitter: EmitterConfig,
        drg_hooks: Dict[str, Any],
    ):
        self.router = router
        self.validator = validator
        self.codec = codec
        self.aggregation = aggregation
        self.emitter = emitter
        self.drg_hooks = drg_hooks

    def as_dict(self) -> Dict[str, Any]:
        return {
            "router": self.router.as_dict(),
            "validator": self.validator.as_dict(),
            "codec": self.codec.as_dict(),
            "aggregation": self.aggregation.as_dict(),
            "emitter": self.emitter.as_dict(),
            "drg_hooks": self.drg_hooks,
        }


def load_config(path: Optional[Path] = None) -> FMIConfig:
    """
    Load FMI configuration from YAML.
    """

    if yaml is None:
        raise RuntimeError("PyYAML is required to load FMI configuration files.")

    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    router_cfg = RouterConfig.from_mapping(raw.get("router", {}))
    validator_cfg = ValidatorConfig.from_mapping(raw.get("validator", {}))
    codec_cfg = CodecConfig.from_mapping(raw.get("codec", {}))
    aggregation_cfg = AggregationConfig.from_mapping(raw.get("aggregation", {}))
    emitter_cfg = EmitterConfig.from_mapping(raw.get("emitter", {}))
    drg_hooks = raw.get("drg_hooks", {"enable_adaptation": False})

    return FMIConfig(
        router=router_cfg,
        validator=validator_cfg,
        codec=codec_cfg,
        aggregation=aggregation_cfg,
        emitter=emitter_cfg,
        drg_hooks=drg_hooks,
    )


__all__ = [
    "AggregationConfig",
    "AggregationResult",
    "ConceptCausalSummary",
    "CodecConfig",
    "FMIConfig",
    "FMIContractRegistry",
    "FMIEncoder",
    "FMIService",
    "FMIEmitter",
    "FMIAggregator",
    "FMITelemetry",
    "FMIRouter",
    "FMIValidator",
    "MetaPolicyHint",
    "MetaPriorUpdate",
    "MetaSignalPack",
    "PolicyOutcomePack",
    "ProcessOutcome",
    "Precision",
    "RouterConfig",
    "ValidationResult",
    "ValidatorConfig",
    "WarmStartProfile",
    "load_config",
]



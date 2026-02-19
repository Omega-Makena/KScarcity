"""
FMI packet contracts and schema registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar, Union

JsonDict = Dict[str, Any]
T = TypeVar("T", bound="PacketBase")


class PacketType(str, Enum):
    MSP = "msp"
    POP = "pop"
    CCS = "ccs"


def _normalise_type(value: Union[str, PacketType]) -> PacketType:
    if isinstance(value, PacketType):
        return value
    try:
        return PacketType(value.lower())
    except ValueError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unsupported packet type: {value!r}") from exc


@dataclass
class PacketBase:
    type: PacketType
    schema_hash: str
    rev: int
    domain_id: str
    profile_class: str
    timestamp: Optional[int] = None
    provenance: JsonDict = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload

    @classmethod
    def from_mapping(cls: Type[T], payload: Mapping[str, Any]) -> T:
        kwargs = dict(payload)
        kwargs["type"] = _normalise_type(kwargs["type"])
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class MetaSignalPack(PacketBase):
    window_span: Tuple[int, int] = (0, 0)
    metrics: JsonDict = field(default_factory=dict)
    controller: JsonDict = field(default_factory=dict)
    evaluator: JsonDict = field(default_factory=dict)
    operators: JsonDict = field(default_factory=dict)
    evidence: JsonDict = field(default_factory=dict)


@dataclass
class PolicyOutcomePack(PacketBase):
    bundle: JsonDict = field(default_factory=dict)
    before: JsonDict = field(default_factory=dict)
    after: JsonDict = field(default_factory=dict)
    windows: int = 0
    confidence: float = 0.0


@dataclass
class ConceptCausalSummary(PacketBase):
    causal_pairs: List[Tuple[str, str, float, str]] = field(default_factory=list)
    concepts: List[JsonDict] = field(default_factory=list)
    stability_delta: float = 0.0
    trust: float = 0.0


@dataclass
class MetaPriorUpdate:
    rev: int
    prior: JsonDict
    contexts: List[JsonDict]
    confidence: float
    cohorts: List[str]

    def as_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class WarmStartProfile:
    profile_class: str
    init: JsonDict
    context_selector: JsonDict

    def as_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class MetaPolicyHint:
    hint_id: str
    bundle: JsonDict
    bounds: JsonDict
    reason: str
    confidence: float

    def as_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class SchemaDefinition:
    packet_type: PacketType
    rev: int
    required_fields: Tuple[str, ...]
    optional_fields: Tuple[str, ...] = ()

    def validate(self, payload: Mapping[str, Any]) -> List[str]:
        missing = [field for field in self.required_fields if field not in payload]
        return missing

    def as_dict(self) -> JsonDict:
        return {
            "packet_type": self.packet_type.value,
            "rev": self.rev,
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
        }


class FMIContractRegistry:
    """
    Registry for FMI packet schemas.
    """

    def __init__(self) -> None:
        self._schemas: Dict[PacketType, SchemaDefinition] = {}
        self.register_default_schemas()

    def register(self, schema: SchemaDefinition) -> None:
        self._schemas[schema.packet_type] = schema

    def register_default_schemas(self) -> None:
        self.register(
            SchemaDefinition(
                packet_type=PacketType.MSP,
                rev=3,
                required_fields=(
                    "type",
                    "schema_hash",
                    "rev",
                    "domain_id",
                    "profile_class",
                    "metrics",
                    "controller",
                    "evaluator",
                    "operators",
                ),
                optional_fields=("window_span", "timestamp", "evidence", "provenance"),
            )
        )
        self.register(
            SchemaDefinition(
                packet_type=PacketType.POP,
                rev=2,
                required_fields=(
                    "type",
                    "schema_hash",
                    "rev",
                    "domain_id",
                    "profile_class",
                    "bundle",
                    "before",
                    "after",
                    "windows",
                    "confidence",
                ),
                optional_fields=("timestamp", "provenance"),
            )
        )
        self.register(
            SchemaDefinition(
                packet_type=PacketType.CCS,
                rev=1,
                required_fields=(
                    "type",
                    "schema_hash",
                    "rev",
                    "domain_id",
                    "profile_class",
                    "causal_pairs",
                    "concepts",
                    "stability_delta",
                    "trust",
                ),
                optional_fields=("timestamp", "provenance"),
            )
        )

    def get(self, packet_type: Union[str, PacketType]) -> SchemaDefinition:
        packet_type = _normalise_type(packet_type)
        if packet_type not in self._schemas:
            raise KeyError(f"No schema registered for {packet_type.value}")
        return self._schemas[packet_type]

    def validate(self, payload: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        packet_type = _normalise_type(payload.get("type", ""))
        schema = self.get(packet_type)
        missing = schema.validate(payload)
        if missing:
            return False, missing
        if payload.get("rev") != schema.rev:
            return False, [f"rev:{payload.get('rev')}!=schema_rev:{schema.rev}"]
        return True, []

    @staticmethod
    def coerce(packet: Union[PacketBase, Mapping[str, Any]]) -> PacketBase:
        if is_dataclass(packet):
            return packet  # type: ignore[return-value]

        packet_type = _normalise_type(packet.get("type"))
        mapping = dict(packet)

        constructors: Dict[PacketType, Type[PacketBase]] = {
            PacketType.MSP: MetaSignalPack,
            PacketType.POP: PolicyOutcomePack,
            PacketType.CCS: ConceptCausalSummary,
        }

        constructor = constructors.get(packet_type)
        if constructor is None:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported packet type {packet_type.value}")
        mapping["type"] = packet_type
        return constructor.from_mapping(mapping)  # type: ignore[return-value]


__all__ = [
    "ConceptCausalSummary",
    "FMIContractRegistry",
    "MetaPolicyHint",
    "MetaPriorUpdate",
    "MetaSignalPack",
    "PacketBase",
    "PacketType",
    "PolicyOutcomePack",
    "SchemaDefinition",
    "WarmStartProfile",
]



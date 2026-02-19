"""
Encoding and quantisation utilities for FMI packets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, MutableMapping, Union

import numpy as np  # type: ignore

try:  # pragma: no cover - optional dependency
    import zstandard as zstd  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    zstd = None

import zlib

from .contracts import FMIContractRegistry, PacketBase


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    Q8 = "q8"


@dataclass
class CodecConfig:
    precision: Precision = Precision.FP16
    compression: str = "zstd"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CodecConfig":
        precision = data.get("precision", cls.precision.value).lower()
        compression = data.get("compression", "zstd").lower()
        return cls(precision=Precision(precision), compression=compression)

    def as_dict(self) -> Dict[str, Any]:
        return {"precision": self.precision.value, "compression": self.compression}


class FMIEncoder:
    """
    Serialises FMI packets with configurable numeric precision and compression.
    """

    def __init__(
        self,
        registry: FMIContractRegistry | None = None,
        config: CodecConfig | None = None,
    ) -> None:
        self.registry = registry or FMIContractRegistry()
        self.config = config or CodecConfig()
        self._compressor = self._build_compressor(self.config.compression)
        self._decompressor = self._build_decompressor(self.config.compression)

    @staticmethod
    def _build_compressor(name: str):
        name = name.lower()
        if name == "zstd" and zstd is not None:  # pragma: no branch - runtime check
            compressor = zstd.ZstdCompressor(level=6)
            return compressor.compress
        if name in {"zstd", "zlib"}:
            return lambda data: zlib.compress(data, level=6)
        raise ValueError(f"Unsupported compression codec: {name}")

    @staticmethod
    def _build_decompressor(name: str):
        name = name.lower()
        if name == "zstd" and zstd is not None:  # pragma: no branch - runtime check
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress
        if name in {"zstd", "zlib"}:
            return zlib.decompress
        raise ValueError(f"Unsupported compression codec: {name}")

    def encode(self, packet: Union[PacketBase, Mapping[str, Any]]) -> bytes:
        packet_instance = self.registry.coerce(packet)
        payload = packet_instance.as_dict()
        quantised = self._quantise_payload(payload)
        raw = json.dumps(
            quantised,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return self._compressor(raw)

    def decode(self, blob: bytes) -> Dict[str, Any]:
        decompressed = self._decompressor(blob)
        data: Dict[str, Any] = json.loads(decompressed.decode("utf-8"))
        return self._dequantise_payload(data)

    # --------------------------------------------------------------------- #
    # Quantisation helpers
    # --------------------------------------------------------------------- #

    def _quantise_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if self.config.precision == Precision.FP32:
            return dict(payload)

        def quantise(value: Any) -> Any:
            if isinstance(value, float):
                if self.config.precision == Precision.FP16:
                    return float(np.float16(value))
                if self.config.precision == Precision.Q8:
                    # Round to the nearest cent (10^-2) which ensures ≤1e-2 tolerance.
                    return float(np.round(value, 2))
            if isinstance(value, list):
                return [quantise(item) for item in value]
            if isinstance(value, dict):
                return {k: quantise(v) for k, v in value.items()}
            return value

        return {k: quantise(v) for k, v in payload.items()}

    def _dequantise_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        # The quantisation strategy keeps floats as floats, so we simply deep copy.
        def copy(value: Any) -> Any:
            if isinstance(value, list):
                return [copy(item) for item in value]
            if isinstance(value, dict):
                return {k: copy(v) for k, v in value.items()}
            return value

        return {k: copy(v) for k, v in payload.items()}


__all__ = [
    "CodecConfig",
    "FMIEncoder",
    "Precision",
]



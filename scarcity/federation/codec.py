"""
Compression and quantisation helper for federation payloads.

This module provides the `PayloadCodec` class, which handles the serialization,
compression (e.g., zstd, lz4), and quantization (e.g., fp16, int8) of data packets
to optimize network bandwidth usage in federated learning.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np


@dataclass
class CodecConfig:
    """Configuration for payload encoding."""
    compression: str = "zstd"
    quantisation: str = "fp16"


class PayloadCodec:
    """
    Handles encoding/decoding and quantization of payloads.
    """

    def __init__(self, config: CodecConfig):
        """
        Initialize the codec.

        Args:
            config: Codec configuration object.
        """
        self.config = config

    def encode(self, payload: Dict[str, Any]) -> bytes:
        """
        Serialize and compress a dictionary payload.

        Args:
            payload: Dictionary to encode.

        Returns:
            Compressed byte string.
        """
        serialised = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return self._compress(serialised)

    def decode(self, blob: bytes) -> Dict[str, Any]:
        """
        Decompress and deserialize a byte string.

        Args:
            blob: Compressed byte string.

        Returns:
            The decoded dictionary payload.
        """
        decompressed = self._decompress(blob)
        return json.loads(decompressed.decode("utf-8"))

    def quantise(self, array: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Quantize a numpy array to reduce size.

        Args:
            array: Input numpy array.

        Returns:
            Tuple of (quantised_array, metadata_string).
        """
        if self.config.quantisation == "q8":
            scale = np.max(np.abs(array)) / 127.0 + 1e-8
            quantised = np.round(array / scale).astype(np.int8)
            metadata = f"q8:{scale}"
            return quantised, metadata
        if self.config.quantisation == "fp16":
            return array.astype(np.float16), "fp16"
        return array.astype(np.float32), "fp32"

    def dequantise(self, array: np.ndarray, metadata: str) -> np.ndarray:
        """
        Dequantize a numpy array back to float32.

        Args:
            array: Quantised numpy array.
            metadata: Metadata string returned during quantization.

        Returns:
            Reconstructed float32 numpy array.
        """
        if metadata.startswith("q8:"):
            scale = float(metadata.split(":")[1])
            return array.astype(np.float32) * scale
        if metadata == "fp16":
            return array.astype(np.float32)
        return array.astype(np.float32)

    def _compress(self, data: bytes) -> bytes:
        """Apply configured compression."""
        if self.config.compression == "zstd":
            return zlib.compress(data, level=9)
        if self.config.compression == "lz4":
            return zlib.compress(data, level=3)
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data using configured algorithm."""
        if self.config.compression in {"zstd", "lz4"}:
            return zlib.decompress(data)
        return data

"""
Context encoder for meta-learning memory retrieval.

Converts heterogeneous context dicts into fixed-size float32 embedding vectors
used as keys in the episodic memory store. Encoding is fully deterministic —
identical inputs always produce identical outputs.

Output layout (default 32 dims):
  [0:16]  named numeric slots (known system metrics and params)
  [16:24] domain identity embedding (hash-projected from domain_id string)
  [24:32] overflow region (unknown numeric keys, hash-accumulated)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Ordered list of known numeric fields and their value scales.
# Scale is the divisor applied before tanh — keeps values in a useful range.
_NAMED_SLOTS: List[Tuple[str, float]] = [
    ("gain_p50",        1.0),
    ("gain_mean",       1.0),
    ("stability_mean",  1.0),
    ("stability_avg",   1.0),
    ("vram_high",       1.0),
    ("latency_ms",      1000.0),
    ("bandwidth_free",  1.0),
    ("bandwidth_low",   1.0),
    ("tau",             1.0),
    ("gamma_diversity", 1.0),
    ("g_min",           0.1),
    ("lambda_ci",       1.0),
    ("meta_score",      1.0),
    ("confidence",      1.0),
    ("score_delta",     0.5),
    ("reward_ema",      1.0),
]

_NAMED_DIM: int = len(_NAMED_SLOTS)  # 16
_SLOT_INDEX: Dict[str, int] = {name: i for i, (name, _) in enumerate(_NAMED_SLOTS)}
_SLOT_SCALE: Dict[str, float] = {name: scale for name, scale in _NAMED_SLOTS}


@dataclass
class ContextEncoderConfig:
    """Configuration for ContextEncoder."""
    domain_dim: int = 8     # dims reserved for domain identity
    overflow_dim: int = 8   # dims reserved for unknown keys
    normalize: bool = True  # L2-normalize the final vector

    @property
    def output_dim(self) -> int:
        return _NAMED_DIM + self.domain_dim + self.overflow_dim


class ContextEncoder:
    """
    Encodes a context dict into a fixed-size float32 embedding vector.

    The vector has three regions:
    - Named slots: known numeric fields mapped to fixed indices, scaled and
      passed through tanh to bound values.
    - Domain embedding: deterministic hash-projection of the domain_id string
      into a unit-norm vector, capturing domain identity.
    - Overflow: unknown numeric keys are hash-accumulated into a shared region,
      capturing any extra signal without requiring schema changes.

    Encoding is fully deterministic and requires no training.
    """

    def __init__(self, config: Optional[ContextEncoderConfig] = None):
        self.config = config or ContextEncoderConfig()
        self._domain_cache: Dict[str, np.ndarray] = {}

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def encode(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Encode a context dict into an embedding vector.

        Args:
            context: Flat dict containing any mix of known and unknown numeric
                     fields, plus an optional 'domain_id' string.

        Returns:
            float32 array of shape (output_dim,).
        """
        cfg = self.config
        vec = np.zeros(cfg.output_dim, dtype=np.float32)

        # --- named slots ---
        for key, value in context.items():
            if key == "domain_id":
                continue
            idx = _SLOT_INDEX.get(key)
            if idx is not None:
                scale = _SLOT_SCALE[key]
                vec[idx] = np.tanh(float(value) / scale)

        # --- domain embedding ---
        domain_id = str(context.get("domain_id", ""))
        if domain_id:
            d_start = _NAMED_DIM
            d_end = _NAMED_DIM + cfg.domain_dim
            vec[d_start:d_end] = self._domain_embedding(domain_id)

        # --- overflow (unknown numeric keys) ---
        o_start = _NAMED_DIM + cfg.domain_dim
        overflow = np.zeros(cfg.overflow_dim, dtype=np.float32)
        has_overflow = False
        for key, value in context.items():
            if key == "domain_id" or key in _SLOT_INDEX:
                continue
            if not isinstance(value, (int, float)):
                continue
            proj = self._key_projection(key, cfg.overflow_dim)
            overflow += float(value) * proj
            has_overflow = True
        if has_overflow:
            norm = np.linalg.norm(overflow)
            if norm > 1e-8:
                overflow /= norm
        vec[o_start:] = overflow

        if cfg.normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm

        return vec

    def encode_batch(self, contexts: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode a list of context dicts.

        Returns:
            float32 array of shape (len(contexts), output_dim).
        """
        return np.stack([self.encode(c) for c in contexts], axis=0)

    def named_slots(self) -> List[str]:
        """Return the ordered list of named slot keys."""
        return [name for name, _ in _NAMED_SLOTS]

    def _domain_embedding(self, domain_id: str) -> np.ndarray:
        """Deterministic unit-norm embedding for a domain_id string."""
        cached = self._domain_cache.get(domain_id)
        if cached is not None:
            return cached
        seed = int(hashlib.sha256(domain_id.encode()).hexdigest(), 16) % (2 ** 31)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.config.domain_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        self._domain_cache[domain_id] = vec
        return vec

    def _key_projection(self, key: str, dim: int) -> np.ndarray:
        """Deterministic unit-norm projection vector for an unknown key string."""
        seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

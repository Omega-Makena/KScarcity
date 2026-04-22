"""
Episodic memory store for meta-learning.

Stores experience episodes as (key, value, context, delta, policy) tuples in a
fixed-capacity ring buffer. Retrieval uses cosine similarity against key embeddings
produced by ContextEncoder.

Capacity policy: when the buffer is full, the oldest entry is evicted (FIFO).
Tie-breaking in retrieval: higher insertion timestamp wins (most recent first).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EpisodicEntry:
    """A single stored episode."""
    key: np.ndarray          # float32 embedding from ContextEncoder
    value: Dict[str, Any]    # retrieved model parameters or adaptation target
    context: Dict[str, Any]  # raw context dict that produced this episode
    delta: Dict[str, float]  # performance deltas observed after applying value
    policy: Dict[str, Any]   # metadata: domain_id, step, source, etc.
    timestamp: int           # monotonic insertion counter — used for tie-breaking


@dataclass
class RetrievalResult:
    """A single retrieval hit."""
    entry: EpisodicEntry
    similarity: float
    rank: int


@dataclass
class EpisodicMemoryConfig:
    capacity: int = 1024       # max episodes to retain
    top_k: int = 5             # episodes returned per query
    min_similarity: float = 0.0  # cosine threshold; 0.0 returns unconditionally


class EpisodicMemory:
    """
    Fixed-capacity episodic memory buffer with cosine-similarity retrieval.

    Thread-safe: a single lock guards all mutations and full scans. Retrieval
    copies the key matrix before computing similarities, so writers are blocked
    only during the copy, not during numpy operations.
    """

    def __init__(self, config: Optional[EpisodicMemoryConfig] = None):
        self.config = config or EpisodicMemoryConfig()
        self._entries: List[EpisodicEntry] = []
        self._head: int = 0          # next write position (ring buffer)
        self._counter: int = 0       # monotonic timestamp
        self._lock = threading.Lock()

        cfg = self.config
        if cfg.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {cfg.capacity}")
        if cfg.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {cfg.top_k}")

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def store(
        self,
        key: np.ndarray,
        value: Dict[str, Any],
        context: Dict[str, Any],
        delta: Dict[str, float],
        policy: Dict[str, Any],
    ) -> None:
        """
        Store an episode. Evicts the oldest entry when capacity is reached.

        Args:
            key:     float32 embedding vector from ContextEncoder.encode()
            value:   model parameter snapshot or adaptation target
            context: raw context dict used to produce key
            delta:   performance deltas observed after applying value
            policy:  freeform metadata (domain_id, step, source, ...)
        """
        key = np.array(key, dtype=np.float32)
        entry = EpisodicEntry(
            key=key,
            value=value,
            context=context,
            delta=delta,
            policy=policy,
            timestamp=self._counter,
        )
        with self._lock:
            self._counter += 1
            cap = self.config.capacity
            if len(self._entries) < cap:
                self._entries.append(entry)
            else:
                self._entries[self._head] = entry
                self._head = (self._head + 1) % cap

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_key: np.ndarray,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Return the top-k most similar episodes.

        Args:
            query_key:      float32 embedding from ContextEncoder.encode()
            top_k:          overrides config.top_k for this call
            min_similarity: cosine threshold override; entries below are excluded

        Returns:
            List of RetrievalResult ordered by similarity descending, ties
            broken by recency (higher timestamp first).
        """
        k = top_k if top_k is not None else self.config.top_k
        threshold = min_similarity if min_similarity is not None else self.config.min_similarity

        with self._lock:
            if not self._entries:
                return []
            entries = list(self._entries)   # snapshot under lock

        query = np.array(query_key, dtype=np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-8:
            return []

        keys = np.stack([e.key for e in entries], axis=0)  # (n, dim)
        norms = np.linalg.norm(keys, axis=1)               # (n,)
        valid = norms > 1e-8
        sims = np.where(valid, keys @ query / np.maximum(norms * q_norm, 1e-8), -2.0)

        candidates: List[Tuple[float, int, int]] = []  # (sim, -timestamp, idx)
        for i, sim in enumerate(sims):
            if sim >= threshold:
                candidates.append((float(sim), -entries[i].timestamp, i))

        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        top = candidates[:k]

        return [
            RetrievalResult(entry=entries[idx], similarity=sim, rank=rank)
            for rank, (sim, _, idx) in enumerate(top)
        ]

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def capacity(self) -> int:
        return self.config.capacity

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._head = 0
            self._counter = 0

    def keys_matrix(self) -> np.ndarray:
        """Return a snapshot of all stored keys as a (n, dim) float32 array."""
        with self._lock:
            if not self._entries:
                return np.empty((0,), dtype=np.float32)
            return np.stack([e.key for e in self._entries], axis=0)

    def timestamps(self) -> List[int]:
        with self._lock:
            return [e.timestamp for e in self._entries]

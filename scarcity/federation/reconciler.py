"""
Utilities for merging aggregated federation updates into the local store.

This module provides the `StoreReconciler` class, which takes validated packets
(PathPacks, EdgeDeltas, CausalSemanticPacks) and applies their content to the
local `HypergraphStore`. It handles weighting, decay factors, and regime mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from scarcity.engine.store import HypergraphStore
from .layers import Layer2Config
from .packets import EdgeDelta, PathPack, CausalSemanticPack


@dataclass
class ReconcilerConfig:
    """Configuration for StoreReconciler."""
    decay_factor: float = 0.05
    min_weight: float = 1e-4
    min_support_baskets: int = 2
    min_support_from_layer2: Optional[int] = None
    dynamic_support_enabled: bool = True
    dynamic_support_ratio: float = 0.1
    dynamic_support_min: int = 2
    dynamic_support_max: Optional[int] = None


class StoreReconciler:
    """
    Merging logic for incoming federation updates.
    """

    def __init__(self, store: HypergraphStore, config: Optional[ReconcilerConfig] = None):
        """
        Initialize the reconciler.

        Args:
            store: The target HypergraphStore.
            config: Reconciler configuration.
        """
        self.store = store
        self.config = config or ReconcilerConfig()
        self._edge_support: Dict[tuple[int, int], set[int]] = {}
        self._known_baskets: set[int] = set()

    def set_layer2_min_support(self, value: int) -> None:
        """Optionally align support threshold with Layer2 config."""
        self.config.min_support_from_layer2 = value

    def _effective_min_support(self) -> int:
        base = self.config.min_support_baskets
        layer2 = self.config.min_support_from_layer2 or 0
        dynamic = 0
        if self.config.dynamic_support_enabled:
            dynamic = max(
                self.config.dynamic_support_min,
                int(np.ceil(self.config.dynamic_support_ratio * max(1, len(self._known_baskets)))),
            )
            if self.config.dynamic_support_max is not None:
                dynamic = min(dynamic, self.config.dynamic_support_max)
        return max(base, layer2, dynamic)

    def _record_support(self, edge: tuple[int, int], basket_id: int) -> int:
        if edge not in self._edge_support:
            self._edge_support[edge] = set()
        self._edge_support[edge].add(basket_id)
        self._known_baskets.add(basket_id)
        return len(self._edge_support[edge])

    def merge_path_pack(self, pack: PathPack) -> Dict[str, int]:
        """
        Merge a PathPack into the store.

        Args:
            pack: The PathPack to merge.

        Returns:
            Dictionary with counts of updated edges and inserted hyperedges.
        """
        inserted = 0
        updated = 0

        for src, dst, weight, ci, stability, regime in pack.edges:
            support = self._record_support((int(src), int(dst)), int(pack.domain_id))
            if support < self._effective_min_support():
                continue
            if abs(weight) < self.config.min_weight:
                continue
            self.store.upsert_edge(
                src_id=int(src),
                dst_id=int(dst),
                effect=weight,
                ci_lo=-ci,
                ci_hi=ci,
                stability=stability,
                regime_id=regime,
            )
            updated += 1

        for hyper in pack.hyperedges:
            sources = hyper.get("S", [])
            if not sources:
                continue
            self.store.upsert_hyperedge(
                sources=[int(s) for s in sources],
                effect=hyper.get("w", 0.0),
                ci_lo=hyper.get("ci", 0.0),
                ci_hi=hyper.get("ci", 0.0),
                stability=hyper.get("st", 0.5),
                regime_id=hyper.get("reg", -1),
            )
            inserted += 1

        return {"edges_updated": updated, "hyperedges_inserted": inserted}

    def merge_edge_delta(self, delta: EdgeDelta) -> Dict[str, int]:
        """
        Apply an EdgeDelta (upserts and prunes) to the store.

        Args:
            delta: The EdgeDelta to apply.

        Returns:
            Dictionary with counts of updated and pruned edges.
        """
        updated = 0
        pruned = 0
        for key, weight_delta, stability_delta, hits_delta, regime, last_seen in delta.upserts:
            src, dst = key.split("->")
            support = self._record_support((int(src), int(dst)), int(delta.domain_id))
            if support < self._effective_min_support():
                continue
            edge = self.store.get_edge(int(src), int(dst))
            weight = weight_delta
            stability = stability_delta
            if edge is not None:
                # Apply decay and update
                weight = (1 - self.config.decay_factor) * edge.weight + weight_delta
                stability = max(edge.stability, stability_delta)
            self.store.upsert_edge(
                src_id=int(src),
                dst_id=int(dst),
                effect=weight,
                ci_lo=edge.ci_lo if edge else -0.1,
                ci_hi=edge.ci_hi if edge else 0.1,
                stability=stability,
                regime_id=regime,
                ts=last_seen,
            )
            updated += 1

        for key in delta.prunes:
            try:
                src, dst = key.split("->")
                if self.store.remove_edge(int(src), int(dst)):
                    pruned += 1
            except ValueError:
                continue # Malformed key

        return {"edges_updated": updated, "edges_pruned": pruned}

    def merge_causal_pack(self, pack: CausalSemanticPack) -> Dict[str, int]:
        """
        Merge causal pairs from a CausalSemanticPack.

        Args:
            pack: The CausalSemanticPack to merge.

        Returns:
            Dictionary with count of accepted pairs.
        """
        accepted = 0
        for pair in pack.pairs:
            support = self._record_support((int(pair.source), int(pair.target)), int(pack.domain_id))
            if support < self._effective_min_support():
                continue
            self.store.upsert_edge(
                src_id=int(pair.source),
                dst_id=int(pair.target),
                effect=pair.probability,
                ci_lo=-0.1,
                ci_hi=0.1,
                stability=max(0.5, pair.probability),
                regime_id=pair.regime or -1,
            )
            accepted += 1
        return {"causal_pairs": accepted}


def build_reconciler(
    store: HypergraphStore,
    reconciler_config: Optional[ReconcilerConfig] = None,
    layer2_config: Optional[Layer2Config] = None,
) -> StoreReconciler:
    """
    Build a StoreReconciler wired to Layer2 minimum support settings.

    Args:
        store: Target HypergraphStore.
        reconciler_config: Optional reconciler configuration.
        layer2_config: Optional Layer2 configuration (for min support alignment).

    Returns:
        Configured StoreReconciler instance.
    """
    reconciler = StoreReconciler(store, config=reconciler_config)
    l2 = layer2_config or Layer2Config()
    reconciler.set_layer2_min_support(l2.min_basket_support)
    return reconciler

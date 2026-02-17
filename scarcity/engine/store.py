"""
HypergraphStore — Online hypergraph memory for MPIE.

Maintains compact, evolving graph of relationships with edges, hyperedges,
regimes, partitions, indexing, decay, and schema versioning.
"""

import logging
import numpy as np
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import heapq

logger = logging.getLogger(__name__)


@dataclass
class EdgeRec:
    """
    Data structure representing a stored causal edge.

    Attributes:
        weight: The exponential moving average (EMA) of the effect size.
        var: Running variance of the effect size.
        stability: EMA of the stability score ([0, 1] agreement metric).
        ci_lo: The lower bound of the confidence interval (fp16 range).
        ci_hi: The upper bound of the confidence interval (fp16 range).
        regime_id: Identifier for the regime where this edge is valid (-1 = global).
        last_seen: The window ID when this edge was last updated.
        hits: Total number of times this edge has been accepted/verified.
    """
    weight: float  # EMA of effect (fp32)
    var: float  # running variance (fp32)
    stability: float  # EMA of stability (fp32)
    ci_lo: float  # fp16
    ci_hi: float  # fp16
    regime_id: int  # -1 if global
    last_seen: int  # window_id
    hits: int  # acceptance count
    
    def to_dict(self) -> Dict:
        """Converts the edge record to a dictionary serialization."""
        return asdict(self)


@dataclass
class HyperRec:
    """
    Data structure representing a stored hyperedge.
    
    Used to track multi-variable interactions where multiple sources jointly
    influence a target, or complex dependency structures.

    Attributes:
        order: The number of source variables in the hyperedge.
        weight: The aggregate effect size of the interaction.
        stability: Stability score of the hyperedge relationship.
        ci_lo: Lower confidence bound.
        ci_hi: Upper confidence bound.
        regime_id: Regime validity identifier.
        last_seen: Last update window ID.
        hits: Count of verifications.
    """
    order: int  # number of sources
    weight: float
    stability: float
    ci_lo: float
    ci_hi: float
    regime_id: int
    last_seen: int
    hits: int
    
    def to_dict(self) -> Dict:
        """Converts the hyperedge record to a dictionary serialization."""
        return asdict(self)


class HypergraphStore:
    """
    Online hypergraph store with bounded memory management.

    This class serves as the long-term memory for the inference engine. It persists
    discovered causal relationships (edges) and complex interactions (hyperedges).
    It implements:
    - Bounded capacity: Automatically pruning weak or stale edges when limits are reached.
    - Decay: Exponential decay of weights over time to favor recent evidence.
    - Indexing: Efficient lookups for neighbors and incoming/outgoing edges.
    - Partitioning: Organizing edges by domain for federated or segmented views.
    - Garbage Collection: Managing lifecycle via periodic pruning cycles.
    """
    
    def __init__(
        self,
        max_edges: int = 10000,
        max_hyperedges: int = 1000,
        topk_per_node: int = 32,
        decay_factor: float = 0.995,
        alpha_weight: float = 0.2,
        alpha_stability: float = 0.2,
        gc_interval: int = 25
    ):
        """
        Initializes the hypergraph store with capacity and learning parameters.

        Args:
            max_edges: The soft limit on the number of edges to store before aggressive pruning.
            max_hyperedges: The soft limit on the number of hyperedges.
            topk_per_node: The maximum number of top-ranked neighbors to maintain in indices per node.
            decay_factor: The multiplicative factor applied to weights each window (e.g., 0.995).
            alpha_weight: The learning rate (EMA alpha) for updating edge weights.
            alpha_stability: The learning rate (EMA alpha) for updating stability scores.
            gc_interval: The number of windows between garbage collection (pruning) cycles.
        """
        self.max_edges = max_edges
        self.max_hyperedges = max_hyperedges
        self.topk_per_node = topk_per_node
        self.decay_factor = decay_factor
        self.alpha_weight = alpha_weight
        self.alpha_stability = alpha_stability
        self.gc_interval = gc_interval
        
        # Node registry
        self.nodes: Dict[int, Dict[str, Any]] = {}  # node_id -> {name, domain, schema_ver, flags}
        
        # Edge storage: (src_id, dst_id) -> EdgeRec
        self.edges: Dict[Tuple[int, int], EdgeRec] = {}
        
        # Hyperedge storage: frozenset(sources) -> HyperRec
        self.hyperedges: Dict[frozenset, HyperRec] = {}
        
        # Indexes: node_id -> heap of (score, neighbor_id, key)
        self.out_index: Dict[int, List[Tuple[float, int, Tuple[int, int]]]] = defaultdict(list)
        self.in_index: Dict[int, List[Tuple[float, int, Tuple[int, int]]]] = defaultdict(list)
        
        # Domain partitions: domain_id -> Set[edge_keys]
        self.domain_partitions: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        
        # Regime buffers: edge_key -> List[regime_counts]
        self.regime_buffers: Dict[Tuple[int, int], List[int]] = defaultdict(lambda: [0] * 4)
        
        # State
        self.current_window = 0
        self.schema_hash = None
        self.next_node_id = 0
        self.gc_counter = 0
        
        # Statistics
        self._stats = {
            'edges_added': 0,
            'edges_pruned': 0,
            'hyperedges_added': 0,
            'hyperedges_pruned': 0,
            'promotions': 0,
            'demotions': 0,
            'gc_cycles': 0
        }
        
        logger.info(
            f"HypergraphStore initialized: max_edges={max_edges}, "
            f"max_hyperedges={max_hyperedges}, topk={topk_per_node}"
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize variable names to avoid duplicate IDs."""
        return " ".join(name.split()).strip().lower()
    
    def get_or_create_node(self, name: str, domain: int = 0, schema_ver: int = 0) -> int:
        """
        Retrieves an existing node ID or creates a new one for the given name.

        Ensures that variable names are uniquely mapped to integer IDs within the
        store. Checks against name and schema version to handle schema evolution.

        Args:
            name: The string name of the variable/node.
            domain: The integer identifier for the domain the node belongs to.
            schema_ver: The schema version number to distinguish changes over time.

        Returns:
            The unique integer ID for the node.
        """
        norm_name = self._normalize_name(name)

        # Check if node exists
        for node_id, node_data in self.nodes.items():
            existing_norm = node_data.get("name_norm") or self._normalize_name(node_data["name"])
            if existing_norm == norm_name and node_data['schema_ver'] == schema_ver:
                return node_id
        
        # Create new node
        node_id = self.next_node_id
        self.next_node_id += 1
        
        self.nodes[node_id] = {
            'name': name,
            'name_norm': norm_name,
            'domain': domain,
            'schema_ver': schema_ver,
            'flags': 0
        }
        
        return node_id
    
    def upsert_edge(
        self,
        src_id: int,
        dst_id: int,
        effect: float,
        ci_lo: float,
        ci_hi: float,
        stability: float,
        regime_id: int = -1,
        ts: Optional[int] = None
    ) -> None:
        """
        Inserts a new edge or updates an existing one with new evidence.

        If the edge exists, its weight and stability are updated using EMA.
        Confidence intervals and metadata are overwritten with the latest values.
        If the edge is new, a new record is created.

        Also performs sanity checks, such as down-weighting edges with excessive
        uncertainty (wide CIs).

        Args:
            src_id: The ID of the source node.
            dst_id: The ID of the target node.
            effect: The estimated causal effect size.
            ci_lo: The lower bound of the confidence interval.
            ci_hi: The upper bound of the confidence interval.
            stability: The stability score from the evaluator.
            regime_id: The regime ID under which this relationship was observed.
            ts: The timestamp (window ID) of the observation. Defaults to current.
        """
        if ts is None:
            ts = self.current_window
        
        key = (src_id, dst_id)
        
        if key in self.edges:
            # Update existing edge
            edge = self.edges[key]
            
            # EMA update for weight
            edge.weight = (1 - self.alpha_weight) * edge.weight + self.alpha_weight * effect
            
            # EMA update for stability
            edge.stability = (1 - self.alpha_stability) * edge.stability + self.alpha_stability * stability
            
            # Update CI
            edge.ci_lo = ci_lo
            edge.ci_hi = ci_hi
            
            # Update metadata
            edge.regime_id = regime_id
            edge.last_seen = ts
            edge.hits += 1
            
            # Update CI sanity check (only down-weight if too uncertain)
            ci_width = ci_hi - ci_lo
            if abs(edge.weight) > 1e-6 and ci_width > 0.5 * abs(edge.weight):
                edge.weight *= 0.5  # Down-weight uncertain edges
                logger.debug(f"Edge {key} down-weighted due to high CI width: {ci_width}")
            
        else:
            # Create new edge
            self.edges[key] = EdgeRec(
                weight=effect,
                var=0.0,
                stability=stability,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                regime_id=regime_id,
                last_seen=ts,
                hits=1
            )
            
            self._stats['edges_added'] += 1
        
        # Update indexes
        self._update_indexes(key)
        
        # Update domain partition
        if src_id in self.nodes:
            domain = self.nodes[src_id]['domain']
            self.domain_partitions[domain].add(key)
    
    def upsert_hyperedge(
        self,
        sources: List[int],
        effect: float,
        ci_lo: float,
        ci_hi: float,
        stability: float,
        regime_id: int = -1,
        ts: Optional[int] = None,
    ) -> None:
        """
        Inserts or updates a hyperedge record.

        Similar to `upsert_edge` but handles multi-source relationships identified
        by a set of source node IDs.

        Args:
            sources: A list of source node IDs involved in the hyperedge.
            effect: The estimated aggregate effect size.
            ci_lo: Lower confidence bound.
            ci_hi: Upper confidence bound.
            stability: Stability score.
            regime_id: Regime identifier.
            ts: Timestamp (window ID) of the observation.
        """
        if not sources:
            return
        key = frozenset(int(src) for src in sources)
        timestamp = ts if ts is not None else self.current_window

        if key in self.hyperedges:
            hyper = self.hyperedges[key]
            hyper.weight = (1 - self.alpha_weight) * hyper.weight + self.alpha_weight * effect
            hyper.stability = max(hyper.stability, stability)
            hyper.ci_lo = ci_lo
            hyper.ci_hi = ci_hi
            hyper.regime_id = regime_id
            hyper.last_seen = timestamp
            hyper.hits += 1
        else:
            self.hyperedges[key] = HyperRec(
                order=len(key),
                weight=effect,
                stability=stability,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                regime_id=regime_id,
                last_seen=timestamp,
                hits=1,
            )
            self._stats['hyperedges_added'] += 1

    def _update_indexes(self, key: Tuple[int, int]) -> None:
        """
        Updates the max-heap indexes for efficient neighbor lookup.

        Maintains `out_index` (source -> targets) and `in_index` (target -> sources)
        for the given edge key. Ensures only the top-k highest scoring edges
        are kept in the heaps for each node.

        Args:
            key: The tuple (src_id, dst_id) identifying the edge.
        """
        src_id, dst_id = key
        edge = self.edges[key]
        score = edge.weight * edge.stability
        
        # Update out_index for src
        heap = self.out_index[src_id]
        heapq.heappush(heap, (-score, dst_id, key))  # Negative for max-heap
        
        # Keep only top-k
        if len(heap) > self.topk_per_node:
            self.out_index[src_id] = heapq.nsmallest(self.topk_per_node, heap)
            heapq.heapify(self.out_index[src_id])
        
        # Update in_index for dst
        heap = self.in_index[dst_id]
        heapq.heappush(heap, (-score, src_id, key))
        
        if len(heap) > self.topk_per_node:
            self.in_index[dst_id] = heapq.nsmallest(self.topk_per_node, heap)
            heapq.heapify(self.in_index[dst_id])
    
    def top_k_neighbors(self, node_id: int, k: int, direction: str = "out", domain: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Retrieves the top-k highest scoring neighbors for a node.

        Args:
            node_id: The ID of the node to query.
            k: The maximum number of neighbors to return.
            direction: Direction of edges to consider ("out" for children, "in" for parents).
            domain: Optional domain ID to filter neighbors. If None, returns neighbors from all domains.

        Returns:
            A list of tuples (neighbor_id, score), sorted by score descending.
        """
        index = self.out_index[node_id] if direction == "out" else self.in_index[node_id]
        
        neighbors = []
        for neg_score, neighbor_id, key in index[:k]:
            # Filter by domain if specified
            if domain is not None:
                if direction == "out":
                    src_domain = self.nodes[node_id].get('domain')
                else:
                    src_domain = self.nodes[neighbor_id].get('domain')
                if src_domain != domain:
                    continue
            
            neighbors.append((neighbor_id, -neg_score))  # Convert back to positive
        
        return neighbors
    
    def decay(self, ts: int) -> None:
        """
        Applies temporal decay to all edges in the store.

        Multiplies weights by `decay_factor` to reduce the influence of old information.
        Also applies separate stability decay and penalizes "stale" edges that haven't
        been seen for a long time (e.g., > 600 windows).

        Args:
            ts: The current timestamp (window ID).
        """
        self.current_window = ts
        
        # Decay edges
        for key, edge in self.edges.items():
            edge.weight *= self.decay_factor
            edge.stability *= 0.997  # Slightly different decay for stability
            
            # Check for stale edges
            age = ts - edge.last_seen
            if age > 600:  # T_stale
                edge.weight *= 0.5
        
        # Decay hyperedges
        for hyper_rec in self.hyperedges.values():
            hyper_rec.weight *= self.decay_factor
            hyper_rec.stability *= 0.997
    
    def prune(self) -> None:
        """
        Removes edges that fall below minimum quality standards.

        Scans all edges and hyperedges and deletes those that:
        - Have negligible effect size (weight < epsilon).
        - Are unstable (stability < threshold).
        - Have not been updated for a very long time (T_prune).
        
        Also enforces capacity limits via aggressive pruning when needed.
        Updates internal pruning statistics.
        """
        # Phase 1: Quality-based pruning
        to_remove = []
        
        for key, edge in self.edges.items():
            # Check pruning criteria
            if (abs(edge.weight) < 0.02 or  # epsilon_weight
                edge.stability < 0.4 or  # epsilon_stability
                (self.current_window - edge.last_seen) > 5000):  # T_prune
                to_remove.append(key)
        
        for key in to_remove:
            del self.edges[key]
            self._stats['edges_pruned'] += 1
        
        # Phase 2: Capacity-based pruning (if still over limit)
        if len(self.edges) > self.max_edges:
            self._capacity_prune_edges()
        
        # Clean up indexes for removed edges
        self._rebuild_indexes()

        # Prune hyperedges
        to_remove_hyper = []
        for key, hyper_rec in self.hyperedges.items():
            if (abs(hyper_rec.weight) < 0.02 or
                hyper_rec.stability < 0.4 or
                (self.current_window - hyper_rec.last_seen) > 5000):
                to_remove_hyper.append(key)
        
        for key in to_remove_hyper:
            del self.hyperedges[key]
            self._stats['hyperedges_pruned'] += 1
        
        # Phase 2 for hyperedges
        if len(self.hyperedges) > self.max_hyperedges:
            self._capacity_prune_hyperedges()
    
    def _capacity_prune_edges(self) -> None:
        """
        Aggressively prune edges when over capacity.
        
        Removes the weakest edges (by weight * stability score) until
        within 80% of max capacity to leave room for new edges.
        """
        target_count = int(self.max_edges * 0.8)
        n_to_remove = len(self.edges) - target_count
        
        if n_to_remove <= 0:
            return
        
        # Score all edges and find the weakest
        scored_edges = [
            (abs(edge.weight) * edge.stability, key)
            for key, edge in self.edges.items()
        ]
        # Find the n_to_remove weakest edges
        weakest = heapq.nsmallest(n_to_remove, scored_edges)
        
        for _, key in weakest:
            del self.edges[key]
            self._stats['edges_pruned'] += 1
        
        logger.info(f"Capacity pruned {len(weakest)} edges, now at {len(self.edges)}")
    
    def _capacity_prune_hyperedges(self) -> None:
        """
        Aggressively prune hyperedges when over capacity.
        """
        target_count = int(self.max_hyperedges * 0.8)
        n_to_remove = len(self.hyperedges) - target_count
        
        if n_to_remove <= 0:
            return
        
        scored_hypers = [
            (abs(hyper.weight) * hyper.stability, key)
            for key, hyper in self.hyperedges.items()
        ]
        weakest = heapq.nsmallest(n_to_remove, scored_hypers)
        
        for _, key in weakest:
            del self.hyperedges[key]
            self._stats['hyperedges_pruned'] += 1
        
        logger.info(f"Capacity pruned {len(weakest)} hyperedges")
    
    def _rebuild_indexes(self) -> None:
        """
        Rebuild indexes after bulk pruning.
        
        This is more efficient than incremental updates after large prune operations.
        """
        # Reset indexes
        self.out_index = defaultdict(list)
        self.in_index = defaultdict(list)
        
        # Rebuild from existing edges
        for key in self.edges.keys():
            self._update_indexes(key)
    
    def remove_edge(self, src_id: int, dst_id: int) -> bool:
        """
        Remove an edge if present.
        """
        key = (src_id, dst_id)
        if key in self.edges:
            del self.edges[key]
            self._stats['edges_pruned'] += 1
            return True
        return False

    def gc(self, ts: int) -> None:
        """
        Orchestrates periodic garbage collection.

        Triggers `decay()` and `prune()` operations if the garbage collection
        interval has been reached.

        Args:
            ts: The current timestamp/window ID.
        """
        self.gc_counter += 1
        if self.gc_counter % self.gc_interval == 0:
            self.decay(ts)
            self.prune()
            self._stats['gc_cycles'] += 1
    
    def get_edge(self, src_id: int, dst_id: int) -> Optional[EdgeRec]:
        """Get edge record."""
        return self.edges.get((src_id, dst_id))
    
    def snapshot(self, include_regimes: bool = True) -> Dict[str, Any]:
        """
        Creates a serializable snapshot of the current store state.

        Captures nodes, edges, hyperedges, and statistics. Useful for standard
        checkpoints or debugging.

        Args:
            include_regimes: If True, includes detailed regime data (currently
                handled as part of edge records).

        Returns:
            A dictionary containing the complete state of the hypergraph.
        """
        return {
            'version': '1.0',
            'schema_hash': self.schema_hash,
            'current_window': self.current_window,
            'nodes': {nid: data for nid, data in self.nodes.items()},
            'edges': {str(k): e.to_dict() for k, e in self.edges.items()},
            'hyperedges': {str(k): h.to_dict() for k, h in self.hyperedges.items()},
            'stats': self.get_stats()
        }
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        edges_active = len(self.edges)
        hyperedges_active = len(self.hyperedges)

        if edges_active == 0:
            avg_weight = 0.0
            avg_stability = 0.0
        else:
            weights = [e.weight for e in self.edges.values()]
            stabilities = [e.stability for e in self.edges.values()]
            avg_weight = float(np.mean(weights))
            avg_stability = float(np.mean(stabilities))

        return {
            'n_edges': edges_active,
            'n_hyperedges': hyperedges_active,
            'edges_active': edges_active,
            'hyperedges_active': hyperedges_active,
            'avg_weight': avg_weight,
            'avg_stability': avg_stability,
            'pruned_last_min': self._stats['edges_pruned'],
            'promotions': self._stats['promotions'],
            'demotions': self._stats['demotions'],
            'gc_cycles': self._stats['gc_cycles']
        }
    
    # Compatibility with old API
    def update_edges(self, evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates the store with a batch of accepted evaluation results.

        This is a convenience (or compatibility) method that parses a list of
        result dictionaries (typically from the Evaluator or an external source)
        and calls `upsert_edge` or `upsert_hyperedge` for each valid item.

        Args:
            evaluation_results: A list of dictionaries containing edge data
                (source, target, gain, stability, etc.).

        Returns:
            A list of simplified dictionaries representing the edges that were
            successfully ingested.
        """
        ingested: List[Dict[str, Any]] = []

        for payload in evaluation_results:
            if not isinstance(payload, dict):
                continue

            source_name = payload.get('source') or payload.get('source_name')
            target_name = payload.get('target') or payload.get('target_name')
            if not source_name or not target_name:
                continue

            domain = int(payload.get('domain', 0) or 0)
            schema_ver = int(payload.get('schema_version', payload.get('schema_ver', 0)) or 0)
            ts = int(payload.get('window_id', self.current_window))
            self.current_window = max(self.current_window, ts)

            effect = float(payload.get('gain', payload.get('effect', 0.0)))
            ci_lo = float(payload.get('ci_lo', payload.get('ci_lower', 0.0)))
            ci_hi = float(payload.get('ci_hi', payload.get('ci_upper', 0.0)))
            stability = float(payload.get('stability', payload.get('stability_score', 0.0)))
            regime_id = int(payload.get('regime_id', -1))

            src_id = self.get_or_create_node(str(source_name), domain=domain, schema_ver=schema_ver)
            dst_id = self.get_or_create_node(str(target_name), domain=domain, schema_ver=schema_ver)
            self.upsert_edge(
                src_id=src_id,
                dst_id=dst_id,
                effect=effect,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                stability=stability,
                regime_id=regime_id,
                ts=ts
            )

            vars_payload = payload.get('vars') or payload.get('variables')
            if isinstance(vars_payload, (list, tuple)) and len(vars_payload) >= 2:
                hyper_nodes = [
                    self.get_or_create_node(str(var_name), domain=domain, schema_ver=schema_ver)
                    for var_name in vars_payload
                ]
                self.upsert_hyperedge(
                    sources=hyper_nodes,
                    effect=effect,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    stability=stability,
                    regime_id=regime_id,
                    ts=ts
                )

            ingested.append({
                'path_id': payload.get('path_id'),
                'source_id': src_id,
                'target_id': dst_id,
                'effect': effect,
                'window_id': ts
            })

        if ingested:
            self.gc(self.current_window)

        return ingested
    
    def get_edge_count(self) -> int:
        """Get current number of edges."""
        return len(self.edges)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return self.stats()


# Alias for backward compatibility
Store = HypergraphStore

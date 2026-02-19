"""
Agent registry for SCARCITY simulation.

Converts MPIE store snapshots into runtime-friendly structures. All agents
originate from learned store entries; nothing is fabricated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class NodeAgent:
    """
    Represents a simulation entity (Node).

    Derived from a discovered variable in the hypergraph store.
    """
    node_id: str
    agent_type: str
    domain: int
    regime: int
    embedding: np.ndarray
    stability: float
    value: float = 0.0


@dataclass
class EdgeLink:
    """
    Represents a causal link between two agents.

    Derived from discovered edges. Used to calculate influence propagation.
    """
    edge_id: str
    source: str
    target: str
    weight: float
    stability: float
    confidence_interval: float
    regime: int


class AgentRegistry:
    """
    Maintains mappings between node/edge identifiers and runtime structures.
    
    Acts as the source of truth for the Simulation Environment. It handles the
    translation of disparate snapshot formats (sim tests vs online engine store)
    into a unified collection of `NodeAgent` and `EdgeLink` objects.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeAgent] = {}
        self._edges: Dict[str, EdgeLink] = {}

    def load_from_store_snapshot(
        self,
        snapshot: Dict[str, any],
        default_value: float = 0.0,
    ) -> None:
        """
        Populate registry using a store snapshot.

        Supports two schemas:
        - Simulation unit tests: nodes already contain embeddings and stability.
        - HypergraphStore.snapshot: nodes/edges are structural, metrics live on edges.
        
        Args:
            snapshot: Dictionary corresponding to the store state.
            default_value: Default initialization value for node states if unspecified.
        """
        self._nodes.clear()
        self._edges.clear()

        nodes = snapshot.get("nodes", {})
        if not nodes:
            return

        first_payload = next(iter(nodes.values()))
        # Branch 1: simulation tests / pre-baked snapshot with embeddings.
        if isinstance(first_payload, dict) and "embedding" in first_payload:
            for node_id, payload in nodes.items():
                embedding = np.asarray(payload.get("embedding", []), dtype=np.float32)
                if embedding.size == 0:
                    embedding = np.zeros(3, dtype=np.float32)
                agent = NodeAgent(
                    node_id=str(node_id),
                    agent_type=payload.get("type", "variable"),
                    domain=int(payload.get("domain", 0)),
                    regime=int(payload.get("regime", -1)),
                    embedding=embedding,
                    stability=float(payload.get("stability", 0.5)),
                    value=float(payload.get("value", default_value)),
                )
                self._nodes[agent.node_id] = agent

            edges = snapshot.get("edges", {})
            for key, payload in edges.items():
                src, dst = payload.get("src"), payload.get("dst")
                if src is None or dst is None:
                    continue
                edge = EdgeLink(
                    edge_id=str(key),
                    source=str(src),
                    target=str(dst),
                    weight=float(payload.get("weight", 0.0)),
                    stability=float(payload.get("stability", 0.5)),
                    confidence_interval=float(payload.get("ci", 0.1)),
                    regime=int(payload.get("regime", -1)),
                )
                self._edges[edge.edge_id] = edge
            return

        # Branch 2: HypergraphStore snapshot from scarcity.engine.store.HypergraphStore.
        # Here, nodes are structural and metrics live on edges.
        # nodes: int_id -> {name, domain, schema_ver, flags}
        # edges: "(src_id, dst_id)" -> {weight, stability, ci_lo, ci_hi, ...}
        node_stats: Dict[int, Dict[str, float]] = {}
        for raw_id, data in nodes.items():
            try:
                int_id = int(raw_id)
            except (TypeError, ValueError):
                # If keys are already ints, this is a no-op; otherwise we skip invalid.
                int_id = raw_id  # type: ignore[assignment]
            domain = int(data.get("domain", 0))
            node_stats[int_id] = {
                "domain": float(domain),
                "degree": 0.0,
                "weight_sum": 0.0,
                "stability_sum": 0.0,
                "hits_sum": 0.0,
            }

        edges = snapshot.get("edges", {})
        # Parse edges, accumulate per-node stats, and build EdgeLink entries.
        for key_str, payload in edges.items():
            # Expect stringified tuple keys like "(0, 1)".
            src_id: Optional[int] = None
            dst_id: Optional[int] = None
            if isinstance(key_str, tuple) and len(key_str) == 2:
                src_id, dst_id = key_str  # type: ignore[assignment]
            elif isinstance(key_str, str):
                trimmed = key_str.strip()
                if trimmed.startswith("(") and trimmed.endswith(")"):
                    inner = trimmed[1:-1]
                    parts = inner.split(",")
                    if len(parts) == 2:
                        try:
                            src_id = int(parts[0].strip())
                            dst_id = int(parts[1].strip())
                        except ValueError:
                            src_id = dst_id = None
            if src_id is None or dst_id is None:
                continue

            weight = float(payload.get("weight", 0.0))
            stability = float(payload.get("stability", 0.5))
            ci_lo = float(payload.get("ci_lo", payload.get("ci_lower", 0.0)))
            ci_hi = float(payload.get("ci_hi", payload.get("ci_upper", 0.0)))
            ci_width = max(0.0, ci_hi - ci_lo)
            regime = int(payload.get("regime_id", payload.get("regime", -1)))

            for nid in (src_id, dst_id):
                stats = node_stats.get(nid)
                if stats is None:
                    continue
                stats["degree"] += 1.0
                stats["weight_sum"] += abs(weight)
                stats["stability_sum"] += stability
                stats["hits_sum"] += float(payload.get("hits", 1.0))

            node_src = nodes.get(src_id, {})
            node_dst = nodes.get(dst_id, {})
            src_name = str(node_src.get("name", src_id))
            dst_name = str(node_dst.get("name", dst_id))

            edge_id = f"{src_name}->{dst_name}"
            edge = EdgeLink(
                edge_id=edge_id,
                source=src_name,
                target=dst_name,
                weight=weight,
                stability=stability,
                confidence_interval=ci_width,
                regime=regime,
            )
            self._edges[edge.edge_id] = edge

        # Build NodeAgent entries with simple 3D embeddings derived from stats.
        # Layout: nodes arranged on a circle, radius ~ log(degree), z ~ avg stability.
        node_items = list(nodes.items())
        total_nodes = len(node_items)
        if total_nodes == 0:
            return

        angles = np.linspace(0.0, 2.0 * np.pi, total_nodes, endpoint=False)
        for idx, (raw_id, data) in enumerate(node_items):
            try:
                int_id = int(raw_id)
            except (TypeError, ValueError):
                int_id = raw_id  # type: ignore[assignment]

            stats = node_stats.get(int_id, {})
            degree = stats.get("degree", 0.0)
            weight_sum = stats.get("weight_sum", 0.0)
            stability_sum = stats.get("stability_sum", 0.0)
            hits_sum = stats.get("hits_sum", 0.0)

            avg_stability = stability_sum / max(1.0, degree)
            avg_weight = weight_sum / max(1.0, degree)

            radius = 4.0 + np.log1p(degree)
            angle = float(angles[idx])
            x = radius * float(np.cos(angle))
            y = radius * float(np.sin(angle))
            z = avg_stability

            embedding = np.asarray([x, y, z], dtype=np.float32)
            node_name = str(data.get("name", raw_id))
            domain = int(data.get("domain", 0))

            agent = NodeAgent(
                node_id=node_name,
                agent_type="variable",
                domain=domain,
                regime=-1,
                embedding=embedding,
                stability=float(avg_stability) if degree > 0 else 0.5,
                value=float(avg_weight) if degree > 0 else default_value,
            )
            self._nodes[agent.node_id] = agent

    def update_edges(self, edges: Iterable[Dict[str, any]]) -> None:
        """
        Merge edge updates (e.g., from `engine.insight` topic).

        Arg:
            edges: List of dictionary payloads containing updated edge metrics.
        """
        for payload in edges:
            src = payload.get("src")
            dst = payload.get("dst")
            if src is None or dst is None:
                continue
            edge_id = payload.get("edge_id", f"{src}->{dst}")
            edge = EdgeLink(
                edge_id=edge_id,
                source=str(src),
                target=str(dst),
                weight=float(payload.get("weight", 0.0)),
                stability=float(payload.get("stability", 0.5)),
                confidence_interval=float(
                    payload.get("ci", payload.get("ci_width", 0.1))
                ),
                regime=int(payload.get("regime", -1)),
            )
            self._edges[edge.edge_id] = edge

    def nodes(self) -> Dict[str, NodeAgent]:
        """Returns the dictionary of active NodeAgents."""
        return self._nodes

    def edges(self) -> Dict[str, EdgeLink]:
        """Returns the dictionary of active EdgeLinks."""
        return self._edges

    def seed_placeholder_graph(self, count: int = 12) -> None:
        """
        Create a synthetic ring of agents so visualisation can warm up even
        before the engine accepts any edges.
        
        Args:
            count: Number of placeholder nodes to generate.
        """

        if count <= 0:
            return

        self._nodes.clear()
        self._edges.clear()
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        radius = 5.0
        for idx in range(count):
            point = np.array(
                [
                    radius * np.cos(angles[idx]),
                    radius * np.sin(angles[idx]),
                    np.sin(angles[idx] * 2.0),
                ],
                dtype=np.float32,
            )
            agent = NodeAgent(
                node_id=f"placeholder_{idx}",
                agent_type="placeholder",
                domain=idx % 4,
                regime=0,
                embedding=point,
                stability=0.6 + 0.3 * np.sin(angles[idx]),
                value=0.5,
            )
            self._nodes[agent.node_id] = agent

        for idx in range(count):
            next_idx = (idx + 1) % count
            self._edges[f"placeholder_{idx}->{next_idx}"] = EdgeLink(
                edge_id=f"placeholder_{idx}->{next_idx}",
                source=f"placeholder_{idx}",
                target=f"placeholder_{next_idx}",
                weight=0.2,
                stability=0.6,
                confidence_interval=0.1,
                regime=0,
            )

    def node_embeddings(self) -> np.ndarray:
        """
        Retrieves a packed array of node embeddings (Nx3).

        If the intrinsic embedding is not 3D, performs basic projection or padding
        to ensure compatibility with the 3D visualizer.

        Returns:
            A numpy array of shape (N, 3), type float32.
        """
        if not self._nodes:
            return np.zeros((0, 3), dtype=np.float32)
        embeddings = np.stack([agent.embedding for agent in self._nodes.values()], axis=0)
        if embeddings.shape[1] > 3:
            # Reduce dimensionality via naive PCA (covariance eigenvectors)
            centered = embeddings - embeddings.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, centered.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1][:3]
            embeddings = centered @ eigvecs[:, order]
        elif embeddings.shape[1] < 3:
            pad = np.zeros((embeddings.shape[0], 3 - embeddings.shape[1]), dtype=np.float32)
            embeddings = np.concatenate([embeddings, pad], axis=1)
        return embeddings.astype(np.float32)

    def adjacency_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Constructs the current adjacency and stability matrices.

        Returns:
            A tuple containing:
            - Adjacency matrix (NxN float32)
            - Stability matrix (NxN float32)
            - List of node IDs corresponding to the matrix indices.
        """
        node_ids = list(self._nodes.keys())
        index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        matrix = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
        stability = np.zeros_like(matrix)
        for edge in self._edges.values():
            if edge.source in index and edge.target in index:
                i, j = index[edge.source], index[edge.target]
                matrix[i, j] = edge.weight
                stability[i, j] = edge.stability
        return matrix, stability, node_ids

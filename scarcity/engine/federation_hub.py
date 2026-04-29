"""
FederationHub — orchestrates a multi-node, multi-basket federated system.

Responsibilities:
  1. Node registry — tracks which nodes participate in which baskets.
  2. Basket-scoped peer routing — when a node processes a row, the hub
     broadcasts it to all other nodes in the SAME basket only. Cross-basket
     data never flows between basket engines.
  3. Pretraining coordination — load a corpus into a basket across all
     (or selected) nodes before live evaluation begins.
  4. Aggregated insights — collect top hypotheses across all nodes for a basket.

Design invariants:
  - The hub never touches engine internals directly; it only calls the
    FederationNode public API.
  - Basket isolation is enforced at both the node (filter_row) and the hub
    (routing is basket-keyed). Double isolation.
  - Adding a new domain never requires changing engine code — only a new
    basket spec in baskets.py and node registration here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .baskets import BasketRegistry, REGISTRY
from .federation_node import FederationNode

logger = logging.getLogger(__name__)


class FederationHub:
    """
    Central coordinator for a federated learning system.

    Usage
    -----
    hub = FederationHub()
    hub.register(FederationNode("KEN", hub=hub))
    hub.register(FederationNode("TZA", hub=hub))

    # Pretrain all nodes' macro engines on a broad corpus
    hub.pretrain_basket("macro", corpus_rows, node_ids=None)  # all nodes

    # Live streaming — node processes its own row, hub fans out to peers
    hub.observe("KEN", row, fan_out=True, peer_weight=0.70)

    # Predict on Kenya
    predictions = hub.predict("KEN", state)

    # Inspect findings
    findings = hub.basket_findings("macro", top_k=10)
    """

    def __init__(self, registry: BasketRegistry = REGISTRY):
        self.registry = registry
        self._nodes: Dict[str, FederationNode] = {}

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def register(self, node: FederationNode) -> "FederationHub":
        """Register a node and set its hub reference. Returns self for chaining."""
        node.hub = self
        self._nodes[node.node_id] = node
        logger.info("Hub: registered node %s (baskets: %s)", node.node_id, node.basket_ids)
        return self

    def node(self, node_id: str) -> FederationNode:
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node: {node_id!r}. Registered: {list(self._nodes)}")
        return self._nodes[node_id]

    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------

    def pretrain_basket(
        self,
        basket_id: str,
        corpus_rows: List[Dict[str, float]],
        node_ids: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Feed a domain corpus into the basket engine of each node.

        Each node independently trains on the same corpus — this is not
        federated averaging; it is independent warm-starting with shared
        historical data (appropriate when the corpus is public / global).

        Parameters
        ----------
        basket_id:   Which basket to pretrain.
        corpus_rows: List of observation dicts (may contain variables from
                     any basket — the node's filter_row will select the right ones).
        node_ids:    Subset of nodes to pretrain. None = all nodes.

        Returns
        -------
        {node_id: rows_ingested} for each node pretrained.
        """
        targets = node_ids if node_ids is not None else list(self._nodes.keys())
        results: Dict[str, int] = {}
        for nid in targets:
            if nid not in self._nodes:
                logger.warning("pretrain_basket: unknown node %s — skipped", nid)
                continue
            n = self._nodes[nid]
            if basket_id not in n.basket_ids:
                logger.warning("Node %s does not participate in basket %s — skipped", nid, basket_id)
                continue
            ingested = n.pretrain(basket_id, corpus_rows)
            results[nid] = ingested
        return results

    def pretrain_all_baskets(
        self,
        corpus_by_basket: Dict[str, List[Dict[str, float]]],
        node_ids: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Pretrain multiple baskets at once.

        Parameters
        ----------
        corpus_by_basket: {basket_id: [rows]} — separate corpus per basket.
        node_ids:         Subset of nodes (None = all).

        Returns
        -------
        {basket_id: {node_id: rows_ingested}}
        """
        results: Dict[str, Dict[str, int]] = {}
        for basket_id, rows in corpus_by_basket.items():
            results[basket_id] = self.pretrain_basket(basket_id, rows, node_ids)
        return results

    # ------------------------------------------------------------------
    # Live streaming
    # ------------------------------------------------------------------

    def observe(
        self,
        node_id: str,
        row: Dict[str, float],
        fan_out: bool = True,
        peer_weight: float = 0.70,
        exclude_baskets: Optional[List[str]] = None,
    ) -> None:
        """
        Process one live observation for a node, then fan-out to peers.

        The row is first processed by the source node's own engines (all baskets).
        Then, if fan_out=True, the hub broadcasts the row to every other node
        IN THE SAME BASKET — never across baskets.

        Parameters
        ----------
        node_id:         Originating node.
        row:             Raw observation dict.
        fan_out:         Whether to broadcast to peers (default True).
        peer_weight:     Trust weight for peer ingestion [0, 1].
        exclude_baskets: Baskets to suppress fan-out for (optional).
        """
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node: {node_id!r}")

        src = self._nodes[node_id]
        src.process_row(row)

        if not fan_out:
            return

        # Route to peers — basket by basket
        routed = self.registry.route_row(row)
        excluded = set(exclude_baskets or [])

        for basket_id, filtered in routed.items():
            if basket_id in excluded:
                continue
            for peer_id, peer_node in self._nodes.items():
                if peer_id == node_id:
                    continue
                if basket_id not in peer_node.basket_ids:
                    continue
                peer_node.receive_peer(basket_id, node_id, filtered, weight=peer_weight)

    def observe_all(
        self,
        rows_by_node: Dict[str, Dict[str, float]],
        fan_out: bool = True,
        peer_weight: float = 0.70,
    ) -> None:
        """
        Process one synchronous time step where all nodes observe simultaneously.

        Each node's row is broadcast to all other nodes. Useful for annual/
        quarterly panel data where all countries report for the same time period.
        """
        for node_id, row in rows_by_node.items():
            self.observe(node_id, row, fan_out=fan_out, peer_weight=peer_weight)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, node_id: str, state: Dict[str, float]) -> Dict[str, float]:
        """Get merged predictions from all basket engines of a node."""
        return self.node(node_id).predict(state)

    # ------------------------------------------------------------------
    # Insights and reporting
    # ------------------------------------------------------------------

    def basket_findings(
        self,
        basket_id: str,
        top_k: int = 10,
        min_confidence: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate the strongest hypotheses across all nodes for one basket.

        Hypotheses are de-duplicated by (variables, type) — the version with
        the highest confidence is kept. This gives a federation-level view of
        what the basket has collectively discovered.
        """
        seen: Dict[Tuple, Dict[str, Any]] = {}
        for node_id, node in self._nodes.items():
            hyps = node.export_state(basket_id, top_k=top_k * 2)
            for h in hyps:
                if h["metrics"]["confidence"] < min_confidence:
                    continue
                key = (tuple(sorted(h["variables"])), h["type"])
                if key not in seen or h["metrics"]["confidence"] > seen[key]["metrics"]["confidence"]:
                    h["source_node"] = node_id
                    seen[key] = h

        findings = sorted(seen.values(), key=lambda x: x["metrics"]["confidence"], reverse=True)
        return findings[:top_k]

    def sync_directions(
        self,
        primary_id: str,
        min_confidence: float = 0.20,
    ) -> int:
        """
        Push peer directional consensus to the primary node.

        For each (source→target) pair where at least one peer has a confident
        CausalHypothesis with direction != 0, compute the confidence-weighted
        majority vote and apply it to any primary CausalHypothesis that is
        currently direction=0 (ambiguous).  Does not override existing non-zero
        directions so the primary's own data always takes precedence.

        Returns the number of direction hints applied.
        """
        from .relationships import CausalHypothesis
        from .discovery import HypothesisState
        from typing import List as _List, Tuple as _Tuple

        if primary_id not in self._nodes:
            return 0

        # Collect confidence-weighted direction votes from all peer nodes
        votes: Dict[Tuple, list] = {}
        for nid, node in self._nodes.items():
            if nid == primary_id:
                continue
            for (src, tgt), (direction, conf, _) in node.get_causal_directions().items():
                if conf < min_confidence:
                    continue
                key = (src, tgt)
                votes.setdefault(key, []).append((direction, conf))

        # Apply consensus to primary node's ambiguous hypotheses
        primary = self._nodes[primary_id]
        n_applied = 0
        for bid in primary.basket_ids:
            eng = primary._engines.get(bid)
            if eng is None:
                continue
            for h in eng.hypotheses.population.values():
                if h.meta.state == HypothesisState.DEAD:
                    continue
                if not isinstance(h, CausalHypothesis):
                    continue
                if h.direction != 0:
                    continue  # primary already has a clear direction
                key = (h.source, h.target)
                peer_votes = votes.get(key, [])
                if not peer_votes:
                    continue
                pos_w = sum(c for d, c in peer_votes if d > 0)
                neg_w = sum(c for d, c in peer_votes if d < 0)
                if pos_w == 0 and neg_w == 0:
                    continue
                h.direction = +1 if pos_w >= neg_w else -1
                n_applied += 1

        logger.info(
            "sync_directions: primary=%s  peer_pairs=%d  applied=%d",
            primary_id, len(votes), n_applied,
        )
        return n_applied

    def system_stats(self) -> Dict[str, Any]:
        """Full stats snapshot: all nodes, all baskets."""
        return {
            "nodes": {nid: n.stats() for nid, n in self._nodes.items()},
            "baskets": self.registry.all_ids(),
            "n_nodes": len(self._nodes),
        }

    def summary(self) -> str:
        """Human-readable system summary."""
        stats = self.system_stats()
        lines = [
            f"FederationHub  nodes={stats['n_nodes']}  baskets={stats['baskets']}",
        ]
        for nid, nstats in stats["nodes"].items():
            lines.append(f"  {nid}:")
            for bid, bstats in nstats["baskets"].items():
                lines.append(
                    f"    {bid:20s}  hyps={bstats['hypotheses']:4d}"
                    f"  pretrain={bstats['pretrain_rows']:4d}"
                    f"  live={bstats['live_rows']:4d}"
                )
        return "\n".join(lines)

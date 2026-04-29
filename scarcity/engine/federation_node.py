"""
FederationNode — one participant in the federated learning system.

A node represents a single data source (country, hospital, firm, etc.).
It holds one isolated OnlineDiscoveryEngine per basket, so sector-specific
priors never contaminate other sectors.

Lifecycle:
  1. pretrain(basket_id, rows)   — warm the basket engine on a corpus before
                                   live evaluation begins.
  2. process_row(row)            — route an observation to all basket engines.
  3. receive_peer(basket_id, …)  — accept a peer's observation (trust-weighted).
  4. predict(row)                — merge predictions from all basket engines.
  5. export_state(basket_id)     — snapshot top hypotheses for external inspection.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .federation_hub import FederationHub

from .baskets import Basket, BasketRegistry, REGISTRY

logger = logging.getLogger(__name__)


class FederationNode:
    """
    One federated participant with isolated per-basket engines.

    Parameters
    ----------
    node_id:    Unique identifier (e.g. "KEN", "hospital_A").
    basket_ids: Which baskets this node participates in.
                Defaults to all registered baskets.
    registry:   BasketRegistry to use (defaults to module singleton).
    hub:        FederationHub this node reports to (set by hub.register).
    engine_kwargs: Extra kwargs forwarded to OnlineDiscoveryEngine.__init__.
    """

    def __init__(
        self,
        node_id: str,
        basket_ids: Optional[List[str]] = None,
        registry: BasketRegistry = REGISTRY,
        hub: Optional["FederationHub"] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = node_id
        self.registry = registry
        self.hub = hub
        self._engine_kwargs = engine_kwargs or {"mode": "balanced"}

        _ids = basket_ids if basket_ids is not None else registry.all_ids()
        self.basket_ids: List[str] = [bid for bid in _ids if bid in registry]

        # One engine per basket — created lazily on first use
        self._engines: Dict[str, Any] = {}
        self._pretrain_counts: Dict[str, int] = {bid: 0 for bid in self.basket_ids}
        self._live_counts: Dict[str, int] = {bid: 0 for bid in self.basket_ids}

        # Per-variable Welford online stats for own data (mean, M2, count).
        # Used to z-score incoming peer rows to own-country scale so that
        # cross-country magnitude differences don't bias the ensemble.
        self._var_stats: Dict[str, Tuple[float, float, int]] = {}  # var→(mean, M2, n)

        # Rolling window of recent own rows (Fix #3): peer renormalization uses
        # the last 15 own observations' statistics instead of all-time Welford
        # stats, so that regime shifts are reflected quickly.
        self._recent_own: "deque[Dict[str, float]]" = deque(maxlen=15)

        self._init_engines()

    def _init_engines(self) -> None:
        from .engine_v2 import OnlineDiscoveryEngine
        for bid in self.basket_ids:
            basket = self.registry.get(bid)
            eng = OnlineDiscoveryEngine(**self._engine_kwargs)
            eng.initialize_v2(basket.schema, use_causal=True)
            self._engines[bid] = eng
        logger.debug("Node %s: initialised engines for baskets %s", self.node_id, self.basket_ids)

    # ------------------------------------------------------------------
    # Per-variable stats helpers (Welford online mean/variance)
    # ------------------------------------------------------------------

    def _update_var_stats(self, row: Dict[str, float]) -> None:
        """Update Welford online stats for own-country data."""
        import math as _math
        for var, val in row.items():
            if not _math.isfinite(val):
                continue
            mean, M2, n = self._var_stats.get(var, (0.0, 0.0, 0))
            n += 1
            delta = val - mean
            mean += delta / n
            delta2 = val - mean
            M2 += delta * delta2
            self._var_stats[var] = (mean, M2, n)

    def _own_std(self, var: str) -> float:
        """Sample std for own-country variable; returns 1.0 if < 3 samples."""
        mean, M2, n = self._var_stats.get(var, (0.0, 0.0, 0))
        if n < 3:
            return 1.0
        return max(math.sqrt(M2 / (n - 1)), 1e-6)

    def _rolling_own_stats(self, var: str) -> Tuple[float, float, int]:
        """Mean, std, n computed from the last 15 own observations (Fix #3)."""
        vals = [r[var] for r in self._recent_own if var in r and math.isfinite(r[var])]
        n = len(vals)
        if n < 3:
            return 0.0, 1.0, n
        mu = sum(vals) / n
        var_ = sum((v - mu) ** 2 for v in vals) / (n - 1)
        return mu, max(math.sqrt(var_), 1e-6), n

    def _renormalize_peer_row(
        self,
        peer_row: Dict[str, float],
        peer_stats: Dict[str, Tuple[float, float, int]],
    ) -> Dict[str, float]:
        """
        Rescale a peer row from peer-country scale to own-country scale.

        Uses rolling-window own stats (last 15 own obs) when ≥10 recent rows
        exist (Fix #3); falls back to Welford all-time stats otherwise.
        This makes renormalization responsive to the current regime rather than
        anchored to the long-run pretrain mean.
        """
        out: Dict[str, float] = {}
        use_rolling = len(self._recent_own) >= 10
        for var, val in peer_row.items():
            if not math.isfinite(val):
                continue
            p_mean, p_M2, p_n = peer_stats.get(var, (0.0, 0.0, 0))
            if p_n < 3:
                out[var] = val
                continue
            p_std = max(math.sqrt(p_M2 / (p_n - 1)), 1e-6)
            z = (val - p_mean) / p_std                    # peer z-score
            if use_rolling:
                own_mean, own_std, own_n = self._rolling_own_stats(var)
            else:
                own_mean, _, own_n = self._var_stats.get(var, (0.0, 0.0, 0))
                own_std = self._own_std(var)
            out[var] = own_mean + z * own_std if own_n >= 3 else val
        return out

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def pretrain(self, basket_id: str, rows: List[Dict[str, float]]) -> int:
        """
        Feed a pretraining corpus into one basket engine.

        This must be called BEFORE live evaluation begins. It builds priors
        from broad historical data so the engine arrives at live data warm.
        The basket engine's variable filter ensures no cross-basket contamination.

        Returns number of rows successfully ingested.
        """
        if basket_id not in self._engines:
            raise KeyError(f"Node {self.node_id!r} does not participate in basket {basket_id!r}")
        basket = self.registry.get(basket_id)
        eng = self._engines[basket_id]
        ingested = 0
        for row in rows:
            filtered = basket.filter_row(row)
            if basket.has_variables(filtered):
                try:
                    eng.process_row(filtered)
                    ingested += 1
                except Exception as exc:
                    logger.debug("pretrain error node=%s basket=%s: %s", self.node_id, basket_id, exc)
        self._pretrain_counts[basket_id] = self._pretrain_counts.get(basket_id, 0) + ingested
        logger.info(
            "Node %s | pretrained basket=%s | %d/%d rows ingested (total pretrain=%d)",
            self.node_id, basket_id, ingested, len(rows), self._pretrain_counts[basket_id],
        )
        return ingested

    def process_row(self, row: Dict[str, float]) -> None:
        """
        Route a live observation to all relevant basket engines.

        Each basket engine only sees variables in its own domain — the
        filter is enforced inside Basket.filter_row().
        """
        # Track per-variable stats for own-country renormalization of peer data
        self._update_var_stats(row)
        self._recent_own.append({k: v for k, v in row.items() if math.isfinite(v)})

        routed = self.registry.route_row(row)
        for bid, filtered in routed.items():
            if bid in self._engines:
                try:
                    self._engines[bid].process_row(filtered)
                    self._live_counts[bid] = self._live_counts.get(bid, 0) + 1
                except Exception as exc:
                    logger.debug("process_row error node=%s basket=%s: %s", self.node_id, bid, exc)

    def receive_peer(
        self,
        basket_id: str,
        peer_id: str,
        peer_row: Dict[str, float],
        weight: float = 0.70,
        peer_stats: Optional[Dict[str, Tuple[float, float, int]]] = None,
    ) -> None:
        """
        Accept a trust-weighted observation from a peer node.

        The basket filter is re-applied here so peers cannot inject
        out-of-basket variables even if their row contains them.

        If peer_stats is provided, the peer row is renormalized from the peer's
        own-country scale to this node's scale before being fed to the engine,
        removing cross-country level differences while preserving relative moves.
        """
        if basket_id not in self._engines:
            return
        basket = self.registry.get(basket_id)
        # Renormalize peer magnitudes to own-country scale when peer stats available
        if peer_stats and len(self._var_stats) >= 3:
            peer_row = self._renormalize_peer_row(peer_row, peer_stats)
        filtered = basket.filter_row(peer_row)
        if not basket.has_variables(filtered):
            return
        try:
            eng = self._engines[basket_id]
            if hasattr(eng, "process_peer_row"):
                eng.process_peer_row(peer_id, filtered, peer_weight=weight)
        except Exception as exc:
            logger.debug("receive_peer error node=%s basket=%s peer=%s: %s",
                         self.node_id, basket_id, peer_id, exc)

    def predict(self, row: Dict[str, float]) -> Dict[str, float]:
        """
        Merge predictions from all basket engines.

        Each basket engine predicts only its own variables. Merged predictions
        cover all variables the node has basket engines for. Later baskets
        overwrite earlier ones for overlapping variables (last-write-wins —
        overlapping variables appear in multiple baskets by design).
        """
        merged: Dict[str, float] = {}
        for bid in self.basket_ids:
            if bid not in self._engines:
                continue
            basket = self.registry.get(bid)
            filtered = basket.filter_row(row)
            if not filtered:
                continue
            try:
                preds = self._engines[bid].predict(filtered)
                merged.update(preds)
            except Exception as exc:
                logger.debug("predict error node=%s basket=%s: %s", self.node_id, bid, exc)
        return merged

    def begin_live_stream(self, pretrain_discount: float = 0.5) -> None:
        """
        Transition from pretraining to live-streaming mode.

        Softens pretrained priors by multiplying all hypothesis confidence
        scores by `pretrain_discount` and capping evidence counts, so the
        live stream can confirm or revise direction without the MetaController
        kill condition (conf < 0.10 AND evidence > 20) firing prematurely.

        Call this after all pretrain() calls and before the first process_row().
        """
        from .relationships import CausalHypothesis
        for bid in self.basket_ids:
            eng = self._engines.get(bid)
            if eng is None:
                continue
            for h in eng.hypotheses.population.values():
                # Clamp evidence so MetaController treats these as still immature
                h.evidence = min(h.evidence, 10)

                if isinstance(h, CausalHypothesis):
                    # For CausalHypothesis, self.confidence is the signed directional
                    # value |conf_fwd - conf_bwd|, not the raw forward confidence.
                    # Discount forward and backward accumulators separately to preserve
                    # the directional meaning, then recompute signed confidence.
                    fwd_total = h.alpha_success + h.beta_failure
                    if fwd_total > 0:
                        conf_fwd = h.alpha_success / fwd_total
                        conf_fwd_new = conf_fwd * pretrain_discount
                        h.alpha_success = conf_fwd_new * fwd_total
                        h.beta_failure = (1.0 - conf_fwd_new) * fwd_total
                    if hasattr(h, '_alpha_bwd'):
                        bwd_total = h._alpha_bwd + h._beta_bwd
                        if bwd_total > 0:
                            conf_bwd = h._alpha_bwd / bwd_total
                            conf_bwd_new = conf_bwd * pretrain_discount
                            h._alpha_bwd = conf_bwd_new * bwd_total
                            h._beta_bwd = (1.0 - conf_bwd_new) * bwd_total
                    h.confidence = h.alpha_success / (h.alpha_success + h.beta_failure)
                    # Reset ECM cointegration fields
                    h._is_coint = False
                    h._coint_alpha = 0.0
                    h._coint_beta = 0.0
                    h._coint_gamma = 0.0
                    h._allow_ecm_refit = False
                else:
                    # Non-CausalHypothesis: confidence = forward confidence directly.
                    h.confidence = h.confidence * pretrain_discount
                    total = h.alpha_success + h.beta_failure
                    if total > 0:
                        h.alpha_success = h.confidence * total
                        h.beta_failure = (1.0 - h.confidence) * total
        logger.info(
            "Node %s: pretrain discount=%.2f applied; ECM fields reset",
            self.node_id, pretrain_discount,
        )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_causal_directions(self) -> Dict[Tuple[str, str], Tuple[int, float, float]]:
        """
        Return the strongest CausalHypothesis directional signal per variable pair.

        Returns {(source, target): (direction, confidence, p_value_forward)} for
        all live CausalHypothesis instances with direction != 0, keeping the
        highest-confidence signal when multiple baskets model the same pair.
        """
        from .relationships import CausalHypothesis
        from .discovery import HypothesisState
        directions: Dict[Tuple[str, str], Tuple[int, float, float]] = {}
        for bid in self.basket_ids:
            eng = self._engines.get(bid)
            if eng is None:
                continue
            for h in eng.hypotheses.population.values():
                if h.meta.state == HypothesisState.DEAD:
                    continue
                if not isinstance(h, CausalHypothesis) or h.direction == 0:
                    continue
                key = (h.source, h.target)
                conf = getattr(h, "confidence", 0.0)
                if key not in directions or conf > directions[key][1]:
                    directions[key] = (h.direction, conf, h.p_value_forward)
        return directions

    def engine(self, basket_id: str) -> Any:
        """Direct access to the basket engine (for introspection / testing)."""
        return self._engines[basket_id]

    def export_state(self, basket_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Return the top-k hypotheses for a basket as serialisable dicts."""
        if basket_id not in self._engines:
            return []
        hyps = self._engines[basket_id].hypotheses.get_strongest(top_k=top_k)
        return [h.to_dict() for h in hyps]

    def stats(self) -> Dict[str, Any]:
        """Summary of engine sizes and observation counts per basket."""
        out: Dict[str, Any] = {"node_id": self.node_id, "baskets": {}}
        for bid in self.basket_ids:
            eng = self._engines.get(bid)
            n_hyps = len(eng.hypotheses.population) if eng else 0
            out["baskets"][bid] = {
                "hypotheses": n_hyps,
                "pretrain_rows": self._pretrain_counts.get(bid, 0),
                "live_rows": self._live_counts.get(bid, 0),
            }
        return out

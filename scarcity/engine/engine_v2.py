"""
Online Relationship Discovery — Main Engine.

The orchestrator that ties:
Stream Row -> [Hypothesis Pool] -> [Measurement] -> [Grouping Updates]

HARDENED v4: Includes Meta-Controller and Explicit Scoring.
"""

import time
import logging
import math
import numpy as np
from typing import Dict, List, Any

from .discovery import HypothesisPool, Hypothesis, RelationshipType, HypothesisState
from .grouping import AdaptiveGrouper
from .arbitration import HypothesisArbiter
from .controller import MetaController
from .algorithms_online import FunctionalLinearHypothesis, TemporalLagHypothesis

# All relationship types — v2 implementations with proper statistics
from .relationships import (
    CausalHypothesis,
    CorrelationalHypothesis,
    TemporalHypothesis,
    FunctionalHypothesis,
    EquilibriumHypothesis,
    CompositionalHypothesis,
    CompetitiveHypothesis,
    SynergisticHypothesis,
    ProbabilisticHypothesis,
    StructuralHypothesis,
)
from .relationships_extended import (
    MediatingHypothesis,
    ModeratingHypothesis,
    GraphHypothesis,
    SimilarityHypothesis,
    LogicalHypothesis,
)
from .types import Candidate

logger = logging.getLogger(__name__)

class OnlineDiscoveryEngine:
    """
    Main entry point for the Online Relationship Discovery System.

    This engine orchestrates the entire lifecycle of causal discovery from streaming
    data. It integrates:
    - Streaming Data Ingestion: Processing rows one by one.
    - Hypothesis Management: Maintaining a pool of competing causal models.
    - Adaptive Grouping: Clustering coherent variables.
    - Arbitration: Resolving conflicts between contradictory hypotheses.
    - Meta-Control: Promoting/pruning hypotheses based on evidence.

    It serves as the high-level API for external consumers to feed data and
    retrieve the learned Knowledge Graph.
    """
    VALID_MODES = {"balanced", "performance"}

    def __init__(self, explore_interval: int = 10, mode: str = "balanced",
                 buffer_size: int = 150, small_dataset_mode: bool = False,
                 vectorized: bool = True, device: str = 'cpu'):
        """
        Initializes the discovery engine and its sub-components.

        Args:
            explore_interval: Steps between exploration phases.
            vectorized: When True (default), delegates process_row() to the
                batch-tensor backend (GPUDiscoveryEngine) instead of iterating
                Python Hypothesis objects.  2-3× faster for N<200.  Uses the
                same lifecycle thresholds as small_dataset_mode when that flag
                is also set.
            device: Tensor device for vectorized mode ('cpu' or 'cuda').
                CPU is the default — CUDA only helps when B_perm ≥ 50.
        """
        self.hypotheses = HypothesisPool()
        self.grouper = AdaptiveGrouper()
        self.arbiter = HypothesisArbiter()
        self.meta_controller = MetaController()

        self.buffer_size = buffer_size
        self.step_count = 0
        self.explore_interval = explore_interval
        self.start_time = time.time()
        self.mode = "balanced"
        self.lifecycle_interval = 10
        self.arbitration_interval = 50
        self.grouping_enabled = True
        self.exploration_enabled = True
        self.update_error_total = 0
        self.small_dataset_mode = small_dataset_mode

        if small_dataset_mode:
            self.hypotheses = HypothesisPool(capacity=2000)
            self.meta_controller = MetaController.small_dataset()

        # Vectorized backend — replaces Python loop in process_row()
        self._vec_engine = None
        if vectorized:
            from .gpu_engine import GPUDiscoveryEngine
            self._vec_engine = GPUDiscoveryEngine(
                device=device,
                small_dataset_mode=small_dataset_mode,
            )

        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """Switch engine runtime mode.

        balanced: full discovery loop behavior.
        performance: reduced overhead for faster pilot-style stream processing.
        """
        mode_norm = str(mode or "balanced").strip().lower()
        if mode_norm not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Valid modes: {sorted(self.VALID_MODES)}")

        self.mode = mode_norm
        if mode_norm == "performance":
            self.lifecycle_interval = 25
            self.arbitration_interval = 100
            self.grouping_enabled = False
            self.exploration_enabled = False
        else:
            self.lifecycle_interval = 10
            self.arbitration_interval = 50
            self.grouping_enabled = True
            self.exploration_enabled = True
        
    def initialize(self, schema: Dict[str, Any]) -> None:
        """
        Sets up the engine based on the data schema.

        Initializes the grouper with variable names and populates the hypothesis
        pool with an initial set of priors.
        - If the variable count is small (< 10), it initializes a dense set of
          pairwise correlational and functional hypotheses (brute-force start).
        - Always adds baseline Temporal Lag (autoregressive) and Equilibrium
          hypotheses for every variable.

        Args:
            schema: The data schema dictionary defining fields and types.
        """
        fields = schema.get('fields', [])
        var_names = [f['name'] for f in fields] if fields else []

        if not var_names:
            logger.warning("No variables found in schema.")
            return

        # Build name→index mapping (used by get_candidate_paths bridge)
        self._var_index = {name: idx for idx, name in enumerate(var_names)}

        self.grouper.initialize(var_names)
        
        if len(var_names) < 10:
            import itertools
            for a, b in itertools.combinations(var_names, 2):
                self.hypotheses.add(CorrelationalHypothesis(a, b, buffer_size=self.buffer_size))
                self.hypotheses.add(FunctionalLinearHypothesis(a, b))
                self.hypotheses.add(FunctionalLinearHypothesis(b, a))

        for v in var_names:
            self.hypotheses.add(TemporalLagHypothesis(v, v))
            self.hypotheses.add(EquilibriumHypothesis(v, buffer_size=self.buffer_size))

    def initialize_v2(self, schema: Dict[str, Any], use_causal: bool = True) -> None:
        """
        Enhanced initialization using production-quality hypothesis classes.
        
        Uses the new Granger-based CausalHypothesis, improved TemporalHypothesis,
        and other advanced relationship types.
        
        Args:
            schema: The data schema dictionary defining fields and types.
            use_causal: Whether to create CausalHypothesis (expensive, O(n²) pairs)
        """
        fields = schema.get('fields', [])
        var_names = [f['name'] for f in fields] if fields else []
        
        if not var_names:
            logger.warning("No variables found in schema.")
            return
        
        # Build name→index mapping (used by get_candidate_paths bridge)
        self._var_index = {name: idx for idx, name in enumerate(var_names)}

        self.grouper.initialize(var_names)

        # Vectorized backend — initialise in parallel with Python pool
        if self._vec_engine is not None:
            self._vec_engine.initialize_v2(schema, use_causal=use_causal)

        logger.info(f"Initializing V2 engine with {len(var_names)} variables")
        
        # 1. For each variable: Temporal (AR) and Equilibrium
        bs = self.buffer_size
        for v in var_names:
            self.hypotheses.add(TemporalHypothesis(v, lag=2, buffer_size=bs))
            self.hypotheses.add(EquilibriumHypothesis(v, buffer_size=bs))

        # 2. For variable pairs — all pairs, no artificial cap
        import itertools
        pairs = list(itertools.combinations(var_names, 2))

        # 3. For variable triples — all combinations seeded at init.
        # Pool capacity (1000) is large enough: for n=9 vars, C(9,3)×5 = 420 triplet
        # hypotheses + 216 pairs + 18 singles = 654 total, well under capacity.
        # For n>12 (C(12,3)=220) cap at 100 to keep init time reasonable.
        all_triplets = list(itertools.combinations(var_names, 3))
        triplets = all_triplets if len(all_triplets) <= 100 else all_triplets[:100]

        # lag=1 in small_dataset_mode saves 2 df per Granger test (df_den +2),
        # improving F-test power when n < 50.
        causal_lag = 1 if self.small_dataset_mode else 2

        for a, b in pairs:
            # Correlational — both directions: each is a distinct predictor for shock propagation.
            # Corr(a,b) uses a to predict b; Corr(b,a) uses b to predict a.
            self.hypotheses.add(CorrelationalHypothesis(a, b, buffer_size=bs))
            self.hypotheses.add(CorrelationalHypothesis(b, a, buffer_size=bs))

            # Functional (linear regression, both directions)
            self.hypotheses.add(FunctionalHypothesis(a, b, degree=1, buffer_size=bs))
            self.hypotheses.add(FunctionalHypothesis(b, a, degree=1, buffer_size=bs))

            # Causal/Granger (both directions)
            if use_causal:
                self.hypotheses.add(CausalHypothesis(a, b, lag=causal_lag, buffer_size=bs))
                self.hypotheses.add(CausalHypothesis(b, a, lag=causal_lag, buffer_size=bs))
            # Note: Competitive, Probabilistic, Structural, Graph are added by _explore_step()
            # for pairs where the above three types show strong signal, keeping init < capacity.

        # 4. Triple-variable hypotheses
        for a, b, c in triplets:
            self.hypotheses.add(CompositionalHypothesis([a, b], c, buffer_size=bs))
            self.hypotheses.add(SynergisticHypothesis(a, b, c, buffer_size=bs))
            self.hypotheses.add(MediatingHypothesis(a, b, c, buffer_size=bs))
            self.hypotheses.add(ModeratingHypothesis(a, b, c, buffer_size=bs))
            self.hypotheses.add(LogicalHypothesis(a, b, c, buffer_size=bs))

        # 5. Similarity hypothesis across a small variable subset
        if len(var_names) >= 3:
            subset = var_names[: min(5, len(var_names))]
            self.hypotheses.add(SimilarityHypothesis(subset, n_clusters=min(3, len(subset)),
                                                     buffer_size=bs))
        
        logger.info(f"Initialized {len(self.hypotheses.population)} hypotheses (V2)")

    def _sanitize_row(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Cleans and normalizes an incoming data row.

        Converts all values to floats, handling None types and string representations
        of numbers. Non-numeric or unparseable fields are converted to NaN.
        Keys are filtered to strings only.

        Args:
            row: The raw input dictionary.

        Returns:
            A clean dictionary mapping variable names to float values (or NaN).
        """
        if not isinstance(row, dict):
            return {}

        clean_row = {}
        for k, v in row.items():
            if not isinstance(k, str): continue 
            try:
                if v is None:
                    clean_row[k] = float('nan')
                elif isinstance(v, (float, int)):
                    clean_row[k] = float(v)
                elif isinstance(v, str):
                    clean_row[k] = float(v)
                else:
                    continue
            except (ValueError, TypeError):
                clean_row[k] = float('nan')
        return clean_row

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single data row through the discovery loop.

        This is the main "tick" of the system. It executes:
        1. Data Sanitization: Cleaning the input.
        2. Hypothesis Update: Updating all active models with the new observation.
        3. Meta-Control: adjusting hypothesis states (Active/Dead) periodically.
        4. Grouping Update: Refining variable clusters.
        5. Arbitration: Resolving conflicts periodically.
        6. Exploration: Triggering exploration strategies (if enabled).

        Args:
            row: The raw data row from the stream.

        Returns:
            A status dictionary containing the current step count, active hypothesis
            count, meta-controller summary, and grouping stats.
        """
        self.step_count += 1

        # 1. Sanitize
        safe_row = self._sanitize_row(row)

        # 2. Update hypotheses — vectorized batch-tensor path or Python loop
        if self._vec_engine is not None:
            self._vec_engine.process_row(safe_row)
        else:
            self.hypotheses.update_all(safe_row)
        row_update_errors = int(getattr(self.hypotheses, "last_update_errors", 0))
        self.update_error_total += row_update_errors
        
        # 3. Meta-Controller Lifecycle
        if self.step_count % self.lifecycle_interval == 0:
            self.meta_controller.manage_lifecycle(self.hypotheses)
        
        # 4. Monitor Grouping (optional in performance mode)
        hypothesis_errors = self._build_group_error_signal()
        drift_pressure = float(np.mean(list(hypothesis_errors.values()))) if hypothesis_errors else 0.0
        drift_alert = bool(drift_pressure > self.grouper.split_threshold)
        if self.grouping_enabled:
            self.grouper.monitor(safe_row, hypothesis_errors)
        
        # 5. Arbitration
        if self.step_count % self.arbitration_interval == 0:
            self._arbitrate_step()
            
        # 6. Exploration
        if self.exploration_enabled and self.step_count % self.explore_interval == 0:
            self._explore_step()

        # 7. IV pass (every 20 steps, only when enough evidence accumulated)
        if self.step_count % 20 == 0 and self.step_count >= 20:
            self._run_iv_pass()

        # Gather Stats
        meta_stats = self.meta_controller.get_summary(self.hypotheses)
        
        return {
            "step": self.step_count,
            "engine_mode": self.mode,
            "update_errors": row_update_errors,
            "update_error_total": self.update_error_total,
            "update_error_details": getattr(self.hypotheses, "last_update_error_details", [])[:5],
            "drift_pressure": drift_pressure,
            "drift_alert": drift_alert,
            "group_error_signal": hypothesis_errors,
            "active_hypotheses": meta_stats['active'],
            "total_hypotheses": len(self.hypotheses.population),
            "meta_summary": meta_stats,
            "groups": len(self.grouper.groups)
        }

    def _build_group_error_signal(self) -> Dict[str, float]:
        """Aggregate per-group model error signal from current hypothesis fit scores."""
        grouped: Dict[str, List[float]] = {}
        for hyp in self.hypotheses.population.values():
            fit = float(getattr(hyp, "fit_score", 0.5))
            if not math.isfinite(fit):
                continue

            # Convert fit quality [0,1] to error pressure [0,1].
            error = max(0.0, min(1.0, 1.0 - fit))

            for var in getattr(hyp, "variables", []):
                gid = self.grouper.get_group_id(var)
                if not gid:
                    continue
                grouped.setdefault(gid, []).append(error)

        return {gid: float(np.mean(vals)) for gid, vals in grouped.items() if vals}
    
    def _arbitrate_step(self) -> None:
        """
        Executes a periodic arbitration phase.

        Invokes the `HypothesisArbiter` to review ACTIVE hypotheses and identify
        conflicts (e.g., cycles, contradictory directions). Only ACTIVE hypotheses
        compete; TENTATIVE ones are still gathering evidence and are not pruned here.
        This ensures rare types (compositional, moderating, similarity) survive long
        enough to accumulate federation evidence before being judged.
        """
        active = [h for h in self.hypotheses.population.values()
                  if h.meta.state == HypothesisState.ACTIVE]
        if not active:
            return
        kept_hyps = self.arbiter.arbitrate(active)
        kept_ids = {h.meta.id for h in kept_hyps}

        for hid, hyp in list(self.hypotheses.population.items()):
            if hyp.meta.state == HypothesisState.ACTIVE and hid not in kept_ids:
                self.hypotheses._kill(hid)

    def _explore_step(self) -> None:
        """
        Active exploration: inject new diverse hypothesis types for under-explored
        variable pairs.

        Strategy (improved from v1 which only ever added CorrelationalHypothesis):
        1. Identify variable pairs with no ACTIVE hypotheses
        2. Sample unexplored pairs and rotate through diverse relationship types
           (Causal, Competitive, Probabilistic, Graph, Functional-degree-2)
        3. For triplets: add Synergistic and Mediating when enough vars exist
        4. Soft-boost weakly-improving hypotheses to survive pruning
        """
        import itertools
        import random

        all_vars = list(self._var_index.keys()) if self._var_index else []
        if len(all_vars) < 2:
            return

        # Pairs already covered by strong hypotheses
        strongest = self.hypotheses.get_strongest(top_k=20)
        covered = set()
        for h in strongest:
            if len(h.variables) >= 2:
                covered.add(tuple(sorted(h.variables[:2])))

        all_pairs = list(itertools.combinations(all_vars, 2))
        unexplored = [p for p in all_pairs if p not in covered]

        if unexplored:
            bs = self.buffer_size
            pair_explore_types = [
                lambda a, b: CausalHypothesis(a, b, lag=2, buffer_size=bs),
                lambda a, b: CausalHypothesis(b, a, lag=2, buffer_size=bs),
                lambda a, b: CompetitiveHypothesis(a, b, buffer_size=bs),
                lambda a, b: ProbabilisticHypothesis(a, b, buffer_size=bs),
                lambda a, b: GraphHypothesis(a, b, buffer_size=bs),
                lambda a, b: StructuralHypothesis(a, b, buffer_size=bs),
                lambda a, b: FunctionalHypothesis(a, b, degree=2, buffer_size=bs),
                lambda a, b: FunctionalHypothesis(b, a, degree=2, buffer_size=bs),
            ]
            n_new = min(3, len(unexplored))
            chosen = random.sample(unexplored, n_new)
            explore_idx = self.step_count % len(pair_explore_types)
            for v1, v2 in chosen:
                try:
                    constructor = pair_explore_types[explore_idx]
                    self.hypotheses.add(constructor(v1, v2))
                    explore_idx = (explore_idx + 1) % len(pair_explore_types)
                except Exception as exc:
                    logger.debug(f"Exploration pair ({v1},{v2}) failed: {exc}")

        # Triplet exploration: signal-driven — prefer (A,B,C) where both (A,B) and
        # (B,C) bivariate hypotheses already have confidence > 0.30.  Falls back to
        # random if no strong pairs found yet.
        if len(all_vars) >= 3:
            bs = self.buffer_size
            strong_pairs: set = set()
            for h in self.hypotheses.population.values():
                if getattr(h, 'confidence', 0.0) > 0.30 and len(h.variables) == 2:
                    strong_pairs.add((h.variables[0], h.variables[1]))
                    strong_pairs.add((h.variables[1], h.variables[0]))
            candidate_triplets = [
                (a, b, c)
                for a, b, c in itertools.combinations(all_vars, 3)
                if (a, b) in strong_pairs and (b, c) in strong_pairs
            ]
            if not candidate_triplets:
                candidate_triplets = list(itertools.combinations(all_vars, 3))
            if candidate_triplets:
                a, b, c = random.choice(candidate_triplets)
                try:
                    self.hypotheses.add(SynergisticHypothesis(a, b, c, buffer_size=bs))
                except Exception:
                    pass
                try:
                    self.hypotheses.add(MediatingHypothesis(a, b, c, buffer_size=bs))
                except Exception:
                    pass

        # Soft-boost improving hypotheses
        for h in list(self.hypotheses.population.values()):
            if getattr(h, 'confidence', 1.0) < 0.3 and hasattr(h, 'is_improving') \
                    and h.is_improving():
                h.confidence = min(0.4, h.confidence + 0.05)

    def _run_iv_pass(self) -> None:
        """
        Instrumental variable approximation for causal direction confidence.

        For each active CausalHypothesis(X→Y), searches the pool for an instrument Z
        such that corr(Z, X) > 0.4 AND corr(Z, Y) < 0.15.  Finding a valid Z provides
        additional evidence that the X→Y direction is correct and yields a small
        confidence boost proportional to the IV strength.

        This is a heuristic (not a formal IV estimator) but provides meaningful
        signal for annual macroeconomic data where Z is typically a policy variable
        or exogenous shock indicator.
        """
        if len(self.hypotheses.population) < 3:
            return

        # build variable → recent values map (up to 20 observations)
        var_series: Dict[str, np.ndarray] = {}
        for h in self.hypotheses.population.values():
            for attr in ('buffer_x', 'buffer_y', 'buffer'):
                buf = getattr(h, attr, None)
                if buf and len(buf) >= 15:
                    var = (h.variables[0] if attr in ('buffer', 'buffer_y') else
                           getattr(h, 'source', h.variables[0]))
                    if var not in var_series:
                        var_series[var] = np.array(list(buf))[-20:]
                    break

        if len(var_series) < 3:
            return

        active_causal = [
            h for h in self.hypotheses.population.values()
            if (h.rel_type == RelationshipType.CAUSAL
                and h.confidence >= 0.15
                and len(getattr(h, 'buffer_x', [])) >= 15)
        ]

        for hyp in active_causal:
            X_name = getattr(hyp, 'source', hyp.variables[0])
            Y_name = getattr(hyp, 'target', hyp.variables[-1])
            if X_name not in var_series or Y_name not in var_series:
                continue

            X = var_series[X_name]
            Y = var_series[Y_name]
            n = min(len(X), len(Y))
            if n < 10:
                continue
            X, Y = X[:n], Y[:n]

            for Z_name, Z_raw in var_series.items():
                if Z_name in (X_name, Y_name):
                    continue
                Z = np.asarray(Z_raw).ravel()[:n]
                if len(Z) != n:
                    continue
                std_Z = float(np.std(Z))
                if std_Z < 1e-9:
                    continue
                X1d = np.asarray(X).ravel()
                Y1d = np.asarray(Y).ravel()
                if len(X1d) != n or len(Y1d) != n:
                    continue
                try:
                    corr_ZX = float(np.corrcoef(Z, X1d)[0, 1]) if np.std(X1d) > 1e-9 else 0.0
                    corr_ZY = float(np.corrcoef(Z, Y1d)[0, 1]) if np.std(Y1d) > 1e-9 else 0.0
                except Exception:
                    continue
                if abs(corr_ZX) > 0.4 and abs(corr_ZY) < 0.15:
                    iv_strength = abs(corr_ZX) * (1.0 - abs(corr_ZY))
                    boost = 0.02 * iv_strength
                    hyp.confidence = min(1.0, hyp.confidence + boost)
                    hyp.alpha_success = max(hyp.alpha_success,
                                            hyp.alpha_success + boost * 5.0)
                    break

    def predict(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Confidence-weighted ensemble prediction.

        Aggregates predictions from all hypotheses with confidence >= 0.25,
        weighting each hypothesis's prediction by its confidence score.
        Falls back to lag-1 (last observed value) for variables with no
        active predictor.

        Args:
            row: The current observation row (used as lag-1 fallback).

        Returns:
            Dict mapping variable names to predicted values.
        """
        safe_row = self._sanitize_row(row)
        weighted_sum: Dict[str, float] = {}
        weight_total: Dict[str, float] = {}

        for h in self.hypotheses.population.values():
            if h.meta.state == HypothesisState.DEAD:
                continue
            if h.confidence < 0.20:
                continue
            result = h.predict_value(safe_row)
            if result is None:
                continue
            var, val = result
            if not np.isfinite(val):
                continue
            w = h.confidence
            weighted_sum[var] = weighted_sum.get(var, 0.0) + w * val
            weight_total[var] = weight_total.get(var, 0.0) + w

        output: Dict[str, float] = {}
        for var in weighted_sum:
            if weight_total[var] > 0:
                output[var] = weighted_sum[var] / weight_total[var]

        # lag-1 fallback for variables with no active predictor
        for var, val in safe_row.items():
            if var not in output and np.isfinite(val):
                output[var] = val

        return output

    # ------------------------------------------------------------------
    # Federation: peer trust weighting + hypothesis-level sharing
    # ------------------------------------------------------------------

    def process_peer_row(self, peer_id: str, row: Dict[str, Any],
                         peer_weight: float = 1.0) -> None:
        """
        Process one observation row from a peer node.

        Identical to process_row() except that the Bayesian signal for each
        hypothesis is scaled by peer_weight before accumulation, allowing
        high-trust peers to contribute more evidence.

        Args:
            peer_id:     Identifier of the sending node (unused internally,
                         reserved for future per-peer trust tracking).
            row:         The peer's raw observation dict.
            peer_weight: Trust weight in [0, 1].  1.0 = full trust (default),
                         0.5 = half-weight evidence from this peer.
        """
        safe_row = self._sanitize_row(row)
        weight = float(np.clip(peer_weight, 0.0, 1.0))

        for h in self.hypotheses.population.values():
            if h.meta.state == HypothesisState.DEAD:
                continue
            try:
                metrics = h.evaluate(safe_row)
                fit = metrics.get('fit_score', 0.5)
                p_val = metrics.get('p_value') or metrics.get('p_value_forward')
                if p_val is not None:
                    signal = max(0.0, 1.0 - float(p_val) * 10.0) * weight
                else:
                    signal = max(0.0, (fit - 0.5) * 2.0) * weight
                h.alpha_success += signal
                h.beta_failure += (1.0 - signal) * weight
                h.confidence = h.alpha_success / (h.alpha_success + h.beta_failure)
            except Exception:
                pass

    def export_hypothesis_summary(self,
                                   min_conf: float = 0.15) -> List[Dict[str, Any]]:
        """
        Export a privacy-preserving hypothesis summary for peer sharing.

        Shares only (variables, type, confidence, evidence) tuples — no raw data.
        Peers can use this to reinforce or suppress their own matching hypotheses
        without receiving the original observation rows.

        Args:
            min_conf: Minimum confidence threshold for inclusion.

        Returns:
            List of hypothesis summary dicts.
        """
        return [
            {
                'vars': h.variables,
                'type': h.rel_type.value,
                'conf': round(h.confidence, 4),
                'evidence': h.evidence,
            }
            for h in self.hypotheses.population.values()
            if h.confidence >= min_conf and h.meta.state != HypothesisState.DEAD
        ]

    def receive_hypothesis_summary(self, peer_summaries: List[Dict[str, Any]],
                                    peer_weight: float = 1.0) -> None:
        """
        Incorporate a peer's hypothesis summary into local beliefs.

        For each incoming summary, finds matching local hypotheses (same variable
        set and relationship type) and nudges their confidence toward the peer's
        reported confidence, scaled by peer_weight.

        Args:
            peer_summaries: Output of a peer's export_hypothesis_summary().
            peer_weight:    Trust weight in [0, 1].
        """
        weight = float(np.clip(peer_weight, 0.0, 1.0))
        for summary in peer_summaries:
            peer_vars = tuple(sorted(summary.get('vars', [])))
            peer_type = summary.get('type', '')
            peer_conf = float(summary.get('conf', 0.0))

            for h in self.hypotheses.population.values():
                if h.meta.state == HypothesisState.DEAD:
                    continue
                if tuple(sorted(h.variables)) != peer_vars:
                    continue
                if h.rel_type.value != peer_type:
                    continue
                # nudge confidence toward peer's reported confidence
                delta = 0.03 * weight * (peer_conf - h.confidence)
                h.confidence = float(np.clip(h.confidence + delta, 0.0, 1.0))
                # reflect nudge in alpha accumulator for consistency
                if delta > 0:
                    h.alpha_success += abs(delta) * 5.0
                break

    def get_knowledge_graph(self) -> List[Dict[str, Any]]:
        """
        Exports the current best understanding of the system as a Knowledge Graph.

        Retrieves the top-k strongest hypotheses from the pool and serializes them.
        This represents the "truth" learned by the system so far.

        Returns:
            A list of dictionaries, each representing a discovered relationship/edge.
        """
        strongest = self.hypotheses.get_strongest(top_k=50)
        return [h.to_dict() for h in strongest]

    def get_candidate_paths(self, top_k: int = 30) -> List[Candidate]:
        """
        Bridge method: export top hypotheses as Candidate objects for MPIEOrchestrator.

        Connects the two previously isolated discovery pipelines:
            OnlineDiscoveryEngine (streaming hypothesis pool)
            ↓  get_candidate_paths()
            MPIEOrchestrator (path-based evaluator + bandit router)

        Constructs Candidate objects using the integer variable index map built
        during initialize() / initialize_v2(), matching the Candidate dataclass
        expected by BanditRouter and Evaluator.

        Args:
            top_k: Maximum candidates to export.

        Returns:
            List of Candidate objects ready for Evaluator.score().
        """
        import hashlib

        var_index = getattr(self, '_var_index', {})
        candidates: List[Candidate] = []
        strongest = self.hypotheses.get_strongest(top_k=top_k)

        for hyp in strongest:
            if len(hyp.variables) < 2:
                continue
            if getattr(hyp, 'confidence', 0.0) < 0.25:
                continue

            # Map variable names to integer indices
            try:
                var_indices = tuple(
                    var_index[v] for v in hyp.variables[:2]
                    if v in var_index
                )
            except KeyError:
                continue
            if len(var_indices) < 2:
                continue

            # Deterministic path_id from (vars, rel_type)
            path_key = f"{var_indices}:{hyp.rel_type.value}"
            path_id = hashlib.md5(path_key.encode()).hexdigest()[:16]

            try:
                cand = Candidate(
                    path_id=path_id,
                    vars=var_indices,
                    lags=(0, 0),
                    ops=('identity', 'identity'),
                    root=var_indices[0],
                    depth=1,
                    domain=0,
                    gen_reason=f'discovery:{hyp.rel_type.value}',
                )
                candidates.append(cand)
            except Exception as exc:
                logger.debug(f"Could not build Candidate for {hyp.meta.id}: {exc}")

        return candidates

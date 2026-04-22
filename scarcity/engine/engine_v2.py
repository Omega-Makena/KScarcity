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

from .discovery import HypothesisPool, Hypothesis, RelationshipType
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

    def __init__(self, explore_interval: int = 10, mode: str = "balanced"):
        """
        Initializes the discovery engine and its sub-components.

        Args:
            explore_interval: The number of steps between exploration phases
                (currently a placeholder for future active learning expansion).
        """
        self.hypotheses = HypothesisPool()
        self.grouper = AdaptiveGrouper()
        self.arbiter = HypothesisArbiter()
        self.meta_controller = MetaController()
        
        self.step_count = 0
        self.explore_interval = explore_interval
        self.start_time = time.time()
        self.mode = "balanced"
        self.lifecycle_interval = 10
        self.arbitration_interval = 50
        self.grouping_enabled = True
        self.exploration_enabled = True
        self.update_error_total = 0
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
                self.hypotheses.add(CorrelationalHypothesis(a, b))
                self.hypotheses.add(FunctionalLinearHypothesis(a, b))
                self.hypotheses.add(FunctionalLinearHypothesis(b, a))

        for v in var_names:
            self.hypotheses.add(TemporalLagHypothesis(v, v))
            self.hypotheses.add(EquilibriumHypothesis(v))

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

        logger.info(f"Initializing V2 engine with {len(var_names)} variables")
        
        # 1. For each variable: Temporal (AR) and Equilibrium
        for v in var_names:
            self.hypotheses.add(TemporalHypothesis(v, lag=2))
            self.hypotheses.add(EquilibriumHypothesis(v))
        
        # 2. For variable pairs (limit to avoid explosion)
        import itertools
        max_pairs = 100
        pairs = list(itertools.combinations(var_names, 2))[:max_pairs]

        # 3. For variable triples (very limited to avoid blow-up)
        max_triplets = 10
        triplets = list(itertools.combinations(var_names, 3))[:max_triplets]
        
        for a, b in pairs:
            # Correlational (always)
            self.hypotheses.add(CorrelationalHypothesis(a, b))
            
            # Functional (linear relationship)
            self.hypotheses.add(FunctionalHypothesis(a, b, degree=1))
            self.hypotheses.add(FunctionalHypothesis(b, a, degree=1))
            
            # Causal/Granger (expensive but valuable)
            if use_causal:
                self.hypotheses.add(CausalHypothesis(a, b, lag=2))
                self.hypotheses.add(CausalHypothesis(b, a, lag=2))
            
            # Competitive (trade-off detection)
            self.hypotheses.add(CompetitiveHypothesis(a, b))

            # Probabilistic / Structural (lightweight, pairwise)
            self.hypotheses.add(ProbabilisticHypothesis(a, b))
            self.hypotheses.add(StructuralHypothesis(a, b))
            self.hypotheses.add(GraphHypothesis(a, b))

        # 4. Triple-variable hypotheses (limited)
        for a, b, c in triplets:
            self.hypotheses.add(CompositionalHypothesis([a, b], c))
            self.hypotheses.add(SynergisticHypothesis(a, b, c))
            self.hypotheses.add(MediatingHypothesis(a, b, c))
            self.hypotheses.add(ModeratingHypothesis(a, b, c))
            self.hypotheses.add(LogicalHypothesis(a, b, c))

        # 5. Similarity hypothesis across a small variable subset
        if len(var_names) >= 3:
            subset = var_names[: min(5, len(var_names))]
            self.hypotheses.add(SimilarityHypothesis(subset, n_clusters=min(3, len(subset))))
        
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
        
        # 2. Update Hypotheses (Evaluate -> Fit -> UpdateConf)
        # Note: This calls the new Hypothesis.update which returns the Dict of metrics
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

        Invokes the `HypothesisArbiter` to review all ACTIVE hypotheses and identify
        conflicts (e.g., cycles, contradictory directions). Conflicted or weaker
        hypotheses are killed.
        """
        active = list(self.hypotheses.population.values())
        kept_hyps = self.arbiter.arbitrate(active)
        kept_ids = {h.meta.id for h in kept_hyps}
        
        all_ids = list(self.hypotheses.population.keys())
        for hid in all_ids:
            if hid not in kept_ids:
                self.hypotheses._kill(hid)

    # Map of pair-level relationship types to constructors for exploration
    _PAIR_EXPLORE_TYPES = [
        lambda a, b: CausalHypothesis(a, b, lag=2),
        lambda a, b: CausalHypothesis(b, a, lag=2),
        lambda a, b: CompetitiveHypothesis(a, b),
        lambda a, b: ProbabilisticHypothesis(a, b),
        lambda a, b: GraphHypothesis(a, b),
        lambda a, b: FunctionalHypothesis(a, b, degree=2),
        lambda a, b: FunctionalHypothesis(b, a, degree=2),
    ]

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

        all_vars = list(self.grouper.groups.keys()) if self.grouper.groups else []
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
            n_new = min(3, len(unexplored))
            chosen = random.sample(unexplored, n_new)
            # Rotate through diverse type constructors
            explore_idx = self.step_count % len(self._PAIR_EXPLORE_TYPES)
            for v1, v2 in chosen:
                try:
                    constructor = self._PAIR_EXPLORE_TYPES[explore_idx]
                    self.hypotheses.add(constructor(v1, v2))
                    explore_idx = (explore_idx + 1) % len(self._PAIR_EXPLORE_TYPES)
                except Exception as exc:
                    logger.debug(f"Exploration pair ({v1},{v2}) failed: {exc}")

        # Triplet exploration: Synergistic and Mediating
        if len(all_vars) >= 3:
            triplets = list(itertools.combinations(all_vars, 3))
            if triplets:
                a, b, c = random.choice(triplets)
                try:
                    self.hypotheses.add(SynergisticHypothesis(a, b, c))
                except Exception:
                    pass
                try:
                    self.hypotheses.add(MediatingHypothesis(a, b, c))
                except Exception:
                    pass

        # Soft-boost improving hypotheses
        for h in list(self.hypotheses.population.values()):
            if getattr(h, 'confidence', 1.0) < 0.3 and hasattr(h, 'is_improving') \
                    and h.is_improving():
                h.confidence = min(0.4, h.confidence + 0.05)

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

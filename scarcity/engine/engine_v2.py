"""
Online Relationship Discovery — Main Engine.

The orchestrator that ties:
Stream Row -> [Hypothesis Pool] -> [Measurement] -> [Grouping Updates]

HARDENED v4: Includes Meta-Controller and Explicit Scoring.
"""

import time
import logging
import math
from typing import Dict, List, Any

from .discovery import HypothesisPool, Hypothesis, RelationshipType
from .grouping import AdaptiveGrouper
from .arbitration import HypothesisArbiter
from .controller import MetaController
from .algorithms_online import FunctionalLinearHypothesis, CorrelationalHypothesis, TemporalLagHypothesis, EquilibriumHypothesis

# New production-quality hypothesis classes
from .relationships import (
    CausalHypothesis,
    CorrelationalHypothesis as CorrelationalHypothesisV2,
    TemporalHypothesis,
    FunctionalHypothesis,
    EquilibriumHypothesis as EquilibriumHypothesisV2,
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
    def __init__(self, explore_interval: int = 10):
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

        self.grouper.initialize(var_names)
        
        if len(var_names) < 10:
            import itertools
            for a, b in itertools.combinations(var_names, 2):
                self.hypotheses.add(CorrelationalHypothesis(a, b))
                self.hypotheses.add(FunctionalLinearHypothesis(a, b)) # A -> B
                self.hypotheses.add(FunctionalLinearHypothesis(b, a)) # B -> A
                
        for v in var_names:
             self.hypotheses.add(TemporalLagHypothesis(v, v)) # Autoregression
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
        
        self.grouper.initialize(var_names)
        
        logger.info(f"Initializing V2 engine with {len(var_names)} variables")
        
        # 1. For each variable: Temporal (AR) and Equilibrium
        for v in var_names:
            self.hypotheses.add(TemporalHypothesis(v, lag=2))
            self.hypotheses.add(EquilibriumHypothesisV2(v))
        
        # 2. For variable pairs (limit to avoid explosion)
        import itertools
        max_pairs = 100
        pairs = list(itertools.combinations(var_names, 2))[:max_pairs]

        # 3. For variable triples (very limited to avoid blow-up)
        max_triplets = 10
        triplets = list(itertools.combinations(var_names, 3))[:max_triplets]
        
        for a, b in pairs:
            # Correlational (always)
            self.hypotheses.add(CorrelationalHypothesisV2(a, b))
            
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
        
        # 3. Meta-Controller Lifecycle (Every 10 steps)
        if self.step_count % 10 == 0:
            self.meta_controller.manage_lifecycle(self.hypotheses)
        
        # 4. Monitor Grouping
        hypothesis_errors = {}
        self.grouper.monitor(safe_row, hypothesis_errors)
        
        # 5. Arbitration (Every 50 steps)
        if self.step_count % 50 == 0:
            self._arbitrate_step()
            
        # 6. Exploration
        if self.step_count % self.explore_interval == 0:
            self._explore_step()
        
        # Gather Stats
        meta_stats = self.meta_controller.get_summary(self.hypotheses)
        
        return {
            "step": self.step_count,
            "active_hypotheses": meta_stats['active'],
            "total_hypotheses": len(self.hypotheses.population),
            "meta_summary": meta_stats,
            "groups": len(self.grouper.groups)
        }
    
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

    def _explore_step(self) -> None:
        """
        Active exploration: Generate new hypotheses based on emerging patterns.
        
        Called periodically to inject diversity into the hypothesis pool.
        This helps discover relationships that weren't in the initial hypothesis set.
        
        Strategy:
        1. Identify unexplored variable pairs (not covered by strong hypotheses)
        2. Sample a few new pairs and create exploratory hypotheses
        3. Optionally promote weak hypotheses that showed improvement
        """
        # Get all variables from grouper
        all_vars = list(self.grouper.groups.keys()) if self.grouper.groups else []
        
        if len(all_vars) < 2:
            return
        
        # Get existing strong hypotheses and their variable pairs
        strongest = self.hypotheses.get_strongest(top_k=20)
        existing_pairs = set()
        for h in strongest:
            if len(h.variables) >= 2:
                existing_pairs.add(tuple(sorted(h.variables[:2])))
        
        # Generate all possible pairs
        import itertools
        all_pairs = list(itertools.combinations(all_vars, 2))
        
        # Find unexplored pairs
        unexplored = [p for p in all_pairs if p not in existing_pairs]
        
        if not unexplored:
            logger.debug(f"Explore step: No unexplored pairs remaining (step {self.step_count})")
            return
        
        # Sample a few new pairs (limit exploration rate)
        import random
        n_new = min(3, len(unexplored))
        new_pairs = random.sample(unexplored, n_new)
        
        # Create exploratory hypotheses for new pairs
        for (v1, v2) in new_pairs:
            # Add correlational hypothesis (low cost, baseline)
            try:
                hyp = CorrelationalHypothesisV2(v1, v2)
                self.hypotheses.add(hyp)
                logger.debug(f"Exploration: Added CorrelationalHypothesis for {v1} <-> {v2}")
            except Exception as e:
                logger.debug(f"Exploration: Failed to create hypothesis for {v1}, {v2}: {e}")
        
        # Optionally look for weak but improving hypotheses to keep alive
        # This prevents premature killing of slow-to-converge relationships
        all_hyps = list(self.hypotheses.population.values())
        for h in all_hyps:
            # Check if hypothesis is weak but showing improvement
            if hasattr(h, 'is_improving') and h.is_improving():
                if h.confidence < 0.3:
                    # Give it a small confidence boost to survive longer
                    h.confidence = min(0.4, h.confidence + 0.05)
                    logger.debug(f"Exploration: Boosted improving hypothesis {h.meta.id}")

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

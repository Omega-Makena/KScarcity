"""
economic engine wrapper.

specialized engine for macroeconomic modeling.
handling mapping between user-friendly variable names and raw world bank codes.
"""

from typing import Dict, Any, List
import logging
from .engine_v2 import OnlineDiscoveryEngine
from ..economic_config import ECONOMIC_VARIABLES, CODE_TO_NAME

logger = logging.getLogger(__name__)

class EconomicDiscoveryEngine:
    def __init__(self):
        self.core = OnlineDiscoveryEngine()
        self.whitelist_codes = set(ECONOMIC_VARIABLES.values())
        logger.info(f"initialized economic engine with {len(self.whitelist_codes)} variables.")
        self._populate_initial_hypotheses()

    def _populate_initial_hypotheses(self):
        """
        for small economic datasets, we don't need random exploration.
        we can just brute-force track all pairwise relationships.
        n=18 -> 18*17 = 306 hypotheses.
        PLUS: We MUST add Autoregressive (Self-Lag) hypotheses to capture momentum.
        """
        import itertools
        from .algorithms_online import VectorizedFunctionalHypothesis, TemporalLagHypothesis
        from .discovery import RelationshipType
        
        friendly_names = list(CODE_TO_NAME.values())
        count = 0
        pool = self.core.hypotheses
        
        # 1. Functional (Cross-Sectional): A -> B
        for a, b in itertools.combinations(friendly_names, 2):
            # A -> B
            idx1 = pool.vec_pool.get_or_create(a, b)
            h1 = VectorizedFunctionalHypothesis(a, b, idx1, pool.vec_pool.engine)
            pool.add(h1)
            
            # B -> A
            idx2 = pool.vec_pool.get_or_create(b, a)
            h2 = VectorizedFunctionalHypothesis(b, a, idx2, pool.vec_pool.engine)
            pool.add(h2)
            count += 2
            
        # 2. Autoregressive (Temporal): A(t-1) -> A(t)
        # This gives the system "inertia" and "memory".
        for a in friendly_names:
            # We use TemporalLagHypothesis for this
            # It needs to be initialized.
            # Note: TemporalLagHypothesis is NOT vectorized in the current hybrid engine (it's legacy OOP)
            # unless we upgraded it. Based on algorithms_online.py, it is a legacy Hypothesis subclass.
            h_ar = TemporalLagHypothesis(a, a, lag=1)
            pool.add(h_ar)
            count += 1
            
        logger.info(f"pre-populated {count} economic hypotheses (functional full mesh + AR1 diagonals).")

    def process_row_raw(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        process a raw world bank row (keys are codes).
        1. filter to whitelist.
        2. rename to friendly names.
        3. feed to core engine.
        """
        # 1. filter & rename
        clean_row = {}
        for code, val in row.items():
            if code in self.whitelist_codes:
                friendly = CODE_TO_NAME[code]
                clean_row[friendly] = val
                
        # 2. process
        # if the row is empty after filtering (e.g. non-economic year), skip
        if not clean_row:
            return {"status": "skipped"}
            
        return self.core.process_row(clean_row)

    def get_simulation_handle(self):
        """returns a policysimulator initialized with current knowledge."""
        from .simulation import PolicySimulator
        return PolicySimulator(self.core.hypotheses)
    
    def print_top_relationships(self, k: int = 10):
        """debug helper."""
        hyps = self.core.hypotheses.get_strongest(k)
        print(f"\n--- top {k} economic relationships ---")
        for h in hyps:
            # vars are already friendly names
            print(f"[{h.rel_type.name}] {h.variables} (conf: {h.confidence:.2f})")
            if hasattr(h, 'idx'):
                 # print weights if vectorized
                 w = self.core.hypotheses.vec_pool.engine.W[h.idx]
                 print(f"    weights: {w}")

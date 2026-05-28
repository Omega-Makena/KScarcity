"""
Meta-Controller for Hypothesis Lifecycle.

Decides the state (ACTIVE, TENTATIVE, DECAYING, DEAD) of hypotheses
based on their explicit metrics (Confidence, Stability, Evidence).
"""

import logging
from typing import Dict, List, Any
from .discovery import Hypothesis, HypothesisState, HypothesisPool

logger = logging.getLogger(__name__)

class MetaController:
    """
    Manages the lifecycle state of causal hypotheses.

    The MetaController acts as a state machine arbiter for the hypothesis pool.
    It transitions hypotheses between states (ACTIVE, TENTATIVE, DECAYING, DEAD)
    based on their accumulated evidence, stability metrics, and confidence scores.
    It ensures that only robust hypotheses remain active while weaker ones are
    pruned or put on probation.
    """
    def __init__(self,
                 confidence_threshold: float = 0.7,
                 stability_threshold: float = 0.6,
                 min_evidence: int = 20,
                 kill_threshold: float = 0.10):
        """
        Initializes the meta-controller with lifecycle thresholds.

        Args:
            confidence_threshold: The minimum confidence score required for a
                hypothesis to become or remain ACTIVE.
            stability_threshold: The minimum stability score required for a
                hypothesis to become or remain ACTIVE.
            min_evidence: The minimum number of observations required before a
                TENTATIVE hypothesis can be promoted to ACTIVE.
        """
        self.conf_thresh = confidence_threshold
        self.stab_thresh = stability_threshold
        self.min_evidence = min_evidence
        self.kill_threshold = kill_threshold

    def manage_lifecycle(self, pool: HypothesisPool) -> None:
        """
        Scans the hypothesis pool and applies state transition logic.

        Iterates through all hypotheses in the pool and updates their state based
        on current metrics.
        - TENTATIVE -> ACTIVE: If evidence, confidence, and stability are sufficient.
        - ACTIVE -> DECAYING: If metrics drop below maintenance thresholds.
        - DECAYING -> ACTIVE: If metrics recover.
        - * -> DEAD: If metrics drop below critical survival levels.

        Also handles the permanent removal (killing) of DEAD hypotheses.

        Args:
            pool: The `HypothesisPool` containing the population to manage.
        """
        dead_ids = []
        
        for hid, hyp in pool.population.items():
            current_state = hyp.meta.state
            
            # Metrics
            conf = hyp.confidence
            stab = hyp.stability
            evid = hyp.evidence
            
            new_state = current_state
            
            # State Machine
            if current_state == HypothesisState.TENTATIVE:
                if evid > self.min_evidence:
                    if conf > self.conf_thresh and stab > self.stab_thresh:
                        new_state = HypothesisState.ACTIVE
                    elif conf < self.kill_threshold:
                        dead_ids.append(hid)  # Kill early failure
            
            elif current_state == HypothesisState.ACTIVE:
                if conf < (self.conf_thresh - 0.1) or stab < (self.stab_thresh - 0.1):
                    new_state = HypothesisState.DECAYING
            
            elif current_state == HypothesisState.DECAYING:
                if conf > self.conf_thresh and stab > self.stab_thresh:
                    new_state = HypothesisState.ACTIVE # Recovered
                elif conf < self.kill_threshold:
                    dead_ids.append(hid)  # Dead
            
            # Apply transition
            if new_state != current_state:
                hyp.meta.state = new_state
                # logger.info(f"Hypothesis {hid} transitioned {current_state} -> {new_state}")

        # Execute Kill Logic
        for hid in dead_ids:
            pool._kill(hid)

    @classmethod
    def small_dataset(cls) -> 'MetaController':
        """Tuned thresholds for datasets with n < 100 observations.

        Key decisions:
        - min_evidence=10 (not 20): allows hypothesis graduation before the stream ends.
        - confidence_threshold=0.55 (not 0.70): compensates for low F-test power at n<50.
        - kill_threshold=0.0 (disabled): with λ=0.99 and pure null signal the Bayesian
          accumulator decays as α≈0.1·λⁿ, β≈100·(1−λⁿ).  After 34 steps conf≈0.0024;
          after 102 steps (federated) conf≈0.0006.  Any positive threshold kills null-
          signal hypotheses (compositional, moderating, similarity, competitive) before
          they can accumulate federation evidence.  Setting threshold=0.0 disables killing
          entirely — pool capacity (2000) is the only pruning mechanism.  Null hypotheses
          stay in the pool but their conf<0.01 means they are filtered out at graph
          extraction (conf_threshold=0.45) and never used in forecasting (threshold=0.20).
        """
        return cls(
            confidence_threshold=0.55,
            stability_threshold=0.50,
            min_evidence=10,
            kill_threshold=0.0,
        )

    def get_summary(self, pool: HypothesisPool) -> Dict[str, int]:
        """
        Generates a summary of the pool's state distribution.

        Args:
            pool: The hypothesis pool to summarize.

        Returns:
            A dictionary counting the number of hypotheses in each state
            (active, tentative, decaying, dead).
        """
        counts = {"active": 0, "tentative": 0, "decaying": 0, "dead": len(pool.graveyard)}
        for hyp in pool.population.values():
            counts[hyp.meta.state.value] += 1
        return counts

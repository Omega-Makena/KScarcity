"""
online relationship discovery engine — core abstractions.

implements the 'hypothesis survival' paradigm where relationships are treated 
as active constraints that must survive the stream of data.

hardened v4: explicit scoring (score, conf, evidence, stability).
"""

from __future__ import annotations

import abc
import numpy as np
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
from enum import Enum

# lazy import to avoid circular dependency if possible, or direct
from .vectorized_core import VectorizedHypothesisPool

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    FUNCTIONAL = "functional"
    PROBABILISTIC = "probabilistic"
    COMPOSITIONAL = "compositional"
    COMPETITIVE = "competitive"
    SYNERGISTIC = "synergistic"
    MEDIATING = "mediating"
    MODERATING = "moderating"
    GRAPH = "graph"
    SIMILARITY = "similarity"
    EQUILIBRIUM = "equilibrium"
    LOGICAL = "logical"

class HypothesisState(Enum):
    """
    Lifecycle states for a causal hypothesis.
    """
    TENTATIVE = "tentative" #: Newly created, gathering initial evidence.
    ACTIVE = "active"       #: Proven, high-confidence, currently used for predictions.
    DECAYING = "decaying"   #: Previously active but performance is degrading.
    DEAD = "dead"           #: Discarded due to lack of evidence or falsification.

@dataclass
class HypothesisMetadata:
    """mlops metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    state: HypothesisState = HypothesisState.TENTATIVE
    generation: int = 0
    parents: List[str] = field(default_factory=list)

class Hypothesis(abc.ABC):
    """
    Abstract base class for all relational hypotheses.

    A Hypothesis represents a proposed relationship between variables (e.g., A causes B,
    A correlates with B). It encapsulates:
    - The structural definition (variables, type).
    - The metadata (ID, state, lineage).
    - The online metrics (fit score, confidence, stability).
    - The learning logic (`fit_step`, `evaluate`).

    Subclasses implement specific mathematical models for different relationship types.
    """
    
    def __init__(self, variables: List[str], rel_type: RelationshipType):
        self.variables = variables
        self.rel_type = rel_type
        self.meta = HypothesisMetadata()
        
        # core metrics (the "4 pillars")
        self.fit_score = 0.5   # how well it explains current data (0-1)
        self.confidence = 0.5  # bayesian probability of truth
        self.evidence = 0      # n samples
        self.stability = 0.5   # 1.0 - cv(error)
        
        # bayesian priors
        self.alpha_success = 1.0
        self.beta_failure = 1.0

    @abc.abstractmethod
    def fit_step(self, row: Dict[str, float]) -> None:
        """
        Updates the internal parameters of the hypothesis with a single new data row.
        
        This is the "learning" step. For example, in a linear regression hypothesis,
        this would update the weights using Recursive Least Squares (RLS).
        
        Args:
            row: A dictionary mapping variable names to their current values.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
        """
        Measures how well the current data row aligns with the hypothesis.

        This is the "testing" step. It computes metrics like prediction error,
        likelihood, or alignment without modifying the internal model parameters.

        Args:
            row: A dictionary mapping variable names to their current values.

        Returns:
            A dictionary of metrics including:
            - 'fit_score': Normalized goodness-of-fit (0.0 to 1.0).
            - 'confidence': Bayesian belief in the hypothesis (0.0 to 1.0).
            - 'evidence': Count of observations seen.
            - 'stability': Measure of metric consistency (0.0 to 1.0).
        """
        pass

    def predict_value(self, row: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """
        predict the target variable's value based on the input row.
        used for simulation. returns (target_variable_name, predicted_value) or none.
        default implementation returns none (non-predictive hypothesis).
        """
        return None

    def update(self, row: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes the full online update cycle for the hypothesis.

        Sequence:
        1. Evaluate the hypothesis against the new data (compute fit/error).
        2. Update internal model parameters (learn from data).
        3. Update Bayesian confidence scores based on success/failure of the fit.
        
        Args:
            row: The incoming data row.

        Returns:
            The metrics dictionary containing the latest fit_score, confidence, etc.
        """
        # 1. evaluate (read-only measurement)
        metrics = self.evaluate(row)
        self.fit_score = metrics['fit_score']
        # note: 'confidence' and 'stability' in metrics might be pre-calculated 
        # based on history, or we update them here.
        
        # 2. fit (update internal state)
        self.fit_step(row)
        self.evidence += 1
        
        # 3. update bayesian confidence
        # we use fit_score as a "soft" success
        self.alpha_success += self.fit_score
        self.beta_failure += (1.0 - self.fit_score)
        
        self.confidence = self.alpha_success / (self.alpha_success + self.beta_failure)
        
        # update reported metrics
        metrics['confidence'] = self.confidence
        metrics['evidence'] = self.evidence
        # stability is usually tracked inside the subclass (cv of error), passed up in metrics
        self.stability = metrics.get('stability', 0.5)
        
        return metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.meta.id,
            "type": self.rel_type.value,
            "state": self.meta.state.value,
            # Surface metadata so UIs can show "new vs old" findings.
            "created_at": self.meta.created_at,
            "generation": self.meta.generation,
            "variables": self.variables,
            "metrics": {
                "fit_score": self.fit_score,
                "confidence": self.confidence,
                "evidence": self.evidence,
                "stability": self.stability
            }
        }

class HypothesisPool:
    """manages population."""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.population: Dict[str, Hypothesis] = {}
        self.graveyard: List[Dict[str, Any]] = [] 
        
        # vectorized backend
        self.vec_pool = VectorizedHypothesisPool(capacity=capacity)
    
    def add(self, hypothesis: Hypothesis) -> None:
        if len(self.population) >= self.capacity:
            self._prune_weakest(force=True)
        self.population[hypothesis.meta.id] = hypothesis
        
    def update_all(self, row: Dict[str, float]) -> None:
        """
        Updates all active hypotheses in the pool with the new data row.

        Implements a hybrid execution model:
        - Vectorized hypotheses are batched and updated via a high-performance backend (if available).
        - Standard (OOP) hypotheses are updated individually in a loop.
        
        Args:
            row: The new data observation.
        """
        # 1. identify active vectorized hypotheses
        # in a real heavy-load system, we'd cache these lists. for now, iterate.
        # to optimize, we maintain a list of 'vectorized_ids'.
        
        vec_indices = []
        vec_x: List[List[float]] = []
        vec_y: List[float] = []
        
        legacy_hyps = []
        
        # we need to distinguish types.
        # let's check if 'idx' attr exists or use isinstance.
        # isinstance check is fast enough.
        
        for hyp in self.population.values():
            if hasattr(hyp, 'idx') and hasattr(hyp, 'engine'):
                # it is vectorized
                # check if we have data for it
                # hyp.input, hyp.target
                if hyp.input in row and hyp.target in row:
                    x_val = row[hyp.input]
                    y_val = row[hyp.target]
                    # basic safety
                    if np.isfinite(x_val) and np.isfinite(y_val):
                         vec_indices.append(hyp.idx)
                         vec_x.append([1.0, x_val]) # bias, feature
                         vec_y.append(y_val)
                
                # still need to run evaluate/meta update logic
                # but fit_step is no-op.
                hyp.update(row) 
            else:
                legacy_hyps.append(hyp)

        # 2. batch update vectorized
        if vec_indices:
            X_batch = np.array(vec_x, dtype=np.float32)
            Y_batch = np.array(vec_y, dtype=np.float32)
            idxs = np.array(vec_indices, dtype=np.int32)
            
            
            self.vec_pool.engine.update_subset(idxs, X_batch, Y_batch)
        else:
             # DEBUG
             # logger.warning("No vectorized items found in this row.")
             pass

        # 3. update legacy
        for hyp in legacy_hyps:
            hyp.update(row)

    def _kill(self, hid: str) -> None:
        if hid in self.population:
            hyp = self.population.pop(hid)
            record = hyp.to_dict()
            record['death_time'] = time.time()
            self.graveyard.append(record)
            if len(self.graveyard) > 500: self.graveyard.pop(0)

    def _prune_weakest(self, force: bool = False) -> None:
        if not self.population: return
        # simple kill lowest confidence
        sorted_hyps = sorted(self.population.items(), key=lambda item: item[1].confidence)
        self._kill(sorted_hyps[0][0])

    def get_strongest(self, top_k: int = 10) -> List[Hypothesis]:
        # return only active or high conf
        return sorted(self.population.values(), 
                     key=lambda h: h.confidence, 
                     reverse=True)[:top_k]

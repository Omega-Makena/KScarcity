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
from collections import deque
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

class RegimeTracker:
    """
    Page's CUSUM structural break detector.

    Feeds residuals from a hypothesis model; tracks how stable the relationship
    is across regime shifts.  consistency_score = 1.0 means no breaks ever
    detected; 0.0 means the relationship breaks constantly.
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._threshold = threshold
        self._drift = drift
        self._sigma_ema = 1.0
        self._sigma_alpha = 0.05
        self._n_breaks = 0
        self._n_consistent = 0

    def update(self, residual: float) -> bool:
        """Feed one residual. Returns True if a regime break is detected."""
        r = abs(float(residual))
        self._sigma_ema = ((1.0 - self._sigma_alpha) * self._sigma_ema
                           + self._sigma_alpha * r)
        normed = r / (self._sigma_ema + 1e-8)
        self._cusum_pos = max(0.0, self._cusum_pos + normed - self._drift)
        self._cusum_neg = max(0.0, self._cusum_neg - normed - self._drift)
        break_detected = (self._cusum_pos > self._threshold
                          or self._cusum_neg > self._threshold)
        if break_detected:
            self._n_breaks += 1
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0
        else:
            self._n_consistent += 1
        return break_detected

    @property
    def consistency_score(self) -> float:
        """Fraction of steps without a detected break. [0, 1]"""
        total = self._n_breaks + self._n_consistent
        if total == 0:
            return 0.5
        return self._n_consistent / total


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

        # skeptical bayesian prior — hypotheses must earn confidence
        # alpha=0.1, beta=1.0 → initial confidence ≈ 0.09
        self.alpha_success = 0.1
        self.beta_failure = 1.0

        # optional regime tracker; subclasses assign self._regime_tracker = RegimeTracker()
        self._regime_tracker: Optional[RegimeTracker] = None

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

    def observe(self, state: Dict[str, float]) -> None:
        """
        Notify the hypothesis of a new simulation state.
        Default is a no-op; subclasses may override to track simulation history.
        """
        pass

    def _update_regime_tracker(self, residual: float) -> None:
        """Subclasses call this in fit_step() with their prediction residual."""
        if self._regime_tracker is not None:
            self._regime_tracker.update(residual)

    def _regime_consistency(self) -> float:
        """
        Returns a consistency score in [0, 1] reflecting regime stability.
        Default delegates to the optional _regime_tracker.
        TemporalHypothesis overrides this to use its built-in CUSUM.
        """
        if self._regime_tracker is not None:
            return self._regime_tracker.consistency_score
        return 0.5

    def update(self, row: Dict[str, float]) -> Dict[str, Any]:
        """
        Full online update cycle.

        1. Evaluate the hypothesis (read-only statistical measurement).
        2. Derive signal from p-value (E[signal]≈0.025 on null data — calibration-safe)
           or fit-score deviation from 0.5 null baseline.
        3. Update internal model parameters via fit_step.
        4. Accumulate signal with λ=0.99 exponential forgetting so the effective
           memory window is ~100 steps.  Without forgetting, 1000 past observations
           overwhelm any regime shift; with λ=0.99 old evidence decays gracefully.
        5. Apply regime-consistency as a conservative multiplicative lift (max 5%).
           Proportional form keeps low-confidence hypotheses from being falsely
           promoted and guarantees the result stays in [0, 1].
        """
        # 1. evaluate (read-only)
        metrics = self.evaluate(row)
        self.fit_score = metrics['fit_score']

        # 2. signal — prefer proper p-value; fall back to fit-score deviation
        p_val = metrics.get('p_value')
        if p_val is None:
            p_val = metrics.get('p_value_forward')

        if p_val is not None:
            # p < 0.10 → positive signal.
            # M=10: E[null signal] ≈ 0.05  (∫₀^{0.10} (1−10p) dp = 0.05).
            # Calibrated for annual macro data: genuine relationships yield
            # p=0.05-0.10 on rolling 5-10yr windows; M=10 lets those accumulate.
            # Null SS confidence ≈ 5% — safely below predict() threshold (20%)
            # and MetaController graduation threshold (70%).
            signal = max(0.0, 1.0 - float(p_val) * 10.0)
        else:
            # fit-score fallback: null baseline at 0.5.
            # RandomPredictor and i.i.d. noise typically yield fit_score ≈ 0.5,
            # so signal ≈ 0.  Structured data yields fit_score > 0.5 → signal > 0.
            signal = max(0.0, (self.fit_score - 0.5) * 2.0)

        # 3. update internal model parameters
        self.fit_step(row)
        self.evidence += 1

        # 4. exponentially-weighted Bayesian accumulators (λ = 0.99).
        # Steady-state confidence converges to signal_mean, allowing the
        # accumulator to track regime changes rather than being permanently
        # anchored by early observations.
        _lambda = 0.99
        self.alpha_success = _lambda * self.alpha_success + signal
        self.beta_failure  = _lambda * self.beta_failure  + (1.0 - signal)
        self.confidence = self.alpha_success / (self.alpha_success + self.beta_failure)

        # 5. regime-consistency: conservative multiplicative lift (max +5%)
        # Proportional form: a hypothesis at conf=0.9 gets at most 0.9*1.05=0.945;
        # one at conf=0.01 gets at most 0.0105.  Low-confidence hypotheses are not
        # falsely promoted.  Result is guaranteed ≤ 1.0 since conf ≤ 1.
        if self.evidence > 15:
            rc = self._regime_consistency()
            self.confidence = min(1.0, self.confidence * (1.0 + 0.05 * rc))

        metrics['confidence'] = self.confidence
        metrics['evidence'] = self.evidence
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
        self.last_update_errors = 0
        self.last_update_error_details: List[Dict[str, Any]] = []
        self.total_update_errors = 0
        self._fdr_step: int = 0

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
        # reset row-level error accounting
        self.last_update_errors = 0
        self.last_update_error_details = []

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
                try:
                    hyp.update(row)
                except Exception as exc:
                    self.last_update_errors += 1
                    self.last_update_error_details.append(
                        {
                            "hypothesis_id": getattr(hyp.meta, "id", "unknown"),
                            "stage": "vectorized_update",
                            "error": str(exc),
                        }
                    )
            else:
                legacy_hyps.append(hyp)

        # 2. batch update vectorized
        if vec_indices:
            X_batch = np.array(vec_x, dtype=np.float32)
            Y_batch = np.array(vec_y, dtype=np.float32)
            idxs = np.array(vec_indices, dtype=np.int32)
            
            
            try:
                self.vec_pool.engine.update_subset(idxs, X_batch, Y_batch)
            except Exception as exc:
                self.last_update_errors += 1
                self.last_update_error_details.append(
                    {
                        "hypothesis_id": "vectorized_batch",
                        "stage": "vectorized_batch_update",
                        "error": str(exc),
                    }
                )
        else:
             # DEBUG
             # logger.warning("No vectorized items found in this row.")
             pass

        # 3. update legacy
        for hyp in legacy_hyps:
            try:
                hyp.update(row)
            except Exception as exc:
                self.last_update_errors += 1
                self.last_update_error_details.append(
                    {
                        "hypothesis_id": getattr(hyp.meta, "id", "unknown"),
                        "stage": "legacy_update",
                        "error": str(exc),
                    }
                )

        self.total_update_errors += self.last_update_errors

        # FDR correction every 10 steps: penalise low-evidence hypotheses that
        # only appear significant due to multiple testing across the pool.
        self._fdr_step += 1
        if self._fdr_step % 10 == 0:
            self._apply_fdr_correction()

    def _apply_fdr_correction(self) -> None:
        """
        Soft Benjamini-Hochberg FDR correction at q=0.05.

        BH ranking uses FORWARD confidence (alpha_success / total).
        After deflation, h.confidence is reset to conf_fwd so that the
        ensemble threshold and arbitrator continue to see the true forward
        confidence, not an intermediate signed value.
        Hypotheses that do not clear the BH rank threshold and have fewer than
        15 observations have their alpha accumulator gently deflated (8%),
        pulling them back toward the skeptical prior.
        """
        n = len(self.population)
        if n < 10:
            return
        q = 0.05

        def _fwd(h) -> float:
            denom = h.alpha_success + h.beta_failure
            return h.alpha_success / denom if denom > 0 else 0.0

        hyps = sorted(self.population.values(), key=_fwd, reverse=True)
        bh_cutoff = 0
        for k, h in enumerate(hyps, start=1):
            if (1.0 - _fwd(h)) <= (k / n) * q:
                bh_cutoff = k
        for h in hyps[bh_cutoff:]:
            if h.evidence >= 15:
                continue
            h.alpha_success = max(0.1, h.alpha_success * 0.92)
            conf_fwd = h.alpha_success / (h.alpha_success + h.beta_failure)
            h.confidence = conf_fwd

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

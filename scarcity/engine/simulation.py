"""
policy simulation engine.

allows "what-if" scenarios by propagating shocks through the 
discovered knowledge graph (hypothesis pool).
"""

import copy
import logging
from typing import Dict, List, Any, Optional
from .discovery import HypothesisPool, HypothesisState, Hypothesis

logger = logging.getLogger(__name__)

class PolicySimulator:
    """
    sandboxed simulation environment.
    """
    # Minimum confidence for TENTATIVE hypotheses to be included
    # when no ACTIVE hypotheses exist (small-data fallback).
    _MIN_TENTATIVE_CONFIDENCE = 0.35

    def __init__(self, engine_hypotheses: HypothesisPool):
        # Prefer ACTIVE hypotheses (strict promotion gate passed).
        self.active_pool = [
            copy.deepcopy(h) 
            for h in engine_hypotheses.population.values() 
            if h.meta.state == HypothesisState.ACTIVE
        ]
        
        # Small-data fallback: if NO hypotheses reached ACTIVE, include
        # high-confidence TENTATIVE ones so the simulator is not empty.
        if not self.active_pool:
            self.active_pool = [
                copy.deepcopy(h)
                for h in engine_hypotheses.population.values()
                if h.meta.state == HypothesisState.TENTATIVE
                and h.confidence >= self._MIN_TENTATIVE_CONFIDENCE
            ]
            if self.active_pool:
                logger.info(
                    f"No ACTIVE hypotheses found; using {len(self.active_pool)} "
                    f"TENTATIVE hypotheses (confidence >= {self._MIN_TENTATIVE_CONFIDENCE})."
                )
        
        self.momentum = 0.5 # Inertia factor (0.0 = instant, 1.0 = frozen)
        
        self.state: Dict[str, float] = {}
        self.policies: Dict[str, float] = {} # Persistent overrides
        self.history: List[Dict[str, float]] = []
        
        logger.info(f"initialized simulation with {len(self.active_pool)} active hypotheses.")

    def set_initial_state(self, state: Dict[str, float]) -> None:
        """set the starting values for variables."""
        self.state = state.copy()
        self.history = [self.state.copy()]

    def perturb(self, variable: str, value: float) -> None:
        """inject a shock/override."""
        self.state[variable] = value
        logger.info(f"perturbation: {variable} set to {value}")

    def set_policy(self, variable: str, value: float) -> None:
        """
        set a sustained policy (variable lock).
        the variable will be forced to this value at every step,
        overriding any endogenous predictions.
        """
        self.policies[variable] = value
        # Apply immediately
        self.state[variable] = value
        logger.info(f"policy set: lock {variable} = {value}")

    
    def add_constraint(self, variable: str, limit: float, operator: str = 'max') -> None:
        """add a fiscal rule (e.g. debt < 100)."""
        if not hasattr(self, 'constraints'): self.constraints = []
        self.constraints.append({'var': variable, 'limit': limit, 'op': operator})

    def check_constraints(self, state: Dict[str, float]) -> List[str]:
        """return list of violated constraints."""
        violations = []
        if not hasattr(self, 'constraints'): return []
        
        for c in self.constraints:
            val = state.get(c['var'], 0.0)
            if c['op'] == 'max' and val > c['limit']:
                violations.append(f"⚠️ {c['var']} ({val:.1f}) exceeds limit {c['limit']}")
            elif c['op'] == 'min' and val < c['limit']:
                violations.append(f"⚠️ {c['var']} ({val:.1f}) below limit {c['limit']}")
        return violations

    def calculate_metrics(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        derive stability, stress, and confidence scores.
        """
        #system Stability (Average Confidence of Active Hypotheses)
        avg_conf = 0.0
        if self.active_pool:
            avg_conf = sum(h.confidence for h in self.active_pool) / len(self.active_pool)
            
        #system Stress (Distance from "Normal")
        stress = 0.0
        if self.history:
            initial = self.history[0]
            count = 0
            for k, loop_v in state.items():
                if k in initial and abs(initial[k]) > 1e-3:
                    # Deviation percent
                    dev = abs(loop_v - initial[k]) / abs(initial[k])
                    stress += dev
                    count += 1
            if count > 0: stress /= count
            
        return {
            "system_confidence": avg_conf,
            "system_stress": stress, 
            "policy_efficiency": avg_conf * (1.0 - min(stress, 1.0)) # heuristic
        }

    def step(self) -> Dict[str, float]:
        """
        advance simulation by one time step.
        each hypothesis votes on the next value of its target.
        """
        next_state = self.state.copy()
        updates: Dict[str, List[float]] = {}
        
        # collect predictions from all hypotheses
        for hyp in self.active_pool:
            prediction = hyp.predict_value(self.state)
            if prediction:
                target, val = prediction
                if target not in updates:
                    updates[target] = []
                updates[target].append(val)
        
       
        #  resolve updates with MOMENTUM (Physics)
        predicted_vars = set()
        
        # Pre-calculate drifts (Vitality)
        import random
        drifts = {}
        for var in self.state:
            curr = self.state[var]
            if abs(curr) > 1e-6:
                drifts[var] = curr * 0.005 * (random.random() - 0.5) * 2
            else:
                drifts[var] = 0.001 * (random.random() - 0.5) * 2

        # Combine Prediction + Drift + Inertia
        all_vars = set(self.state.keys()) | set(updates.keys())
        
        for var in all_vars:
            if var.lower() == 'year': continue
            if var in self.policies: continue
            
            # 1. Determine Target Value
            if var in updates and updates[var]:
                # Causal Prediction
                raw_target = sum(updates[var]) / len(updates[var])
                predicted_vars.add(var)
            else:
                 # No prediction? Stay where we are (with drift)
                raw_target = self.state.get(var, 0.0)
            
            # 2. Add Vitality (Organic Noise)
            # We add drift to the TARGET, so the system "wants" to move there
            target_val = raw_target + drifts.get(var, 0.0)
            
            #pply Momentum (Continuity Rule)
            # next = p * current + (1-p) * target
            current_val = self.state.get(var, 0.0)
            next_val = self.momentum * current_val + (1.0 - self.momentum) * target_val
            
            next_state[var] = next_val
                
        #  Apply Propagated Time
        if 'Year' in self.state:
             next_state['Year'] = self.state['Year'] + 1
        if 'year' in self.state:
             next_state['year'] = self.state['year'] + 1

        # 4. RE-APPLY POLICIES (Exogenous Overrides)
        for var, val in self.policies.items():
            next_state[var] = val
        
        # 5. Check Constraints & Metrics
        alerts = self.check_constraints(next_state)
        metrics = self.calculate_metrics(next_state)
        
        # 6. Commit
        self.state = next_state
        self.history.append(self.state.copy())
        
        # meta_history for metrics
        if not hasattr(self, 'meta_history'): self.meta_history = []
        self.meta_history.append({'metrics': metrics, 'alerts': alerts})
        
        # 7. Notify Hypotheses
        self.observe(self.state)
        
        return self.state

    def observe(self, state: Dict[str, float]) -> None:
        """
        externally force hypotheses to observe a state 
        (e.g., after manual scrub/rewind).
        """
        for hyp in self.active_pool:
            hyp.observe(state)

    def run(self, steps: int) -> List[Dict[str, float]]:
        """run for n steps."""
        for _ in range(steps):
            self.step()
        return self.history

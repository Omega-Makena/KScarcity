"""
Vectorized Policy Evaluation Engine (V4).
Compiles N individual policies into Matrices for O(1) Batch Evaluation.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from .policies import EconomicPolicy

logger = logging.getLogger("kshield.tensor")


@dataclass
class TensorEngineConfig:
    """
    Configuration for PolicyTensorEngine.
    
    Allows runtime tuning of crisis response and other parameters
    without modifying individual policies.
    """
    # Crisis response configuration
    crisis_multiplier: float = 5.0  # Weight multiplier during crisis (was hardcoded 5.0)
    normal_weight: float = 1.0      # Weight multiplier during normal conditions
    
    # PID tuning overrides (applied globally if > 0)
    global_kp_scale: float = 1.0    # Scale factor for proportional gains
    global_ki_scale: float = 1.0    # Scale factor for integral gains
    global_kd_scale: float = 1.0    # Scale factor for derivative gains
    
    # Integral windup prevention
    integral_max: float = 100.0     # Maximum integral accumulation
    integral_min: float = -100.0    # Minimum integral accumulation
    
    # Output limiting
    max_magnitude: float = 10.0     # Maximum action magnitude per policy
    min_action_threshold: float = 1e-6  # Minimum magnitude to report


@dataclass
class PolicyMatrix:
    """
    Represents the compiled policy logic.
    """
    # Matrix Dimensions: (N_policies, N_metrics)
    # Allows calculating: MetricVector . Mask = Triggered
    
    # 1. State Mapping
    metric_map: Dict[str, int] # "inflation" -> 0
    action_map: Dict[str, int] # "tighten" -> 0
    
    # 2. Thresholds (N_policies, )
    thresholds: np.ndarray
    
    # 3. Directions (N_policies, ) 1 for >, -1 for <
    directions: np.ndarray
    
    # 4. Metric Selectors (N_policies, N_metrics) One-hot
    metric_selector: np.ndarray
    
    # 5. PID Gains (N_policies, 3) [Kp, Ki, Kd]
    pid_gains: np.ndarray
    
    # 6. Weights (N_policies, )
    weights: np.ndarray
    
    # 7. Action Mapping (N_policies, ) -> Index in action_map
    action_indices: np.ndarray
    
    # 8. Authority Mapping (List[str] of length N_policies)
    authorities: List[str]

    # 9. Crisis (N_policies, )
    thesis_thresholds: np.ndarray
    
    num_policies: int


class PolicyTensorEngine:
    def __init__(self, config: Optional[TensorEngineConfig] = None):
        """
        Initialize PolicyTensorEngine with optional configuration.
        
        Args:
            config: Engine configuration. Uses defaults if None.
        """
        self.config = config or TensorEngineConfig()
        self.matrix: Optional[PolicyMatrix] = None
        # State tracking for PID integration: (N_policies, )
        self.integrals: np.ndarray = None
        self.last_errors: np.ndarray = None

    def compile(self, policies_dict: Dict[str, List[EconomicPolicy]], all_metrics: List[str]):
        """
        Compiles the dictionary of policy objects into numpy arrays.
        """
        flat_policies = []
        for cat, rules in policies_dict.items():
            flat_policies.extend(rules)
            
        n_p = len(flat_policies)
        n_m = len(all_metrics)
        
        metric_map = {m: i for i, m in enumerate(all_metrics)}
        
        # Arrays
        thresholds = np.zeros(n_p)
        directions = np.zeros(n_p)
        selector = np.zeros((n_p, n_m))
        pid_gains = np.zeros((n_p, 3))
        weights = np.ones(n_p)
        action_indices = np.zeros(n_p, dtype=int)
        crisis_thresh = np.zeros(n_p)
        authorities = []
        
        # Action Map (Dynamic)
        action_set = set(p.action for p in flat_policies)
        action_map = {a: i for i, a in enumerate(sorted(list(action_set)))}
        
        for i, p in enumerate(flat_policies):
            # Threshold
            thresholds[i] = p.threshold
            
            # Direction (> is 1, < is -1)
            directions[i] = 1.0 if p.direction == ">" else -1.0
            
            # Metric Selector
            if p.metric in metric_map:
                col = metric_map[p.metric]
                selector[i, col] = 1.0
                
            # PID
            pid_gains[i] = [p.kp, p.ki, p.kd]
            
            # Action
            action_indices[i] = action_map.get(p.action, 0)
            
            # Crisis
            crisis_thresh[i] = p.crisis_threshold
            
            # Misc
            authorities.append(p.authority)
            
        self.matrix = PolicyMatrix(
            metric_map=metric_map,
            action_map=action_map,
            thresholds=thresholds,
            directions=directions,
            metric_selector=selector,
            pid_gains=pid_gains,
            weights=weights,
            action_indices=action_indices,
            authorities=authorities,
            thesis_thresholds=crisis_thresh,
            num_policies=n_p
        )
        
        # Init State
        self.integrals = np.zeros(n_p)
        self.last_errors = np.zeros(n_p)
        
        logger.info(f"Compiled {n_p} policies into Tensor Engine (State Dim: {n_m})")

    def evaluate(self, state_values: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
        """
        Vectorized Step.
        Returns aggregated action vectors {action_name: magnitude}.
        
        Uses configuration for crisis multiplier and output limiting.
        """
        if not self.matrix: return {}
        
        M = self.matrix
        cfg = self.config
        
        # 1. Extract Relevant Values for each policy
        # (N_policies, N_metrics) dot (N_metrics, ) -> (N_policies, )
        current_vals = M.metric_selector @ state_values
        
        # 2. Calculate Error
        # Direction 1 (>): Error = Val - Thresh
        # Direction -1 (<): Error = Thresh - Val
        # Formula: Direction * (Val - Thresh)
        raw_diff = current_vals - M.thresholds
        errors = M.directions * raw_diff
        
        # Clamp negative errors (we only act if boundary crossed)
        # Note: If continuous PID is desired without threshold, remove clamp.
        # But per specs, it's threshold based.
        active_errors = np.maximum(0.0, errors) 
        
        # 3. Crisis Check
        # Check against crisis thresholds in same way
        crisis_diff = current_vals - M.thesis_thresholds
        # Direction logic applies to crisis too
        crisis_breach = M.directions * crisis_diff
        is_crisis = crisis_breach > 0
        
        # Apply Weights (Vectorized) - now using configurable multiplier
        current_weights = np.where(is_crisis, cfg.crisis_multiplier, cfg.normal_weight)
        
        # 4. PID Calculation
        # I (with windup prevention)
        self.integrals += active_errors * dt
        self.integrals = np.clip(self.integrals, cfg.integral_min, cfg.integral_max)
        
        # D
        derivs = (active_errors - self.last_errors) / dt
        self.last_errors = active_errors
        
        # PID Sum with configurable scaling
        # (N_p, ) * (N_p, )
        p_term = active_errors * M.pid_gains[:, 0] * cfg.global_kp_scale
        i_term = self.integrals * M.pid_gains[:, 1] * cfg.global_ki_scale
        d_term = derivs * M.pid_gains[:, 2] * cfg.global_kd_scale
        
        magnitudes = p_term + i_term + d_term
        
        # Apply magnitude limiting
        magnitudes = np.clip(magnitudes, -cfg.max_magnitude, cfg.max_magnitude)
        
        # 5. Aggregation
        # We need to sum magnitudes * weights per Action.
        # This is effectively a "Group By" operation.
        # In pure matrix terms, we can use an Action Selector Matrix (N_actions, N_policies).
        
        weighted_mags = magnitudes * current_weights
        
        results = {}
        # Iterate unique actions (usually small number < 10)
        for action_name, idx in M.action_map.items():
            # Mask for policies belonging to this action
            mask = (M.action_indices == idx)
            net_mag = np.sum(weighted_mags[mask])
            if abs(net_mag) > cfg.min_action_threshold:
                results[action_name] = net_mag
                
        return results

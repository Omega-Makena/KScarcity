"""
Type definitions for Controller ⇆ Evaluator interaction.

Defines shared data structures for online bandit learning and path evaluation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class Candidate:
    """
    Represents a proposed causal path for evaluation.

    A `Candidate` is a structured definition of a potential causal relationship,
    specifying the source indices, time lags, and transformation operations
    that map input variables to a target.

    Attributes:
        path_id: A deterministic unique identifier derived from the path's
            structure (vars, lags, ops, and schema).
        vars: A tuple of integer indices identifying the variables involved
            in the path.
        lags: A tuple of integer lags corresponding to each variable in `vars`.
        ops: A tuple of string identifiers for the operations applied to the
            variables (e.g., "sketch", "attn").
        root: The index of the primary root variable (used for bandit arm
            assignment).
        depth: The length or complexity of the path.
        domain: The integer identifier for the domain/shard this path belongs to.
        gen_reason: A string tag indicating the strategy that generated this
            candidate (e.g., "UCB", "random", "diversity").
    """
    path_id: str
    vars: Tuple[int, ...]
    lags: Tuple[int, ...]
    ops: Tuple[str, ...]
    root: int
    depth: int
    domain: int
    gen_reason: str


@dataclass
class EvalResult:
    """
    Encapsulates the outcome of validating a candidate path.

    Attributes:
        path_id: The ID of the candidate path being evaluated.
        gain: The predictive improvement score (e.g., R² gain or negative log-likelihood reduction).
            Positive values indicate a valid signal.
        ci_lo: The lower bound of the bootstrapped confidence interval for the gain.
        ci_hi: The upper bound of the bootstrapped confidence interval for the gain.
        stability: A score in [0, 1] representing the robustness of the relationship
            across resamples or time windows.
        cost_ms: The execution time of the evaluation step in milliseconds.
        accepted: A boolean flag indicating if the path met all acceptance criteria
            (thresholds for gain, stability, uncertainty).
        extras: A dictionary for additional metadata or debug metrics (e.g.,
            'granger_p', 'holdout_rows', 'error').
    """
    path_id: str
    gain: float
    ci_lo: float
    ci_hi: float
    stability: float
    cost_ms: float
    accepted: bool
    extras: Dict[str, Any]


@dataclass
class Reward:
    """
    A shaped feedback signal for the bandit controller.

    Attributes:
        path_id: The ID of the candidate path associated with this reward.
        arm_key: The composite key identifying the bandit arm (e.g., `(root_var, depth)`)
            that should receive this feedback.
        value: The final scalar reward value, normalized to [-1, +1]. Includes
            core performance metrics and shaping adjustments.
        latency_penalty: The specific penalty component applied due to processing cost (non-negative).
        diversity_bonus: The specific bonus component added for structural novelty (non-negative).
        accepted: A copy of the acceptance status from the evaluation result, used
            to update success/failure counts in the bandit.
    """
    path_id: str
    arm_key: Tuple[int, int]
    value: float
    latency_penalty: float
    diversity_bonus: float
    accepted: bool

"""
BanditRouter — Multi-Armed Bandit Path Selection.

Provides exploration-exploitation tradeoff for causal path selection
using Thompson Sampling (default) or Upper Confidence Bound (UCB) algorithms.

Thompson Sampling is preferred because:
- Better theoretical regret bounds in many scenarios
- Natural uncertainty quantification via posterior sampling
- More robust to non-stationary rewards (common in causal discovery)
- Implicit exploration without tuning exploration parameters
"""

import hashlib
import logging
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from scarcity.engine.types import Candidate, Reward
from scarcity.config import ENGINE_CONFIG

logger = logging.getLogger(__name__)


class BanditAlgorithm(str, Enum):
    """Available bandit algorithms."""
    THOMPSON = "thompson"
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class ArmStats:
    """
    Statistics for a single bandit arm.
    
    Uses Beta distribution parameters for Thompson Sampling:
    - alpha (wins): Successes, initialized to 1 (Beta prior)
    - beta (losses): Failures, initialized to 1 (Beta prior)
    
    This gives a uniform prior Beta(1,1) which is updated as data arrives.
    """
    alpha: float = 1.0  # Successes (Beta prior α)
    beta: float = 1.0   # Failures (Beta prior β)
    observations: int = 0
    cumulative_reward: float = 0.0
    last_pulled: int = -1
    
    @property
    def mean(self) -> float:
        """Expected value under Beta posterior."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """Variance under Beta posterior."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    @property
    def ucb_score(self) -> float:
        """Upper confidence bound score."""
        if self.observations == 0:
            return float('inf')
        return self.mean + np.sqrt(2 * np.log(self.observations + 1) / self.observations)


@dataclass
class BanditConfig:
    """Configuration for BanditRouter."""
    algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON
    n_arms: int = 1000
    epsilon: float = 0.1  # For epsilon-greedy
    ucb_c: float = 2.0    # UCB exploration constant
    decay_factor: float = 0.999  # Reward decay for non-stationarity
    min_observations: int = 5  # Minimum pulls before exploitation


class BanditRouter:
    """
    Multi-armed bandit router for path proposal selection.
    
    Implements Thompson Sampling (default), UCB, and epsilon-greedy
    strategies for selecting causal paths to evaluate.
    
    Thompson Sampling is recommended for causal discovery because:
    1. It naturally handles uncertainty in early exploration
    2. It adapts to changing reward distributions (non-stationary)
    3. It has provably good regret bounds
    4. It doesn't require tuning exploration parameters
    
    Usage:
        router = BanditRouter(config=BanditConfig(algorithm=BanditAlgorithm.THOMPSON))

        # Get candidate paths ranked by the bandit policy
        candidates = router.propose(n_proposals=10, context={'schema': schema})

        # After evaluation, feed shaped rewards back
        router.apply_rewards(rewards)
    """
    
    def __init__(
        self,
        config: Optional[BanditConfig] = None,
        n_arms: int = 1000,
        drg: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize BanditRouter.
        
        Args:
            config: Configuration object. If None, uses defaults with Thompson Sampling.
            n_arms: Number of arms (ignored if config provided).
        """
        self.config = config or BanditConfig(n_arms=n_arms)
        self.arms: Dict[int, ArmStats] = {}
        self._rng = rng or np.random.default_rng()
        self.drg = drg or {}
        self._step = 0
        self._total_reward = 0.0
        self._arm_id_counter = 0
        
        # Path ID to arm ID mapping for named paths
        self._path_to_arm: Dict[str, int] = {}

        # Candidate-proposal state (for the Candidate-returning propose contract)
        self._proposer = ENGINE_CONFIG.proposer
        self._diversity = ENGINE_CONFIG.diversity
        # How many times each path_id has been proposed (frequency novelty).
        self._proposal_counts: Dict[str, int] = {}
        # Variable-sets of recently accepted candidates (set novelty).
        self._recent_varsets: deque = deque(maxlen=self._diversity.recent_memory)
        # Last known variable count, reused when a window carries no schema.
        self._last_n_vars: Optional[int] = None

        logger.info(f"BanditRouter initialized with {self.config.algorithm.value} algorithm")

    def apply_meta_update(self, tau: Optional[float] = None, gamma_diversity: Optional[float] = None) -> None:
        if tau is not None:
            self.config.epsilon = float(np.clip(tau, 0.0, 1.0))
        if gamma_diversity is not None:
            self.config.ucb_c = float(max(0.0, gamma_diversity))
    
    def register_arms(self, n_arms: int) -> List[int]:
        """
        Register a batch of new arms.
        
        Args:
            n_arms: Number of arms to register.
            
        Returns:
            List of assigned arm IDs.
        """
        arm_ids = []
        for _ in range(n_arms):
            arm_id = self._arm_id_counter
            self.arms[arm_id] = ArmStats()
            arm_ids.append(arm_id)
            self._arm_id_counter += 1
        return arm_ids
    
    def register_path(self, path_id: str) -> int:
        """
        Register a named path and get its arm ID.
        
        Args:
            path_id: Unique identifier for the path (e.g., "A->B@lag2").
            
        Returns:
            Arm ID for this path.
        """
        if path_id in self._path_to_arm:
            return self._path_to_arm[path_id]
        
        arm_id = self._arm_id_counter
        self.arms[arm_id] = ArmStats()
        self._path_to_arm[path_id] = arm_id
        self._arm_id_counter += 1
        return arm_id
    
    def get_arm_id(self, path_id: str) -> Optional[int]:
        """Get arm ID for a path, or None if not registered."""
        return self._path_to_arm.get(path_id)
    
    def propose(
        self,
        n_proposals: int,
        context: Optional[Dict[str, Any]] = None,
        exclude: Optional[Set[int]] = None,
    ) -> List[Candidate]:
        """
        Propose candidate paths to evaluate, ranked by the bandit policy.

        Generates directed variable-pair paths from the window schema, registers
        each as a bandit arm, scores every arm under the configured policy
        (Thompson / UCB / epsilon-greedy), and returns the top ``n_proposals`` as
        ``Candidate`` objects for the Evaluator.

        Args:
            n_proposals: Number of candidate paths to return.
            context: Optional dict; ``context['schema']`` supplies the variable
                set (via its ``fields``) used to enumerate candidate paths.
            exclude: Optional set of ``path_id`` strings to skip.

        Returns:
            List of ``Candidate`` objects, highest-priority first.
        """
        n_vars = self._infer_n_vars(context)
        if n_vars < 2:
            return []

        exclude = exclude or set()
        candidates = self._generate_candidate_paths(n_vars)
        candidates = [c for c in candidates if c.path_id not in exclude]
        if not candidates:
            return []

        # Score each candidate by its arm's policy value, then take the top-n.
        scored: List[Tuple[float, Candidate]] = []
        for cand in candidates:
            arm_id = self.register_path(cand.path_id)
            scored.append((self._arm_score(arm_id), cand))
        scored.sort(reverse=True, key=lambda x: x[0])

        selected = [c for _, c in scored[: max(0, n_proposals)]]
        for cand in selected:
            self._proposal_counts[cand.path_id] = (
                self._proposal_counts.get(cand.path_id, 0) + 1
            )
        return selected

    def _infer_n_vars(self, context: Optional[Dict[str, Any]]) -> int:
        """Determine the variable count for candidate generation from context."""
        schema = (context or {}).get('schema', {}) if context else {}
        fields = schema.get('fields') if isinstance(schema, dict) else None
        # An explicit field list is authoritative, even if it yields < 2 vars.
        if isinstance(fields, (dict, list)):
            n_vars = len(fields)
            if n_vars >= 2:
                self._last_n_vars = n_vars
            return n_vars
        # No field list: honour an explicit count key if present.
        if isinstance(schema, dict):
            for key in ('n_features', 'n_vars', 'width'):
                val = schema.get(key)
                if isinstance(val, int):
                    if val >= 2:
                        self._last_n_vars = val
                    return val
        # No schema info at all: reuse the last known count, else fall back.
        if self._last_n_vars:
            return self._last_n_vars
        return self._proposer.fallback_n_vars

    def _generate_candidate_paths(self, n_vars: int) -> List[Candidate]:
        """
        Enumerate directed variable-pair candidate paths, bounded by ``max_arms``.

        For each ordered pair (source i, target j) and each configured source
        lag, builds one contemporaneous-target path. Deterministic ``path_id``
        keeps arms stable across windows.
        """
        ops = tuple(self._proposer.candidate_ops)
        cands: List[Candidate] = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                for lag in self._proposer.candidate_lags:
                    lags = (int(lag), 0)
                    vars_t = (i, j)
                    path_id = hashlib.md5(
                        f"{vars_t}:{lags}:{ops}".encode()
                    ).hexdigest()[:16]
                    cands.append(Candidate(
                        path_id=path_id,
                        vars=vars_t,
                        lags=lags,
                        ops=ops,
                        root=i,
                        depth=1,
                        domain=0,
                        gen_reason=self.config.algorithm.value,
                    ))
                    if len(cands) >= self._proposer.max_arms:
                        return cands
        return cands

    def _arm_score(self, arm_id: int) -> float:
        """Priority score for an arm under the configured bandit policy."""
        stats = self.arms[arm_id]
        if self.config.algorithm == BanditAlgorithm.THOMPSON:
            return float(self._rng.beta(stats.alpha, stats.beta))
        if self.config.algorithm == BanditAlgorithm.UCB:
            if stats.observations == 0:
                return float('inf')
            mean = stats.cumulative_reward / stats.observations
            exploration = self.config.ucb_c * np.sqrt(
                np.log(self._step + 1) / stats.observations
            )
            return float(mean + exploration)
        # epsilon-greedy: random priority with prob epsilon, else exploit mean.
        if self._rng.random() < self.config.epsilon:
            return float(self._rng.random())
        return float(stats.cumulative_reward / max(1, stats.observations))

    def diversity_score(self, candidate: Candidate) -> float:
        """
        Structural novelty of a candidate in [0, 1].

        Blends frequency novelty (rarely proposed paths score higher) with
        variable-set novelty (paths whose variables differ from recently
        accepted ones score higher). Weight is ``diversity.novelty_weight``.
        """
        count = self._proposal_counts.get(candidate.path_id, 0)
        freq_novelty = 1.0 / (1.0 + count)

        var_set = frozenset(candidate.vars)
        if var_set and self._recent_varsets:
            max_overlap = 0.0
            for seen in self._recent_varsets:
                union = var_set | seen
                if not union:
                    continue
                jaccard = len(var_set & seen) / len(union)
                if jaccard > max_overlap:
                    max_overlap = jaccard
            set_novelty = 1.0 - max_overlap
        else:
            set_novelty = 1.0

        w = self._diversity.novelty_weight
        return float(np.clip(w * freq_novelty + (1.0 - w) * set_novelty, 0.0, 1.0))

    def apply_rewards(self, rewards: List[Reward]) -> None:
        """
        Feed shaped evaluation rewards back to the arms.

        Each ``Reward`` carries a ``path_id``; its arm's Beta stats are updated
        with the (rescaled to [0, 1]) reward value and explicit accepted flag.
        """
        for r in rewards:
            arm_id = self.register_path(r.path_id)
            v01 = float(np.clip((r.value + 1.0) / 2.0, 0.0, 1.0))
            self.update(arm_id, reward=v01, success=bool(r.accepted))

    def register_acceptances(self, candidates: List[Candidate]) -> None:
        """Record accepted candidates' variable-sets for diversity scoring."""
        for cand in candidates:
            self._recent_varsets.append(frozenset(cand.vars))

    def update_resource_profile(self, profile: Dict[str, Any]) -> None:
        """Refresh the resource profile context (bounded, side-effect free)."""
        if isinstance(profile, dict) and profile:
            self.drg = profile
    
    def update(self, arm_id: int, reward: float, success: bool = None) -> None:
        """
        Update arm statistics with observed reward.
        
        Args:
            arm_id: The arm that was pulled.
            reward: The observed reward (should be in [0, 1] for Beta updates).
            success: Optional explicit success/failure for Beta update.
                     If None, uses reward > 0.5 as threshold.
        """
        if arm_id not in self.arms:
            logger.warning(f"Unknown arm ID: {arm_id}")
            return
        
        stats = self.arms[arm_id]
        stats.observations += 1
        stats.cumulative_reward += reward
        stats.last_pulled = self._step
        
        # Update Beta parameters
        if success is None:
            success = reward > 0.5
        
        if success:
            stats.alpha += 1
        else:
            stats.beta += 1
        
        self._step += 1
        self._total_reward += reward
    
    def update_batch(self, arm_rewards: List[Tuple[int, float]]) -> None:
        """
        Batch update multiple arms.
        
        Args:
            arm_rewards: List of (arm_id, reward) tuples.
        """
        for arm_id, reward in arm_rewards:
            self.update(arm_id, reward)
    
    def decay(self) -> None:
        """
        Apply temporal decay to arm statistics.
        
        Useful for non-stationary environments where older
        observations should have less influence.
        """
        factor = self.config.decay_factor
        for stats in self.arms.values():
            stats.alpha = max(1.0, stats.alpha * factor)
            stats.beta = max(1.0, stats.beta * factor)
            stats.cumulative_reward *= factor
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the router.
        
        Returns:
            Dictionary with router metrics.
        """
        if not self.arms:
            return {"n_arms": 0, "total_observations": 0}
        
        total_obs = sum(s.observations for s in self.arms.values())
        means = [s.mean for s in self.arms.values() if s.observations > 0]
        
        return {
            "algorithm": self.config.algorithm.value,
            "n_arms": len(self.arms),
            "total_observations": total_obs,
            "total_reward": self._total_reward,
            "step": self._step,
            "mean_reward": self._total_reward / max(1, total_obs),
            "best_arm_mean": max(means) if means else 0.0,
            "explored_arms": sum(1 for s in self.arms.values() if s.observations > 0),
        }
    
    def get_top_arms(self, k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-k arms by mean reward.
        
        Args:
            k: Number of arms to return.
            
        Returns:
            List of (arm_id, mean_reward) tuples.
        """
        scored = [
            (arm_id, stats.mean)
            for arm_id, stats in self.arms.items()
            if stats.observations >= self.config.min_observations
        ]
        scored.sort(reverse=True, key=lambda x: x[1])
        return scored[:k]
    
    def reset(self) -> None:
        """Reset all arm statistics."""
        self.arms.clear()
        self._path_to_arm.clear()
        self._step = 0
        self._total_reward = 0.0
        self._arm_id_counter = 0

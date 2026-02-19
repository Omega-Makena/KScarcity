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

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

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
        
        # Get proposals
        arms = router.propose(n_proposals=10)
        
        # After evaluation, update with rewards
        for arm_id, reward in zip(arms, rewards):
            router.update(arm_id, reward)
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
        exclude: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Select arms to pull based on bandit policy.
        
        Args:
            n_proposals: Number of arms to select.
            context: Optional context for contextual bandits (future extension).
            exclude: Set of arm IDs to exclude from selection.
            
        Returns:
            List of selected arm IDs.
        """
        if not self.arms:
            # Auto-register arms if none exist
            self.register_arms(self.config.n_arms)
        
        exclude = exclude or set()
        available_arms = [aid for aid in self.arms.keys() if aid not in exclude]
        
        if not available_arms:
            return []
        
        n_proposals = min(n_proposals, len(available_arms))
        
        if self.config.algorithm == BanditAlgorithm.THOMPSON:
            return self._thompson_sampling(available_arms, n_proposals)
        elif self.config.algorithm == BanditAlgorithm.UCB:
            return self._ucb_selection(available_arms, n_proposals)
        else:
            return self._epsilon_greedy(available_arms, n_proposals)
    
    def _thompson_sampling(self, available: List[int], n: int) -> List[int]:
        """
        Thompson Sampling selection.
        
        Samples from Beta posterior for each arm and selects top-n.
        """
        samples = []
        for arm_id in available:
            stats = self.arms[arm_id]
            # Sample from Beta(alpha, beta) posterior
            sample = self._rng.beta(stats.alpha, stats.beta)
            samples.append((sample, arm_id))
        
        # Sort by sampled value descending
        samples.sort(reverse=True, key=lambda x: x[0])
        return [arm_id for _, arm_id in samples[:n]]
    
    def _ucb_selection(self, available: List[int], n: int) -> List[int]:
        """
        Upper Confidence Bound selection.
        
        Selects arms with highest UCB score.
        """
        scores = []
        for arm_id in available:
            stats = self.arms[arm_id]
            if stats.observations == 0:
                score = float('inf')
            else:
                mean = stats.cumulative_reward / stats.observations
                exploration = self.config.ucb_c * np.sqrt(
                    np.log(self._step + 1) / stats.observations
                )
                score = mean + exploration
            scores.append((score, arm_id))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [arm_id for _, arm_id in scores[:n]]
    
    def _epsilon_greedy(self, available: List[int], n: int) -> List[int]:
        """
        Epsilon-greedy selection.
        
        With probability epsilon, explore randomly.
        Otherwise, exploit best known arms.
        """
        selected = []
        
        for _ in range(n):
            remaining = [a for a in available if a not in selected]
            if not remaining:
                break
            
            if self._rng.random() < self.config.epsilon:
                # Explore: random selection
                arm_id = self._rng.choice(remaining)
            else:
                # Exploit: best mean reward
                best_arm = max(
                    remaining,
                    key=lambda a: (
                        self.arms[a].cumulative_reward / max(1, self.arms[a].observations)
                    )
                )
                arm_id = best_arm
            
            selected.append(arm_id)
        
        return selected
    
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

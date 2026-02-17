# bandit_router.py — Multi-Armed Bandit Path Selection

The `BanditRouter` implements the **exploration-exploitation tradeoff** for selecting which candidate paths to evaluate. Instead of testing all hypotheses equally, it allocates more budget to promising paths.

---

## The Problem

Given hundreds of potential paths (variable pairs × lags), we can't evaluate all of them every window. We need to:

1. **Explore**: Try new paths to discover surprises
2. **Exploit**: Focus on paths that have worked well
3. **Forget**: Reduce confidence in paths that haven't been tested recently

This is the classic **Multi-Armed Bandit** problem.

---

## Algorithm Options

### Thompson Sampling (Default)

Uses Bayesian inference with Beta distribution:

1. Each arm has (α, β) parameters (wins, losses)
2. Sample from Beta(α, β) for each arm
3. Select arms with highest samples
4. Update α, β based on observed rewards

**Why Thompson Sampling?**
- Probability-matching: Allocates exploration proportionally to uncertainty
- Natural exploration decay: Confident arms get exploited more
- No tuning parameters needed

### Upper Confidence Bound (UCB)

```
UCB(arm) = mean_reward + c * sqrt(ln(t) / n_arm)
```

- Adds "optimism bonus" for under-explored arms
- `c` controls exploration-exploitation balance

### Epsilon-Greedy

- With probability ε: random exploration
- With probability (1-ε): exploit best known

Simplest, but requires tuning ε.

---

## Core Classes

### `ArmStats` (Dataclass)

Statistics for a single bandit arm:

```python
@dataclass
class ArmStats:
    alpha: float = 1.0           # Beta parameter (wins + 1)
    beta: float = 1.0            # Beta parameter (losses + 1)
    observations: int = 0        # Total pulls
    cumulative_reward: float = 0 # Sum of rewards
    last_pulled: int = -1        # Window ID
```

Properties:
- `mean`: Expected value = α / (α + β)
- `variance`: Uncertainty measure
- `ucb_score`: UCB formula value

### `BanditConfig` (Dataclass)

Configuration:

```python
@dataclass
class BanditConfig:
    algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON
    n_arms: int = 1000
    epsilon: float = 0.1         # For epsilon-greedy
    ucb_c: float = 2.0           # UCB exploration constant
    decay_factor: float = 0.999  # Forgetting factor
    min_observations: int = 5    # Min pulls before exploitation
```

---

## Class: `BanditRouter`

### Initialization

```python
router = BanditRouter(
    config=BanditConfig(algorithm=BanditAlgorithm.THOMPSON),
    n_arms=500
)
```

Creates initial arm population with uniform priors.

### Arm Registration

#### `register_arms(n_arms) -> List[int]`

Batch registration:

```python
arm_ids = router.register_arms(100)
# Returns: [0, 1, 2, ..., 99]
```

#### `register_path(path_id) -> int`

Named path registration:

```python
arm_id = router.register_path("gdp->inflation@lag2")
# Returns: 42
# Remembers mapping: "gdp->inflation@lag2" → 42
```

---

## Selection Methods

### `propose(n_proposals, context=None, exclude=None) -> List[int]`

Main selection method:

```python
selected = router.propose(
    n_proposals=10,      # How many arms to pull
    exclude={5, 12, 23}  # Arms to skip (e.g., recently pulled)
)
# Returns: [7, 14, 2, 45, ...]
```

**Selection process** (Thompson Sampling):

```python
def _thompson_sampling(self, available, n):
    # Sample from Beta posterior for each arm
    samples = []
    for arm_id in available:
        stats = self.arms[arm_id]
        sample = self.rng.beta(stats.alpha, stats.beta)
        samples.append((sample, arm_id))
    
    # Select top-n by sample value
    samples.sort(reverse=True)
    return [arm_id for _, arm_id in samples[:n]]
```

---

## Reward Updates

### `update(arm_id, reward, success=None)`

Update arm after observing reward:

```python
router.update(
    arm_id=7,
    reward=0.85,     # Reward in [0, 1]
    success=True     # Optional explicit win/loss
)
```

**Beta update**:
- If success: α += 1
- If failure: β += 1
- Cumulative reward and observation count updated

**Success determination** (if not explicit):
```python
success = reward > 0.5  # Default threshold
```

### `update_batch(arm_rewards)`

Batch update:

```python
router.update_batch([
    (7, 0.85),
    (14, 0.23),
    (2, 0.91)
])
```

---

## Temporal Decay

### `decay()`

Apply forgetting to all arms:

```python
router.decay()
```

**What happens**:
```python
for arm in self.arms.values():
    # Shrink (α, β) toward prior (1, 1)
    arm.alpha = 1 + decay_factor * (arm.alpha - 1)
    arm.beta = 1 + decay_factor * (arm.beta - 1)
```

**Effect**: Over time, unused arms return to uncertainty, enabling re-exploration of old paths.

---

## Meta-Learning Integration

### `apply_meta_update(tau=None, gamma_diversity=None)`

Meta-learner can adjust bandit behavior:

- `tau`: Temperature for Thompson Sampling (higher = more exploration)
- `gamma_diversity`: Bonus for selecting diverse arms

```python
router.apply_meta_update(tau=1.5, gamma_diversity=0.1)
```

---

## Statistics

### `get_top_arms(k=10)`

Returns best-performing arms:

```python
top = router.get_top_arms(10)
# Returns: [(arm_id, mean_reward), ...]
```

### `get_stats() -> Dict`

Router statistics:

```python
{
    "n_arms": 500,
    "n_active": 342,         # Arms with observations > 0
    "avg_alpha": 2.3,
    "avg_beta": 1.8,
    "total_observations": 15000
}
```

---

## Contextual Extension (Future)

The `propose()` method accepts a `context` argument for **contextual bandits**:

```python
selected = router.propose(
    n_proposals=10,
    context={"volatility": 0.8, "trend": "up"}
)
```

This enables regime-aware exploration: different arms optimal in different contexts.

Currently a placeholder — full contextual bandit is a future enhancement.

---

## Usage Example

```python
from scarcity.engine import BanditRouter, BanditConfig, BanditAlgorithm

# Initialize with Thompson Sampling
router = BanditRouter(config=BanditConfig(
    algorithm=BanditAlgorithm.THOMPSON,
    n_arms=200
))

# Register paths for all variable pairs
for i, source in enumerate(variables):
    for target in variables:
        if source != target:
            router.register_path(f"{source}->{target}")

# Main loop
for window in data_windows:
    # Select arms to evaluate
    arms = router.propose(n_proposals=20)
    
    # Evaluate each selected arm
    rewards = []
    for arm_id in arms:
        result = evaluate_arm(arm_id, window)
        rewards.append((arm_id, result.reward))
    
    # Update bandit with rewards
    router.update_batch(rewards)
    
    # Periodic decay
    if window_id % 100 == 0:
        router.decay()
```

---

## When to Tune

| Problem | Solution |
|---------|----------|
| Too much exploration | Increase `min_observations`, decrease UCB `c` |
| Too little exploration | Increase `epsilon`, increase UCB `c` |
| Slow adaptation | Increase `decay_factor` (toward 1.0) |
| Fast adaptation needed | Decrease `decay_factor` |
| Memory pressure | Reduce `n_arms`, prune cold arms |

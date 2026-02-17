# evaluator.py — Path Scoring

The `Evaluator` is responsible for **scoring candidate paths** — measuring whether a proposed relationship between variables has genuine predictive value, or is just noise.

---

## Purpose

When the system proposes "does X→Y have predictive value?", the Evaluator answers:

- **Predictive Gain**: Does knowing X help predict Y better than just using Y's mean?
- **Statistical Confidence**: How certain are we? (Bootstrap confidence intervals)
- **Stability**: Is the effect consistent across different samples?

It also produces **shaped rewards** for the BanditRouter to learn which paths are worth exploring.

---

## Core Challenge: Online Scoring with Uncertainty

Unlike batch regression where you have all the data, the Evaluator works with **sliding windows** of limited size. It must:

1. Build valid regression problems from windowed time series
2. Estimate uncertainty via bootstrap resampling
3. Avoid false positives from random correlations
4. Adapt thresholds when too few candidates pass

---

## Architecture

```
Window Tensor + Candidates
         │
         ▼
┌─────────────────────────────────────────┐
│            Evaluator                     │
│                                          │
│  For each candidate:                     │
│    1. Build design matrix (X, y)         │
│    2. Bootstrap N resamples              │
│    3. Compute gain statistics            │
│    4. Check stability                    │
│    5. Accept/reject decision             │
│                                          │
└─────────────────────────────────────────┘
         │
         ▼
   List[EvalResult]
         │
         ▼
┌─────────────────────────────────────────┐
│         Reward Shaping                   │
│   (Normalize, weight, return rewards)    │
└─────────────────────────────────────────┘
         │
         ▼
   List[Reward] → BanditRouter
```

---

## Class: `Evaluator`

### Initialization

```python
def __init__(
    self,
    drg: Optional[Dict[str, Any]] = None,
    operators: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None
):
```

**DRG Configuration** (Dynamic Resource Governor):
- `n_resamples`: Number of bootstrap resamples (default: 10)
- `holdout_frac`: Fraction held out for validation (default: 0.2)
- `gain_min`: Minimum gain to accept (default: 0.02)
- `stability_min`: Minimum stability score (default: 0.6)
- `lambda_ci`: Confidence interval width multiplier

---

## Key Methods

### `score(window_tensor, candidates) -> List[EvalResult]`

Main entry point. Scores a batch of candidate paths.

```python
results = evaluator.score(window_tensor, candidates)
```

**Input**:
- `window_tensor`: NumPy array of shape `(time, variables)`
- `candidates`: List of `Candidate` objects (source, target, lags)

**Output**:
- List of `EvalResult` objects with scores and accept/reject flags

### `_score_single(window_tensor, candidate) -> EvalResult`

Scores a single candidate through the full pipeline:

#### Step 1: Build Design Matrix

```python
X, y = self._build_design_matrix(window_tensor, candidate)
```

Creates lagged feature matrix:
- For lag=2 on source variable: uses `source[t-1]`, `source[t-2]`
- Target: `target[t]`
- Aligns arrays by trimming edges

If construction fails (window too small, all-constant values), returns error result.

#### Step 2: Bootstrap Resampling

Runs `n_resamples` bootstrap iterations:

```python
for i in range(n_resamples):
    gain = self._bootstrap_gain(X, y, holdout_rows)
    gains.append(gain)
```

Each iteration:
1. Randomly splits data into train/holdout
2. Fits linear regression on train
3. Computes R² gain on holdout
4. Returns gain (or None if solve fails)

#### Step 3: Compute Statistics

From the bootstrap gains:
- `median_gain`: Median across resamples
- `p10`, `p90`: 10th and 90th percentiles
- `ci_lo`, `ci_hi`: Confidence interval bounds
- `stability`: Sign consistency across resamples

#### Step 4: Accept/Reject Decision

```python
accepted = (
    median_gain >= self.gain_min and
    stability >= self.stability_min and
    ci_lo > -self.gain_min  # CI doesn't include "no effect"
)
```

---

## Adaptive Threshold Relaxation

### `_maybe_relax_thresholds()`

If acceptance rate drops critically low (< 5% over 500 evaluations), the Evaluator automatically relaxes thresholds:

- `gain_min` reduced by 10%
- `stability_min` reduced by 10%
- `lambda_ci` widened by 10%

This prevents the system from starving (producing zero hypotheses).

Relaxation is logged and bounded — thresholds can't drop below safety floors.

---

## Stability Score

### `_compute_stability(current_gain, gains)`

Measures consistency:

1. **Sign consistency**: What fraction of bootstrap gains have the same sign as the median?
2. **Historical agreement**: Does the current gain agree with the previous window?

Final stability is weighted average of these signals.

High stability (>0.9) = Effect is consistent, trustworthy
Low stability (<0.5) = Effect flips sign, unreliable

---

## Reward Shaping

### `make_rewards(results, D_lookup, candidates) -> List[Reward]`

Converts raw evaluation results into shaped rewards for the BanditRouter.

**Shaping Logic**:

1. **Normalize gain**: Map gain to [0, 1] using sigmoid squashing
2. **Stability bonus**: Multiply by stability score
3. **Latency penalty**: Penalize slow evaluations
4. **Diversity bonus**: Boost rewards for underexplored paths

The shaped reward helps the bandit learn not just "what works" but "what works reliably and quickly."

```python
shaped_reward = (
    w_gain * normalized_gain +
    w_stability * stability +
    w_latency * (1 - latency_penalty)
)
```

Weights are configurable via DRG.

---

## Statistics and Telemetry

### `get_stats() -> Dict`

Returns aggregate evaluator statistics:

```python
{
    "total_evaluated": 15000,
    "acceptance_rate": 0.23,
    "current_gain_min": 0.018,
    "p50_gain": 0.045,
    "p90_gain": 0.12,
    "avg_ci_width": 0.03,
    "stability_floor": 0.58
}
```

Used for monitoring and meta-learning adjustments.

---

## Meta-Learning Integration

### `apply_meta_update(g_min, lambda_ci)`

Allows the meta-learner to dynamically adjust thresholds:

```python
evaluator.apply_meta_update(g_min=0.01, lambda_ci=0.8)
```

This enables the system to:
- Loosen standards during exploration phases
- Tighten standards once high-quality paths are found

---

## Edge Cases

### Insufficient Data

If window size < required lags:
- Returns `EvalResult(accepted=False, error="insufficient_data")`
- No crash, graceful degradation

### Collinearity

If feature matrix is singular:
- `_bootstrap_gain` returns None
- Candidate scored as unreliable

### All-Constant Features

If all values are the same:
- R² computation is undefined
- Candidate rejected with low gain

---

## Integration Points

- **`BanditRouter`**: Receives shaped rewards to update arm statistics
- **`MPIEOrchestrator`**: Calls `score()` in the main processing loop
- **`HypergraphStore`**: Accepted paths become stored edges
- **`operators.evaluation_ops`**: Provides `r2_gain()` computation

# relationships.py — Relationship Implementations

This file contains the **concrete implementations** for relationship types. Each class extends the base `Hypothesis` and implements specific statistical algorithms.

---

## Overview

The engine recognizes 15 relationship types, split across two files:
- **`relationships.py`**: 10 core types
- **`relationships_extended.py`**: 5 advanced types

Each implementation follows the same pattern:
1. Initialize with variable names and configuration
2. Maintain internal buffers/statistics
3. Implement `fit_step()`, `evaluate()`, and optionally `predict_value()`

---

## 1. CausalHypothesis — Granger Causality

**What it detects**: X Granger-causes Y if past values of X help predict Y beyond what past Y alone can predict.

**Algorithm**:
1. Maintain rolling buffers for X and Y
2. Fit two regressions:
   - Restricted: `Y ~ Y_lags` (autoregressive only)
   - Augmented: `Y ~ Y_lags + X_lags` (include source)
3. Compare R² scores — "gain" = R²_augmented - R²_restricted

**Key parameters**:
- `lag`: Number of lags (default: 2)
- `buffer_size`: Rolling window size (default: 100)

**Interpretation**:
- High gain (>0.05) + stable = Strong causal evidence
- Low gain = X doesn't help predict Y
- Note: Granger causality ≠ true causality (correlation in time)

**Prediction**: Uses learned coefficients to forecast target

```python
hyp = CausalHypothesis("interest_rate", "inflation", lag=2)
```

---

## 2. CorrelationalHypothesis — Pearson Correlation

**What it detects**: Linear correlation between two variables (contemporaneous, not lagged).

**Algorithm**:
1. Maintain running statistics: means, variances, covariance
2. Compute Pearson r using Welford's online algorithm
3. Confidence from evidence count

**Important**: Correlation ≠ causation. High correlation could mean:
- X causes Y
- Y causes X
- Both caused by hidden Z
- Spurious coincidence

**Prediction**: Returns `None` — correlation is not predictive

```python
hyp = CorrelationalHypothesis("gdp", "employment")
```

---

## 3. TemporalHypothesis — Autoregressive (AR)

**What it detects**: Self-predictive patterns — Y depends on its own past.

**Algorithm**:
1. Recursive Least Squares (RLS) for online AR fitting
2. Predict Y_t from [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
3. Track prediction error to compute fit score

**Common patterns detected**:
- Momentum (positive AR coefficients)
- Mean reversion (negative AR coefficients)
- Seasonality (periodic patterns in residuals)

**Prediction**: Extrapolates using learned AR weights

```python
hyp = TemporalHypothesis("stock_price", lag=5)
```

---

## 4. FunctionalHypothesis — Polynomial Regression

**What it detects**: Y = f(X) functional relationships.

**Algorithm**:
1. Fit polynomial regression (default: linear)
2. RLS online update for coefficients
3. Track residuals for fit score

**Extension**: Higher-order polynomials for nonlinear relationships

**Prediction**: Evaluates polynomial at current X value

```python
hyp = FunctionalHypothesis("temperature", "energy_consumption", degree=2)
```

---

## 5. ProbabilisticHypothesis — Mutual Information

**What it detects**: Statistical dependence (captures nonlinear relationships).

**Algorithm**:
1. Discretize continuous values into bins
2. Estimate joint probability P(X, Y)
3. Compute I(X; Y) = Σ P(x,y) log[P(x,y) / (P(x)P(y))]

**Advantages over correlation**:
- Detects nonlinear dependence
- Symmetric (no direction assumption)

**Limitation**: Requires discretization, less precise with small samples

```python
hyp = ProbabilisticHypothesis("input", "output", n_bins=10)
```

---

## 6. CompositionalHypothesis — Multi-Input Regression

**What it detects**: Multiple sources jointly explain target.

**Algorithm**:
1. Fit multivariate regression: Y ~ X1 + X2 + ... + Xn
2. RLS update for coefficient vector
3. Track relative importance of each source

**Use case**: Identify factor models (what combination of indicators predicts GDP?)

```python
hyp = CompositionalHypothesis(
    sources=["inflation", "unemployment", "interest_rate"],
    target="gdp_growth"
)
```

---

## 7. CompetitiveHypothesis — Competing Influences

**What it detects**: When two sources compete for influence on target.

**Algorithm**:
1. Compare explanatory power of X1 vs X2 for Y
2. Measure "crowding out" — does including one reduce the other's importance?
3. Track which source dominates over time

**Use case**: Does monetary policy or fiscal policy have more effect on inflation?

```python
hyp = CompetitiveHypothesis("x1", "x2", target="y")
```

---

## 8. SynergisticHypothesis — Interaction Effects

**What it detects**: X1 and X2 have synergistic (or antagonistic) effects on Y.

**Algorithm**:
1. Fit regression with interaction term: Y ~ X1 + X2 + X1*X2
2. Measure significance of the interaction coefficient
3. Positive interaction = synergy, negative = antagonism

**Use case**: Does high inflation *combined with* low growth (stagflation) have worse effects than either alone?

```python
hyp = SynergisticHypothesis("inflation", "gdp_growth", target="unemployment")
```

---

## 9. EquilibriumHypothesis — Long-Run Balance

**What it detects**: Variables that return to equilibrium relationship over time.

**Algorithm**:
1. Estimate long-run relationship (cointegration-like)
2. Track deviations from equilibrium
3. Measure mean-reversion speed

**Economic interpretation**: Purchasing power parity, interest rate parity, etc.

```python
hyp = EquilibriumHypothesis("exchange_rate", "price_ratio")
```

---

## 10. StructuralHypothesis — Graph Patterns

**What it detects**: Global properties of the relationship graph.

**Algorithm**:
1. Analyze network of discovered relationships
2. Detect: cycles, hubs, isolated nodes, hierarchies
3. Flag structural anomalies

**Not about individual pairs** — this hypothesis looks at the entire graph topology.

```python
hyp = StructuralHypothesis(all_variables=["A", "B", "C", "D"])
```

---

## Configuration System

Each hypothesis type has an optional `*Config` dataclass:

```python
@dataclass
class CausalConfig:
    min_buffer: int = 20
    granger_threshold: float = 0.02
    confidence_growth_rate: float = 0.01
```

Pass to constructor:
```python
config = CausalConfig(min_buffer=50)
hyp = CausalHypothesis("X", "Y", config=config)
```

---

## Common Patterns

### Metric Updating

All hypotheses track these metrics:
- `fit_score`: Current alignment (0-1)
- `confidence`: Bayesian belief (starts 0.5, grows with confirming evidence)
- `stability`: Consistency over recent history
- `evidence`: Observation count

### Stability Tracking

Uses `OnlineMAD` (Median Absolute Deviation) for robust stability:
```python
self.stability_tracker = OnlineMAD()
self.stability_tracker.update(current_fit_score)
# High MAD → low stability
```

### Effect Size

Causal and functional hypotheses track learned coefficients:
```python
self.effect_size = self.coefficients[-1]  # Coefficient on source variable
```

---

## Choosing the Right Hypothesis

| Question | Appropriate Type |
|----------|------------------|
| Does X cause Y? | CausalHypothesis |
| Do X and Y move together? | CorrelationalHypothesis |
| Does Y depend on its past? | TemporalHypothesis |
| Is Y a function of X? | FunctionalHypothesis |
| Are X, Y dependent (nonlinear)? | ProbabilisticHypothesis |
| Do multiple inputs explain Y? | CompositionalHypothesis |
| Do X and Y compete for influence? | CompetitiveHypothesis |
| Do X and Y interact? | SynergisticHypothesis |
| Is there long-run equilibrium? | EquilibriumHypothesis |
| What's the global structure? | StructuralHypothesis |

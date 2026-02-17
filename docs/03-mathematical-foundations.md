# Mathematical Foundations

## Overview

SCARCITY is built on rigorous mathematical foundations from causal inference, online learning, resource optimization, and federated learning.

## 1. Causal Discovery

### 1.1 Structural Causal Models (SCM)

A Structural Causal Model is defined as:
```
M = (U, V, F, P(U))
```

Where:
- `U`: Exogenous (unobserved) variables
- `V`: Endogenous (observed) variables  
- `F`: Set of functions `v_i = f_i(pa_i, u_i)`
- `P(U)`: Probability distribution over U

### 1.2 Causal Graphs

A causal graph `G = (V, E)` represents causal relationships:
- Nodes `V`: Variables
- Directed edges `E`: Causal relationships
- `X → Y`: X causes Y

**D-separation**: Variables X and Y are d-separated by Z if all paths between X and Y are blocked by Z.

### 1.3 Conditional Independence Testing

SCARCITY uses conditional independence (CI) tests to discover causal structure:

```
X ⊥ Y | Z  ⟺  P(X, Y | Z) = P(X | Z) · P(Y | Z)
```

**Implementation**: Bootstrap-based permutation test
- Null hypothesis: X ⊥ Y | Z
- Test statistic: Mutual information I(X; Y | Z)
- P-value: Proportion of permutations with larger test statistic

### 1.4 PC Algorithm (Constraint-Based)

The PC algorithm discovers causal structure through CI tests:

```
1. Start with complete undirected graph
2. For each pair (X, Y):
   - Test X ⊥ Y | Z for all subsets Z
   - Remove edge if independent
3. Orient edges using v-structures
4. Apply orientation rules
```

**Complexity**: O(n^k) where k is maximum conditioning set size

### 1.5 Multi-Path Inference

SCARCITY extends PC with multi-path inference:

```
For each candidate path P = (X₁ → X₂ → ... → Xₙ):
  1. Compute path strength: ∏ᵢ w(Xᵢ → Xᵢ₊₁)
  2. Compute path stability: min_i stability(Xᵢ → Xᵢ₊₁)
  3. Rank paths by combined score
```

**Path Score**:
```
score(P) = α · strength(P) + β · stability(P) + γ · novelty(P)
```

## 2. Online Learning

### 2.1 Streaming Data Model

Data arrives as a stream of windows:
```
W₁, W₂, W₃, ..., Wₜ
```

Each window `Wₜ ∈ ℝ^(n×d)`:
- n: Number of samples
- d: Feature dimension

### 2.2 Sliding Window

SCARCITY uses sliding windows for temporal adaptation:

```
Window size: w
Stride: s
Window t: Wₜ = {xₜ₋ᵥ₊₁, ..., xₜ}
```

**Overlap**: `overlap = (w - s) / w`

### 2.3 Incremental Updates

Hypergraph updates incrementally:

```
Gₜ = update(Gₜ₋₁, Wₜ)
```

**Edge Weight Update**:
```
wₜ(X → Y) = (1 - α) · wₜ₋₁(X → Y) + α · evidence(X → Y, Wₜ)
```

Where α is the learning rate.

### 2.4 Concept Drift Detection

Detect distribution shifts using:

```
KL(Pₜ || Pₜ₋₁) > threshold
```

Where KL is Kullback-Leibler divergence.

## 3. Resource Optimization

### 3.1 Resource Model

System resources modeled as:
```
R = (CPU, Memory, GPU, VRAM)
```

Each resource has:
- Current utilization: uₜ ∈ [0, 1]
- Capacity: cₘₐₓ
- Threshold: θ

### 3.2 Control Theory

DRG uses PID control for resource management:

```
u(t) = Kₚ·e(t) + Kᵢ·∫e(τ)dτ + Kₐ·de(t)/dt
```

Where:
- e(t) = θ - uₜ (error)
- Kₚ: Proportional gain
- Kᵢ: Integral gain  
- Kₐ: Derivative gain

### 3.3 Adaptive Throttling

Adjust processing rate based on resources:

```
rateₜ = rateₘₐₓ · (1 - uₜ/θ)^β
```

Where β controls aggressiveness.

### 3.4 Predictive Forecasting

Forecast future resource usage:

```
ûₜ₊ₖ = f(uₜ, uₜ₋₁, ..., uₜ₋ₙ)
```

Using exponential smoothing:
```
ûₜ₊₁ = α·uₜ + (1-α)·ûₜ
```


## 4. Federated Learning

### 4.1 Federated Averaging (FedAvg)

Aggregate models from K domains:

```
w_global = Σₖ (nₖ/n) · wₖ
```

Where:
- wₖ: Model weights from domain k
- nₖ: Number of samples in domain k
- n = Σₖ nₖ: Total samples

### 4.2 Weighted Aggregation

Weight by domain performance:

```
w_global = Σₖ αₖ · wₖ
```

Where:
```
αₖ = (1/lossₖ) / Σⱼ(1/lossⱼ)
```

Better performing domains get higher weight.

### 4.3 Differential Privacy

Add Gaussian noise for privacy:

```
w̃ₖ = wₖ + N(0, σ²I)
```

Where:
```
σ = √(2·ln(1.25/δ)) · Δf / ε
```

- ε: Privacy budget
- δ: Privacy parameter
- Δf: Sensitivity

**Privacy Guarantee**: (ε, δ)-differential privacy

### 4.4 Secure Aggregation

Aggregate without revealing individual models:

```
1. Each domain k generates random mask rₖ
2. Share masked model: w̃ₖ = wₖ + rₖ
3. Aggregate: w̃ = Σₖ w̃ₖ = Σₖ wₖ + Σₖ rₖ
4. If Σₖ rₖ = 0, then w̃ = Σₖ wₖ
```

## 5. Meta-Learning

### 5.1 Model-Agnostic Meta-Learning (MAML)

Learn initialization that adapts quickly:

```
θ* = argmin_θ Σₜ L_τ(θ - α∇L_τ(θ))
```

Where:
- τ: Task
- L_τ: Loss on task τ
- α: Inner learning rate

### 5.2 Cross-Domain Transfer

Transfer knowledge between domains:

```
θ_target = θ_source + Δθ
```

Where Δθ is learned transformation.

### 5.3 Domain Similarity

Measure domain similarity using:

```
sim(D₁, D₂) = exp(-KL(P₁ || P₂))
```

Or Maximum Mean Discrepancy (MMD):

```
MMD(D₁, D₂) = ||μ(D₁) - μ(D₂)||_H
```

Where μ(D) is mean embedding in RKHS H.

### 5.4 Meta-Prior

Maintain prior distribution over model parameters:

```
p(θ) = N(μ_meta, Σ_meta)
```

Updated after each domain:

```
μ_meta ← (1-β)·μ_meta + β·θ_domain
Σ_meta ← (1-β)·Σ_meta + β·(θ_domain - μ_meta)²
```

## 6. Statistical Validation

### 6.1 Bootstrap Resampling

Estimate confidence intervals:

```
1. Resample data B times with replacement
2. Compute statistic on each resample
3. CI = [percentile(2.5), percentile(97.5)]
```

### 6.2 Permutation Testing

Test null hypothesis:

```
1. Compute test statistic T on original data
2. Permute labels B times
3. Compute T* on each permutation
4. p-value = (1 + Σ(T* ≥ T)) / (B + 1)
```

### 6.3 Multiple Testing Correction

Control false discovery rate using Benjamini-Hochberg:

```
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find largest k where pₖ ≤ (k/m)·α
3. Reject H₁, ..., Hₖ
```

### 6.4 Effect Size

Measure practical significance:

```
Cohen's d = (μ₁ - μ₂) / σ_pooled
```

Where:
```
σ_pooled = √((σ₁² + σ₂²) / 2)
```

## 7. Optimization

### 7.1 Stochastic Gradient Descent

Update parameters:

```
θₜ₊₁ = θₜ - η·∇L(θₜ; xₜ)
```

With momentum:
```
vₜ₊₁ = β·vₜ + ∇L(θₜ; xₜ)
θₜ₊₁ = θₜ - η·vₜ₊₁
```

### 7.2 Adam Optimizer

Adaptive learning rates:

```
mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ
vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²
m̂ₜ = mₜ/(1-β₁ᵗ)
v̂ₜ = vₜ/(1-β₂ᵗ)
θₜ = θₜ₋₁ - η·m̂ₜ/(√v̂ₜ + ε)
```

### 7.3 Learning Rate Scheduling

Decay learning rate over time:

```
ηₜ = η₀ / (1 + decay·t)
```

Or cosine annealing:
```
ηₜ = η_min + (η_max - η_min)·(1 + cos(πt/T))/2
```

## 8. Information Theory

### 8.1 Mutual Information

Measure dependence between X and Y:

```
I(X; Y) = Σₓ Σᵧ p(x,y)·log(p(x,y)/(p(x)·p(y)))
```

### 8.2 Conditional Mutual Information

```
I(X; Y | Z) = Σₓ Σᵧ Σᵤ p(x,y,z)·log(p(x,y|z)/(p(x|z)·p(y|z)))
```

### 8.3 Transfer Entropy

Measure directed information flow:

```
TE(X→Y) = I(Yₜ; Xₜ₋₁ | Yₜ₋₁)
```

### 8.4 Entropy

Measure uncertainty:

```
H(X) = -Σₓ p(x)·log p(x)
```

## 9. Graph Theory

### 9.1 Hypergraph

A hypergraph H = (V, E) where:
- V: Set of nodes
- E: Set of hyperedges (subsets of V)

### 9.2 Graph Metrics

**Degree Centrality**:
```
C_D(v) = deg(v) / (n-1)
```

**Betweenness Centrality**:
```
C_B(v) = Σₛ≠ᵥ≠ₜ (σₛₜ(v) / σₛₜ)
```

**PageRank**:
```
PR(v) = (1-d)/n + d·Σᵤ∈In(v) PR(u)/deg(u)
```

### 9.3 Graph Laplacian

```
L = D - A
```

Where:
- D: Degree matrix
- A: Adjacency matrix

Eigenvalues of L reveal graph structure.

## 10. Probability Theory

### 10.1 Bayes' Theorem

```
P(A|B) = P(B|A)·P(A) / P(B)
```

### 10.2 Law of Total Probability

```
P(A) = Σᵢ P(A|Bᵢ)·P(Bᵢ)
```

### 10.3 Markov Property

```
P(Xₜ₊₁ | X₁, ..., Xₜ) = P(Xₜ₊₁ | Xₜ)
```

### 10.4 Central Limit Theorem

For i.i.d. random variables:

```
(X̄ₙ - μ) / (σ/√n) → N(0, 1)
```

As n → ∞.

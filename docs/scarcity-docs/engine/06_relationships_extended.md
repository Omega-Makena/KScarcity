# relationships_extended.py — Advanced Relationship Types

Extends the core relationship types with more sophisticated patterns: mediating chains, moderating effects, graph patterns, similarity, and logical constraints.

---

## 11. MediatingHypothesis — Indirect Effects

**What it detects**: X affects Y *through* an intermediary M (mediation chain).

**Pattern**: X → M → Y

**Algorithm**:
1. Estimate direct path: X → Y
2. Estimate two-stage path: X → M, M → Y
3. Compare total effect via M vs direct effect
4. Mediation strength = how much of X's effect goes through M

**Example**: Does education → income happen through skill acquisition?
- Direct: education → income
- Mediated: education → skills → income

**Psychology research** uses this heavily (Baron & Kenny method, Sobel test).

```python
hyp = MediatingHypothesis(
    source="education",
    mediator="skills",
    target="income"
)
```

**Key metrics**:
- `direct_effect`: X→Y ignoring M
- `indirect_effect`: X→M→Y path strength
- `mediation_ratio`: indirect / (direct + indirect)

---

## 12. ModeratingHypothesis — Context-Dependent Effects

**What it detects**: The effect of X on Y changes depending on moderator M.

**Pattern**: X → Y, but the arrow's strength depends on M

**Algorithm**:
1. Split data by moderator levels (e.g., high M vs low M)
2. Estimate X→Y effect in each stratum
3. Compare effect sizes across strata
4. Large difference = strong moderation

**Example**: Does advertising → sales differ by region?
- High-income regions: strong effect
- Low-income regions: weak effect
- Region "moderates" the relationship

```python
hyp = ModeratingHypothesis(
    source="advertising",
    target="sales",
    moderator="region_income"
)
```

**Key metrics**:
- `high_regime_effect`: Effect when M is high
- `low_regime_effect`: Effect when M is low
- `moderation_strength`: Difference between regimes

---

## 13. GraphHypothesis — Network Patterns

**What it detects**: Structural patterns in the relationship network.

**Patterns detected**:
- **Cycles**: A → B → C → A
- **Hubs**: One variable influences many
- **Clusters**: Groups with dense internal connections
- **Bridges**: Variables connecting otherwise separate clusters
- **Hierarchies**: Layered influence structure

**Algorithm**:
1. Build adjacency matrix from confirmed relationships
2. Apply graph algorithms (DFS for cycles, degree centrality, etc.)
3. Flag anomalies and interesting structures

**Not about single pairs** — this analyzes the full network.

```python
hyp = GraphHypothesis(all_variables=variables_list)
```

**Key metrics**:
- `n_cycles`: Number of feedback loops
- `hub_score`: Maximum centrality in network
- `clustering_coefficient`: Network density

---

## 14. SimilarityHypothesis — Behavioral Similarity

**What it detects**: Two variables that behave alike (move similarly over time).

**Different from correlation**: Similarity can capture:
- Lagged similarity (X behaves like Y did yesterday)
- Nonlinear similarity (same volatility patterns)
- Profile similarity (similar response to shocks)

**Algorithm**:
1. Compute distance metric (DTW, cosine, Euclidean) over rolling window
2. Track similarity over time
3. Cluster similar variables

**Use case**: Find "peer groups" of similar assets, similar economic indicators.

```python
hyp = SimilarityHypothesis("stock_A", "stock_B", metric="dtw")
```

**Key metrics**:
- `similarity_score`: 0=different, 1=identical
- `stability`: How consistent is the similarity over time

---

## 15. LogicalHypothesis — Boolean Constraints

**What it detects**: Logical/rule-based relationships between discretized variables.

**Pattern types**:
- **Implication**: If X_high then Y_high
- **Exclusion**: Not(X_high and Y_high)
- **Equivalence**: X_high ↔ Y_high
- **Coverage**: X_high or Y_high always true

**Algorithm**:
1. Discretize variables into states (e.g., high/low/normal)
2. Build contingency tables
3. Check logical rules via conditional probabilities
4. Rules with high support and confidence survive

**Example**: "When inflation > 3%, interest rates never stay below 2%"

```python
hyp = LogicalHypothesis(
    variables=["inflation", "interest_rate"],
    rule_type="implication"
)
```

**Key metrics**:
- `support`: How often the rule's antecedent occurs
- `confidence`: P(consequent | antecedent)
- `lift`: Improvement over random chance

---

## When to Use Extended Types

| Scenario | Appropriate Type |
|----------|------------------|
| Indirect influence chains | MediatingHypothesis |
| Context-dependent effects | ModeratingHypothesis |
| Network structure analysis | GraphHypothesis |
| Find similar variables | SimilarityHypothesis |
| Business rules / constraints | LogicalHypothesis |

---

## Implementation Notes

### Buffer Management

Like core types, extended hypotheses maintain rolling buffers:

```python
def fit_step(self, row):
    self.x_buffer.append(row.get(self.source))
    self.m_buffer.append(row.get(self.mediator))
    self.y_buffer.append(row.get(self.target))
    
    # Trim to max size
    if len(self.x_buffer) > self.buffer_size:
        self.x_buffer.pop(0)
        # etc.
```

### Evaluation Requirements

Extended types often need more evidence before producing meaningful results:

| Type | Minimum Evidence |
|------|------------------|
| Mediating | 50+ observations |
| Moderating | 100+ (enough in each stratum) |
| Graph | Enough confirmed edges |
| Similarity | 30+ consecutive observations |
| Logical | 100+ for rule confidence |

### Computational Cost

Extended types are generally more expensive:

| Type | Cost | Reason |
|------|------|--------|
| Mediating | O(n) | Two regression fits |
| Moderating | O(n) | Stratified analysis |
| Graph | O(edges) | Graph traversal |
| Similarity | O(n²) | Distance computation |
| Logical | O(states²) | Contingency tables |

The system typically creates fewer extended hypotheses than core types.

---

## Integration with Core Types

Extended types **do not replace** core types — they complement them.

Typical discovery flow:
1. Core types (Causal, Correlational) identify basic relationships
2. Extended types probe deeper patterns in confirmed relationships

**Example**:
- CausalHypothesis confirms: "interest_rate → inflation"
- ModeratingHypothesis asks: "Does this effect vary by GDP growth level?"
- MediatingHypothesis asks: "Does this effect go through bank lending?"

---

## Creating Instances

Extended hypotheses are created during `initialize_v2()`:

```python
# In OnlineDiscoveryEngine.initialize_v2():

# After creating causal hypotheses...
for source in variables:
    for mediator in variables:
        for target in variables:
            if len({source, mediator, target}) == 3:
                hyp = MediatingHypothesis(source, mediator, target)
                pool.add(hyp)
```

This creates O(n³) mediating hypotheses — use sparingly for large schemas.

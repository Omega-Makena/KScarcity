# arbitration.py — Hypothesis Arbitration

The `HypothesisArbiter` resolves **conflicts between hypotheses** to produce a parsimonious knowledge graph without redundancy or contradiction.

---

## The Problem

After hypotheses pass the MetaController's lifecycle checks, you might have:

- **Redundancy**: Both "A correlates with B" and "A causes B" are ACTIVE
- **Contradiction**: Both "A → B" and "B → A" are ACTIVE (feedback loops are rare)
- **Multiple claims**: Three hypotheses all claim to explain the same relationship

The Arbiter decides which hypotheses survive.

---

## Hierarchy

The Arbiter enforces a **strength hierarchy**:

```
CAUSAL > TEMPORAL > CORRELATIONAL > PROBABILISTIC > FUNCTIONAL
```

**Interpretation**: If we have both:
- "A causes B" (Causal, confidence 0.7)
- "A correlates with B" (Correlational, confidence 0.9)

The Causal hypothesis wins, even with lower confidence. **Causation subsumes correlation.**

---

## Class: `HypothesisArbiter`

### Initialization

```python
arbiter = HypothesisArbiter()
```

Sets up internal mappings:
- `TYPE_HIERARCHY`: Ranking of relationship types
- `CONFLICT_MATRIX`: Which types conflict with each other

---

## Arbitration

### `arbitrate(hypotheses: List[Hypothesis]) -> List[Hypothesis]`

Filters redundant hypotheses:

```python
survivors = arbiter.arbitrate(active_hypotheses)
```

**Algorithm**:

1. Group hypotheses by variable pair (ignoring direction)
2. For each pair:
   - Sort by hierarchy level (Causal first)
   - Within same level, sort by confidence
   - Keep only the best
3. Return filtered list

```python
def arbitrate(self, hypotheses):
    pair_groups = defaultdict(list)
    
    # Group by variable pair
    for h in hypotheses:
        pair = frozenset(h.variables)
        pair_groups[pair].append(h)
    
    survivors = []
    for pair, group in pair_groups.items():
        # Sort by hierarchy, then confidence
        group.sort(key=lambda h: (-get_strength(h), -h.confidence))
        survivors.append(group[0])  # Keep best
    
    return survivors
```

---

## Conflict Detection

### `detect_conflicts(hypotheses: List[Hypothesis]) -> List[Dict]`

Identifies surviving contradictions:

```python
conflicts = arbiter.detect_conflicts(hypotheses)
```

**Types of conflicts detected**:

### 1. Bidirectional Causality

Both "A → B" and "B → A" are ACTIVE:

```python
{
    "type": "bidirectional",
    "pair": ("A", "B"),
    "hypothesis_ids": ["h1", "h2"],
    "note": "Possible feedback loop or spurious"
}
```

**Interpretation**: True feedback loops exist (interest rates ↔ inflation), but they're rare. Often indicates one direction is spurious.

### 2. Type Conflicts

Same pair has incompatible types:

```python
{
    "type": "type_conflict",
    "pair": ("X", "Y"),
    "types": ["causal", "independence"],
    "hypothesis_ids": ["h3", "h4"]
}
```

**Interpretation**: Can't be both causal AND independent.

### 3. Effect Sign Disagreement

Same pair, same direction, different signs:

```python
{
    "type": "sign_conflict",
    "pair": ("A", "B"),
    "effects": [0.35, -0.28],
    "hypothesis_ids": ["h5", "h6"]
}
```

**Interpretation**: One says positive effect, other says negative. Likely regime-dependent.

---

## Arbitration Triggers

Called periodically by `OnlineDiscoveryEngine._arbitrate_step()`:

```python
def _arbitrate_step(self):
    active = [h for h in self.pool.population.values() 
              if h.meta.state == HypothesisState.ACTIVE]
    
    # Run arbitration
    survivors = self.arbiter.arbitrate(active)
    
    # Kill non-survivors
    survivor_ids = {h.meta.id for h in survivors}
    for h in active:
        if h.meta.id not in survivor_ids:
            self.pool._kill(h.meta.id)
```

Runs every ~50 ticks by default.

---

## Edge Cases

### Empty Input

```python
arbiter.arbitrate([])  # Returns []
```

### Single Hypothesis Per Pair

If only one hypothesis per pair exists, no arbitration needed — just returns input.

### All Same Type

If all hypotheses are same type (e.g., all Correlational), arbitration is purely by confidence.

---

## Customizing Hierarchy

To change the hierarchy (e.g., prefer Functional over Causal):

```python
arbiter.TYPE_HIERARCHY = {
    RelationshipType.FUNCTIONAL: 1,
    RelationshipType.CAUSAL: 2,
    # ...
}
```

Lower number = higher priority.

---

## Integration Points

- **`OnlineDiscoveryEngine`**: Calls `arbitrate()` periodically
- **`HypothesisPool`**: Non-survivors are killed
- **Export**: Only survivors appear in knowledge graph

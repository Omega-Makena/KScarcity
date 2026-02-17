# grouping.py — Adaptive Variable Grouping

The `AdaptiveGrouper` manages **dynamic hierarchical clustering** of variables, allowing the system to reason at different granularity levels.

---

## Purpose

Sometimes you don't want to reason about individual variables, but about **groups**:

- "Macro indicators" as a group → Asset A
- "Tech sector" → "Market index"

The Grouper:
1. Initializes variables into groups (atomic by default)
2. Monitors prediction errors for each group
3. **Splits** groups that can't be predicted well (incoherent)
4. **Merges** groups that interact strongly (future feature)

---

## Core Concepts

### Coarse-to-Fine

Start coarse (big groups), refine to fine (individual variables) as needed:

```
Initial:  [A, B, C, D]  →  One big group
          
Split:    [A, B] [C, D]  →  Two groups
          
Split:    [A] [B] [C, D]  →  Fine-grained where needed
```

### Residual Pressure

A group's "residual" is how much prediction error it has:

- Low residual → Group is coherent, good predictions
- High residual → Group is incoherent, should split

---

## Data Structures

### `VariableGroup` (Dataclass)

```python
@dataclass
class VariableGroup:
    id: str                   # UUID
    variables: Set[str]       # Member variables
    variance_stats: OnlineMAD # Track internal variance
    residual_stats: OnlineMAD # Track prediction errors
```

### `OnlineMAD`

Tracks **Median Absolute Deviation** in streaming fashion. More robust than variance for detecting outliers.

---

## Class: `AdaptiveGrouper`

### Initialization

```python
grouper = AdaptiveGrouper(
    split_threshold=1.0,     # Residual threshold for splitting
    minimize_groups=False    # If True, prefer merging
)
```

### `initialize(variable_names: List[str])`

Sets up initial groups:

```python
grouper.initialize(["gdp", "inflation", "unemployment", "rates"])
```

**Default behavior**: Creates **atomic groups** (one variable per group):
- Group_1: {gdp}
- Group_2: {inflation}
- Group_3: {unemployment}
- Group_4: {rates}

This is optimal for "Online Relationship Discovery" where we want maximum resolution.

### `monitor(row, hypothesis_errors)`

Checks for split/merge triggers:

```python
grouper.monitor(
    row=current_data,
    hypothesis_errors={"group_1_id": 0.8, "group_2_id": 0.1, ...}
)
```

**Logic**:

```python
for group_id, error in hypothesis_errors.items():
    group = self.groups[group_id]
    group.add_residual(error)
    
    # Check split condition
    if group.residual_stats.median > self.split_threshold:
        self._split_group(group_id)
```

### `_split_group(group_id)`

Shatters a group into atomic variables:

```python
# Before: Group "macro" = {gdp, inflation, unemployment}
# After:  Group "gdp_1" = {gdp}
#         Group "inflation_1" = {inflation}
#         Group "unemployment_1" = {unemployment}
```

**Logged** for debugging:
```
INFO: Splitting group abc123 (vars={'gdp', 'inflation'}) due to high residual
```

### `get_group_id(variable: str) -> Optional[str]`

Lookup which group a variable belongs to:

```python
group_id = grouper.get_group_id("inflation")
```

---

## Integration with Discovery

The Grouper doesn't currently drive hypothesis creation directly (we use atomic groups by default). Its future use cases:

1. **Macro-to-micro discovery**: First find "Macro → Asset", then refine to "Inflation → Asset"
2. **Dimensionality reduction**: With 1000 variables, group similar ones
3. **Hierarchical hypotheses**: "Group X → Group Y" as first-class hypotheses

---

## Statistics Tracking

### `VariableGroup.update_stats(row)`

Tracks internal variance of group members:

```python
# Average value of group members
vals = [row[v] for v in group.variables if v in row]
mean_val = np.mean(vals)
group.variance_stats.update(mean_val)
```

**High internal variance** = Diverse members, maybe shouldn't be grouped

### `VariableGroup.add_residual(error)`

Tracks unexplained variance:

```python
group.residual_stats.update(error)
```

**Rising median residual** = Group predictions getting worse = Split candidate

---

## Merge Logic (Placeholder)

Merging is the inverse of splitting:

```python
# Future: If two groups always co-move and interact
# merge them into one larger group
```

Currently not implemented — splitting is more commonly needed.

---

## Usage Example

```python
from scarcity.engine.grouping import AdaptiveGrouper

# Initialize with atomic groups
grouper = AdaptiveGrouper(split_threshold=1.5)
grouper.initialize(["A", "B", "C", "D"])

# Simulate processing
for step in range(100):
    row = get_data_row(step)
    errors = get_hypothesis_errors(row)  # {group_id: error}
    grouper.monitor(row, errors)

# Check final grouping
for gid, group in grouper.groups.items():
    print(f"Group {gid}: {group.variables}")
```

---

## When Grouping Matters

| Scenario | Grouping Benefit |
|----------|------------------|
| 1000+ variables | Reduces hypothesis count from O(n²) to O(g²) |
| Known factor structure | Group known factors together |
| Hierarchical domains | Macro → Sector → Company |
| Streaming concept drift | Groups can split as relationships change |

For typical use (10-50 variables), atomic grouping is fine and grouping adds little value.

# store.py — HypergraphStore

The `HypergraphStore` is the **long-term memory** of the relationship discovery system. It persists discovered edges, applies temporal decay, and provides efficient neighbor lookups.

---

## Purpose

When a hypothesis is confirmed (high confidence, stable), it becomes an **edge** in the knowledge graph. The HypergraphStore:

- Stores edges with effect sizes, confidence intervals, and stability scores
- Applies temporal decay so old relationships fade unless re-confirmed
- Manages memory with bounded capacity and LRU pruning
- Supports hyperedges (multi-input relationships) and regime tracking

---

## Core Data Structures

### `EdgeRec` (Dataclass)

Represents a single causal edge:

```python
@dataclass
class EdgeRec:
    weight: float      # EMA of effect size
    var: float         # Running variance
    stability: float   # Consistency metric
    ci_lo: float       # Confidence interval lower bound
    ci_hi: float       # Confidence interval upper bound
    regime_id: int     # Which regime (for regime-switching models)
    last_seen: int     # Window ID when last updated
    hits: int          # Number of times confirmed
```

### `HyperRec` (Dataclass)

Represents a hyperedge (multiple sources → target):

```python
@dataclass
class HyperRec:
    order: int         # Number of sources (edge order)
    weight: float      # Combined effect
    stability: float
    ci_lo: float
    ci_hi: float
    regime_id: int
    last_seen: int
    hits: int
```

---

## Class: `HypergraphStore`

### Initialization

```python
def __init__(
    self,
    max_edges: int = 10000,
    max_hyperedges: int = 1000,
    topk_per_node: int = 32,
    decay_factor: float = 0.995,
    alpha_weight: float = 0.2,
    alpha_stability: float = 0.2,
    gc_interval: int = 25
):
```

**Parameters**:
- `max_edges`: Soft limit before aggressive pruning
- `max_hyperedges`: Limit for multi-input edges
- `topk_per_node`: How many neighbors to index per node
- `decay_factor`: Multiplicative decay per window (0.995 = slow decay)
- `alpha_weight`: EMA smoothing for edge weights
- `alpha_stability`: EMA smoothing for stability
- `gc_interval`: How often to run garbage collection

---

## Node Management

### `get_or_create_node(name, domain=0, schema_ver=0)`

Maps variable names to integer IDs:

```python
node_id = store.get_or_create_node("gdp", domain=0, schema_ver=1)
# Returns: 42
```

**Schema versioning**: If schema changes (new columns added), `schema_ver` distinguishes old vs new variable definitions.

**Domains**: Group variables by domain (e.g., domain 0 = macro, domain 1 = micro).

---

## Edge Operations

### `upsert_edge(src_id, dst_id, effect, ci_lo, ci_hi, stability, ...)`

Insert or update an edge:

```python
store.upsert_edge(
    src_id=2,           # interest_rate
    dst_id=5,           # inflation
    effect=-0.23,       # Negative effect
    ci_lo=-0.30,
    ci_hi=-0.16,
    stability=0.89,
    regime_id=0,
    ts=1234             # Current window ID
)
```

**If edge exists**: Updates using EMA
- `new_weight = α * effect + (1-α) * old_weight`
- Increments hit counter
- Updates last_seen timestamp

**If edge is new**: Creates fresh `EdgeRec`

### Hyperedges

### `upsert_hyperedge(sources, effect, ci_lo, ci_hi, stability, ...)`

For multi-input relationships:

```python
store.upsert_hyperedge(
    sources=[2, 3, 7],   # Multiple source node IDs
    effect=0.45,
    ci_lo=0.35,
    ci_hi=0.55,
    stability=0.92
)
```

Sources are frozen into a sorted tuple as the key.

---

## Indexing

### Efficient Neighbor Lookup

The store maintains indices for fast queries:

- `out_index[node_id]`: Heap of top-k outgoing neighbors
- `in_index[node_id]`: Heap of top-k incoming neighbors

### `top_k_neighbors(node_id, k, direction="out", domain=None)`

```python
neighbors = store.top_k_neighbors(
    node_id=5,
    k=10,
    direction="out"   # or "in"
)
# Returns: [(neighbor_id, score), (neighbor_id, score), ...]
```

This enables fast traversal without scanning all edges.

---

## Temporal Decay

### `decay(ts)`

Applies time-based forgetting:

```python
store.decay(current_window_id)
```

**What happens**:
1. All edge weights multiplied by `decay_factor` (e.g., 0.995)
2. Stability also decays (slower)
3. "Stale" edges (not seen in 600+ windows) get extra penalty
4. Edges with weight below threshold are candidates for pruning

**Why decay matters**:
- Relationships change over time (regime shifts, structural breaks)
- Old evidence should count less than recent evidence
- Prevents stale edges from dominating

---

## Garbage Collection

### `_gc()`

Runs periodically (every `gc_interval` windows):

1. Identify edges below weight threshold
2. Identify edges not updated in N windows
3. Remove from storage
4. Rebuild indices

### `_maybe_gc()`

Conditional GC — runs only if:
- Edge count exceeds soft limit
- Or regular interval elapsed

---

## Query Methods

### `get_edge(src_id, dst_id) -> Optional[EdgeRec]`

Direct lookup:

```python
edge = store.get_edge(2, 5)
if edge:
    print(f"Effect: {edge.weight}, Stability: {edge.stability}")
```

### `get_all_edges() -> List[Dict]`

Export all edges for serialization:

```python
edges = store.get_all_edges()
# Returns: [{"source": "gdp", "target": "inflation", "weight": 0.23, ...}, ...]
```

### `get_statistics() -> Dict`

Store health metrics:

```python
{
    "n_nodes": 25,
    "n_edges": 342,
    "n_hyperedges": 45,
    "avg_weight": 0.12,
    "avg_stability": 0.78,
    "sparsity": 0.05
}
```

---

## Regime Tracking

Edges can be tagged with `regime_id` for regime-switching models:

- Regime 0: Normal times
- Regime 1: Crisis mode
- Regime 2: Recovery

The store supports filtering edges by regime:

```python
crisis_edges = [e for e in store.edges.values() if e.regime_id == 1]
```

This enables **regime-aware** relationship tracking.

---

## Persistence

### `to_dict() -> Dict`

Serialize entire store:

```python
state = store.to_dict()
json.dump(state, file)
```

### `from_dict(state) -> HypergraphStore`

Restore from serialized state:

```python
store = HypergraphStore.from_dict(json.load(file))
```

---

## Performance Characteristics

| Operation | Complexity |
|-----------|------------|
| Get edge | O(1) hash lookup |
| Upsert edge | O(log k) for index update |
| Top-k neighbors | O(k) |
| Decay all | O(n_edges) |
| GC | O(n_edges) |

**Memory**: Each edge ~100 bytes, 10K edges ≈ 1MB

---

## Integration Points

- **`MPIEOrchestrator`**: Writes accepted paths as edges
- **`Exporter`**: Reads edges for knowledge graph output
- **`federation`**: Merges edge stores across distributed nodes
- **`simulation`**: Uses edges to build predictive models

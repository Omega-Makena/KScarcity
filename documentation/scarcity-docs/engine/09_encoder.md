# encoder.py — Feature Encoding

The `Encoder` transforms raw data windows and candidate paths into **compact latent representations** suitable for fast scoring and comparison.

---

## Purpose

When evaluating paths, we need to:
1. Extract features from time-series windows
2. Incorporate path metadata (source, target, lags)
3. Produce fixed-size vectors for downstream scoring

The Encoder handles this with:
- Variable identity embeddings
- Lag positional encodings
- Attention and pooling operations
- Sketch-based dimensionality reduction

---

## Architecture

```
Raw Window + Path Metadata
           │
           ▼
    ┌──────────────────┐
    │ Variable Embedder │ ← Identity vectors for variables
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  Lag Encoder     │ ← Positional encoding for lags
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Attention + Pool │ ← Aggregate time dimension
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  Sketch Project  │ ← Reduce to fixed dimension
    └──────────────────┘
           │
           ▼
     Latent Vector (d=128)
```

---

## Supporting Classes

### `VariableEmbeddingMapper`

Maps variable indices to learnable dense vectors:

```python
mapper = VariableEmbeddingMapper(n_vars=20, id_dim=64)
embedding = mapper.get_embedding(var_idx=5)  # Shape: (64,)
```

**Initialization**: Random from Xavier/Glorot distribution, scaled by variance hooks.

### `LagPositionalEncoder`

Encodes temporal lags as dense vectors (like Transformer positional encoding):

```python
encoder = LagPositionalEncoder(max_lag=6, lag_dim=16)
lag_vec = encoder.encode_lag(lag=2)  # Shape: (16,)
```

Uses sinusoidal encoding + learned components.

### `PrecisionManager`

Controls mixed-precision (FP16/FP32) computing:

```python
pm = PrecisionManager()
x_fp16 = pm.autocast_fp16(x)  # Cast to FP16 if enabled
x_fp32 = pm.accumulate_fp32(x)  # Force FP32 for accumulations
```

Falls back to FP32 if instability detected (NaN/Inf values).

### `SketchCache`

LRU cache for sketch projection matrices:

```python
cache = SketchCache(capacity=8)
params = cache.get((input_dim, output_dim))
if params is None:
    params = generate_sketch_params(...)
    cache.put((input_dim, output_dim), params)
```

Avoids regenerating random matrices for repeated dimensions.

---

## Main Class: `Encoder`

### Initialization

```python
encoder = Encoder(drg={
    "id_dim": 64,        # Variable embedding size
    "lag_dim": 16,       # Lag encoding size
    "sketch_dim": 128,   # Output latent size
    "attn_heads": 8,     # Attention heads
    "pool_type": "avg",  # Pooling strategy
    "cache_cap": 8,      # Sketch cache capacity
    "fp16": True         # Use mixed precision
})
```

### `encode_batch(window, candidates) -> EncodedBatch`

Main encoding method:

```python
result = encoder.encode_batch(
    window=window_tensor,    # Shape: (time, variables)
    candidates=candidates     # List of Candidate objects
)

# Access results
latents = result.latents    # List of (128,) vectors
meta = result.meta          # Metadata for each candidate
stats = result.stats        # Timing/performance stats
```

---

## Encoding Pipeline

For each candidate:

### Step 1: Extract Window Slice

Get relevant columns for source and target variables:

```python
source_series = window[:, source_idx]  # Shape: (time,)
target_series = window[:, target_idx]  # Shape: (time,)
```

### Step 2: Apply Lag Structure

Create lagged feature matrix:

```python
# For lag=2:
# X[t] = [source[t-1], source[t-2]]
# y[t] = target[t]
```

### Step 3: Add Embeddings

Concatenate:
- Variable identity embedding (who is source/target)
- Lag positional encoding (what lag structure)
- Time series features

```python
features = np.concatenate([
    time_series_features,   # From window data
    source_embedding,       # Variable identity
    lag_encoding            # Temporal position
])
```

### Step 4: Attention/Pooling

Aggregate across time dimension:

| Pool Type | Method |
|-----------|--------|
| `avg` | Simple mean |
| `lastk` | Mean of last K timesteps |
| `attn` | Learned attention weights |

Attention uses query-key-value mechanism from `attention_ops`.

### Step 5: Sketch Projection

Reduce to fixed dimension using random projection:

```python
latent = sketch_project(pooled_features, target_dim=128)
```

Uses `poly_sketch` or `tensor_sketch` for efficient, deterministic projection.

### Step 6: Clip and Normalize

```python
latent = latent_clip(latent, max_norm=10.0)
latent = latent / (np.linalg.norm(latent) + 1e-8)
```

Prevents numerical issues downstream.

---

## Telemetry

### `get_stats() -> Dict`

```python
{
    "sketch_dim": 128,
    "fp16_enabled": True,
    "total_encoded": 15000,
    "avg_latency_ms": 2.3,
    "cache_hit_rate": 0.85
}
```

---

## Performance Optimizations

### Mixed Precision

FP16 operations use half the memory and are faster on modern hardware:

```python
# In PrecisionManager
if self.fp16_enabled and not self.fallback_mode:
    return x.astype(np.float16)
```

Falls back to FP32 if NaN/Inf detected.

### Batch Processing

Encodes multiple candidates together when possible:

```python
# Stack similar candidates
batch_input = np.stack([features_1, features_2, ...])
batch_output = sketch_project(batch_input)
```

### Caching

SketchCache prevents regenerating random projection matrices:

```python
# Cache hit rate typically 80%+
# Especially for fixed dimension combinations
```

---

## Integration with Operators

The Encoder uses operators from `engine/operators/`:

| Operator | Purpose |
|----------|---------|
| `attn_linear` | Attention weights |
| `pooling_avg`, `pooling_lastk` | Time aggregation |
| `layernorm`, `rmsnorm` | Normalization |
| `poly_sketch`, `tensor_sketch` | Dimensionality reduction |
| `countsketch` | Alternative sketch method |
| `latent_clip` | Bound output values |

See [operators documentation](./operators/00_overview.md) for details.

---

## Usage Context

You typically don't call the Encoder directly. It's used internally by:

- **`MPIEOrchestrator`**: Encodes candidates before evaluation
- **`federation`**: Encodes local updates before transmission

Direct use for debugging:

```python
from scarcity.engine.encoder import Encoder

encoder = Encoder()
result = encoder.encode_batch(window, candidates)
print(f"Latent shape: {result.latents[0].shape}")
```

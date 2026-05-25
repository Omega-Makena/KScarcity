# engine.py — MPIE Orchestrator

The **MPIEOrchestrator** (Multi-Path Inference Engine) is the **event-driven coordinator** that ties together all engine subsystems. It orchestrates the flow from raw data windows through encoding, evaluation, storage, and export.

---

## Purpose

While `OnlineDiscoveryEngine` processes single rows for hypothesis discovery, `MPIEOrchestrator` operates at a higher level:

- Listens to event bus topics for incoming data windows
- Coordinates the **Controller → Encoder → Evaluator → Store → Exporter** pipeline
- Responds to resource constraints from the Dynamic Resource Governor
- Publishes telemetry metrics for monitoring

Think of it as the **production-grade wrapper** around the core discovery logic.

---

## Architecture

```
Event Bus
    │
    ├── "data_window" ──────────┐
    ├── "resource_profile" ─────┼───▶ MPIEOrchestrator
    ├── "meta_policy_update" ───┘           │
    │                                       ▼
    │                           ┌───────────────────────┐
    │                           │    Processing Loop     │
    │                           │                        │
    │                           │  1. BanditRouter       │
    │                           │     proposes paths     │
    │                           │                        │
    │                           │  2. Encoder            │
    │                           │     creates latents    │
    │                           │                        │
    │                           │  3. Evaluator          │
    │                           │     scores paths       │
    │                           │                        │
    │                           │  4. Store              │
    │                           │     persists edges     │
    │                           │                        │
    │                           │  5. Rewards            │
    │                           │     update bandits     │
    │                           └───────────────────────┘
    │                                       │
    ◀── "mpie_metrics" ─────────────────────┘
```

---

## Core Class: `MPIEOrchestrator`

### Initialization

```python
def __init__(self, bus: Optional[EventBus] = None):
```

Creates all subsystems:
- **`BanditRouter`**: Proposes which paths (variable combinations + lags) to test
- **`Encoder`**: Converts data windows and paths into latent vectors
- **`Evaluator`**: Scores paths for predictive gain and stability
- **`HypergraphStore`**: Persists discovered relationships
- **`Exporter`**: Generates outputs (knowledge graph, reports)

If no event bus is provided, uses the global singleton from `scarcity.runtime.bus`.

### Lifecycle Methods

| Method | Purpose |
|--------|---------|
| `start()` | Activates event subscriptions, marks orchestrator as running |
| `stop()` | Unsubscribes from events, allows graceful shutdown |

Both are idempotent — safe to call multiple times.

---

## Event Handlers

### `_handle_data_window(topic, data)`

The **core processing loop**. When a new data window arrives:

1. **Extract** window tensor, schema, metadata from payload
2. **Propose** N candidate paths via BanditRouter
3. **Encode** each path into latent representation
4. **Evaluate** paths against the window data
   - Computes predictive gain, uncertainty, stability
5. **Filter** accepted paths (passed thresholds)
6. **Store** accepted edges in HypergraphStore
7. **Reward** BanditRouter based on which paths succeeded
8. **Export** updated knowledge graph
9. **Publish** telemetry metrics

The entire loop is designed to be **non-blocking** with bounded state.

### `_handle_resource_profile(topic, data)`

Receives updates from the Dynamic Resource Governor (DRG):
- `n_proposals`: How many paths to propose per window
- `n_resamples`: Bootstrap resampling count
- `allocation_limit`: Memory cap

Propagates these constraints to all subsystems.

### `_handle_meta_policy_update(topic, data)`

High-level policy changes from the Meta-Learning layer:
- `controller.tau`: Temperature for Thompson Sampling
- `evaluator.g_min`: Minimum gain threshold
- `evaluator.lambda_ci`: Confidence interval width

---

## Integration with FMI

Three handlers support the **Functional Model Interface** (FMI):

| Handler | Purpose |
|---------|---------|
| `_handle_fmi_policy_hint` | Apply FMI policy bundles to meta-learner |
| `_handle_fmi_warm_start` | Initialize subsystems from FMI state |
| `_handle_fmi_telemetry` | Hook for FMI telemetry (currently no-op) |

This allows external systems to inject pre-trained weights or override policies.

---

## Telemetry

After each window, the orchestrator publishes to `mpie_metrics`:

```python
{
    "window_id": 1234,
    "latency_ms": 45.2,
    "latency_ema_ms": 42.1,
    "n_candidates": 10,
    "n_accepted": 3,
    "controller_stats": {...},  # From BanditRouter
    "evaluator_stats": {...},   # From Evaluator
    "store_stats": {...}        # From HypergraphStore
}
```

---

## State Management

The orchestrator maintains minimal state:
- `_running`: Boolean flag
- `_current_schema`: Latest schema version
- `_step_counter`: Windows processed
- `_latency_ema`: Exponential moving average of processing time
- `_resource_profile`: Current DRG constraints

This bounded state enables horizontal scaling where different orchestrators process different shards.

---

## Usage Context

You typically don't interact with `MPIEOrchestrator` directly. Instead:

1. A **data streaming layer** publishes windows to the event bus
2. The orchestrator processes them automatically
3. A **dashboard or API** reads the exported knowledge graph

For simpler use cases (single-row processing), use `OnlineDiscoveryEngine` instead.

---

## Error Handling

- **Missing schema**: Logs warning, skips window
- **Empty window**: Returns early without processing
- **Subsystem failures**: Caught, logged, partial results returned
- **Resource exhaustion**: Evaluator relaxes thresholds to avoid starvation

---

## Performance Notes

- **Average latency**: Tracked via EMA with α=0.1
- **Lazy initialization**: Subsystems created on first use
- **Vectorized paths**: Encoder processes paths in batches where possible

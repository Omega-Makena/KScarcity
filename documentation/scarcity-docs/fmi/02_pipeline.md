# FMI Pipeline — Router, Aggregator, Encoder, Emitter, Validator, Service, Telemetry

End-to-end processing pipeline for `scarcity.fmi`.

---

## FMIRouter (router.py)

Groups incoming packets into cohorts for aggregation.

### `RouterConfig`

```python
@dataclass
class RouterConfig:
    cohort_key: Sequence[str] = ("schema_hash", "profile_class")
    cold_cohort_merge_interval_windows: int = 20
```

| Field | Description |
|-------|-------------|
| `cohort_key` | Ordered list of packet fields used to construct the cohort key string |
| `cold_cohort_merge_interval_windows` | MSP packets are ready when span ≥ this value OR buffer count ≥ this value |

### `FMIRouter`

```python
class FMIRouter:
    def route(packet: Mapping | PacketBase) -> str
    def ready() -> Dict[str, List[PacketBase]]
    def flush_all() -> Dict[str, List[PacketBase]]
    def clear(cohort: str) -> None
    def cohort_count() -> int
```

**`route`** — coerces the payload to a typed packet, builds a key by joining the configured fields with `/` (falling back to `provenance` dict if the field is not on the packet directly, then to `"na"`), appends to the cohort buffer, and tracks the `window_span` bounds. Returns the cohort key string.

**`ready`** — inspects each cohort buffer and marks it ready if:
- Any packet is non-MSP (POP/CCS are always ready immediately), or
- MSP: `window_span` range ≥ `cold_cohort_merge_interval_windows`, or
- MSP: buffer has accumulated ≥ `cold_cohort_merge_interval_windows` packets

Ready cohorts are cleared from the buffer. Returns `{cohort_key: [packets]}`.

---

## FMIAggregator (aggregator.py)

Merges a cohort of packets into a `MetaPriorUpdate`, `WarmStartProfile`, and `MetaPolicyHint`.

### `AggregationConfig`

```python
@dataclass
class AggregationConfig:
    metrics_trim_alpha: float = 0.1
    vote_min_sites: int = 3
    metrics_aggregation: str = "trimmed_mean"  # trimmed_mean | mean | fedavg | fedprox | scarcityw
    dp_noise_sigma: float = 0.0
    dp_epsilon: float = 0.0
    dp_delta: float = 0.0
    dp_sensitivity: float = 1.0
```

| Field | Description |
|-------|-------------|
| `metrics_trim_alpha` | Trim fraction for trimmed-mean (removes top and bottom α fraction) |
| `vote_min_sites` | Minimum distinct site_ids required for a POP bundle to be selected |
| `metrics_aggregation` | Aggregation method: `trimmed_mean`, `mean`/`fedavg`/`fedprox`, or `scarcityw`/`weighted` |
| `dp_noise_sigma` | Fixed Gaussian noise σ for differential privacy. If 0, derived from ε/δ |
| `dp_epsilon` | DP privacy budget ε |
| `dp_delta` | DP failure probability δ |
| `dp_sensitivity` | Global sensitivity for DP clipping |

### `AggregationResult`

```python
@dataclass
class AggregationResult:
    prior_update: Optional[MetaPriorUpdate] = None
    warm_start: Optional[WarmStartProfile] = None
    policy_hint: Optional[MetaPolicyHint] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def has_output() -> bool
```

### `FMIAggregator.aggregate(cohort, packets) -> AggregationResult`

Partitions packets by type and runs three independent aggregations:

**MSP → `_build_prior`**

- Aggregates `controller`, `evaluator`, `operators`, `metrics` dicts across all MSP packets using the configured method
- Packet weight = max of `provenance.trust` and `metrics.gain_p50` / `metrics.accept_rate` / `metrics.gain_p90`
- Confidence = harmonic mean of `evidence.confidence` values across packets
- CCS packets supply `contexts`: for each `(regime, concept_vectors)` group, computes the element-wise median across concept score vectors (up to 16 dimensions per concept)
- DP noise is applied to each aggregated float value after aggregation

**POP → `_build_policy_hint`**

- Votes policy bundles by `provenance.site_id`; gain per POP = `after.accept_rate − before.accept_rate`
- Selects the bundle with the highest mean gain that has support from ≥ `vote_min_sites` distinct sites
- Generates `±10%` bounds for each numeric bundle parameter
- Confidence = `clip(best_gain, 0, 1)`

**`_build_warm_start`** — derives a `WarmStartProfile` from the prior update: sets `profile_class` from the first MSP packet, copies `controller`/`evaluator` from the prior, and sets `context_selector` to the first aggregated regime if available.

**DP noise resolution**: if `dp_noise_sigma > 0`, use it directly; if `dp_epsilon > 0` and `dp_delta > 0`, compute `σ = sensitivity × sqrt(2 × ln(1.25/δ)) / ε`; otherwise no noise is added.

---

## FMIEncoder (encoder.py)

Serializes packets with configurable numeric precision and compression.

### `Precision`

```python
class Precision(str, Enum):
    FP32 = "fp32"   # No quantization
    FP16 = "fp16"   # Truncate to np.float16
    Q8 = "q8"       # Round to 2 decimal places
```

### `CodecConfig`

```python
@dataclass
class CodecConfig:
    precision: Precision = Precision.FP16
    compression: str = "zstd"   # "zstd" or "zlib"
```

### `FMIEncoder`

```python
class FMIEncoder:
    def encode(packet: Union[PacketBase, Mapping]) -> bytes
    def decode(blob: bytes) -> Dict[str, Any]
```

**`encode`** pipeline: coerce → `as_dict()` → quantize floats (recursive, handles nested dicts/lists) → JSON serialize → compress (zstd preferred; falls back to zlib if zstandard not installed).

**`decode`** pipeline: decompress → JSON parse → deep-copy pass (dequantize is identity — quantized values remain as-is post JSON round-trip).

---

## FMIEmitter (emitter.py)

Publishes aggregated outputs to the event bus.

### `EmitterConfig`

```python
@dataclass
class EmitterConfig:
    prior_broadcast_interval_windows: int = 10
```

Prior updates are suppressed if fewer than `prior_broadcast_interval_windows` windows have elapsed since the last emission.

### `FMIEmitter`

Published topics:

| Method | Primary topic | Legacy bridge topic |
|--------|--------------|---------------------|
| `emit_prior_update` | `fmi.meta_prior_update` | `meta_prior_update` |
| `emit_warm_start` | `fmi.warm_start_profile` | — |
| `emit_policy_hint` | `fmi.meta_policy_hint` | `meta_policy_update` (hint.bundle) |

**`emit_result(result, window)`** — calls all three emit methods conditionally based on which fields in `AggregationResult` are non-None. Also publishes `result.telemetry` to `fmi.telemetry`.

---

## FMIValidator (validator.py)

Applies a chain of checks before a packet is routed.

### `ValidatorConfig`

```python
@dataclass
class ValidatorConfig:
    dp_required: bool = False
    trust_min: float = 0.2
    max_packet_kb: int = 256
    max_age_seconds: Optional[float] = None
    max_value_norm: Optional[float] = None
```

### `ValidationResult`

```python
@dataclass
class ValidationResult:
    ok: bool
    reason: Optional[str] = None
    quarantined: bool = False
    dropped: bool = False
    warnings: list[str] = field(default_factory=list)
    payload: Optional[PacketBase] = None
```

### Validation chain (`validate(payload)`)

Checks run in order; first failure short-circuits:

| Check | Failure action | Reason string |
|-------|---------------|---------------|
| Schema (required fields + rev match) | `dropped=True` | `"schema:field1|field2"` |
| Size (JSON-encoded bytes ≤ max_packet_kb × 1024) | `dropped=True` | `"size_limit"` |
| Staleness (timestamp age ≤ max_age_seconds; ms timestamps auto-detected) | `dropped=True` | `"stale_packet"` |
| Value norm (L2 norm of all numeric fields ≤ max_value_norm) | `dropped=True` | `"norm_exceeded"` |
| Trust (`provenance.trust` or `provenance.trust_score` or `trust` field) | `quarantined=True` (payload still available) | `"trust<{min}"` |
| DP flag (if dp_required: `dp_flag=True` or `privacy.dp.enabled=True`) | `dropped=True` | `"dp_flag_missing"` |
| DP params (if dp_required: positive epsilon and delta) | `dropped=True` | `"dp_params_missing"` |

If trust is between `trust_min` and `0.5`, a `"low_trust"` warning is added without failing.

When no trust field is present, trust defaults to `1.0` (fully trusted).

---

## FMIService (service.py)

Orchestrates the full pipeline: validate → route → aggregate → emit.

### `ProcessOutcome`

```python
@dataclass
class ProcessOutcome:
    accepted: bool
    reason: Optional[str] = None
    cohort: Optional[str] = None
    aggregation: List[AggregationResult] = field(default_factory=list)
    quarantined: bool = False
```

### `FMIService.ingest(payload) -> ProcessOutcome`

1. Extract `trust` and compute JSON size
2. Run `validator.validate(payload)` — on failure return rejected `ProcessOutcome`
3. If `_suspend_pop` and packet type is POP or CCS → reject with `"suspended_by_drg"`
4. Route via `router.route(packet)` and update active cohort count
5. If `_defer_aggregation` → re-emit last prior (if available) and return without aggregating
6. Call `router.ready()` — for each ready cohort: `aggregator.aggregate()` → `emitter.emit_result()`
7. Return `ProcessOutcome(accepted=True, cohort=..., aggregation=[...])`

### `FMIService.apply_drg_signal(signal)`

Responds to DRG pressure signals (only when `config.drg_hooks.enable_adaptation=True`):

| Signal | Effect |
|--------|--------|
| `bandwidth_low` | Switch encoder precision to Q8 |
| `latency_high` | `_suspend_pop = True` (drop POP/CCS packets) |
| `vram_high` | `_defer_aggregation = True` |
| `util_low` | Restore FP16, clear suspend and defer flags |

### `FMIService.snapshot() -> Dict`

Returns `FMITelemetry.snapshot()`.

---

## FMITelemetry (telemetry.py)

### `TelemetryCounters`

```python
@dataclass
class TelemetryCounters:
    packets_in: int = 0
    packets_out: int = 0
    drops: int = 0
    quarantines: int = 0
    last_emit_ts: float
    cohorts_active: int = 0
```

### `FMITelemetry.snapshot() -> Dict`

Returns:

```python
{
    "packets_in": int,
    "packets_out": int,
    "drops": int,
    "quarantines": int,
    "cohorts_active": int,
    "type_breakdown": {"msp": int, "pop": int, "ccs": int},
    "latency": {
        "last_ingress_ts": float,
        "last_packet_kb": float,
        "last_trust": float,
    },
    "meta_gain_delta": float,
}
```

`meta_gain_delta` is updated from `result.telemetry["meta_gain_delta"]` on each `record_emit` call.

---

## End-to-End Usage

```python
from scarcity.fmi import FMIService

svc = FMIService()
outcome = await svc.ingest(msp_packet_dict)
# outcome.accepted, outcome.cohort, outcome.aggregation

# React to resource pressure
svc.apply_drg_signal("bandwidth_low")

# Inspect counters
snap = svc.snapshot()
print(snap["packets_in"], snap["drops"])
```

Subscribing to FMI output on the event bus:

```python
from scarcity.runtime import get_bus

bus = get_bus()

async def on_prior(topic, data):
    print(f"New prior rev={data['rev']} confidence={data['confidence']:.3f}")

bus.subscribe("fmi.meta_prior_update", on_prior)
bus.subscribe("fmi.warm_start_profile", on_warm_start)
bus.subscribe("fmi.meta_policy_hint", on_hint)
```

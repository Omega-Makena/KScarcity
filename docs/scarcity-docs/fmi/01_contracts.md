# FMI Contracts — Packet Schema Layer

API reference for `scarcity.fmi.contracts` — packet types, field definitions, schema registry.

---

## PacketType

```python
class PacketType(str, Enum):
    MSP = "msp"   # Meta Signal Pack — current learned state
    POP = "pop"   # Policy Outcome Pack — policy experiment results
    CCS = "ccs"   # Concept Causal Summary — discovered causality
```

---

## PacketBase

Base dataclass for all FMI packets.

```python
@dataclass
class PacketBase:
    type: PacketType
    schema_hash: str
    rev: int
    domain_id: str
    profile_class: str
    timestamp: Optional[int] = None
    provenance: JsonDict = field(default_factory=dict)
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `PacketType` | Packet category (MSP, POP, CCS) |
| `schema_hash` | `str` | Hash of the schema version that produced this packet |
| `rev` | `int` | Schema revision number; must match the registered schema rev |
| `domain_id` | `str` | Originating domain or site identifier |
| `profile_class` | `str` | Cohort profile class used for grouping |
| `timestamp` | `Optional[int]` | Unix timestamp (ms) of packet creation |
| `provenance` | `dict` | Arbitrary metadata: trust scores, site_id, regime_tag |

**Methods:**
- `as_dict() -> JsonDict` — serializes to a plain dict; `type` is converted to its string value
- `from_mapping(cls, payload) -> T` — constructs from a mapping, normalizing `type` string to `PacketType`

---

## MetaSignalPack (MSP)

Carries a snapshot of the local meta-learner state.

```python
@dataclass
class MetaSignalPack(PacketBase):
    window_span: Tuple[int, int] = (0, 0)
    metrics: JsonDict = field(default_factory=dict)
    controller: JsonDict = field(default_factory=dict)
    evaluator: JsonDict = field(default_factory=dict)
    operators: JsonDict = field(default_factory=dict)
    evidence: JsonDict = field(default_factory=dict)
```

| Field | Description |
|-------|-------------|
| `window_span` | `(start_window, end_window)` integer range covered by this packet |
| `metrics` | Operational metrics (latency_ms, accept_rate, gain_p50, etc.) |
| `controller` | Controller parameter snapshot |
| `evaluator` | Evaluator parameter snapshot |
| `operators` | Operator configuration snapshot |
| `evidence` | Confidence and evidence quality: `{"confidence": 0.85, ...}` |

---

## PolicyOutcomePack (POP)

Reports the outcome of a policy experiment.

```python
@dataclass
class PolicyOutcomePack(PacketBase):
    bundle: JsonDict = field(default_factory=dict)
    before: JsonDict = field(default_factory=dict)
    after: JsonDict = field(default_factory=dict)
    windows: int = 0
    confidence: float = 0.0
```

| Field | Description |
|-------|-------------|
| `bundle` | The policy bundle that was applied |
| `before` | Metrics snapshot before applying the policy (accept_rate, latency, etc.) |
| `after` | Metrics snapshot after applying the policy |
| `windows` | Number of windows the policy was evaluated over |
| `confidence` | Reporter confidence in the outcome measurement |

---

## ConceptCausalSummary (CCS)

Summarizes discovered causal relationships and concept vectors.

```python
@dataclass
class ConceptCausalSummary(PacketBase):
    causal_pairs: List[Tuple[str, str, float, str]] = field(default_factory=list)
    concepts: List[JsonDict] = field(default_factory=list)
    stability_delta: float = 0.0
    trust: float = 0.0
```

| Field | Description |
|-------|-------------|
| `causal_pairs` | List of `(source_var, target_var, strength, direction)` tuples |
| `concepts` | List of concept dicts; each with `id` and `score` fields |
| `stability_delta` | Change in causal graph stability since last packet |
| `trust` | Overall trust score for this summary |

---

## Output Packet Types

These are produced by the aggregator (not transmitted between sites).

### `MetaPriorUpdate`

```python
@dataclass
class MetaPriorUpdate:
    rev: int
    prior: JsonDict       # {"controller": ..., "evaluator": ..., "operators": ..., "metrics": ..., "signals": ...}
    contexts: List[JsonDict]   # [{"regime": "nairobi", "vector": [...]}]
    confidence: float
    cohorts: List[str]
```

### `WarmStartProfile`

```python
@dataclass
class WarmStartProfile:
    profile_class: str
    init: JsonDict        # {"controller": ..., "evaluator": ...}
    context_selector: JsonDict  # {"nearest_regime": "..."}
```

### `MetaPolicyHint`

```python
@dataclass
class MetaPolicyHint:
    hint_id: str          # "FMI-XXXX" deterministic ID
    bundle: JsonDict
    bounds: JsonDict      # per-parameter ±10% intervals
    reason: str
    confidence: float
```

All three expose `as_dict() -> JsonDict`.

---

## SchemaDefinition

```python
@dataclass
class SchemaDefinition:
    packet_type: PacketType
    rev: int
    required_fields: Tuple[str, ...]
    optional_fields: Tuple[str, ...] = ()

    def validate(self, payload: Mapping) -> List[str]  # returns list of missing fields
    def as_dict() -> JsonDict
```

Default schema revisions:
- MSP rev 3 — requires: type, schema_hash, rev, domain_id, profile_class, metrics, controller, evaluator, operators
- POP rev 2 — requires: type, schema_hash, rev, domain_id, profile_class, bundle, before, after, windows, confidence
- CCS rev 1 — requires: type, schema_hash, rev, domain_id, profile_class, causal_pairs, concepts, stability_delta, trust

---

## FMIContractRegistry

Central registry for packet schemas and coercion.

```python
class FMIContractRegistry:
    def register(schema: SchemaDefinition) -> None
    def get(packet_type: Union[str, PacketType]) -> SchemaDefinition
    def validate(payload: Mapping) -> Tuple[bool, List[str]]
    def coerce(packet: Union[PacketBase, Mapping]) -> PacketBase
```

**`validate`** — checks: (1) type field resolves to a known PacketType, (2) all required fields present, (3) `rev` matches registered schema rev. Returns `(ok, issues_list)`.

**`coerce`** — if the input is already a dataclass, returns it unchanged. Otherwise resolves `type` to the correct constructor (`MetaSignalPack`, `PolicyOutcomePack`, or `ConceptCausalSummary`) and calls `from_mapping`.

---

## Usage

```python
from scarcity.fmi.contracts import (
    FMIContractRegistry, MetaSignalPack, PacketType
)

registry = FMIContractRegistry()

# Validate an incoming payload dict
ok, issues = registry.validate(payload)
if not ok:
    print(f"Schema errors: {issues}")

# Coerce dict to typed packet
packet = registry.coerce(payload)  # returns MetaSignalPack / PolicyOutcomePack / ConceptCausalSummary

# Build an MSP manually
msp = MetaSignalPack(
    type=PacketType.MSP,
    schema_hash="abc123",
    rev=3,
    domain_id="site_001",
    profile_class="urban_ke",
    metrics={"accept_rate": 0.15, "latency_ms": 45.0},
    controller={"lr": 0.01},
    evaluator={"threshold": 0.5},
    operators={},
    evidence={"confidence": 0.78},
)
payload = msp.as_dict()
```

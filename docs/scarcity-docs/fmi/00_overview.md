# Scarcity FMI Module — Overview

The **FMI (Federated Metadata Interchange)** module provides a standardized protocol for exchanging metadata between federated sites. It defines packet types, validation schemas, and aggregation logic.

---

## Purpose

In federated learning, sites need to exchange:
- **Meta signals**: Current hyperparameters, metrics, evidence
- **Policy outcomes**: Results of policy changes
- **Causal summaries**: Discovered causal relationships

The FMI module:
1. **Defines packet contracts**: Typed, versioned data structures
2. **Validates exchanges**: Schema checking before aggregation
3. **Aggregates knowledge**: Combines packets into meta priors
4. **Routes updates**: Distributes aggregated knowledge

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FMIService                                │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ Emitter  │───►│ Validator│───►│Aggregator│───►│  Router  │ │
│   │(packets) │    │ (schema) │    │ (merge)  │    │(dispatch)│ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        │                                               │        │
│        ▼                                               ▼        │
│   ┌──────────┐                                   ┌──────────┐  │
│   │ Encoder  │                                   │ Telemetry│  │
│   │(compress)│                                   │(metrics) │  │
│   └──────────┘                                   └──────────┘  │
│                                                                  │
│                     ┌───────────────┐                           │
│                     │ ContractReg   │                           │
│                     │ (schemas)     │                           │
│                     └───────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Packet Types

### `PacketType` (Enum)

```python
class PacketType(str, Enum):
    MSP = "msp"  # Meta Signal Pack
    POP = "pop"  # Policy Outcome Pack
    CCS = "ccs"  # Concept Causal Summary
```

### MetaSignalPack (MSP)

Current system state from a federated site:

```python
@dataclass
class MetaSignalPack(PacketBase):
    window_span: Tuple[int, int]  # Time window covered
    metrics: Dict                  # Performance metrics
    controller: Dict               # Controller parameters
    evaluator: Dict                # Evaluator parameters
    operators: Dict                # Operator states
    evidence: Dict                 # Supporting evidence
```

### PolicyOutcomePack (POP)

Results of a policy change:

```python
@dataclass
class PolicyOutcomePack(PacketBase):
    bundle: Dict          # Policy parameters that changed
    before: Dict          # Metrics before change
    after: Dict           # Metrics after change
    windows: int          # Windows observed
    confidence: float     # Confidence in outcome
```

### ConceptCausalSummary (CCS)

Discovered causal relationships:

```python
@dataclass
class ConceptCausalSummary(PacketBase):
    causal_pairs: List[Tuple[str, str, float, str]]  # (source, target, weight, type)
    concepts: List[Dict]                              # Concept definitions
    stability_delta: float                            # Stability change
    trust: float                                      # Trust score
```

---

## Key Components

### FMIContractRegistry (`contracts.py`)

Schema registry and validation:

```python
registry = FMIContractRegistry()

# Validate a packet
is_valid, missing = registry.validate(packet_dict)

# Coerce dict to typed packet
packet = registry.coerce(packet_dict)
```

### FMIAggregator (`aggregator.py`)

Merges validated packets:

```python
aggregator = FMIAggregator(config=AggregationConfig())

result = aggregator.aggregate(
    cohort="healthcare",
    packets=[msp1, msp2, pop1, ccs1]
)
# result.prior_update: Aggregated priors
# result.warm_start: Warm start profile
# result.policy_hint: Policy recommendations
```

**Aggregation features**:
- Trimmed mean for robustness
- Weighted by confidence
- Optional differential privacy noise

### FMIValidator (`validator.py`)

Deep validation of packets:

```python
validator = FMIValidator(registry=registry)

errors = validator.validate(packet)
# Returns list of validation errors
```

### FMIEmitter (`emitter.py`)

Creates packets from system state:

```python
emitter = FMIEmitter(domain_id="site_001")

packet = emitter.emit_msp(
    metrics=current_metrics,
    controller=controller_state,
    evaluator=evaluator_state
)
```

### FMIRouter (`router.py`)

Routes packets to destinations:

```python
router = FMIRouter(config=RouterConfig())

await router.route(packet, destination="aggregator")
```

---

## Configuration

### `AggregationConfig`

```python
@dataclass
class AggregationConfig:
    metrics_trim_alpha: float = 0.1   # Trim percentage
    vote_min_sites: int = 3           # Min sites for voting
    metrics_aggregation: str = "trimmed_mean"
    dp_noise_sigma: float = 0.0       # DP noise
    dp_epsilon: float = 0.0
    dp_delta: float = 0.0
    dp_sensitivity: float = 1.0
```

---

## Output Types

### `MetaPriorUpdate`

Aggregated prior knowledge:

```python
@dataclass
class MetaPriorUpdate:
    rev: int               # Revision number
    prior: Dict            # Aggregated prior values
    contexts: List[Dict]   # Context information
    confidence: float      # Aggregate confidence
    cohorts: List[str]     # Contributing cohorts
```

### `WarmStartProfile`

Initialization hints for new sites:

```python
@dataclass
class WarmStartProfile:
    profile_class: str     # Profile type
    init: Dict             # Initial parameters
    context_selector: Dict # Context matching
```

### `MetaPolicyHint`

Policy recommendations:

```python
@dataclass
class MetaPolicyHint:
    hint_id: str          # Unique ID
    bundle: Dict          # Recommended parameters
    bounds: Dict          # Parameter bounds
    reason: str           # Explanation
    confidence: float     # Confidence score
```

---

## File Guide

| File | Purpose |
|------|---------|
| `contracts.py` | Packet types and schema registry |
| `aggregator.py` | FMIAggregator merging logic |
| `validator.py` | Deep packet validation |
| `emitter.py` | Packet creation from state |
| `encoder.py` | Packet compression |
| `router.py` | Packet routing |
| `service.py` | FMIService orchestrator |
| `telemetry.py` | FMI metrics publishing |

---

## Integration

### With Federation

```python
# Federation receives packets
bus.subscribe("federation.packet", handle_packet)

# Aggregated results go to meta layer
bus.publish("meta.fmi_update", result.prior_update)
```

### With Meta Layer

```python
# Meta layer receives aggregated priors
def handle_fmi_update(topic, payload):
    prior = payload["prior"]
    # Update hyperparameters
```

---

## Usage Example

```python
from scarcity.fmi import FMIService, FMIEmitter, FMIAggregator

# Site emits its state
emitter = FMIEmitter(domain_id="site_001")
packet = emitter.emit_msp(
    metrics={"gain_p50": 0.15},
    controller={"tau": 0.9}
)

# Central aggregator receives from multiple sites
aggregator = FMIAggregator(config=AggregationConfig())
result = aggregator.aggregate("healthcare", [packet1, packet2, packet3])

print(result.prior_update.prior)
# {"tau": 0.91, "g_min": 0.008}
```

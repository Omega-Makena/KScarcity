# Protocol Bridge — Adaptation & Sync Packets

`scarcity/federation/packets.py` (Phase 4 additions) · `scarcity/federation/codec.py`

---

## Purpose

The **protocol bridge** extends the federation wire format with three new typed packet dataclasses that carry adaptation signals between domain servers and the global meta-learner:

| Packet | Direction | Purpose |
|--------|-----------|---------|
| `AdaptationRequest` | Client → Domain Server | Request a warm-start prior for a new task context |
| `AdaptationResponse` | Domain Server → Client | Return the retrieved prior (or passthrough if unavailable) |
| `DomainSyncPacket` | Domain Server → Global | Sync domain state (base params, performance, memory health) |

These sit alongside the existing `PathPack`, `EdgeDelta`, `PolicyPack`, and `CausalSemanticPack` packets — all additive, no existing APIs changed.

---

## `AdaptationRequest`

A client asks its domain server for a warm-start parameter prior.

```python
@dataclass
class AdaptationRequest:
    basket_id: str          # Federation basket the client belongs to
    domain_id: str          # Logical domain name (e.g. "healthcare")
    context: Dict[str, Any] # Feature context for the current task
    round_id: int = 0       # Federation round counter
```

**Topic:** `federation.adaptation_request`

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `basket_id` | `str` | Identifies which basket (and thus which DomainServer) to query |
| `domain_id` | `str` | Domain label for logging and routing |
| `context` | `dict` | Context features used for episodic memory retrieval (e.g. `{"gain_p50": 0.3}`) |
| `round_id` | `int` | Monotonic round counter for replay guard compatibility |

### Round-trip

```python
req = AdaptationRequest("basket_hc", "healthcare", {"gain_p50": 0.3}, round_id=5)
d   = req.to_dict()
req2 = AdaptationRequest.from_dict(d)
assert req2.basket_id == "basket_hc"
```

---

## `AdaptationResponse`

The domain server responds with a prior, or signals that no prior is available.

```python
@dataclass
class AdaptationResponse:
    basket_id: str                  # Echo of the requesting basket
    domain_id: str                  # Echo of the domain
    prior_params: Dict[str, float]  # Retrieved parameter prior (may be empty)
    source: str                     # "global_memory" | "passthrough"
    round_id: int = 0
```

**Topic:** `federation.adaptation_response`

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `prior_params` | `dict` | Flat `{param_name: float}` prior for warm-starting |
| `source` | `str` | `"global_memory"` when memory had a match; `"passthrough"` when it fell back to defaults |

---

## `DomainSyncPacket`

Carries a full snapshot of a domain server's state upward to the global aggregator.

```python
@dataclass
class DomainSyncPacket:
    basket_id: str
    domain_id: str
    base_params: Dict[str, float]      # Current domain base model
    performance: Dict[str, float]      # Recent performance metrics
    memory_size: int                   # Episodes stored
    hit_rate: float                    # EMA adaptation hit rate
    round_id: int
```

**Topic:** `federation.domain_sync`

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `base_params` | `dict` | Named parameters of the domain's base model |
| `performance` | `dict` | Arbitrary performance metrics (e.g. `{"gain": 0.8, "stability": 0.9}`) |
| `memory_size` | `int` | How many adaptation episodes the domain has accumulated |
| `hit_rate` | `float` | EMA fraction of adapt calls that found a useful prior |

---

## `serialise_packet(packet)` — Topic Routing

Routes any packet to its canonical topic string:

```python
from scarcity.federation.packets import serialise_packet, AdaptationRequest

req = AdaptationRequest("b1", "healthcare", {})
topic, payload = serialise_packet(req)
# topic  → "federation.adaptation_request"
# payload → {"basket_id": "b1", "domain_id": "healthcare", "context": {}, "round_id": 0}
```

Supports all packet types (existing + new):

| Packet class | Topic |
|--------------|-------|
| `PathPack` | `federation.path_pack` |
| `EdgeDelta` | `federation.edge_delta` |
| `PolicyPack` | `federation.policy_pack` |
| `CausalSemanticPack` | `federation.causal_pack` |
| `AdaptationRequest` | `federation.adaptation_request` |
| `AdaptationResponse` | `federation.adaptation_response` |
| `DomainSyncPacket` | `federation.domain_sync` |

---

## `normalise_packets(packets)` — Grouping by Type

Groups a mixed list of packets into type-keyed buckets:

```python
from scarcity.federation.packets import normalise_packets

grouped = normalise_packets([req1, resp1, sync1, req2])
# {
#   "AdaptationRequest":  [req1, req2],
#   "AdaptationResponse": [resp1],
#   "DomainSyncPacket":   [sync1],
# }
```

---

## Codec Round-Trip

`PayloadCodec` in `codec.py` handles JSON serialization of all packets:

```python
from scarcity.federation.codec import PayloadCodec

codec = PayloadCodec()
encoded = codec.encode(req)          # bytes
decoded = codec.decode(encoded)      # dict
```

The decoded dict can be passed back to `AdaptationRequest.from_dict()`, `AdaptationResponse.from_dict()`, or `DomainSyncPacket.from_dict()` to reconstruct the typed object.

---

## End-to-End Flow

```
Client needs warm-start
        │
        ▼
AdaptationRequest(basket_id, domain_id, context)
        │
        ├─► serialise_packet() → "federation.adaptation_request"
        │
        ▼
DomainServer.suggest_prior(context)
        │
        ├─ memory hit  → source = "global_memory"
        └─ memory miss → source = "passthrough", prior_params = {}
        │
        ▼
AdaptationResponse(basket_id, domain_id, prior_params, source)
        │
        └─► serialise_packet() → "federation.adaptation_response"
                │
                ▼
        Client applies prior_params as warm-start initialization
```

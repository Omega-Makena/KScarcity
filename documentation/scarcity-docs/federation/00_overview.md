# Scarcity Federation Module — Overview

The **federation module** enables distributed, privacy-preserving learning across multiple clients. It implements a **hierarchical federated learning** system with domain-aware grouping, gossip protocols, and secure aggregation.

---

## Purpose

Real-world relationship discovery often involves data that:
- Is distributed across multiple parties (hospitals, banks, countries)
- Cannot be centralized due to privacy regulations (GDPR, HIPAA)
- Operates under adversarial conditions (Byzantine clients, poisoning attacks)

The federation module solves this by letting each client run a local `OnlineDiscoveryEngine` and **share only aggregated, privatized updates** — never raw data.

---

## Architecture: Hierarchical Federation

```
                    ┌─────────────────────────────────┐
                    │        Global Meta-Model         │
                    │      (crosses all domains)       │
                    └─────────────────────────────────┘
                                    ▲
                                    │ Layer 2 Aggregation
                                    │ (Secure Aggregation)
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
            │ Basket: Health│ │Basket: Fin│ │ Basket: Retail│
            │  Aggregate    │ │ Aggregate │ │   Aggregate   │
            └───────────────┘ └───────────┘ └───────────────┘
                    ▲               ▲               ▲
                    │ Layer 1       │ Layer 1       │ Layer 1
                    │ (Gossip+DP)   │               │
            ┌───────┴───────┐       │               │
    ┌───────▼───────┐   ┌───▼───┐   │               │
    │ Client A      │◄──│Client B│   ▼               ▼
    │ (Hospital 1)  │──►│(Hosp 2)│  ...             ...
    └───────────────┘   └───────┘
         Gossip
```

### Two-Layer Aggregation

1. **Layer 1 (Intra-Basket)**: Clients within the same domain gossip updates with each other. Local differential privacy (DP) applied to each message.

2. **Layer 2 (Cross-Basket)**: Basket aggregates are combined into a global model using secure aggregation. Central DP applied before distribution.

---

## Core Components

### HierarchicalFederation

The **main entry point** that orchestrates everything:

```python
from scarcity.federation import HierarchicalFederation, HierarchicalFederationConfig

fed = HierarchicalFederation(HierarchicalFederationConfig(
    total_epsilon=10.0,    # Total privacy budget
    auto_aggregate=True    # Aggregate when triggers fire
))

# Register clients
fed.register_client("hospital_1", domain_id="healthcare")
fed.register_client("hospital_2", domain_id="healthcare")
fed.register_client("bank_1", domain_id="finance")

# Submit updates
fed.submit_update("hospital_1", update_vector)
fed.run_gossip_round()
global_model = fed.maybe_aggregate()
```

### Domain Baskets (BasketManager)

Groups clients by domain for efficient within-domain communication:
- **Healthcare basket**: Hospitals, clinics
- **Finance basket**: Banks, exchanges
- **Retail basket**: Stores, e-commerce

Clients within a basket share more freely (similar data distributions).

### Gossip Protocol (GossipProtocol)

Decentralized communication within baskets:
- **Push**: When local state changes significantly, push to K random peers
- **Pull**: Periodically request updates from K random peers
- **Local DP**: All messages clipped and noised before transmission

### Secure Aggregation (SecureAggClient, SecureAggCoordinator)

Cryptographic protocol for Layer 2:
- Clients secret-share their updates
- Coordinator can only see the **sum**, not individual contributions
- Dropout-tolerant (works if some clients go offline)

### Privacy Accounting (PrivacyAccountant)

Tracks cumulative privacy loss:
- Monitors (ε, δ) budget consumption
- Blocks operations that would exceed budget
- Supports composition across rounds

---

## Data Flow

### Client Submits Update

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Update Flow                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Local Engine ──► 2. Clip to norm ──► 3. Store in Buffer      │
│                                                                   │
│  4. Check Trigger ──► 5. If ready: Layer 1 Aggregation           │
│                                                                   │
│  6. Aggregate baskets ──► 7. Layer 2 if threshold met            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Gossip Round

```
For each basket:
  1. PeerSampler selects K peers (with rotation for privacy)
  2. MaterialityDetector checks if push needed
  3. If push: Clip+Noise local state, send to peers
  4. Receive and average incoming messages
```

### Layer 2 Secure Aggregation

```
1. Each basket encrypts its aggregate with ephemeral keys
2. Coordinator collects encrypted shares
3. After threshold reached: sum revealed, individual inputs hidden
4. Central DP noise added to global sum
5. Global model distributed to all baskets
```

---

## File Guide

| File | Purpose |
|------|---------|
| `hierarchical.py` | Main orchestrator |
| `aggregator.py` | Byzantine-robust aggregation (Krum, Bulyan, etc.) |
| `gossip.py` | Push-pull gossip with local DP |
| `basket.py` | Domain basket management |
| `buffer.py` | Staleness-aware update buffer |
| `layers.py` | Layer 1 & 2 aggregation logic |
| `secure_aggregation.py` | Cryptographic secure aggregation |
| `packets.py` | Wire format for federation messages |
| `privacy_guard.py` | DP mechanism enforcement |
| `trust_scorer.py` | Client trust scoring |
| `validator.py` | Packet validation |
| `client_agent.py` | Client-side federation agent |
| `coordinator.py` | Server-side coordinator |
| `reconciler.py` | Merge distributed HypergraphStores |
| `scheduler.py` | Federation round scheduling |
| `codec.py` | Serialization/deserialization |
| `transport.py` | Network transport abstraction |

---

## Key Concepts

### Privacy Budget

Federation consumes privacy budget at each interaction:
- Local DP at gossip: ~ε per message
- Secure aggregation: ~ε per round
- Central DP at Layer 2: additional ε

Total budget (e.g., ε=10) is finite — once exhausted, no more sharing.

### Byzantine Robustness

Clients can be:
- **Honest but curious**: Follow protocol but try to infer others' data
- **Byzantine/malicious**: Send arbitrary updates to poison the model

Aggregation methods like **Krum** and **Bulyan** detect and exclude outliers.

### Staleness

Updates may arrive out of order or delayed. The buffer tracks:
- `round_id`: Which training round the update belongs to
- Age-weighted aggregation: Older updates get lower weight

---

## Usage Context

Federation is used when:
1. Data is **distributed** and cannot be centralized
2. **Privacy** is a regulatory or competitive requirement
3. Multiple parties want to **collaborate** without sharing raw data

For single-party scenarios, use `OnlineDiscoveryEngine` directly.

---

## Integration with Engine Module

```python
# Client side
from scarcity.engine import OnlineDiscoveryEngine
from scarcity.federation import FederationClientAgent

engine = OnlineDiscoveryEngine()
agent = FederationClientAgent(engine, client_id="client_1")

# Process local data
for row in local_data:
    engine.process_row(row)

# Periodically share with federation
update = agent.get_shareable_update()
federation.submit_update("client_1", update)
```

The federation layer wraps the engine — it doesn't replace it.

# Federation Module — Documentation Index

Complete documentation for the `scarcity.federation` module — distributed, privacy-preserving learning.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Architecture and concepts |
| [01_hierarchical.md](./01_hierarchical.md) | HierarchicalFederation — Main orchestrator |
| [02_aggregator.md](./02_aggregator.md) | Byzantine-robust aggregation methods |
| [03_gossip.md](./03_gossip.md) | Push-pull gossip with local DP |
| [04_basket.md](./04_basket.md) | Domain basket management |
| [05_buffer.md](./05_buffer.md) | Staleness-aware update buffer |
| [06_layers.md](./06_layers.md) | Layer 1 & 2 aggregation |
| [07_secure_aggregation.md](./07_secure_aggregation.md) | Cryptographic secure sum |
| [08_utilities.md](./08_utilities.md) | Packets, codec, trust, etc. |

---

## Key Concepts

### Hierarchical Federation

Two-layer architecture:
- **Layer 1**: Intra-basket aggregation (gossip + local DP)
- **Layer 2**: Cross-basket aggregation (secure sum + central DP)

### Privacy Stack

Multiple layers of protection:
1. **Local DP**: Noise added before gossip
2. **Secure Aggregation**: Coordinator sees only sum
3. **Central DP**: Additional noise on global model
4. **Privacy Accounting**: Budget tracking

### Byzantine Robustness

Aggregation methods that resist poisoning:
- Trimmed Mean, Median
- Krum, Multi-Krum
- Bulyan

---

## Integration with Engine

```python
from scarcity.engine import OnlineDiscoveryEngine
from scarcity.federation import HierarchicalFederation

# Local engine
engine = OnlineDiscoveryEngine()

# Federation layer
fed = HierarchicalFederation()
fed.register_client("client_1", domain_id="healthcare")

# Process local data
for row in local_data:
    engine.process_row(row)

# Share with federation
update = extract_update(engine)
fed.submit_update("client_1", update)
```

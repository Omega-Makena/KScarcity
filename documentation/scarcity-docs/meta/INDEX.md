# Meta Module — Documentation Index

Complete documentation for the `scarcity.meta` module — meta-learning and governance.

---

## Quick Links

| File | Topic |
|------|-------|
| [00_overview.md](./00_overview.md) | **Start here** — Architecture and concepts |
| [01_utilities.md](./01_utilities.md) | All module files explained |

---

## Key Concepts

### Meta-Governance (Tier-5)

- Adjusts hyperparameters based on performance
- Applies cooldowns and rollbacks
- Rule-based policy engine

### Meta-Learning

- Domain-specific priors
- Cross-domain aggregation
- Reptile optimization

---

## Quick Start

```python
from scarcity.meta import MetaIntegrativeLayer

layer = MetaIntegrativeLayer()

result = layer.update({
    "gain_p50": 0.15,
    "stability_mean": 0.8
})

print(result["policy_updates"])
```

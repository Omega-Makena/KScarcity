# Synthetic Data Module — Overview

> `scarcity.synthetic` — Generates realistic synthetic social media data for testing and development.

---

## Purpose

The synthetic module creates artificial social media posts, user accounts, and interaction patterns that mimic real-world data. This enables:

- **Pulse Engine testing** without requiring live social media API access
- **Repeatable benchmarks** with deterministic seed control
- **Stress testing** with configurable crisis scenarios

---

## Architecture

```
SyntheticPipeline
├── AccountGenerator    → Fake user profiles with demographics
├── ContentGenerator    → Posts with signal-laden text
├── BehaviorSimulator   → Timing, spread, engagement patterns
├── scenarios.py        → Pre-built crisis templates
└── vocabulary.py       → Signal-specific word banks (13 KB)
```

---

## Key Classes

### `AccountGenerator` — `accounts.py`

Creates synthetic social media accounts with realistic demographics.

```python
from scarcity.synthetic import AccountGenerator

gen = AccountGenerator(seed=42)
accounts = gen.generate(n=100)
# Each account: username, display_name, location, followers, etc.
```

### `ContentGenerator` — `content.py`

Generates text content that triggers specific Pulse signal detectors.

```python
from scarcity.synthetic import ContentGenerator

gen = ContentGenerator(seed=42)
posts = gen.generate(
    signal_type="food_insecurity",
    count=50,
    intensity="high"
)
```

### `BehaviorSimulator` — `behavior.py`

Simulates temporal patterns — posting frequency, retweets, cascading behavior.

```python
from scarcity.synthetic import BehaviorSimulator

sim = BehaviorSimulator(seed=42)
timeline = sim.simulate(
    accounts=accounts,
    posts=posts,
    duration_hours=72
)
```

### `SyntheticPipeline` — `pipeline.py`

End-to-end pipeline that combines all generators:

```python
from scarcity.synthetic import SyntheticPipeline

pipeline = SyntheticPipeline(seed=42)
dataset = pipeline.run(
    n_accounts=200,
    n_posts=1000,
    scenario="election_crisis"
)
```

---

## Supporting Files

| File | Purpose |
|------|---------|
| `vocabulary.py` | Signal-specific keyword banks (13 KB) — words and phrases that trigger each of the 15 Pulse signal detectors |
| `scenarios.py` | Pre-built crisis scenario templates (election, drought, economic shock, etc.) |

---

## Integration

The synthetic module feeds directly into the Pulse Engine for testing:

```
SyntheticPipeline.run()
    → List[SocialPost]
        → PulseSensor.process_text()
            → SignalDetections
```

---

*Source: `scarcity/synthetic/` · Last updated: 2026-02-11*

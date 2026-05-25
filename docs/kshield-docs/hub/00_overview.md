# KShieldHub — Overview

> `kshiked.hub` — Central orchestrator for the SENTINEL platform.

---

## Purpose

`KShieldHub` is a **singleton** that wires together the Intelligence Layer components:

1. **Pulse Engine** — social signal detection
2. **Scarcity Simulation** — SFC economic model
3. **Economic Bridge** — converts threat signals to simulation shocks
4. **Dashboard** — provides aggregated data via `get_dashboard_data()`

---

## Class: `KShieldHub`

| Method | Description |
|--------|-------------|
| `__init__()` | Initialises Pulse sensor, economic bridge, and optionally the simulation engine |
| `process_pulse_update(text_input)` | Runs text through Pulse → computes threat indices → triggers economic bridge |
| `get_dashboard_data()` | Returns aggregated state dict for the Streamlit dashboard |
| `_init_simulation()` | Creates `SimulationEngine` + `EventBus` and links the bridge |
| `_apply_shock_to_sim(shock)` | Converts a Pulse `ShockEvent` to a bus event and publishes to simulation |

### Singleton Pattern

```python
from kshiked.hub import get_hub

hub = get_hub()          # Returns cached singleton
hub2 = get_hub()         # Same instance
assert hub is hub2
```

### Processing Pipeline

```python
hub = get_hub()

# Ingest new social text
hub.process_pulse_update("Fuel prices have tripled in Nairobi")

# Read aggregated state for dashboard
data = hub.get_dashboard_data()
# data["pulse"]["threat_level"]     → "ELEVATED"
# data["pulse"]["indices"]          → {polarization, legitimacy, ...}
# data["simulation"]["active"]      → True/False
```

---

## Internal Data Flow

```
process_pulse_update(text)
   │
   ├── PulseSensor.process_text(text)
   ├── PulseSensor.update_state()
   ├── compute_threat_report(state, history)
   │       → ThreatIndexReport
   └── KShieldEconomicBridge.process_signals(report, state)
           │
           └── [if simulation linked]
               EventBus.publish("simulation.shock", payload)
               SFCEconomy.step()
```

---

## Dependencies

| Import | Source | Conditional |
|--------|--------|-------------|
| `PulseSensor` | `kshiked.pulse` | Required |
| `compute_threat_report` | `kshiked.pulse.indices` | Required |
| `KShieldEconomicBridge` | `kshiked.pulse.simulation_connector` | Required |
| `SimulationEngine` | `scarcity.simulation.engine` | Optional — graceful fallback if missing |
| `EventBus` | `scarcity.runtime` | Optional |

---

*Source: `kshiked/hub.py` · Last updated: 2026-02-11*

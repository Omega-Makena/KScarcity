# controller.py — Meta-Controller

The `MetaController` manages the **lifecycle state machine** for hypotheses, transitioning them between states based on their accumulated evidence and metrics.

---

## Purpose

Hypotheses don't just exist — they have a lifecycle:

1. **Birth**: Created with some prior belief
2. **Testing**: Accumulate evidence from data
3. **Promotion**: If strong enough, become "active"
4. **Decay**: If metrics drop, enter probation
5. **Death**: If irredeemable, killed and archived

The MetaController enforces these transitions.

---

## State Machine

```
                    ┌───────────────────────────────┐
                    │                               │
   Create           ▼           metrics recover    │
     │        ┌──────────┐           │              │
     └──────► │ TENTATIVE │ ──────────┤              │
              └──────────┘           │              │
                    │                │              │
                    │ enough evidence│              │
                    │ + high metrics │              │
                    ▼                │              │
              ┌──────────┐           │              │
              │  ACTIVE  │ ◀─────────┘              │
              └──────────┘                          │
                    │                               │
                    │ metrics drop                  │
                    ▼                               │
              ┌──────────┐                          │
              │ DECAYING │ ─────────────────────────┘
              └──────────┘
                    │
                    │ metrics collapse
                    ▼
              ┌──────────┐
              │   DEAD   │ ──► Graveyard
              └──────────┘
```

---

## Class: `MetaController`

### Initialization

```python
controller = MetaController(
    confidence_threshold=0.7,   # Minimum to become/stay ACTIVE
    stability_threshold=0.6,    # Minimum stability score
    min_evidence=20             # Minimum observations before promotion
)
```

### `manage_lifecycle(pool: HypothesisPool)`

Main method — scans all hypotheses and applies transitions:

```python
controller.manage_lifecycle(pool)
```

**For each hypothesis**:

1. **Read current metrics**: confidence, stability, evidence
2. **Apply transition rules** (see below)
3. **Update state** if transition occurs
4. **Kill dead hypotheses** (move to graveyard)

---

## Transition Rules

### TENTATIVE → ACTIVE

```python
if evidence > min_evidence:
    if confidence > conf_thresh and stability > stab_thresh:
        new_state = ACTIVE
    elif confidence < 0.3:
        kill(hypothesis)  # Early failure, don't waste time
```

**Interpretation**: New hypotheses need to prove themselves. If they gather enough evidence and pass thresholds, they're promoted. If they fail badly early, they're killed quickly.

### ACTIVE → DECAYING

```python
if confidence < (conf_thresh - 0.1) or stability < (stab_thresh - 0.1):
    new_state = DECAYING
```

**Interpretation**: Active hypotheses can lose their status if metrics drop. The 0.1 hysteresis prevents oscillation at threshold boundaries.

### DECAYING → ACTIVE

```python
if confidence > conf_thresh and stability > stab_thresh:
    new_state = ACTIVE  # Recovered!
```

**Interpretation**: Decaying hypotheses get a second chance. If metrics recover, they're re-promoted.

### DECAYING → DEAD

```python
if confidence < 0.2:
    kill(hypothesis)
```

**Interpretation**: If confidence drops too low during decay, the hypothesis is terminated.

---

## Kill Logic

When a hypothesis is killed:

1. `pool._kill(hid)` called
2. Hypothesis serialized to `to_dict()`
3. Moved to `pool.graveyard` (for debugging/analysis)
4. Removed from `pool.population`

**Graveyard keeps history** — you can analyze why hypotheses failed.

---

## Summary Statistics

### `get_summary(pool: HypothesisPool) -> Dict`

```python
summary = controller.get_summary(pool)
# Returns:
{
    "active": 42,
    "tentative": 120,
    "decaying": 8,
    "dead": 15  # From graveyard size
}
```

Used for monitoring and dashboard display.

---

## When Transitions Happen

The MetaController is called **once per tick** in `OnlineDiscoveryEngine.process_row()`:

```python
def process_row(self, row):
    # Update all hypotheses
    self.pool.update_all(row)
    
    # Manage lifecycle transitions
    self.meta_controller.manage_lifecycle(self.pool)
    
    # ...
```

---

## Threshold Tuning

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `confidence_threshold` | Fewer ACTIVE hypotheses, higher quality |
| `stability_threshold` | More consistent hypotheses promoted |
| `min_evidence` | Longer burn-in before promotion |

**Conservative settings** (high thresholds): Fewer false positives, slower discovery
**Aggressive settings** (low thresholds): More discoveries, more noise

---

## Integration Points

- **`OnlineDiscoveryEngine`**: Calls `manage_lifecycle()` each tick
- **`HypothesisPool`**: Controller reads/writes hypothesis states
- **`HypothesisArbiter`**: Operates on ACTIVE hypotheses only

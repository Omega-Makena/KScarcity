# Causal Module Utilities

This page outlines the production modules used in the Scarcity causal pipeline.

If you find yourself repeating implementation details, use:
1) purpose, 2) data flow, 3) invariants, 4) failure modes, 5) pseudocode.

---

## engine.py — Pipeline Entry Point

Purpose:
- Orchestrate multi-estimand execution.
- Enforce runtime policies (parallelism, failure, time-series).
- Bundle results and emit artifacts.

Data flow:
- Specs → validation → identification → estimation → refutation → artifacts.

Invariants:
- Results are bundled; no raw list return.
- Fail policies apply per spec and per chunk.

Failure modes:
- Per-spec errors are isolated.
- Worker failures are captured as chunk errors.

Pseudocode:
```
run_causal(data, specs, runtime)
→ per-spec pipeline
→ bundle results + artifacts
```

---

## specs.py — Specification Classes

Purpose:
- `EstimandSpec` defines the causal question and temporal constraints.
- `RuntimeSpec` defines execution and safety policies.

Invariants:
- Estimand specs must be self-consistent.
- Runtime policies normalize to canonical values.

Failure modes:
- Missing required fields cause spec rejection.

---

## identification.py — Identification Layer

Purpose:
- Build DoWhy CausalModel using DOT structure when provided.
- Run identification per estimand type.

Invariants:
- DoWhy remains the identification owner.

Failure modes:
- Unidentifiable effects surface as spec errors.

---

## estimation.py — Estimation Backends

Purpose:
- Select backend per estimand type and runtime preferences.

Invariants:
- EconML is only used through DoWhy integration.

Failure modes:
- Missing EconML dependency raises a spec error.

---

## validation.py — Refutation Layer

Purpose:
- Run configured refuters and return structured diagnostics.

Invariants:
- Refuter failures do not crash the spec; they are recorded.

---

## time_series.py — Temporal Safety

Purpose:
- Validate time columns, lags, and temporal DAG constraints.

Invariants:
- `strict` blocks execution on temporal violations.
- `warn` annotates diagnostics and continues.

---

## graph.py — DOT Integration

Purpose:
- Load DOT templates, parse edges, validate temporal direction.
- Export learned graph snapshots.

Invariants:
- DOT is structure-only and version controlled.

---

## artifacts.py — Artifact Emission

Purpose:
- Emit run bundles, per-spec artifacts, errors, and graph snapshots.

Failure modes:
- Artifact emission does not modify model state; failures are logged.

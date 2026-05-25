# Scarcity Causal Module — Overview

The **causal module** is a production pipeline for identification, estimation, refutation, and artifact generation.
It is **DoWhy-first**, with **EconML** used behind DoWhy for heterogeneous estimands.

---

## Purpose

- Provide a multi-estimand, multi-backend causal engine.
- Enforce graph-first reasoning (DOT → DoWhy → Estimation → Refutation → Artifacts).
- Support time-series safety and deterministic execution.

---

## Data Flow

Input → specification validation → feature validation → identification → estimation → refutation → artifact emission.

Pseudocode:
```
run_causal(data, specs, runtime)
→ identify → estimate → refute
→ artifacts + run bundle
```

---

## Invariants

- DoWhy owns identification and graph semantics.
- DOT is structure-only and version controlled (no weights).
- Every run emits artifacts, even with partial failure.
- Results are bundled; never return raw lists alone.

---

## Failure Modes

- Spec-level failures do not corrupt other specs.
- `fail_fast` stops on the first failure.
- `continue` emits partial success with error records.
- Time-series violations are fatal in `strict`, warnings in `warn`.

---

## Public API

- `run_causal(data, spec | list[spec], runtime)` returns `CausalRunResult`.
- Supports parallel execution with process pool by default.

---

## Result Bundle

`CausalRunResult` includes:
- `results`: list of `EffectArtifact` (per-spec artifacts)
- `errors`: list of `SpecError` (per-spec failures)
- `summary`: run-level status and counts
- `metadata`: runtime, versions, data signature

---

## Time-Series Safety

Policy modes:
- `strict`: require time column, lag, temporal DAG, mediator timing.
- `warn`: emit diagnostics and proceed.
- `none`: skip time-series checks.

---

## DOT Graph Integration

- DOT templates are loaded per spec or runtime.
- Temporal direction is validated when policy is time-aware.
- Learned graph snapshots are exported per run.

DOT source file:
- `documentation/scarcity-docs/causal/scarcity_causal_flow.dot`

---

## Artifacts

Every run emits:
- `summary.json`
- `effects.jsonl`
- `errors.jsonl`
- `graphs/input.dot`
- `graphs/learned.dot`

# Engine Identity Resolution

**Date:** 2026-04-27
**Status:** RESOLVED — discrepancy documented and explained; benchmark numbers are reproducible.

---

## Summary

The benchmark scripts instantiate **`OnlineDiscoveryEngine`** from
`scarcity.engine.engine_v2` with **`FederationHub` / `FederationNode`** from
`scarcity.engine.federation_*`.

The architecture documentation describes a separate, newer subsystem:
**`MPIEOrchestrator`** (from `scarcity.engine.engine`) + **`HierarchicalFederation`**
(from `scarcity.federation.hierarchical`). These are co-existing subsystems — not
conflicting alternatives. The benchmark exercises the older, annual-frequency path.

All claim numbers in the benchmark findings report are produced by
`OnlineDiscoveryEngine` + `FederationHub/FederationNode`.

---

## Engine Audit (by script)

Every benchmark script that instantiates an engine class uses:

| Script | Engine imported | Federation imported |
|--------|----------------|---------------------|
| `benchmark_basket_fl.py` | (none — uses FederationNode directly) | `FederationHub`, `FederationNode` |
| `benchmark_discovery.py` | `OnlineDiscoveryEngine` | `FederationHub`, `FederationNode` |
| `benchmark_proper.py` | `OnlineDiscoveryEngine` | `FederationHub`, `FederationNode` |
| `experiment_east_africa_federation.py` | `OnlineDiscoveryEngine` | `FederationHub`, `FederationNode` |
| `benchmark_reviewer.py` | `OnlineDiscoveryEngine` | `FederationHub`, `FederationNode` |
| `benchmark_federation_ablations.py` | `OnlineDiscoveryEngine` | `FederationHub`, `FederationNode` |

Zero benchmark scripts instantiate `MPIEOrchestrator` or `HierarchicalFederation`.

---

## MetaController Thresholds (from `scarcity/engine/controller.py`)

| Parameter | Value | Role |
|-----------|-------|------|
| `confidence_threshold` | 0.70 | Minimum confidence to promote TENTATIVE → ACTIVE |
| `stability_threshold` | 0.60 | Minimum stability score |
| `min_evidence` | 20 | Minimum evidence count before promotion |
| `kill_threshold` | 0.10 | Confidence below which hypothesis is killed |

**Note:** The promotion threshold in `controller.py` (`MetaController`) is **0.70**, not the
simulation gate of **0.25**. The 0.25 gate is in `discovery.py`
(`get_candidate_paths()` filter). These are different thresholds at different layers:
MetaController manages hypothesis lifecycle; the simulation gate controls what reaches
the PolicySimulator.

---

## Relationship Between the Two Engine Subsystems

```
scarcity/engine/engine_v2.py — OnlineDiscoveryEngine
    Hypothesis survival paradigm (Bayesian accumulators, λ=0.99)
    FederationHub/FederationNode (scarcity/engine/federation_*)
    Used by: ALL benchmark scripts
    Frequency: annual macro (N=34)

scarcity/engine/engine.py — MPIEOrchestrator
    Thompson Sampling + bootstrap R² gain (BanditRouter)
    async EventBus architecture (data_window, resource_profile topics)
    HierarchicalFederation (scarcity/federation/hierarchical.py)
    Used by: production runtime only
    Frequency: intended for higher-frequency streaming data

Bridge: MPIEOrchestrator.attach_discovery_engine() calls
    _discovery_engine.get_candidate_paths(top_k=30)
    to merge OnlineDiscoveryEngine hypothesis candidates into BanditRouter
    proposals. This bridge is opt-in and never called in benchmark scripts.
```

The two subsystems operate on different time-scales and serve different purposes:
- `OnlineDiscoveryEngine`: transparent, hypothesis-level, suited to annual macro
- `MPIEOrchestrator`: event-driven, bandit-router, suited to high-frequency streaming

---

## Resolution

The discrepancy is structural, not an error:
1. All benchmark claim numbers come from `OnlineDiscoveryEngine` — reproducible.
2. `MPIEOrchestrator` is a production path, not yet exercised by any benchmark.
3. Stage 3.2 (HierarchicalFederation) is now separately tested in isolation.
4. The MetaController thresholds documented here (0.70 / 0.10) differ from the
   simulation gate (0.25) — both values are correct for their respective roles.

Stage 0 status: **RESOLVED — PASS**

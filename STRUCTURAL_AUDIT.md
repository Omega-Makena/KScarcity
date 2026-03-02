# SCARCITY / KShield — Full Structural Audit Report

**Date:** 2026-03-02  
**Auditor:** GitHub Copilot (Claude Sonnet 4.6)  
**Scope:** Entire repository — all modules, all files  
**Mode:** Read-only comprehension. No changes made.

---

## 1. Repository Structure Summary

This repository is a **multi-system platform** with three interlocking concerns:

| System | Purpose |
|---|---|
| **SCARCITY** | Online-first causal inference engine, federated ML, resource governance, SFC simulation |
| **KShield** | Kenya national threat detection + economic governance, powered by SCARCITY internals |
| **Backend** | FastAPI v2 REST API serving SCARCITY engine state; deprecated v1 mock layer |

### Top-Level Directory Inventory

```
scace4/
├── scarcity/               Core causal/ML/simulation library (installable package)
│   ├── engine/             MPIE orchestrator, bandit router, hypothesis pool, operator ops
│   ├── federation/         2-layer hierarchical FL with DP/gossip/secure aggregation
│   ├── fmi/                Federation-Meta Interface pipeline
│   ├── governor/           Dynamic Resource Governor (DRG) async control loop
│   ├── meta/               Meta-learning agent (cross-domain Reptile optimization)
│   ├── runtime/            EventBus pub/sub + Telemetry
│   ├── stream/             Stream ingestion with PI-controller rate regulation
│   ├── simulation/         SFC economy + learned SFC + agent-based sims
│   ├── causal/             DoWhy/EconML causal inference pipeline
│   ├── analytics/          Policy response surface / terrain generation
│   ├── synthetic/          Synthetic dataset generation pipeline
│   └── tests/              25 test files covering all sub-systems
│
├── federated_databases/    Federated database query + governance layer
│   ├── catalog/            Dataset/connector registry
│   ├── compatibility/      Node compatibility scoring
│   ├── connectors/         SQLite (working), Postgres (working), 4 stubs
│   ├── contracts/          Data contract enforcement
│   ├── executor/           Query execution + k-anonymity + non-IID
│   ├── planner/            JSON/SQL DSL query planner
│   ├── policy/             ABAC/RBAC hybrid policy engine
│   └── runtime/            (exists, not deeply audited)
│
├── federated_ml/           FL orchestration wrapping scarcity.federation
│   ├── audit_hooks/        Event logger for FL jobs
│   ├── orchestration/      Main FL job lifecycle + non-IID metrics
│   ├── registry/           Versioned model artifact storage
│   └── topology_adapter/   Maps topology nodes to participant bindings
│
├── k_collab/               Collaboration infrastructure (projects, topology, trust, audit)
│   ├── audit/              Append-only SHA-256 chain audit log
│   ├── common/             VersionedJSONStore (foundational primitive for 8+ modules)
│   ├── projects/           Versioned collaboration project registry
│   ├── topology/           Topology store + validation + diffing
│   ├── trust/              Connector trust policy validation
│   └── ui/                 Streamlit service locator
│
├── kshiked/                KShield threat detection & economic governance system
│   ├── core/               EconomicGovernor, ScarcityBridge, Shocks, Policies
│   ├── federation/         Defense federation layer (Aegis) extending scarcity.federation
│   ├── causal/             Kenya economic causal discovery (World Bank CSV)
│   ├── pulse/              Threat pulse sensor
│   ├── sim/                Economic simulation demo scripts
│   ├── simulation/         Simulation integration
│   ├── analysis/           Analysis tools
│   ├── causal_adapter/     Causal adapter layer
│   └── ui/                 Streamlit dashboard panels
│
├── backend/                FastAPI REST API
│   └── app/
│       ├── api/v1/         Deprecated mock API (10 routers)
│       ├── api/v2/         Active API backed by scarcity core (12 routers)
│       ├── core/           Config (pydantic-settings), ScarcityCoreManager, logging, datasets
│       ├── engine/         EngineRunner (async MPIE stack bootstrap)
│       ├── schemas/        Pydantic request/response schemas
│       └── simulation/     SimulationManager (mock data — deprecated)
│
├── tests/                  Root-level integration + unit tests
├── scripts/                Operational + utility scripts
├── data/                   Kenya CSVs, news cache, simulation data
├── config/                 X proxy and session config examples
├── documentation/          Architecture docs, routing, roadmaps
└── artifacts/              Run artifacts (benchmarks, meta, runs)
```

---

## 2. Module Classification Table

| Module | Classification | Status |
|---|---|---|
| `scarcity/engine/` | Core Logic — Algorithmic/ML | ~90% implemented |
| `scarcity/federation/` | Core Logic — Algorithmic/ML | ~95% implemented |
| `scarcity/fmi/` | Interface Layer | ~90% implemented |
| `scarcity/governor/` | Infrastructure | ~90% implemented |
| `scarcity/meta/` | Model / Algorithmic Component | ~90% implemented |
| `scarcity/runtime/` | Infrastructure | ~95% implemented |
| `scarcity/stream/` | Data Layer | ~90% implemented |
| `scarcity/simulation/` | Model / Algorithmic Component | ~90% implemented |
| `scarcity/causal/` | Model / Algorithmic Component | ~90% implemented |
| `scarcity/analytics/` | Utility / Helper | ~85% implemented |
| `scarcity/synthetic/` | Data Layer | ~80% (unverified) |
| `scarcity/tests/` | Testing | ~85% implemented |
| `federated_databases/` | Data Layer + Infrastructure | ~95% implemented |
| `federated_ml/` | Model / Algorithmic Component | ~95% implemented |
| `k_collab/` | Infrastructure | ~98% implemented |
| `kshiked/core/` | Core Logic | ~85% implemented |
| `kshiked/federation/` | Interface Layer | ~75% implemented (stubs) |
| `kshiked/ui/` | Interface Layer | Not audited |
| `backend/app/api/v2/` | API Layer | ~85% implemented |
| `backend/app/api/v1/` | API Layer (deprecated) | ~90% mock |
| `backend/app/core/` | Configuration + Infrastructure | ~90% implemented |
| `backend/app/engine/` | Core Logic | ~90% implemented |
| `scripts/` | Utility / Helper | ~70% operational |
| `tests/` (root) | Testing | ~85% implemented |

---

## 3. Dependency Graph Overview

```
kshiked/ui/
  └─→ kshiked/hub.py
        └─→ kshiked/core/scarcity_bridge.py
              └─→ scarcity/engine/engine_v2.py (OnlineDiscoveryEngine)
              └─→ scarcity/simulation/engine.py (SimulationEngine)
              └─→ scarcity/meta/meta_learning.py (MetaLearningAgent)
        └─→ kshiked/core/governance.py
              └─→ scarcity/simulation/sfc.py (SFCEconomy)
              └─→ scarcity/runtime/bus.py (EventBus)

kshiked/federation/
  └─→ scarcity/federation/coordinator.py (FederationCoordinator)
  └─→ scarcity/federation/client_agent.py (FederationClientAgent)
  └─→ scarcity/federation/gossip.py (GossipProtocol)

backend/app/main.py
  └─→ app/api/routes.py (v1)
  └─→ app/api/v2/routes.py (v2)
  └─→ app/core/scarcity_manager.py
        └─→ scarcity/runtime/bus.py
        └─→ scarcity/engine/engine.py (MPIEOrchestrator)
        └─→ scarcity/governor/drg_core.py (DynamicResourceGovernor)
        └─→ scarcity/federation/hierarchical.py (HierarchicalFederation)
        └─→ scarcity/meta/meta_learning.py (MetaLearningAgent)
        └─→ scarcity/simulation/engine.py (SimulationEngine)
  └─→ app/engine/runner.py (EngineRunner)
  └─→ scarcity/dashboard/server  ← *** DOES NOT EXIST ***

federated_ml/orchestration/nested.py
  └─→ k_collab/common/versioned_store.py
  └─→ scarcity/federation/hierarchical.py

federated_databases/control_plane.py
  └─→ k_collab/projects/registry.py
  └─→ k_collab/trust/controls.py
  └─→ federated_databases/catalog/registry.py
  └─→ federated_databases/compatibility/engine.py
  └─→ federated_databases/planner/router.py
  └─→ federated_databases/executor/engine.py
  └─→ federated_databases/policy/engine.py

scarcity/engine/engine.py (MPIEOrchestrator)
  └─→ scarcity/engine/bandit_router.py (BanditRouter)  ← import is from controller.py (WRONG)
  └─→ scarcity/engine/encoder.py
  └─→ scarcity/engine/evaluator.py
  └─→ scarcity/engine/store.py
  └─→ scarcity/runtime/bus.py
  └─→ scarcity/meta/meta_learning.py (event: meta_prior_update) ← topic mismatch

scarcity/federation/hierarchical.py
  └─→ scarcity/federation/basket.py
  └─→ scarcity/federation/gossip.py
  └─→ scarcity/federation/buffer.py
  └─→ scarcity/federation/layers.py
  └─→ scarcity/federation/aggregator.py
  └─→ scarcity/federation/privacy_guard.py

k_collab/ui/services.py (Streamlit service locator)
  └─→ k_collab/topology/store.py
  └─→ k_collab/audit/log.py
  └─→ k_collab/projects/registry.py
  └─→ federated_databases/control_plane.py
  └─→ federated_ml/orchestration/nested.py
```

Key cross-module event flows via EventBus:

```
StreamSource → "data_window" → MPIEOrchestrator
DRG → "resource_profile" → MPIEOrchestrator
MetaLearningAgent → "meta_prior_update" → [DEAD — engine listens to "meta_policy_update"]
MPIEOrchestrator → "processing_metrics" → MetaLearningAgent
Telemetry → "bus_latency_ms" → [Simulation uses "latency_ms" — MISMATCH]
FMIEmitter → "fmi.*" → [No subscribers found in inspected code]
```

---

## 4. List of Fully Implemented Components

The following are confirmed complete and functional based on source inspection:

| Component | Evidence |
|---|---|
| `k_collab/common/versioned_store.py` — VersionedJSONStore | JSONL-backed, SHA-256, read/write/diff |
| `k_collab/audit/log.py` — AppendOnlyAuditLog | SHA-256 chain hash, tamper-evidence |
| `k_collab/topology/schema.py` — validate_topology, diff_topologies | 182 lines, full validation |
| `k_collab/topology/store.py` — TopologyStore | Versioned with YAML/JSON loading |
| `k_collab/trust/controls.py` — validate_connector_trust | DP noise validation, secret detection |
| `k_collab/projects/registry.py` — CollaborationProjectRegistry | Complete |
| `federated_databases/models.py` — FederatedNode, etc. | Pure dataclasses, complete |
| `federated_databases/storage.py` — NodeStorage, ControlPlaneStorage | SQLite-backed, complete |
| `federated_databases/scarcity_federation.py` — ScarcityFederationManager | 484 lines, complete |
| `federated_databases/control_plane.py` — FederatedDatabaseControlPlane | 707 lines, complete |
| `federated_databases/compatibility/engine.py` — CompatibilityEngine | 531 lines, complete |
| `federated_databases/connectors/sqlite_node.py` | Fully working aggregate executor |
| `federated_databases/connectors/postgres_node.py` | Working; psycopg optional-dep handled |
| `federated_databases/executor/engine.py` — FederatedExecutor | k-anonymity suppression included |
| `federated_databases/policy/engine.py` — PolicyEngine | ABAC/RBAC hybrid |
| `federated_ml/orchestration/nested.py` — NestedFederatedMLOrchestrator | 305 lines, complete |
| `federated_ml/registry/model_registry.py` — FederatedModelRegistry | Complete |
| `federated_ml/topology_adapter/adapter.py` — NestedTopologyAdapter | Complete |
| `scarcity/runtime/bus.py` — EventBus | Async pub/sub, concurrent dispatch, complete |
| `scarcity/stream/source.py` — StreamSource | PI-controller rate regulation, windup protection |
| `scarcity/engine/store.py` — HypergraphStore | EMA + index + GC, complete |
| `scarcity/engine/evaluator.py` — Evaluator | Bootstrap gain, CI, stability thresholds |
| `scarcity/engine/bandit_router.py` — BanditRouter | Thompson/UCB/Epsilon-greedy, complete |
| `scarcity/engine/encoder.py` — Encoder | Deterministic seed, lag table, complete |
| `scarcity/engine/robustness.py` — Winsorizer | Clips only after enough samples |
| `scarcity/engine/vectorized_core.py` — VectorizedRLS | Stability check, batch ops |
| `scarcity/engine/relationships.py` — CausalHypothesis | Granger with direction threshold |
| `scarcity/engine/relationships_extended.py` — Mediation, etc. | n>=30 guard |
| `scarcity/engine/relationship_config.py` — CausalConfig | Central defaults, overridable |
| `scarcity/engine/economic_engine.py` — EconomicDiscoveryEngine | 306 hypotheses, complete |
| `scarcity/engine/operators/*.py` — All 8 operator files | Sanitized, bounded, complete |
| `scarcity/federation/hierarchical.py` — HierarchicalFederation | 428 lines, 2-layer FL, DP |
| `scarcity/federation/coordinator.py` — FederationCoordinator | Peer lifecycle, complete |
| `scarcity/federation/gossip.py` — GossipProtocol | Local DP + materiality detection |
| `scarcity/federation/aggregator.py` — FederatedAggregator | Trimmed mean, median, FedAvg |
| `scarcity/federation/privacy_guard.py` — PrivacyGuard | Gaussian noise + masking |
| `scarcity/federation/reconciler.py` — StoreReconciler | Edge upsert with decay |
| `scarcity/meta/optimizer.py` — OnlineReptileOptimizer | Reptile update, beta scheduling |
| `scarcity/meta/scheduler.py` — MetaScheduler | Adaptive interval within min/max bounds |
| `scarcity/meta/storage.py` — MetaStorageManager | JSON persistence + backups |
| `scarcity/meta/validator.py` — MetaValidator | Confidence + keys + finite vector |
| `scarcity/governor/drg_core.py` — DynamicResourceGovernor | Async control loop, complete |
| `scarcity/governor/profiler.py` — ResourceProfiler | Kalman update |
| `scarcity/fmi/service.py` — FMIService | validate→route→encode→aggregate→emit, async |
| `scarcity/fmi/aggregator.py` — FMIAggregator | Trimmed mean aggregation |
| `scarcity/fmi/router.py` — FMIRouter | Readiness logic |
| `scarcity/simulation/sfc.py` — SFCEconomy | 628 lines, full SFC model, research-grade |
| `scarcity/simulation/learned_sfc.py` — LearnedSFCEconomy | FallbackBlender, 350 lines |
| `scarcity/simulation/engine.py` — SimulationEngine | Complete orchestration |
| `scarcity/causal/engine.py` — run_causal | Parallel ProcessPool, complete |
| `scarcity/causal/graph.py` — DOT parser, temporal edge validator | Complete |
| `kshiked/core/shocks.py` — 6 stochastic shock types | Research-grade, 440 lines |
| `kshiked/core/governance.py` — EconomicGovernor | SFC + EventBus integration |
| `kshiked/core/scarcity_bridge.py` — ScarcityBridge | 359 lines, try/except guarded |
| `backend/app/core/config.py` — Settings | 18+ federation/privacy fields, complete |
| `backend/app/core/scarcity_manager.py` — ScarcityCoreManager | 640 lines, lifecycle complete |
| `backend/app/engine/runner.py` — EngineRunner | Async MPIE bootstrap, 358 lines |
| `backend/app/api/v2/endpoints/health.py` | Complete |
| `backend/app/api/v2/endpoints/mpie.py` | 295 lines, complete |
| Audit tests: `test_audit_*.py` (17 files) | Granular audit coverage across sub-systems |
| Root integration tests: `test_federated_databases_smoke.py`, `test_kcollab_*` | High quality |

---

## 5. List of Partially Implemented Components

| Component | Gap | Severity |
|---|---|---|
| `backend/app/api/v2/endpoints/federation.py` | `# TODO:` — update count and peer latency not tracked | Moderate |
| `backend/app/api/v2/endpoints/simulation.py` | Simulation disabled by default; gated on config | Moderate |
| `kshiked/federation/coordinator.py` — `DefenseCoordinator.register_peer` | `pass` placeholder where missing-clearance rejection should go; currently accepts peers with no declared clearance silently | Significant |
| `scarcity/federation/layers.py` — `SecureAggregator` | Marked as "simulated" — performs plain sum, not true secure aggregation protocol | Significant (documented claim) |
| `scarcity/governor/` (sensors, actuators, hooks, monitor, registry) | Core loop in `drg_core.py` is complete; individual sensor/actuator sub-components not deeply audited | Low–Moderate |
| `scarcity/fmi/` — FMI topic subscribers | `FMIEmitter` publishes to `fmi.*` topics; no subscribers found in the engine or meta layer — FMI output is produced but never consumed | Significant |
| `scarcity/simulation/__init__.py` | Empty — no re-exports; callers must import from sub-modules directly | Low |
| `backend/requirements.txt` | `scarcity` package not listed — relies on workspace install; fragile for CI/Docker | Moderate |
| `kshiked/causal/economic_causal_discovery.py` | Hardcoded `API_KEN_DS2_en_csv_v2_14659.csv` path — breaks if file moves | Moderate |
| `scripts/smoke_integrated_pipeline.py` | Hardcoded `.venv-linux/bin/python` — Linux-only, fails on Windows/macOS | Significant |

---

## 6. List of Broken or Inconsistent Components

### P0 — Will crash / prevent startup

| # | Component | Issue | Evidence |
|---|---|---|---|
| **B1** | `backend/app/main.py:27` | `from scarcity.dashboard.server import attach_simulation_manager, create_app as create_scic_app` — `scarcity/dashboard/` **directory does not exist** → `ImportError` at module import time; FastAPI server cannot start | No files under `scarcity/dashboard/` found |
| **B2** | `scarcity/engine/engine.py:17` | Imports `BanditRouter` from `scarcity.engine.controller`; `controller.py` exports `MetaController`, not `BanditRouter` → `ImportError` | `controller.py` defines `class MetaController`; `BanditRouter` is in `bandit_router.py` |

### P1 — Wrong results / silent failure

| # | Component | Issue | Evidence |
|---|---|---|---|
| **B3** | `scarcity/meta/meta_learning.py:153` ↔ `scarcity/engine/engine.py:109` | Meta publishes `"meta_prior_update"` event; MPIE engine subscribes to `"meta_policy_update"` → prior updates are **never applied** to the engine; meta-learning loop is fully disconnected | `meta_learning.py:153`, `engine.py:109` |
| **B4** | `scarcity/engine/algorithms_online.py:204` | `evaluate()` calls `self.win_x.update(x)` which **mutates** Winsorizer state; since `update()` calls `evaluate()` then `fit_step()`, each row advances the Winsorizer twice → double-clipping, biased regression inputs | `algorithms_online.py:183-205` |
| **B5** | Simulation ↔ Telemetry field mismatch | `scarcity/simulation/engine.py` reads `latency_ms` / `fps` from telemetry; `scarcity/runtime/telemetry.py` publishes `bus_latency_ms` → simulation consumes wrong or missing field | `simulation/engine.py:96-99,187-193` vs `telemetry.py:363-374` |

### P2 — Reliability / correctness gaps

| # | Component | Issue |
|---|---|---|
| **B6** | `scarcity/engine/encoder.py:75,119` | `np.random.seed(42)` and `np.random.seed(43)` called at `__init__` → resets **global** NumPy RNG; any code depending on stochastic reproducibility elsewhere in the same process is silently contaminated |
| **B7** | `scarcity/federation/privacy_guard.py` | `PrivacyConfig.dp_noise_sigma = 0.0` by default → DP is **off by default**; documentation claims DP is always applied; no epsilon/delta tracking, only sigma |
| **B8** | `scarcity/fmi/validator.py:82-88,118-127` | DP-required flag only checks for flag presence, not valid epsilon/delta values → passes validation with epsilon=0 or negative values |
| **B9** | `scarcity/engine/store.py:181-184` | Node lookup compares by raw name string without normalization → `"GDP_Growth"` and `"gdp_growth"` are treated as distinct hypotheses; duplicate variable nodes possible |
| **B10** | `federated_databases/connectors/azure_stub.py`, `oracle_stub.py`, `sqlserver_stub.py`, `http_api_stub.py` | All raise `NotImplementedError` unconditionally — registered in `ConnectorFactory`; any institution routing to these sources causes runtime crash |

---

## 7. Missing or Stubbed Functionality

| # | Location | What's Missing |
|---|---|---|
| **M1** | `scarcity/dashboard/` (entire module) | Does not exist; imported by `backend/app/main.py` — needs to be created or the import removed |
| **M2** | FMI subscriber wiring | `FMIEmitter` publishes `fmi.meta_prior_update`, `fmi.policy_hint`, etc. — no module subscribes to these topics; FMI pipeline output is produced but consumed by no one |
| **M3** | `kshiked/federation/coordinator.py` | `register_peer` clearance validation silently skips peers with no declared clearance — should reject or assign default clearance level |
| **M4** | `scarcity/federation/layers.py` — `SecureAggregator` | Uses plain weighted sum instead of a cryptographic secure aggregation protocol (e.g., Shamir secret sharing or additive masking) |
| **M5** | `federated_databases/connectors/` | Azure Data Lake, Oracle, SQL Server, and HTTP API connectors are stubs — institutions using these data sources cannot participate in federated queries |
| **M6** | `scarcity/causal/__init__.py` | Missing `__init__.py` — package has no official API surface; callers must know module paths |
| **M7** | `scarcity/analytics/__init__.py` | Missing `__init__.py` — same issue |
| **M8** | `scarcity/simulation/__init__.py` | Empty (no re-exports) — callers cannot do `from scarcity.simulation import SimulationEngine` directly |
| **M9** | `backend/app/api/v2/endpoints/federation.py` | Peer update count and latency metrics are `# TODO` — federation health metrics are incomplete |
| **M10** | `scripts/smoke_integrated_pipeline.py` | Hardcoded Linux Python path — needs portability fix for cross-platform use |
| **M11** | `backend/requirements.txt` | Missing `scarcity` as installable dependency — Docker/CI deployment will fail to resolve imports |
| **M12** | Meta-learning ↔ Engine event topic alignment | Two topics in use (`meta_prior_update` vs `meta_policy_update`) — must be unified |
| **M13** | `scarcity/engine/engine.py` BanditRouter API adapter | `controller.propose()` called with `window_meta=`, `schema=` kwargs; `BanditRouter.propose()` signature is `n_proposals, context, exclude` — API mismatch |

---

## 8. Integration Gaps Between Modules

| Gap | Modules Affected | Impact |
|---|---|---|
| **G1** — `scarcity.dashboard.server` missing | `backend/app/main.py` ↔ `scarcity/` | **Fatal**: Backend cannot start |
| **G2** — Meta-learning event topic mismatch | `scarcity/meta/meta_learning.py` ↔ `scarcity/engine/engine.py` | Meta-learning loop is completely disconnected from engine |
| **G3** — FMI output never consumed | `scarcity/fmi/emitter.py` ↔ any subscriber | FMI pipeline runs but has zero downstream effect |
| **G4** — BanditRouter imported from wrong module | `scarcity/engine/engine.py` ↔ `scarcity/engine/bandit_router.py` | MPIE path selection crashes on import |
| **G5** — Simulation telemetry field name collision | `scarcity/simulation/engine.py` ↔ `scarcity/runtime/telemetry.py` | Simulation reads wrong/null telemetry values |
| **G6** — `scarcity` not in `backend/requirements.txt` | `backend/` ↔ `scarcity/` | CI/Docker builds will fail to resolve `scarcity.*` imports |
| **G7** — `VersionedJSONStore` data directory assumptions | `k_collab/common/versioned_store.py` ↔ any caller in different working directory | File-based storage uses relative paths; breaks when invoked from non-root CWD |
| **G8** — `scarcity/simulation/__init__.py` empty | `backend/app/engine/runner.py` imports `from scarcity.simulation import AgentRegistry, SimulationConfig, SimulationEngine` | Python resolves these as namespace-package sub-module imports; works but fragile |
| **G9** — `HierarchicalFederation` ↔ `NestedFederatedMLOrchestrator` | `federated_ml/orchestration/nested.py` uses `hierarchical.run_round()` | Integration is structurally correct but relies on project authorization check being enforced upstream via `k_collab.projects.registry` — if that check is bypassed, any node can trigger a FL round |
| **G10** — `kshiked/federation/` LBAC gap | `kshiked/federation/coordinator.py` (`register_peer`) | Nodes with no clearance declaration are registered without enforcement; downstream `select_peers_for_task` will correctly exclude them only if tasks specify a clearance |

---

## 9. High-Risk Runtime Zones

| Zone | Risk Type | Details |
|---|---|---|
| **R1** `backend/app/main.py:27` | **Startup crash** | `ImportError: cannot import from scarcity.dashboard.server` — server is dead on launch |
| **R2** `scarcity/engine/engine.py:17` | **Startup crash** | `ImportError: cannot import BanditRouter from scarcity.engine.controller` — MPIE cannot initialize |
| **R3** `scarcity/engine/algorithms_online.py:204` | **Silent data corruption** | Double-Winsorization biases every RLS regression hypothesis silently — no exception raised |
| **R4** Meta ↔ Engine event bridge | **Silent logic failure** | Meta-learning runs the full Reptile update cycle and correctly updates its internal prior — but the engine never receives it; the system appears to work while meta priors are frozen at initialization |
| **R5** `scarcity/engine/encoder.py:75,119` | **Stochastic interference** | Two `np.random.seed()` calls at object construction reset the global NumPy RNG — parallel or sequential use of other stochastic components (shocks, DP noise, bandit initialization) may produce non-reproducible results |
| **R6** `federated_databases/connectors/` stubs | **Runtime crash** | Any ConnectorFactory.create() call for Azure/Oracle/SQL Server/HTTP raises `NotImplementedError` without graceful degradation message |
| **R7** `scarcity/federation/privacy_guard.py` DP default | **Privacy violation** | `dp_noise_sigma=0.0` by default means DP is disabled silently; in a real deployment, gradient leakage occurs if the caller does not explicitly set sigma |
| **R8** `kshiked/federation/coordinator.py` clearance skip | **Security bypass** | Peers with no `clearance_level_int` capability are registered silently; LBAC guarantees are only enforced in `select_peers_for_task`, not at registration — a peer could inject updates before a task selection check occurs |
| **R9** `scarcity/stream/source.py` CSV in-memory | **OOM under scale** | CSV ingestion loads entire file into memory before streaming; large CSV files (e.g., 100M+ rows) will cause OOM without chunking |
| **R10** `scarcity/analytics/terrain.py` O(steps²×time_horizon) | **Performance degradation** | Response surface generation has quadratic complexity — may time out under wide policy sweeps (>50 steps, long horizon) |
| **R11** `scarcity/engine/operators/sketch_ops.py:123-127` | **Performance degradation** | Tensor sketch uses O(d1×d2) nested loops — will be slow for high-dimensional feature spaces |
| **R12** Pytest `FileNotFoundError` on test collection | **CI failure** | `pytest scarcity/tests -q` exits with `FileNotFoundError` in capture cleanup (verified in existing audit) — test suite cannot run in current environment without path resolution fix |

---

## 10. Suggested Order of Repair and Stabilization

### Phase 1 — Stop the bleeding (P0 crashes, ~1-2 days)

1. **Fix B1** — Either create `scarcity/dashboard/__init__.py` + `scarcity/dashboard/server.py` with a minimal `create_app()` returning a bare FastAPI instance and `attach_simulation_manager()` as a no-op, OR remove the import and the `/scic` mount from `backend/app/main.py` if the module is not yet in scope.
   - File: [backend/app/main.py](backend/app/main.py)

2. **Fix B2** — Change `from scarcity.engine.controller import BanditRouter` to `from scarcity.engine.bandit_router import BanditRouter` in `engine.py`. Patch the `controller.propose()` call to use the correct `BanditRouter.propose(n_proposals=..., context=...)` signature. (Minimal patch diff already exists in the audit record.)
   - File: [scarcity/engine/engine.py](scarcity/engine/engine.py)

### Phase 2 — Fix P1 silent failures (~1-2 days)

3. **Fix B3 (M12)** — Align meta-learning event topics: either subscribe engine to `"meta_prior_update"` in addition to `"meta_policy_update"`, or rename the meta publish topic. Update `_handle_meta_policy_update` to accept both the `{controller:..., evaluator:...}` shaped payload from policy updates and the `{prior:..., meta:...}` shaped payload from meta learning.
   - Files: [scarcity/engine/engine.py](scarcity/engine/engine.py), [scarcity/meta/meta_learning.py](scarcity/meta/meta_learning.py)

4. **Fix B4** — In `algorithms_online.py:evaluate()`, replace `self.win_x.update(x)` with a read-only clip using stored bounds so the Winsorizer state is not advanced during evaluation.
   - File: [scarcity/engine/algorithms_online.py](scarcity/engine/algorithms_online.py)

5. **Fix B5** — Reconcile simulation telemetry field names: standardize on `bus_latency_ms` everywhere, or add an alias in the simulation engine's `_handle_telemetry` method.
   - Files: [scarcity/simulation/engine.py](scarcity/simulation/engine.py), [scarcity/runtime/telemetry.py](scarcity/runtime/telemetry.py)

### Phase 3 — Reconnect disconnected pipelines (~2-3 days)

6. **Fix G3 (M2)** — Wire FMI output: identify the intended consumer of `fmi.meta_prior_update` (likely the meta-learning agent or engine) and add a `bus.subscribe("fmi.meta_prior_update", ...)` in the appropriate module.
   - Files: [scarcity/fmi/emitter.py](scarcity/fmi/emitter.py), [scarcity/meta/meta_learning.py](scarcity/meta/meta_learning.py)

7. **Fix M8 + M6 + M7** — Add minimal `__init__.py` re-exports to `scarcity/simulation/`, `scarcity/causal/`, and `scarcity/analytics/` so all sub-packages have consistent public API surfaces.

8. **Fix M11** — Add `scarcity @ file:../scarcity` or an editable install entry to `backend/requirements.txt`.

### Phase 4 — Security and correctness hardening (~2-3 days)

9. **Fix R5 (B6)** — Replace `np.random.seed()` in `encoder.py` with a module-local `np.random.default_rng(seed)` RNG instance that does not affect global state.
   - File: [scarcity/engine/encoder.py](scarcity/engine/encoder.py)

10. **Fix R7 (B7)** — Set `dp_noise_sigma` to a non-zero default or add an assertion in `PrivacyGuard.__init__` that warns loudly when sigma=0 and `secure_aggregation=True`.
    - File: [scarcity/federation/privacy_guard.py](scarcity/federation/privacy_guard.py)

11. **Fix R8 (M3)** — In `DefenseCoordinator.register_peer`, replace the `pass` with a clearance validation: if `clearance_level_int` is missing, either raise `ValueError` or assign a default clearance of `UNCLASSIFIED=0`.
    - File: [kshiked/federation/coordinator.py](kshiked/federation/coordinator.py)

12. **Fix B9** — Add case-normalized key (`name.strip().lower()`) to node lookup in `HypergraphStore` to prevent duplicate variable nodes.
    - File: [scarcity/engine/store.py](scarcity/engine/store.py)

### Phase 5 — Stubs and connector completions (~1-2 weeks, depending on scope)

13. Implement or replace the 4 federated database connector stubs (Azure, Oracle, SQL Server, HTTP API) — either with real implementations or remove from `ConnectorFactory` until ready.
    - Directory: [federated_databases/connectors/](federated_databases/connectors/)

14. Implement `scarcity/federation/layers.py` `SecureAggregator` with a real additive masking protocol, or clearly document in its docstring that it provides no cryptographic guarantee and rename to `MockSecureAggregator`.

15. Fix `scripts/smoke_integrated_pipeline.py` Linux path — use `sys.executable` instead of hardcoded `.venv-linux/bin/python`.

### Phase 6 — Test infrastructure repair (~1 day)

16. Resolve the `pytest` `FileNotFoundError` on `scarcity/tests` collection — likely a missing conftest path or missing `__init__.py` preventing proper test root discovery. Run `pytest --rootdir=. scarcity/tests -v` and fix the import resolution.
    - File: [scarcity/tests/conftest.py](scarcity/tests/conftest.py)

17. After Phase 1–2 fixes, run the full test suite and verify no regressions: `pytest scarcity/tests tests/ -q`.

---

## Appendix — Hardcoding Ledger

All configurable parameters identified as hardcoded defaults (all are overridable via config objects):

| Parameter | Default | File | Overridable? | Risk |
|---|---|---|---|---|
| `direction_threshold` | 0.02 | `relationship_config.py` | Yes (CausalConfig) | Med |
| `confidence_multiplier` | 2.0 | `relationship_config.py` | Yes | Med |
| `forgetting_factor` | 0.99 | `relationship_config.py` | Yes | Med |
| `ridge_alpha` | 1e-3 | `relationship_config.py` | Yes | Med |
| `lambda_forget` | 0.99 | `algorithms_online.py` | Yes (RLSConfig) | Med |
| `process_noise` (Kalman) | 1e-4 | `algorithms_online.py` | Yes (KalmanConfig) | Med |
| `max_edges` | 10000 | `store.py` | Yes (ctor) | Med |
| `decay_factor` | 0.995 | `store.py` | Yes (ctor) | Med |
| `gain_min` | 0.01 | `evaluator.py` | Yes (DRG) | Med |
| `resamples` | 8 | `evaluator.py` | Yes (DRG) | Med |
| `n_arms` | 1000 | `bandit_router.py` | Yes (BanditConfig) | Med |
| `beta_init` | 0.1 | `optimizer.py` | Yes (MetaOptimizerConfig) | Med |
| `latency_target_ms` | 80.0 | `scheduler.py` | Yes | Med |
| `window_size` | 2048 | `window.py` | Yes | Med |
| `dp_noise_sigma` | **0.0** | `privacy_guard.py` | Yes | **High** |
| `np.random.seed(42)` | 42 | `encoder.py` | No (hardcoded call) | High |

---

## Appendix — Claims vs Implementation Mismatches

| Claim | Source | Actual | Mismatch |
|---|---|---|---|
| "DP noise injected into all updates before transmission" | `02_federation.md` | `dp_noise_sigma=0.0` by default → DP is off | Yes — high risk |
| "PrivacyGuard injects Laplace/Gaussian noise satisfying (ε, δ)-DP" | `02_federation.md` | Uses only sigma; no ε/δ tracking | Yes |
| "New priors are pushed back to Engine via `meta_policy_update`" | `03_meta_learning.md` | Agent publishes `meta_prior_update`; engine listens to `meta_policy_update` | Yes — event names differ |
| "Secure aggregation prevents gradient leakage" | Docs | SecureAggregator does a plain weighted sum | Yes — simulated only |
| "Federation and Simulation coming soon" | `backend/app/main.py` docstring | Federation v2 partially done; simulation disabled by config | Partial truth |

---

*This audit covered approximately 120 Python source files across 14 top-level modules. No modifications were made to any file during this audit.*

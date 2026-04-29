# Architecture Diagrams — K-Scarcity Core Subsystems

> Accurate diagrams derived from source code. All class names, method signatures,
> and event topics match the current implementation.

---

## Table of Contents

1. [Meta-Learning Layer](#1-meta-learning-layer)
2. [Online Learning Engine (MPIE)](#2-online-learning-engine-mpie)
3. [Simulation Engine](#3-simulation-engine)
4. [Dynamic Resource Governor (DRG)](#4-dynamic-resource-governor-drg)
5. [Cross-Subsystem Interaction](#5-cross-subsystem-interaction)
6. [Federated Learning (Gossip + Security Hardening)](#6-federated-learning-gossip--security-hardening)

---

## 1. Meta-Learning Layer

### 1.1 Component Map

```
scarcity/meta/
├── meta_learning.py       MetaLearningAgent      — top-level orchestrator
├── domain_meta.py         DomainMetaLearner      — per-domain EMA + delta tracking
├── cross_meta.py          CrossDomainMetaAggregator — trimmed-mean / median fusion
│                          CrossDomainMetaLearner    — memory-backed blend (Phase 5b)
│                          CrossDomainMetaLearnerConfig
├── domain_server_meta.py  DomainServerMeta       — federation→meta bridge (Phase 5a)
│                          DomainServerMetaConfig
├── optimizer.py           OnlineReptileOptimizer — EMA prior update + rollback
├── scheduler.py           MetaScheduler          — window-count gated, adaptive interval
├── validator.py           MetaPacketValidator    — confidence + finiteness guards
├── storage.py             MetaStorageManager     — JSON persistence, versioned backups
├── encoder.py             ContextEncoder         — context embedding
├── memory.py              EpisodicMemory         — local adaptation episode buffer
├── adaptation.py          AdaptationEngine       — context-driven parameter retrieval
├── integrative_meta.py    MetaIntegrativeLayer   — rule-based hyperparameter governance
│                          MetaSupervisor         — EventBus bridge to MetaIntegrativeLayer
├── integrative_config.py  IntegrativeMetaConfig  — typed config dataclass tree
├── telemetry_hooks.py     build_meta_metrics_snapshot / publish_meta_metrics
└── __init__.py            package export surface (__all__), version metadata
```

### 1.2 Internal Data Flow

```
                          EventBus: "federation.policy_pack"
                                        │
                                        ▼
                          ┌─────────────────────────┐
                          │   MetaLearningAgent      │
                          │   .start() / .stop()     │
                          └──────────┬──────────────┘
                                     │ _handle_policy_pack()
                                     ▼
                          ┌─────────────────────────┐
                          │   DomainMetaLearner      │
                          │   .observe(domain_id,    │
                          │     metrics, params)     │
                          │                          │
                          │  Per-domain state:       │
                          │  · ema_score (EMA)       │
                          │  · confidence ∈ [0,1]    │
                          │  · history (last 20)     │
                          │                          │
                          │  meta_lr = lr_min +      │
                          │   (lr_max−lr_min)×conf   │
                          │  delta = meta_lr ×       │
                          │   (params − prev_params) │
                          │                          │
                          │  sign_agreement =        │
                          │   sign(score_delta) ==   │
                          │   sign(confidence)       │
                          │  if sign_agreement and   │
                          │   score_delta > 0:       │
                          │   confidence += 0.05     │
                          │  confidence clipped      │
                          │   to [0, 1]              │
                          └────────┬────────────────┘
                                   │ DomainMetaUpdate
                                   │ {vector, keys,
                                   │  confidence,
                                   │  score_delta}
                                   ▼
                          ┌────────────────────────────┐
                          │   MetaPacketValidator       │
                          │   .validate_update(update)  │
                          │                             │
                          │  Rejects if:                │
                          │  · confidence < 0.1         │
                          │  · |score_delta| > 1.0      │
                          │  · vector has non-finite    │
                          │  · len(keys) > 32           │
                          └────────┬───────────────────┘
                                   │ valid updates →
                                   │ _pending_updates[domain_id]
                                   │
                          EventBus: "processing_metrics"
                                   │
                                   ▼
                          ┌─────────────────────────────────┐
                          │   MetaScheduler                  │
                          │   .record_window()               │
                          │   .should_update(telemetry)      │
                          │                                  │
                          │  adaptive interval:              │
                          │  · latency > 80ms → slower       │
                          │  · latency < 56ms → faster       │
                          │  · bandwidth_low  → slower       │
                          │  range: [min_iv, max_iv]         │
                          └────────┬────────────────────────┘
                                   │ gate: window_counter ≥ interval
                                   ▼
                          ┌─────────────────────────────────┐
                          │   CrossDomainMetaAggregator      │
                          │   .aggregate(pending_updates[])  │
                          │                                  │
                          │  0. pre-filter updates where     │
                          │     confidence >= 0.05 and       │
                          │     len(vector) > 0              │
                          │  1. union all keys               │
                          │  2. zero-pad mismatched dims     │
                          │  3. stack → (N_domains × K) mat  │
                          │  4. trimmed_mean(alpha=0.1)      │
                          │     OR median                    │
                          │                                  │
                          │  → (agg_vector, keys, meta)      │
                          └────────┬────────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────────────────────┐
                          │   OnlineReptileOptimizer             │
                          │   .apply(agg_vector, keys,           │
                          │          reward, drg_profile)        │
                          │                                      │
                          │  · _update_beta(drg_profile):        │
                          │    vram/latency high → β × 0.8       │
                          │    bandwidth free   → β × 1.1        │
                          │    clamp to [β_init×0.5, β_max]      │
                          │                                      │
                          │  · _record_history() → backup stack  │
                          │                                      │
                          │  · prior += β × agg_vector           │
                          │                                      │
                          │  · reward_ema = (1−α)×ema + α×r      │
                          │                                      │
                          │  .should_rollback(reward):           │
                          │    ema − reward > 0.1 → True         │
                          │  .rollback() → restore prior[-1]     │
                          └────────┬────────────────────────────┘
                                   │ flat prior dict
                                   │ {tau, gamma_diversity,
                                   │  g_min, lambda_ci, ...}
                                   ▼
                          ┌─────────────────────────────────────┐
                          │   _structure_prior()                 │
                          │                                      │
                          │  CONTROLLER_KEYS = {tau,             │
                          │                    gamma_diversity}  │
                          │  EVALUATOR_KEYS  = {g_min,           │
                          │                    lambda_ci}        │
                          │                                      │
                          │  → {controller: {...},               │
                          │     evaluator:  {...}}               │
                          └────────┬────────────────────────────┘
                                   │
                    ┌──────────────┼───────────────────────┐
                    │              │                        │
                    ▼              ▼                        ▼
             MetaStorageManager   EventBus:          EventBus:
             .save_prior()        "meta_prior_update" "meta_update"
             JSON + versioned     → engine applies    → raw prior
             backup (ns stamp)      controller &       for debug
                                    evaluator updates
```

### 1.3 MetaIntegrativeLayer (Parallel Governance Path)

```
MetaSupervisor.start()
        subscribe("processing_metrics", _handle_processing_metrics)
        subscribe("telemetry", _handle_telemetry)
        subscribe("meta_rollback_active", _handle_meta_rollback_active)

EventBus: "processing_metrics" -> _handle_processing_metrics(data)
        · cache last_processing
        · track low_accept_windows: accept_rate < 0.03 increments
        · trigger _maybe_update()

EventBus: "telemetry" -> _handle_telemetry(data)
        · cache last_telemetry (bus/gpu metrics)

EventBus: "meta_rollback_active" -> _handle_meta_rollback_active(...)
        · _rollback_suppression_cycles = 2

MetaSupervisor._maybe_update()
        · if not running or no processing metrics: return
        · if suppression_cycles > 0: decrement and return
        · meta_input = _build_meta_input()
        · outputs = MetaIntegrativeLayer.update(meta_input)

MetaSupervisor._build_meta_input()
        · latency_ms = engine_latency_ms OR bus_latency_ms fallback
        · vram_util from gpu_memory_util OR vram_used/total OR vram_util
        · accept_low_windows from rolling counter
        · gain_prev from prev_gain_p50 or current gain_p50
        · defaults for ci_width_target, stability_avg, rcl_contrast, oom_flag

MetaIntegrativeLayer.update(telemetry)
        ├─ _compute_reward()   ← typed MetaScoreConfig weights
        │  accept × 0.35 + stability × 0.25 + contrast × 0.10
        │  − latency_norm × 0.15 − vram × 0.10 − oom × 0.20
        │  clipped to [−1, +1]
        │
        ├─ _update_ema(reward) → ema_reward
        │
        ├─ _apply_policies(telemetry, reward, ema_reward)
        │  · Controller knobs: tau ∈ [0.5,1.2], gamma ∈ [0.1,0.5]
        │  · Evaluator knobs: g_min ∈ [0.006,0.02], lambda_ci ∈ [0.4,0.6]
        │  · Operator tiers: tier2_enabled, tier3_topk
        │  · cooldown map gates repeated knob updates (5 cycles)
        │
        ├─ _resource_policy(telemetry)
        │  · vram > 0.85 or oom → n_paths_delta −15%, sketch_dim target
        │  · vram < 0.55 + latency < 100ms → n_paths_delta +10%, resamples +2
        │
        └─ _safety_checks(reward, ema, prev_snapshot, changed_knobs)
                 · ema drop > 0.1 → _rollback_previous(prev_snapshot)
                 · rollback_count++, logger.warning

MetaSupervisor._apply_resource_hint(resource_profile_hint)
        · n_paths = clamp(current * (1 + n_paths_delta), 1, n_paths_max)
        · apply sketch_dim_target and resamples_target when provided
        · publish "resource_profile" only when profile actually changes

Published events
        · "meta_policy_update" -> engine
        · "resource_profile"   -> DRG / engine (on profile change)
        · "meta_metrics"       -> telemetry/dashboards
```

### 1.4 Rollback Coordination

```
MetaLearningAgent                    MetaSupervisor
      │                                    │
      │  reward drops > rollback_delta     │
      │  optimizer.rollback()              │
      │                                    │
      ├─► EventBus: "meta_rollback_active" │
      │              │                     │
      │              └────────────────────►│
      │                                    │ _handle_meta_rollback_active()
      │                                    │ _rollback_suppression_cycles = 2
      │                                    │
      │                                    │ next 2 processing_metrics cycles
      │                                    │ → _maybe_update() returns early
      │                                    │ → MetaIntegrativeLayer.update()
      │                                    │   NOT called
      │                                    │   (prevents double-rollback)
```

### 1.5 Typed Config and Runtime State

```
IntegrativeMetaConfig dataclass tree
├── MetaScoreConfig
├── ControllerPolicyConfig
├── EvaluatorPolicyConfig
├── DRGPolicyConfig
└── SafetyConfig

MetaState core runtime fields
├── tau, gamma_diversity, g_min, lambda_ci
├── tier2_enabled, tier3_topk
├── cooldowns (per-knob lockout counters)
└── decision_count, rollback_count

MetaSupervisor runtime fields
├── low_accept_windows
├── prev_gain_p50
└── _rollback_suppression_cycles
```

---

## 2. Online Learning Engine (MPIE)

### 2.1 Component Map

```
scarcity/engine/
├── engine.py          MPIEOrchestrator   — pipeline coordinator
├── bandit_router.py   BanditRouter       — Thompson Sampling path selection
├── encoder.py         Encoder            — feature extraction + sketching
├── evaluator.py       Evaluator          — bootstrap R² gain + CI bounds
├── store.py           HypergraphStore    — edge persistence + decay
├── exporter.py        Exporter           — insight broadcast
├── discovery.py       Hypothesis         — relational hypothesis base class
│                      HypothesisPool     — hypothesis lifecycle management
├── types.py           Candidate/EvalResult/Reward — runtime contracts
├── controller.py      MetaController     — state machine for hypothesis lifecycle
├── arbitration.py     HypothesisArbiter  — conflict resolution between hypotheses
├── resource_profile.py                  — default profile dict
└── __init__.py        OnlineDiscoveryEngine export surface (Engine alias + __all__)
```

### 2.2 Pipeline: One Data Window

```
EventBus: "data_window"
  {data: np.ndarray[T×V], schema: {fields: {name, domain}}, window_id}
           │
           ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  MPIEOrchestrator._handle_data_window()                          │
  └───────────────────────────────────┬──────────────────────────────┘
                                      │
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 1 — Propose paths                                │
          │                                                       │
          │  BanditRouter.propose(n_proposals, context)           │
          │  → List[Candidate]                                    │
          │                                                       │
          │  Per arm: ArmStats{α, β, observations}                │
          │  Thompson Sampling: sample Beta(α, β) per arm         │
          │  Select top-N by sample score                         │
          │  Apply diversity penalty (gamma_diversity × overlap)  │
          │  Apply depth/domain exploration bias (tau)            │
          │  Runtime contract guard:                              │
          │  if proposals are non-Candidate, skip window          │
          └───────────────────────────┬──────────────────────────┘
                                      │ List[Candidate]
                                      │ {path_id, vars[], lags[],
                                      │  ops[], root, depth, domain}
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 2 — Normalize runtime input                      │
          │                                                       │
          │  window_tensor = data.get("data")                    │
          │  · if missing: log warning and skip window            │
          │  · list -> np.ndarray conversion when needed          │
          │  · schema_obj = data.get("schema", {}) or {}         │
          │  · var_names = _resolve_var_names(schema_obj, width)  │
          └───────────────────────────┬──────────────────────────┘
                                      │ normalized window_tensor + var_names
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 3 — Score                                        │
          │                                                       │
          │  Evaluator.score(window_tensor, candidates)           │
          │  → List[EvalResult]                                   │
          │                                                       │
          │  Per candidate:                                       │
          │  · _build_design_matrix(window, candidate) → (X, y)  │
          │  · _bootstrap_gain(X, y, holdout=resamples) → R²gain  │
          │  · EMA baseline comparison → gain = R²_model − R²_base│
          │  · CI bounds via bootstrap distribution               │
          │  · _compute_stability(gain, history[]) → ∈ [0,1]     │
          │                                                       │
          │  Accept if:                                           │
          │    gain ≥ gain_min (g_min)                            │
          │    AND ci_lo > 0 (signal above noise)                 │
          │    AND stability ≥ stability_min                      │
          └───────────────────────────┬──────────────────────────┘
                                      │ List[EvalResult]
                                      │ {gain, ci_lo, ci_hi,
                                      │  stability, accepted,
                                      │  cost_ms}
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 4 — Shape rewards                                │
          │                                                       │
          │  Evaluator.make_rewards(results, D_lookup)            │
          │  → List[Reward]                                       │
          │                                                       │
          │  Base reward ∈ [−1, +1] from gain                    │
          │  + diversity bonus (BanditRouter.diversity_score)     │
          │  − depth penalty (deeper paths cost more)             │
          │  + stability bonus if stability ↑                     │
          └───────────────────────────┬──────────────────────────┘
                                      │ List[Reward]
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 5 — Update bandit                                │
          │                                                       │
          │  BanditRouter.update(arm_id, reward)                  │
          │  · success = (reward > 0.5)                           │
          │  · success → α += 1 (win)                             │
          │  · failure → β += 1 (loss)                            │
          │  · decay() every window: α, β × 0.999                 │
          │    (non-stationary environment adaptation)            │
          └───────────────────────────┬──────────────────────────┘
                                      │
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 6 — Persist accepted edges                       │
          │                                                       │
          │  HypergraphStore.update_edges(store_payloads)         │
          │                                                       │
          │  EdgeRec: {src, tgt, op_type, weight, stability,      │
          │            ci_lo, ci_hi, regime, timestamp}           │
          │                                                       │
          │  · exponential weight decay per window                │
          │  · GC: prune edges below weight_floor                 │
          │  · bounded capacity (max_edges param)                 │
          └───────────────────────────┬──────────────────────────┘
                                      │
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 7 — Publish insight + exporter cadence           │
          │                                                       │
          │  After store.update_edges(...), MPIE publishes        │
          │  EventBus: "engine.insight" when accepted edges exist │
          │                                                       │
          │  Exporter.emit_insights(...):                         │
          │  · publishes "engine.insight" every window            │
          │  · publishes "inference.path_pack" every              │
          │    export_interval windows (batched edges)            │
          │  · lazy bus resolution (get_bus()) avoids circular    │
          │    imports at construction time                       │
          └───────────────────────────┬──────────────────────────┘
                                      │
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 8 — Publish metrics                              │
          │                                                       │
          │  EventBus: "processing_metrics"                       │
          │  {engine_latency_ms, n_candidates, accepted_count,    │
          │   accept_rate, edges_active, oom_flag,                │
          │   proposal_entropy, diversity_index, arm_mean_r_topk, │
          │   drift_detections, thompson_mode, eval_accept_rate,  │
          │   gain_p50, gain_p90, ci_width_avg, stability_avg,    │
          │   total_evaluated}                                    │
          └──────────────────────────────────────────────────────┘
```

### 2.3 Hypothesis Lifecycle

```
                  HypothesisPool.population
                  {hypothesis_id → Hypothesis}

    New observation
         │
         ▼
    Hypothesis.fit_step(X, y)  ← online update (RLS / EMA / Welford)
    Hypothesis.evaluate()      → fit_score, confidence, stability,
                                  evidence, ready

         │
         ├─ [ready: False — buffer not yet full]
         │   _not_ready() sentinel:
         │   {confidence: 0.0, fit_score: 0.0, stability: 0.0,
         │    evidence: n, ready: False}
         │   → blocked by get_candidate_paths() confidence < 0.25 gate
         │   → never enters proposal pool during cold start
         │
         └─ [ready: True — buffer full]
             → normal evaluate() result with ready: True
         │
         ▼
    MetaController.manage_lifecycle(pool)

    State Machine:
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  TENTATIVE ──[evidence>20 & conf>0.7 & stab>0.6]──► ACTIVE
    │      │                                               │
    │      │ [conf < 0.3]                    [conf<0.6 or stab<0.5]
    │      ▼                                               ▼
    │    DEAD ◄──[metrics critical]────────── DECAYING
    │                                             │
    │                                   [conf>0.7 & stab>0.6]
    │                                             │
    │                                             ▼
    │                                           ACTIVE
    └─────────────────────────────────────────────────────┘

    HypothesisArbiter.resolve_conflicts(pool)
    · type hierarchy: Logical > Functional > Causal > Temporal > Correlational
    · conflicting hypotheses on same (src, tgt) → higher-type wins
    · loser → DECAYING
```

### 2.4 Buffer Size and Data Frequency

`OnlineDiscoveryEngine(buffer_size=N)` controls the observation window for all 15
hypothesis types. `buffer_size` is threaded through every constructor call in
`initialize_v2()` and `_explore_step()`.

```
buffer_size guidelines
──────────────────────────────────────────────────────────────────
50–80    Tick / high-frequency data    cold-start resolves quickly
150      Default (mixed data)          balanced sensitivity
200–300  Monthly macro series          stable distribution tests
```

Internally, window-batch types store the last `buffer_size` observations in a
`deque(maxlen=buffer_size)`. Truly-online types (RLS, Welford, Kalman) use it
only for the `_not_ready()` cold-start count check.

### 2.5 EventBus Topics

```
 SUBSCRIBED                         PUBLISHED
 ─────────────────────              ─────────────────────────
 "data_window"           ──────►   "engine.insight"
 "resource_profile"      ──────►   "inference.path_pack"
 "meta_policy_update"    ──────►   "processing_metrics"
 "meta_prior_update"
 "fmi.meta_prior_update"
 "fmi.meta_policy_hint"
 "fmi.warm_start_profile"
 "fmi.telemetry"
```

Implementation notes:
- "meta_prior_update" and "fmi.meta_prior_update" are routed through `_handle_meta_policy_update()`.
- "fmi.telemetry" is currently handled as a no-op hook.

### 2.6 Causal Modelling Pipeline (Structural Inference)

```
scarcity/causal/
├── engine.py          run_causal(data, spec, runtime) — orchestrates full causal run
├── feature_layer.py   FeatureBuilder.validate_and_clean — schema + NaN guards
├── time_series.py     validate_time_series             — temporal consistency checks
├── identification.py  Identifier.identify              — DoWhy causal graph + estimand ID
├── estimation.py      EstimatorFactory.estimate        — DoWhy/EconML backend routing
├── validation.py      Validator.validate               — refuters (RCC/placebo/subset)
├── artifacts.py       ArtifactWriter.write_run         — run bundle + graph artifacts
└── reporting.py       CausalRunResult / EffectArtifact — typed outputs

kshiked/causal_adapter/
├── runner.py          KShieldCausalRunner              — segment-wise orchestration
├── spec_builder.py    build_estimand_specs             — policy-driven spec generation
├── policy.py          select_estimands                 — ATE/CATE/LATE/mediation gating
└── integration.py     artifact_to_edge / edge_to_simulation_update
```

```
Input DataFrame + Task Specs
        │
        ▼
run_causal(data, specs, runtime)
        │
        ├─ FeatureBuilder.validate_and_clean()
        │   · required columns from EstimandSpec
        │   · drop NaN rows on critical vars
        │
        ├─ validate_time_series(data, spec, dot, policy)
        │   · temporal/DAG consistency gate
        │
        ├─ Identifier(spec, graph).identify(clean_data)
        │   · builds DoWhy CausalModel
        │   · identifies estimand
        │
        ├─ EstimatorFactory.estimate(...)
        │   · backend: DoWhy or EconML
        │   · methods include linear, IV, mediation, CATE/ITE
        │
        ├─ Validator.validate(model, estimate, runtime)
        │   · random_common_cause
        │   · placebo_treatment_refuter
        │   · data_subset_refuter
        │
        └─ ArtifactWriter.write_run(...)
            · learned edges + diagnostics + provenance
            · emits EffectArtifact list

                │
                ▼
   kshiked.causal_adapter.integration
      artifact_to_edge(effect_artifact)
      edge_to_simulation_update(edge)
                │
                ├─► Knowledge graph edges (dashboards)
                └─► SimulationParameterUpdate deltas (policy simulation)
```

---

## 3. Simulation Engine

### 3.1 Two Engine Paths

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SIMULATION ENGINE                                    │
│                                                                         │
│   PATH A — Legacy Aggregate                PATH B — Typed Multi-Sector  │
│   sfc.py                                   sfc_engine.py                │
│                                                                         │
│   SFCEconomy                               MultiSectorSFCEngine         │
│   · 4 balance-sheet sectors                · 8 ordered behavioral blocks│
│   · SFCConfig (40+ params)                 · EconomyState (frozen)      │
│   · Households, Firms,                     · PolicyState  (frozen)      │
│     Banks, Government                      · ShockVector  (frozen)      │
│   · Phillips Curve                         · StepResult   (output)      │
│   · Taylor Rule                            · AllParams    (KNBS-cal.)   │
│   · Okun's Law                                                          │
│   · 4 shock channels:                      Used by:                     │
│     demand, supply,                        ScarcityBridge               │
│     fiscal, fx                             learned_sfc.py               │
│                                                                         │
│   Used by:                                                              │
│   KShield dashboards                                                    │
│   ScarcityBridge (legacy)                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 IO Foundation Layer

```
io_structure.py + parameters.py

KNBS 9-Sector Supply-Use Table (2017)
         │
         │  aggregate_io_to_sfc_sectors()
         │  Standard aggregation formula:
         │  A_agg[I,J] = Σ_{i∈I} Σ_{j∈J} A[i,j] · x_j / X_J
         ▼
4-Sector IO Matrix (InputOutputParams)

  Sector concordance:
  AGRICULTURE   ← agriculture
  MANUFACTURING ← manufacturing + mining + construction + water
  SERVICES      ← services + health + transport + security
  INFORMAL      ← field estimates (not in KNBS SUT)

  A matrix (row = consuming, col = supplying):
                  AGR    MFG    SRV    INF
  AGRICULTURE  [ 0.12   0.03   0.04   0.02 ]
  MANUFACTURING[ 0.17   0.22   0.15   0.03 ]
  SERVICES     [ 0.13   0.21   0.30   0.04 ]
  INFORMAL     [ 0.10   0.08   0.05   0.06 ]

  Column sums: AGR=0.52, MFG=0.54, SRV=0.54, INF=0.15  (all < 1.0 ✓ Hawkins-Simon)

  import_content:
  AGR=0.15  MFG=0.31  SRV=0.11  INF=0.08
         │
         ▼
AllParams (unified parameter container)
  ├── NationalAccountsParams  GDP shares, employment shares, 2023 baselines
  ├── ProductionParams        CES: TFP(A), capital-share(α), substitution(σ)
  ├── InputOutputParams       4×4 IO matrix (above)
  ├── HouseholdParams         MPC=0.82, quintile income shares, food shares
  ├── GovernmentParams        VAT=16%, income tax=12%, debt/GDP=68%
  ├── MonetaryParams          Taylor rule: i_neutral=2.5%, φ_π=1.5, φ_y=0.5
  ├── ExternalParams          export/import GDP ratios, trade elasticities
  └── BankingParams           LTD=0.78, CAR=17.2%, NPL=14.9%
```

### 3.3 MultiSectorSFCEngine: Step Sequence

```
step(state: EconomyState, policy: PolicyState,
     shock: ShockVector, params: AllParams) → StepResult
     │
     ├─ Block 1: compute_labor_market()
     │  expected_output = demand_shock × supply_shock × agg_shock
     │  N = f(expected_output, labor_force)       ← employment
     │  U = labor_force − N                       ← unemployment
     │  Δw = phillips_coef × output_gap           ← wage pressure
     │  w_new = w × (1 + Δw)
     │
     ├─ Block 2: compute_gross_output() + compute_value_added()
     │  Y_gross[s] = A[s] × K[s]^α × N[s]^(1−α) × tfp_shock
     │  Y[s] = Y_gross[s] − Σ_j A[s,j] × Y_gross[j]  ← IO linkages
     │
     ├─ Block 3: compute_potential_output()
     │  Y_pot = A × K^α × N_natural^(1−α) × TFP_trend
     │  output_gap = (Y − Y_pot) / Y_pot
     │
     ├─ Block 4: compute_prices_and_profits()
     │  P[s] = ULC[s] / (1 − markup) + import_cost[s] × E_fx
     │  CPI = Σ_s weight[s] × P[s]
     │  π_cpi = (CPI − CPI_prev) / CPI_prev  [clipped ±50%/quarter]
     │  profits = Y − w×N − interest − depreciation
     │
     ├─ Block 5: compute_monetary_block()
     │  Taylor Rule:
     │    i_target = i_neutral + φ_π×(π−π*) + φ_y×output_gap
     │    i_cb = smoothing×i_prev + (1−smoothing)×i_target
     │    clamp: [i_floor=1.25%, i_ceiling=20%]
     │  Spreads:
     │    i_loan = i_cb + spread_loan (150bps)
     │    i_dep  = i_cb + spread_dep  (−100bps)
     │    i_gov  = i_cb + spread_gov  (50bps)
     │
     ├─ Block 6: compute_households()
     │  income = w×N + dividends + remittances + transfers
     │  taxes  = income × tax_rate
     │  C = MPC × (income − taxes) + wealth_effect × D_h
     │  S_h = income − taxes − C
     │  D_h_new = D_h + S_h − loan_repayment
     │
     ├─ Block 7: compute_government_block()
     │  T_rev = VAT×C + income_tax×w×N + corp_tax×profits
     │        + trade_tax×(IM − EX)
     │  G_exp = wage_bill + transfers + interest + G_inv + other
     │  deficit = G_exp − T_rev
     │  debt_new = debt + deficit
     │  automatic_stabilizers: transfers ↑ if U ↑, tax ↓ if Y ↓
     │
     ├─ Block 8a: compute_foreign_block()
     │  EX[s] = EX_base[s] × (E_fx/E_base)^η_export × world_gdp_growth
     │  IM[s] = IM_base[s] × (E_fx/E_base)^η_import × (C+G+I)^ε_import
     │  CA = Σ_s(EX[s] − IM[s]) + remittances + aid
     │  ΔRE_fx = CA + capital_flows − fx_intervention
     │
     └─ Block 8b: compute_banking_block()
        credit = LTD_ratio × deposits × credit_multiplier
        NPL_new = NPL × (1 + sensitivity×ΔU)
        CAR = equity / (risk_weighted_assets)
        if CAR < min_CAR: credit_rationing → credit × 0.5
        bank_equity += profits − dividends − provisions

     └─ Residual accounting checks:
        S_h + S_firms + S_gov + CA ≈ ΔK  (SFC identity ± tolerance)
        asset_totals ≈ liability_totals   (balance sheet consistency)
```

### 3.4 Shock → Economy Pipeline

```
Shock Sources (KShield)                Kenya Calibration
──────────────────────                 ─────────────────
Pulse threat indices                   kenya_calibration.py
  PI, LEI, MRS, ECI,                   .calibrate_from_data(csv)
  IWI, SFI, ECR, ETM                   → SFCConfig
        │                                     │
        ▼                                     ▼
scenario_templates.py              SFCEconomy(config)
.build_shock_vectors()                    │
  drought, FX crisis,                     │
  fiscal shock,                           │
  insurgency, etc.                        │
        │                                 │
        └──────────── ShockVector ────────┘
                           │
                           ▼
                   SFCEconomy.step() ×N quarters
                           │
                           ▼
                   trajectory: List[Dict]
                     t, shock_vector, policy_vector,
                     outcomes {gdp_growth, inflation,
                               unemployment, CA, debt},
                     sector_balances, flows
```

---

## 4. Dynamic Resource Governor (DRG)

### 4.1 Component Map

```
scarcity/governor/
├── DynamicResourceGovernor  — main async control loop
├── ResourceSensors          — hardware metric sampling (psutil, torch, pynvml)
├── ResourceProfiler         — EMA smoothing + Kalman forecasting
├── PolicyRule               — condition + action declarative rules
├── ResourceActuators        — executes actions on registered subsystems
├── SubsystemRegistry        — subsystem_name → tunable handle
└── DRGMonitor               — historical metrics logging (JSON)
```

### 4.2 Control Loop

```
DRGMonitor._loop()   [async, every 500ms]
        │
        ├─ Step 1: ResourceSensors.sample()
        │  ┌──────────────────────────────────────────────┐
        │  │  CPU:    cpu_util [0,1], cpu_freq (MHz)       │
        │  │  Memory: mem_util [0,1], mem_avail_gb         │
        │  │  GPU:    gpu_util [0,1], vram_util [0,1]      │
        │  │  I/O:    disk_read/write_mb, net_sent/recv_mb │
        │  └──────────────────────────────────────────────┘
        │
        ├─ Step 2: ResourceProfiler.update(metrics)
        │  EMA smoothing:   ema[k] = α×metric[k] + (1−α)×ema_prev[k]
        │  Kalman forecast: predict next 2 steps from ema trajectory
        │  → (ema_metrics, forecast_metrics)
        │
        ├─ Step 3: Evaluate policy rules
        │  For each registered subsystem + its PolicyRules:
        │    if metric[rule.metric] OP rule.threshold:
        │      → triggered: (subsystem, rule)
        │
        │  Built-in policies:
        │  "simulation": vram > 0.90 → scale_down  (factor=0.5)
        │                fps  < 25.0 → increase_lod
        │  "mpie":       cpu  > 0.85 → reduce_batch (factor=0.5)
        │  "meta":       vram > 0.85 → drop_low_priority
        │
        ├─ Step 4: Dispatch
        │  ├─ ResourceActuators.execute(subsystem, action, factor)
        │  │    → subsystem.set_parameter(param_name, new_value)
        │  │
        │  └─ EventBus.publish("resource_profile", profile_dict)
        │       {n_paths, resamples, sketch_dim,
        │        gain_min, stability_min, cache_capacity,
        │        tier2_enabled, tier3_topk}
        │
        └─ Step 5: DRGMonitor.record({metrics, ema})  → JSON
```

### 4.3 Assurance Level Computation

```
Inputs: current metrics + forecast

 Level        Condition                         Meaning
 ─────────────────────────────────────────────────────────────────
 GREEN  (LOW) All metrics < 70%                Full capability
 YELLOW (MED) Any metric 70–85%                Directionally reliable
 ORANGE (HIGH)Any metric 85–95%                Indicative, review recommended
 RED   (CRIT) Any metric ≥ 95% OR             FALLBACK — hardcoded baselines
              forecast → critical in ≤2 steps

Used by ScarcityBridge.validate() to assign DRG assurance tags
to simulation projections and causal relationship confidence scores.
```

### 4.4 Resource Profile → Subsystem Effects

```
resource_profile event payload
        │
        ├──► MPIEOrchestrator (engine)
        │    n_paths     → BanditRouter.propose(n_proposals=n_paths)
        │    sketch_dim  → Encoder projection dimension
        │    resamples   → Evaluator bootstrap samples
        │    gain_min    → Evaluator acceptance threshold (g_min)
        │    tier2/3     → Operator tier enable/disable
        │
        ├──► MetaSupervisor (meta-learning)
        │    n_paths_delta  → scale batch up/down
        │    sketch_dim     → compress representations
        │
        └──► Simulation (indirectly via ScarcityBridge)
             assurance level → DRG annotation on projections
```

---

## 5. Cross-Subsystem Interaction

### 5.1 Full System Interaction Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DATA INPUTS                                           │
│  Social Media  ·  World Bank CSV  ·  KNBS Data  ·  Institution Uploads      │
└──────────┬───────────────────┬──────────────────────────────────────────────┘
           │                   │
           ▼                   ▼
   ┌───────────────┐   ┌───────────────────────┐
   │  Pulse Engine │   │   ScarcityBridge       │
   │  (KShield)    │   │   .train(csv)          │
   │               │   │   .create_learned_     │
   │  15 signal    │   │    economy()           │
   │  detectors    │   └──────────┬────────────┘
   │  8 threat     │              │
   │  indices      │              │ EventBus: "data_window"
   │               │              │
   │  ShockVector  │              ▼
   │  → simulation │   ┌──────────────────────────────────────────┐
   └──────┬────────┘   │   ONLINE LEARNING ENGINE (MPIE)          │
          │            │                                          │
          │            │   BanditRouter ──► Encoder ──► Evaluator │
          │            │        ▲               │           │     │
          │            │        └───────────────┘           │     │
          │            │        (reward feedback)            │     │
          │            │                          HypergraphStore  │
          │            │                               │           │
          │            │                               ▼           │
          │            │                           Exporter        │
          │            └──────────────────────────────┬───────────┘
          │                                           │
          │          "engine.insight"                 │
          │          {causal edges discovered}        │
          │                                           │
          │          "processing_metrics"             │
          │          {accept_rate, gain, latency…}    │
          │                   │                       │
          │         ┌─────────┘           ┌───────────┘
          │         │                     │
          │         ▼                     ▼
          │  ┌─────────────────┐   ┌─────────────────────────────────┐
          │  │   DRG           │   │   META-LEARNING LAYER           │
          │  │                 │   │                                 │
          │  │ ResourceSensors │   │  MetaLearningAgent              │
          │  │ ResourceProfiler│   │  · DomainMetaLearner (per inst) │
          │  │ PolicyRules     │   │  · CrossDomainAggregator        │
          │  │ Actuators       │   │  · OnlineReptileOptimizer       │
          │  │                 │   │  · MetaScheduler                │
          │  │  CPU/GPU/Mem    │   │  · MetaPacketValidator          │
          │  │  monitoring     │   │  · MetaStorageManager           │
          │  │  every 500ms    │   │                                 │
          │  │                 │   │  MetaSupervisor +               │
          │  └────────┬────────┘   │  MetaIntegrativeLayer           │
          │           │            │  (rule-based governance)        │
          │           │            └────────────┬────────────────────┘
          │           │                         │
          │    "resource_profile"       "meta_prior_update"
          │           │                 "meta_policy_update"
          │           │                         │
          │           └──────────┬──────────────┘
          │                      │
          │                      ▼
          │         ┌────────────────────────┐
          │         │   MPIE Orchestrator    │
          │         │   applies updates:     │
          │         │   · n_paths, sketch_dim│
          │         │   · gain_min, lambda_ci│
          │         │   · tau, gamma_div     │
          │         │   · tier2/3 on/off     │
          │         └────────────────────────┘
          │
          ▼
   ┌─────────────────────────────────────────────────────┐
   │   SIMULATION ENGINE                                  │
   │                                                      │
   │   MultiSectorSFCEngine                               │
   │   · 8 behavioral blocks (labor → banking)            │
   │   · IO Foundation (9-sector KNBS → 4-sector SFC)    │
   │   · AllParams (KNBS-calibrated Kenya baselines)      │
   │                                                      │
   │   SFCEconomy (legacy)                                │
   │   · 4 balance-sheet sectors                          │
   │   · Phillips/Taylor/Okun equations                   │
   │                                                      │
   │   Input:  ShockVector + PolicyState                  │
   │   Output: trajectory {GDP, inflation, unemployment,  │
   │                        sector_balances, CA, debt}    │
   └──────────────────────────────┬──────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────────┐
              │   DASHBOARDS  (KShield layer)              │
              │                                           │
              │   K-SHIELD (8505)                         │
              │   · Causal Relationships (engine.insight) │
              │   · Policy Terrain (simulation output)    │
              │   · Simulations (trajectory viewer)       │
              │   · Policy Impact (sentiment + scarcity)  │
              │                                           │
              │   Institution Portal (8506)               │
              │   · Executive briefing                    │
              │   · Cost of Delay Engine (KES billions)   │
              │   · Sector Reports                        │
              │   · FL Dashboard                          │
              │   · Unified PDF Export                    │
              │                                           │
              │   SENTINEL (8507)                         │
              │   · Live threat map (Pulse indices)       │
              │   · Federation gossip topology            │
              │   · Policy chat (LLM)                     │
              └───────────────────────────────────────────┘
```

### 5.2 EventBus Wiring — All Topics

```
Topic                     Published By              Consumed By
─────────────────────────────────────────────────────────────────────────────
"data_window"             ScarcityBridge / streams  MPIEOrchestrator
"resource_profile"        DRG, MetaSupervisor       MPIEOrchestrator
"meta_policy_update"      MetaSupervisor            MPIEOrchestrator
"meta_prior_update"       MetaLearningAgent         MPIEOrchestrator
"meta_rollback_active"    MetaLearningAgent         MetaSupervisor
"meta_update"             MetaLearningAgent         (debug / telemetry)
"meta_metrics"            MetaLearningAgent         DRG monitor / dashboards
"processing_metrics"      MPIEOrchestrator          MetaLearningAgent
                                                    MetaSupervisor (DRG)
"engine.insight"          Exporter (MPIEOrch.)      K-SHIELD causal graph
"inference.path_pack"     Exporter (MPIEOrch.)      Federation / consumers (batched edges)
"telemetry"               Engine internals          MetaSupervisor
"federation.policy_pack"  Aegis Federation nodes    MetaLearningAgent
"federation.path_pack"    Federation agents         Federation peers/coord.
"federation.edge_delta"   Federation agents         Federation peers/coord.
"federation.causal_pack"  Federation agents         Federation peers/coord.
"federation.health"              FederationClientAgent     Federation transport
"federation_update"              FederationClientAgent     Local telemetry/ops
"fmi.meta_prior_update"          Federation bridge (FMI)   MPIEOrchestrator
"fmi.meta_policy_hint"           Federation bridge         MPIEOrchestrator
"fmi.warm_start_profile"         Federation bridge         MPIEOrchestrator
"fmi.telemetry"                  Federation bridge         MPIEOrchestrator (no-op)
"federation.adaptation_request"  Client                    DomainServer (Phase 4)
"federation.adaptation_response" DomainServer              Client (Phase 4)
"federation.domain_sync"         DomainServer              GlobalMetaMemory (Phase 4)
```

### 5.3 Feedback Loop: How the System Self-Regulates

```
                 ┌─────────────────────────────────────────────────┐
                 │              SELF-REGULATION LOOP               │
                 │                                                 │
                 │   1. DRG monitors hardware every 500ms          │
                 │      if vram > 90% → publish resource_profile   │
                 │         n_paths ↓, sketch_dim ↓                 │
                 │                                                 │
                 │   2. MPIE runs lighter with reduced profile      │
                 │      fewer paths → lower accept_rate            │
                 │      publishes processing_metrics                │
                 │                                                 │
                 │   3. MetaScheduler detects low accept_rate      │
                 │      MetaIntegrativeLayer:                       │
                 │        tau ↑ (more exploration)                 │
                 │        g_min ↓ (relax acceptance threshold)     │
                 │      publishes meta_policy_update               │
                 │                                                 │
                 │   4. MPIE applies new tau/g_min                  │
                 │      accept_rate recovers                        │
                 │      gain_p50 improves                           │
                 │                                                 │
                 │   5. MetaLearningAgent:                          │
                 │      reward EMA rises → beta ↑ (learn faster)   │
                 │      aggregates domain updates                   │
                 │      publishes structured prior                  │
                 │      (tau, gamma, g_min, lambda_ci)              │
                 │                                                 │
                 │   6. MPIE applies prior → better initialization  │
                 │      on next federation round                    │
                 │                                                 │
                 │   → System converges to stable operating point   │
                 └─────────────────────────────────────────────────┘
```

### 5.4 Cold Start vs Warm State

```
 COLD START                           WARM STATE
 ──────────────────                   ──────────────────────────────
 DRG: GREEN (all resources free)      DRG: monitors + adapts profile
 Engine: uniform Beta(1,1) priors     Engine: informed Beta posteriors
 Meta: empty _pending_updates         Meta: multi-domain prior loaded
 Optimizer: flat zero prior           Optimizer: prior from disk (JSON)
 MetaInteg: default tau=0.9, g_min    MetaInteg: tuned knobs per history
 Simulation: KNBS 2023 baselines      Simulation: learned SFC params
                                        from ScarcityBridge.train()
 Transition:
 · MetaStorageManager.load_prior()   — reloads prior from artifacts/meta/
 · ScarcityBridge.train(csv)         — calibrates SFC from real data
 · Federation warm-start             — fmi.warm_start_profile event
```

### 5.5 Causal + FL Extended Interaction

```
                     ┌─────────────────────────────────────────────────┐
                     │   CAUSAL + FEDERATION CLOSED LOOP               │
                     └─────────────────────────────────────────────────┘

  Local windows / institution data
                │
                ├─► MPIE (online hypotheses) ──► "engine.insight"
                │        · fast online relational edges
                │
                └─► run_causal(...) via KShield adapter
                           · structural estimands + refutation checks
                           · EffectArtifact + causal edge confidence
                                         │
                                         ▼
           Federation packetization (PathPack / EdgeDelta / CausalSemanticPack)
                                         │
                                         ▼
                         Hierarchical Federation
                         · GossipProtocol (intra-basket)
                         · Layer1Aggregator (basket)
                         · Layer2Aggregator (global)
                                         │
                                         ▼
                          "federation.policy_pack"
                          + optional FMI hints/warm starts
                                         │
                                         ▼
                           MetaLearningAgent + MetaSupervisor
                                         │
                                         ▼
                           "meta_prior_update" / "meta_policy_update"
                                         │
                                         ▼
                                   MPIE tuning

  Result:
  · Causal structure is continuously re-estimated (structural + online paths)
  · Federated consensus improves robustness across institutions
  · Meta-learning adapts exploration/acceptance under resource constraints
```

---

## 6. Federated Learning (Gossip + Security Hardening)

### 6.1 Component Map

```
scarcity/federation/
├── transport.py         BaseTransport/Loopback/Simulated + build_transport router
├── ws_transport.py      WebSocketTransport/WSTransportConfig (distributed runtime)
├── client_agent.py      FederationClientAgent   — node-level export/receive loop
├── coordinator.py       FederationCoordinator   — peer membership + trust routing
├── scheduler.py         FederationScheduler     — adaptive export cadence/backoff
├── packets.py           PathPack/EdgeDelta/PolicyPack/CausalSemanticPack
│                        AdaptationRequest/AdaptationResponse/DomainSyncPacket (Phase 4)
├── validator.py         PacketValidator         — trust + structural + size gates
├── trust_scorer.py      TrustScorer             — reputation decay/penalty/sandbox
├── aggregator.py        FederatedAggregator     — fedavg/weighted/trim/krum/bulyan
├── privacy_guard.py     PrivacyGuard            — DP noise + mask/unmask utilities
├── secure_aggregation.py SecureAggClient/Coordinator — signed pairwise masking
├── gossip.py            GossipProtocol          — push/pull intra-basket gossip
├── buffer.py            UpdateBuffer/ReplayGuard/TriggerEngine
├── layers.py            Layer1Aggregator/Layer2Aggregator/CentralDPMechanism
├── domain_server.py     DomainServer/DomainServerRegistry  — per-domain logical meta agent (Phase 2)
├── global_meta_memory.py GlobalMetaMemory        — cross-domain episodic prior store (Phase 3)
└── hierarchical.py      HierarchicalFederation  — end-to-end orchestrator
                         + run_full_meta_round() (Phase 5)
```

### 6.2 Transport Layer (WebSocket)

```
TransportConfig.protocol
                                │
                                ▼
build_transport(config)
                                │
                                ├─ "loopback" / "local"   -> LoopbackTransport
                                ├─ "sim" / "simulated"    -> SimulatedNetworkTransport
                                └─ "ws" / "websocket"     -> WebSocketTransport
```

```
WebSocketTransport lifecycle
──────────────────────────────────────────────────────────────────────────
start()
        · starts websocket server on host:port
        · pre-connects outbound peer_endpoints with _ensure_connection()

send(topic, payload)
        · no configured peers -> broadcast to inbound active clients
        · with peers          -> concurrent _send_to_peer() fan-out

send_to(endpoint, topic, payload)
        · targeted endpoint send with retry on transient failure

_handle_connection(websocket)
        · decode JSON and enforce optional auth_token gate
        · dispatch inbound packets via registered handler

_listen_peer(endpoint, ws)
        · keeps outbound peer links bidirectional (receive + dispatch)

stop()
        · closes peer connections, inbound clients, and server socket
```

```
Security and resilience hooks in websocket transport
──────────────────────────────────────────────────────────────────────────
- auth_token check blocks unauthenticated inbound packets
- ping_interval/ping_timeout detect dead peers
- reconnect_backoff bounds outbound connection attempts
- max_message_size limits payload pressure
- connected_peers / connected_clients expose runtime health counters
```

### 6.3 Gossip Learning (Intra-Basket)

```
Client local update vector
        │
        ▼
GossipProtocol.create_message(client_id, raw_vector)
        │
        ├─ MessageBudgetTracker.can_send(client_id)
        │   · enforce max_messages_per_day
        │
        ├─ LocalDPMechanism.clip_and_noise(vector)
        │   · L2 clipping to clip_norm
        │   · Gaussian noise for (epsilon, delta)-DP
        │   · sigma = clip_norm * sqrt(2*ln(1.25/delta)) / epsilon
        │   · defaults: clip_norm=1.0, epsilon=1.0, delta=1e-5
        │
        └─ GossipMessage{sender_id, basket_id, summary_vector, seq, round}
                    │
                    ▼
          PeerSampler.sample(k peers, rotation window)
                    │
                    ▼
          push/pull exchange within same basket
                    │
                    ▼
          GossipProtocol.merge_messages(messages)
                    │
                    ▼
          BufferedUpdate -> UpdateBuffer.add()
```

### 6.4 Anti-Poisoning and Security Control Path

```
Inbound packet/update
        │
        ├─ TrustScorer.score(peer_id)
        │   · low-trust peers throttled/sandboxed
        │
        ├─ PacketValidator.validate_* (trust + shape/limits)
        │   · trust_min gate
        │   · max_edges / max_concepts caps
        │
        ├─ ReplayGuard.validate(client_id, sequence_number)
        │   · strict monotonic sequence check
        │   · participation cap per day
        │
        ├─ FederatedAggregator.aggregate(...)
        │   · robust methods: median / trimmed_mean / krum / multi_krum / bulyan
        │   · detect_outliers(..., z_thresh)
        │
        ├─ Layer1Aggregator.aggregate_basket()
        │   · basket-level L2 clipping (bounded influence)
        │
        ├─ Layer2Aggregator.aggregate_global()
        │   · min_basket_support gate
        │   · secure aggregation (masking or crypto pairwise protocol)
        │   · CentralDPMechanism.add_noise(global)
                │   · central sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
                │   · sensitivity = basket_clip_norm * num_baskets
                │   · defaults: basket_clip_norm=5.0, epsilon=1.0, delta=1e-5
                │   · privacy accountant spends (epsilon, delta) per release
        │
        └─ StoreReconciler.merge_*()
            · dynamic min-support filtering across baskets
            · low-support edges suppressed before store upsert
```

### 6.5 Security Coverage Matrix

```
Threat / Failure Mode          Primary Mitigation(s)
──────────────────────────────────────────────────────────────────────────
Model poisoning / Byzantine    trimmed_mean, KRUM, Multi-KRUM, Bulyan
Extreme outliers               FederatedAggregator.detect_outliers()
Sybil / low-quality peers      TrustScorer + PacketValidator.trust_min
Malformed / oversized packets  PacketValidator max_edges/max_concepts
Replay attacks                 ReplayGuard sequence monotonicity check
Over-participation abuse       ReplayGuard + MessageBudgetTracker caps
Single basket dominance        Layer2 basket_clip_norm + min_basket_support
Gradient/info leakage          Local DP (gossip) + Central DP (global)
Raw update exposure            Secure aggregation masking / crypto mode
Stale update drift             UpdateBuffer staleness pruning + decay weights
```

### 6.6 Federation Packet and Event Flow

```
FederationClientAgent.publish_packets()
        │
        ├─ PathPack         -> "federation.path_pack"
        ├─ EdgeDelta        -> "federation.edge_delta"
        ├─ PolicyPack       -> "federation.policy_pack"
        └─ CausalSemanticPack -> "federation.causal_pack"

FederationClientAgent.receive_aggregated(...)
        │
        ├─ merge_path_pack / merge_edge_delta / merge_causal_pack
        └─ publish "federation_update" (local reconciliation telemetry)

FederationClientAgent._on_processing_metrics(...)
        └─ scheduler-triggered health export -> "federation.health"
```

### 6.7 Differential Privacy Profile Used In FL

```
Local DP (gossip layer)
──────────────────────────────────────────────────────────────────────────
Module: scarcity/federation/gossip.py :: LocalDPMechanism
Mechanism: Gaussian local DP after per-message L2 clipping
Formula: sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
Sensitivity source: GossipConfig.clip_norm
Default config:
        clip_norm = 1.0
        local_dp_epsilon = 1.0
        local_dp_delta = 1e-5
Notes:
        - Applied before peer-to-peer gossip send
        - Provides client-side privacy even under untrusted gossip peers

Central DP (global layer)
──────────────────────────────────────────────────────────────────────────
Module: scarcity/federation/layers.py :: CentralDPMechanism
Mechanism: Gaussian noise on global aggregate after secure aggregation
Formula: sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
Sensitivity source: basket_clip_norm * number_of_contributing_baskets
Default config:
        basket_clip_norm = 5.0
        dp_epsilon = 1.0
        dp_delta = 1e-5
Notes:
        - Layer2Aggregator updates sensitivity each round from basket count
        - Noise is added only at release time for global aggregate

Budget accounting (composition)
──────────────────────────────────────────────────────────────────────────
Module: scarcity/federation/buffer.py :: PrivacyAccountant
Composition model: simple additive composition of epsilon and delta
Per-release spend: spend(layer2.dp_epsilon, layer2.dp_delta)
Default total budget (HierarchicalFederationConfig):
        total_epsilon = 10.0
        total_delta = 1e-4
Release gating:
        TriggerEngine.check_layer2() requires accountant.can_release()
        default can_release check: epsilon >= 0.1 and delta >= 1e-6 remaining

Important implementation note
──────────────────────────────────────────────────────────────────────────
PrivacyGuard can also inject DP noise, but in Layer2Aggregator it is initialized
with dp_noise_sigma=0.0 (used there primarily for secure masking utilities).
So the main DP path in hierarchical FL is:
        Local DP in gossip + Central DP at global release.
```

---

---

## 7. Phase 5 — Federated Meta-Learning Pipeline

### 7.1 Component Map

```
scarcity/federation/
├── domain_server.py      DomainServer           — per-domain base model + episodic memory
│                         DomainServerRegistry    — basket_id → DomainServer map
├── global_meta_memory.py GlobalMetaMemory        — cross-domain episode store + prior retrieval
│                         GlobalMetaMemoryConfig
└── packets.py (Phase 4)  AdaptationRequest       — client warm-start query
                          AdaptationResponse      — domain server prior reply
                          DomainSyncPacket        — state snapshot upward

scarcity/meta/
├── domain_server_meta.py DomainServerMeta        — federation→meta bridge via duck typing
│                         DomainServerMetaConfig
└── cross_meta.py         CrossDomainMetaLearner  — memory-backed aggregation
                          CrossDomainMetaLearnerConfig

scarcity/federation/hierarchical.py
└── HierarchicalFederation.run_full_meta_round()  — single-call pipeline entry point
```

### 7.2 Full Pipeline Data Flow

```
HierarchicalFederation.run_full_meta_round(performance_map)
        │
        │  ┌─────────────────────────────────────────────────────────────────┐
        │  │   DomainServerRegistry                                           │
        │  │   basket_hc → DomainServer(healthcare)                          │
        │  │   basket_fin → DomainServer(finance)                            │
        │  │   basket_ret → DomainServer(retail)                             │
        │  │                                                                 │
        │  │   Each DomainServer exposes:                                    │
        │  │   · base_params: Dict[str, float]   (domain base model)         │
        │  │   · hit_rate: float                 (EMA adaptation success)    │
        │  │   · memory_size: int                (episodes stored)           │
        │  │   · round_id: int                   (monotonic counter)         │
        │  └─────────────────────────────────────────────────────────────────┘
        │                      │
        │   Step 1             ▼
        │  DomainServerMeta.observe_registry(registry, performance_map)
        │  ┌──────────────────────────────────────────────────────────────┐
        │  │  For each server (duck typed — no circular imports):          │
        │  │                                                               │
        │  │  confidence = hit_rate_w × hit_rate                          │
        │  │             + mem_w × log1p(mem_size) / log1p(mem_ref)       │
        │  │             + gain_boost × max(0, perf["gain"])              │
        │  │  confidence = max(confidence, min_confidence)                │
        │  │                                                               │
        │  │  meta_lr = lr_min + (lr_max − lr_min) × confidence          │
        │  │  delta   = meta_lr × (base_params − prev_snapshot)          │
        │  │  score_delta = hit_rate − prev_hit_rate                      │
        │  │                                                               │
        │  │  → DomainMetaUpdate{vector, keys, confidence, score_delta}  │
        │  └──────────────────────────────────────────────────────────────┘
        │                      │ List[DomainMetaUpdate]
        │
        │   Step 2             ▼
        │  CrossDomainMetaLearner.aggregate(updates)
        │  ┌──────────────────────────────────────────────────────────────┐
        │  │  fallback_vec = CrossDomainMetaAggregator.aggregate(updates) │
        │  │     · filter: confidence ≥ min_confidence                    │
        │  │     · union keys, zero-pad, stack → (N × K) matrix          │
        │  │     · trimmed_mean(alpha=0.1) OR median                      │
        │  │                                                               │
        │  │  quality = clip(memory.memory_size / ref_cap,                │
        │  │                 min_quality, max_quality)                    │
        │  │                                                               │
        │  │  if quality > 0:                                             │
        │  │    prior = memory.suggest_prior("cross_domain", ctx)         │
        │  │    result = (1−quality)×fallback + quality×prior             │
        │  │    meta["source"] = "memory_backed"                          │
        │  │  else:                                                        │
        │  │    result = fallback_vec                                     │
        │  │    meta["source"] = "fallback"                               │
        │  └──────────────────────────────────────────────────────────────┘
        │                      │ (cross_vec, cross_keys, cross_meta)
        │
        │   Step 3             ▼
        │  GlobalMetaMemory.aggregate(registry, performance_map)
        │  ┌──────────────────────────────────────────────────────────────┐
        │  │  Reads base_params from all DomainServers in registry        │
        │  │  Computes REPTILE-blended global parameter vector:           │
        │  │    global[key] = blend_alpha × retrieved_prior[key]          │
        │  │                + (1−blend_alpha) × mean(domain_params[key]) │
        │  │  Stores episode: {context, params} in EpisodicMemory         │
        │  │  memory_size += 1                                            │
        │  └──────────────────────────────────────────────────────────────┘
        │                      │ global_params: Dict[str, float]
        │
        └──────────────────────▼
        {
            "global_params":  {"gain": 0.48, "tau": 0.86, ...},
            "cross_domain":   (vec, keys, {"source": "memory_backed",
                                           "memory_quality": 0.25,
                                           "prior_keys_matched": 2, ...}),
            "n_updates":      3,
        }
```

### 7.3 Memory Quality Progression

As `GlobalMetaMemory` accumulates episodes, `CrossDomainMetaLearner` blends progressively more prior into each round:

```
Episodes   memory_quality   cross_domain source   Blend ratio
─────────────────────────────────────────────────────────────────────
0          0.000            fallback               100% statistical
16 / 256   0.063            fallback→transitional   94% stat / 6% prior
64 / 256   0.250            memory_backed           75% stat / 25% prior
128 / 256  0.500            memory_backed           50% stat / 50% prior
256+ / 256 0.800 (cap)      memory_backed           20% stat / 80% prior
```

`memory_reference_capacity` (default 256) controls the saturation rate. Set it to `32` for fast-learning scenarios.

### 7.4 CrossDomainMetaLearner Blend Formula

```
Given:
  updates   = [DomainMetaUpdate, ...]   from DomainServerMeta
  memory    = GlobalMetaMemory          (may be empty)
  ref_cap   = CrossDomainMetaLearnerConfig.memory_reference_capacity
  max_q     = CrossDomainMetaLearnerConfig.max_memory_quality  (default 0.8)

Step 1 — Fallback (always computed):
  filter: updates where confidence ≥ min_confidence AND len(vector) > 0
  keys:   union of all filtered update keys (sorted)
  matrix: shape (N_filtered × |keys|), zero-padded for missing keys
  fallback_vec = trimmed_mean(matrix, alpha=0.1)   [or median]

Step 2 — Memory quality:
  quality = clip(memory.memory_size / ref_cap, 0, max_q)

Step 3 — Prior retrieval (only if quality > 0):
  context = {
      "n_domains":       len(filtered_updates),
      "confidence_mean": mean(update.confidence for update in filtered),
      "score_delta_mean": mean(update.score_delta for update in filtered),
      **extra_context,   # caller-supplied, optional
  }
  prior_dict = memory.suggest_prior("cross_domain", context)
  prior_vec  = [prior_dict.get(k, 0.0) for k in keys]   # aligned + zero-filled

Step 4 — Blend:
  result_vec = (1 − quality) × fallback_vec + quality × prior_vec

  If prior is None or keys is empty: result_vec = fallback_vec (source="fallback")
```

### 7.5 Protocol Bridge Packet Flow (Phase 4)

```
Packet                    Topic                           Direction
────────────────────────────────────────────────────────────────────
AdaptationRequest         federation.adaptation_request   Client → DomainServer
  basket_id, domain_id,
  context, round_id

AdaptationResponse        federation.adaptation_response  DomainServer → Client
  basket_id, domain_id,
  prior_params, source,    source: "global_memory" | "passthrough"
  round_id

DomainSyncPacket          federation.domain_sync          DomainServer → Global
  basket_id, domain_id,
  base_params, performance,
  memory_size, hit_rate,
  round_id

All packets: .to_dict() / .from_dict() round-trip
serialise_packet(packet) → (topic, payload_dict)
normalise_packets(list)  → {TypeName: [packets]}
PayloadCodec             → encode(packet) → bytes / decode(bytes) → dict
```

---

*Generated from source: `scarcity/meta/`, `scarcity/engine/`, `scarcity/simulation/`, `scarcity/governor/`, `scarcity/federation/`, `scarcity/causal/`, `kshiked/causal_adapter/`*
*All class names, method signatures, and event topics verified against current implementation.*

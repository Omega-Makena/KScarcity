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

---

## 1. Meta-Learning Layer

### 1.1 Component Map

```
scarcity/meta/
├── meta_learning.py       MetaLearningAgent      — top-level orchestrator
├── domain_meta.py         DomainMetaLearner      — per-domain EMA + delta tracking
├── cross_meta.py          CrossDomainMetaAggregator — trimmed-mean / median fusion
├── optimizer.py           OnlineReptileOptimizer — EMA prior update + rollback
├── scheduler.py           MetaScheduler          — window-count gated, adaptive interval
├── validator.py           MetaPacketValidator    — confidence + finiteness guards
├── storage.py             MetaStorageManager     — JSON persistence, versioned backups
├── integrative_meta.py    MetaIntegrativeLayer   — rule-based hyperparameter governance
│                          MetaSupervisor         — EventBus bridge to MetaIntegrativeLayer
├── integrative_config.py  IntegrativeMetaConfig  — typed config dataclass tree
└── telemetry_hooks.py     build_meta_metrics_snapshot / publish_meta_metrics
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
EventBus: "processing_metrics"
          │
          ▼
  MetaSupervisor._handle_processing_metrics()
          │
          │  [suppressed for 2 cycles if meta_rollback_active received]
          │
          ▼
  MetaSupervisor._maybe_update()
          │
          ▼
  MetaIntegrativeLayer.update(telemetry)
          │
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
          │  · Cooldown per knob: 5 cycles before re-adjusting
          │
          ├─ _resource_policy(telemetry)
          │  · vram > 0.85 → n_paths_delta −15%, sketch_dim ↓
          │  · vram < 0.55 + latency < 100ms → n_paths_delta +10%
          │
          └─ _safety_checks(reward, ema, prev_snapshot, changed_knobs)
             · ema drop > 0.1 → _rollback_previous(prev_snapshot)
             · rollback_count++, logger.warning

                    │
                    ├─ EventBus: "meta_policy_update"  → engine
                    ├─ EventBus: "resource_profile"    → DRG / engine
                    └─ EventBus: "meta_metrics"        → telemetry
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
├── controller.py      MetaController     — state machine for hypothesis lifecycle
├── arbitration.py     HypothesisArbiter  — conflict resolution between hypotheses
└── resource_profile.py                  — default profile dict
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
          └───────────────────────────┬──────────────────────────┘
                                      │ List[Candidate]
                                      │ {path_id, vars[], lags[],
                                      │  ops[], root, depth, domain}
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 2 — Encode                                       │
          │                                                       │
          │  Encoder.step(window, candidates, context)            │
          │  → EncodedBatch                                       │
          │                                                       │
          │  Per candidate:                                       │
          │  · VariableEmbeddingMapper: var → embed + σ scaling   │
          │  · LagPositionalEncoder: lag → positional embed       │
          │  · CountSketch (dim=sketch_dim): dim reduction        │
          │  · PrecisionManager: FP16 with FP32 fallback          │
          │  · SketchCache (LRU): deterministic projection reuse  │
          │                                                       │
          │  → latents[N_cand × sketch_dim]                       │
          │  → meta[N_cand] {var_names, lag_vals, telemetry}      │
          └───────────────────────────┬──────────────────────────┘
                                      │ EncodedBatch
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 3 — Score                                        │
          │                                                       │
          │  Evaluator.score(window, candidates)                  │
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
          │ Step 7 — Broadcast insights                           │
          │                                                       │
          │  Exporter.emit_insights(accepted, profile)            │
          │  → EventBus: "engine.insight"                         │
          │    {edges[], window_id, n_accepted, timestamp}        │
          └───────────────────────────┬──────────────────────────┘
                                      │
          ┌───────────────────────────▼──────────────────────────┐
          │ Step 8 — Publish metrics                              │
          │                                                       │
          │  EventBus: "processing_metrics"                       │
          │  {accept_rate, gain_p50, stability_avg, latency_ms,   │
          │   diversity_index, rcl_contrast, ci_width_avg,        │
          │   n_candidates, n_accepted, total_evaluated,          │
          │   engine_latency_ms, oom_flag}                        │
          └──────────────────────────────────────────────────────┘
```

### 2.3 Hypothesis Lifecycle

```
                  HypothesisPool.population
                  {hypothesis_id → Hypothesis}

    New observation
         │
         ▼
    Hypothesis.fit_step(X, y)  ← online update (RLS / EMA)
    Hypothesis.evaluate()      → confidence, stability, evidence

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

### 2.4 EventBus Topics

```
 SUBSCRIBED                         PUBLISHED
 ─────────────────────              ─────────────────────────
 "data_window"           ──────►   "engine.insight"
 "resource_profile"      ──────►   "processing_metrics"
 "meta_policy_update"
 "meta_prior_update"
 "fmi.meta_prior_update"
 "fmi.meta_policy_hint"
 "fmi.warm_start_profile"
 "fmi.telemetry"
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
"telemetry"               Engine internals          MetaSupervisor
"federation.policy_pack"  Aegis Federation nodes    MetaLearningAgent
"fmi.meta_prior_update"   Federation bridge (FMI)   MPIEOrchestrator
"fmi.meta_policy_hint"    Federation bridge         MPIEOrchestrator
"fmi.warm_start_profile"  Federation bridge         MPIEOrchestrator
"fmi.telemetry"           Federation bridge         MPIEOrchestrator (no-op)
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

---

*Generated from source: `scarcity/meta/`, `scarcity/engine/`, `scarcity/simulation/`, `scarcity/governor/`*
*All class names, method signatures, and event topics verified against current implementation.*

# System Architecture — Full Component Interaction Map
## K-Scarcity / K-SHIELD / Institution Dashboards

---

## 1. High-Level Layer Map

```
 RAW DATA          SCARCITY ENGINE            K-SHIELD LAYER              DASHBOARDS
 ─────────         ───────────────────────    ────────────────────────    ─────────────────────
 CSV Upload    ──> AutoPipeline               ScarcityBridge (backend)    Executive Dashboard
 Pulse / News  ──> Pulse Ingestion Pipeline   AnalyticsEngine             Admin Gov. Console
 World Bank    ──> KenyaCalibration           ExecutiveBridge             Spoke (Local) Dashboard
 Kenya Params  ──> KenyaCalibration           FederationBridge            K-SHIELD Module (all)
```

---

## 2. Scarcity Engine — Internal Component Interactions

```
scarcity/simulation/
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SFCConfig (dataclass)                                                      │
│   shock_vectors: Dict[str, np.ndarray]   ← injected by scenario templates  │
│   policy_vectors: Dict[str, np.ndarray]  ← injected by AutoPipeline        │
│   steps, dt, mpc, crr, tax_rate, gov_spend_ratio, ...                      │
│                    │                                                        │
│                    ▼                                                        │
│  SFCEconomy                                                                 │
│   Sectors: Households, Firms, Banks, Government, Foreign                    │
│   .initialize(gdp) → seeds balance sheets for all 5 sectors                │
│   .step() → one quarter of macro dynamics:                                  │
│      Consumption = MPC × disposable_income + wealth_effect                 │
│      Investment  = acc_coeff × ΔGDP − credit_cost × (r − r_neutral)       │
│      Tax Revenue = tax_rate × GDP                                           │
│      Government Spending = gov_spend_ratio × GDP                           │
│      Net Exports = current_account_adjustment                               │
│      Bank Lending = deposit_base × (1 − crr) × multiplier                 │
│      → publishes: {gdp_growth, inflation, unemployment,                    │
│                    household_welfare, sector_balances, flows}               │
│   .run(steps) → List[frame]   ← used in K-SHIELD sim tabs                 │
│   .apply_shock(type, magnitude) ← used in stress tests                     │
│                    │                                                        │
│                    ▼  (SFCEconomy is EMBEDDED inside ResearchSFCEconomy)   │
│  ResearchSFCEconomy                                                         │
│   Owns: SFCEconomy (core macro)                                             │
│       + HeterogeneousHouseholdEconomy (Q1-Q5 agents)                       │
│       + OpenEconomyModule (exports, imports, REER, reserves)               │
│       + FinancialAcceleratorModule (credit cycles, LTV, leverage)           │
│       + IOStructureModule (agriculture, manufacturing, services, finance)   │
│       + BayesianBeliefUpdater (shock probability distributions)             │
│                                                                             │
│   .initialize(gdp) → calls SFCEconomy.initialize + all sub-modules        │
│   .step() → call sequence each quarter:                                     │
│      1. SFCEconomy.step()         → base macro outcomes                    │
│      2. _step_open_economy()      → trade balance, REER, reserves          │
│      3. _step_financial()         → credit spreads, leverage ratio          │
│      4. _step_io()                → inter-sector demand flows               │
│      5. _step_heterogeneous()     → Q1-Q5 income shares, MPC effects       │
│      6. _record_unified_frame()   → assembles full frame:                  │
│            outcomes: {gdp_growth, inflation, unemployment,                  │
│                       household_welfare, reserves_months}                   │
│            inequality: {gini, palma,                                        │
│                         quintile_incomes: {q1..q5}}                        │
│            sector_balances: {households, firms, banks, govt, foreign}      │
│            flows: {consumption, investment, gov_spend, net_exports}         │
│   .run(steps)  → List[frame]                                               │
│   .stress_test(shocks) → shocked scenario outcomes                         │
│   .twin_deficit_analysis() → fiscal + current account positions            │
│   .external_vulnerability_index() → 0-1 reserve adequacy score            │
│   .financial_stability_index() → 0-1 leverage + credit health score       │
│                                                                             │
│  HeterogeneousHouseholdEconomy                                              │
│   Agents: Q1 (bottom 20%) … Q5 (top 20%)                                  │
│   Kenya calibration:                                                        │
│      income_shares = [0.04, 0.08, 0.12, 0.20, 0.56]                       │
│      MPC           = [0.95, 0.90, 0.85, 0.75, 0.60]                       │
│      formal_labor  = [0.10, 0.25, 0.45, 0.70, 0.90]                       │
│   .step() → each agent: income = share × aggregate_income                  │
│                          consumption = MPC × income + wealth_effect        │
│                          savings update                                     │
│   InequalityMetrics (static helpers)                                        │
│      .gini_from_quintiles(shares) → Lorenz trapezoidal approximation       │
│      .palma_ratio(shares)         → Q5/2 ÷ (Q1+Q2)                        │
│      .theil_index(shares)         → GE(1)                                  │
│   .distributional_impact(policy_var, change)                                │
│         → per-quintile: {income_change, consumption_change,                 │
│                          employment_effect, welfare_score}                  │
└─────────────────────────────────────────────────────────────────────────────┘

scarcity/simulation/whatif.py
┌─────────────────────────────────────────────────────────────────────────────┐
│  WhatIfManager                                                              │
│   Uses SFCEconomy (base engine only) for speed                             │
│   .run_bootstrap(base_cfg, n=8, jitter_pct=8%)                             │
│         → jitters all numeric SFCConfig fields by ±8%                      │
│         → returns (mean−std, mean+std) CI tuple per dimension              │
└─────────────────────────────────────────────────────────────────────────────┘

scarcity/engine/
┌─────────────────────────────────────────────────────────────────────────────┐
│  EventBus (runtime/bus.py)  — async pub/sub backbone                       │
│   Topics:                                                                   │
│      "data_window"              ← new data row arrives                     │
│      "scarcity.anomaly_detected" → OnlineAnomalyDetector result            │
│      "scarcity.forecasted_trends" → PredictiveForecaster result            │
│      "scarcity.drg_extension_profile" → DRG risk profile                  │
│                                                                             │
│  OnlineAnomalyDetector                                                      │
│   Algorithm: RRCF (Robust Random Cut Forest) — streaming, no training      │
│   Subscribes to: "data_window"                                              │
│   Publishes to:  "scarcity.anomaly_detected"                                │
│   Output per row: {anomaly_score: float, is_anomaly: bool, context: dict}  │
│                                                                             │
│  PredictiveForecaster                                                       │
│   Algorithm: GARCH-VARX — multi-variate with exogenous variables           │
│   Subscribes to: "data_window", "scarcity.anomaly_detected"                │
│   Publishes to:  "scarcity.forecasted_trends"                               │
│   Output: {forecasts: List[float], variances: List[float], horizon: int}   │
│                                                                             │
│  OnlineDiscoveryEngine (engine_v2.py)                                       │
│   Maintains: HypothesisPool, AdaptiveGrouper, HypothesisArbiter,           │
│              MetaController                                                  │
│   Hypothesis types: Functional, Correlational, TemporalLag, Equilibrium    │
│   .initialize(schema) → seeds all variable-pair hypotheses                 │
│   .process_row(row)   → updates all active hypotheses, arbitrates,         │
│                          promotes/prunes via MetaController                 │
│   .get_knowledge_graph() → top-K confirmed relationships as JSON            │
└─────────────────────────────────────────────────────────────────────────────┘

scarcity/causal/
┌─────────────────────────────────────────────────────────────────────────────┐
│  run_causal(specs, runtime)                                                 │
│   FeatureBuilder     → feature engineering + time series validation        │
│   Identifier         → do-calculus identification (backdoor, frontdoor)    │
│   EstimatorFactory   → DoWhy / statsmodels OLS / IV / DML estimators       │
│   Validator          → refutation tests (placebo, subset bootstrap)        │
│   ArtifactWriter     → saves DAG + effect estimates to artifacts/runs/     │
│   Output: CausalRunResult {effect_size, ci_lower, ci_upper, p_value,       │
│                             dag: dot string}                                │
└─────────────────────────────────────────────────────────────────────────────┘

scarcity/federation/
┌─────────────────────────────────────────────────────────────────────────────┐
│  FederatedAggregator                                                        │
│   Methods: FedAvg, TrimmedMean, Krum, Bulyan (Byzantine-robust)            │
│   .aggregate(spoke_weight_vectors) → global_weights + metadata             │
│   .detect_outliers(vectors, reference) → poisoning detection               │
│                                                                             │
│  PrivacyGuard                                                               │
│   .add_dp_noise(weights, epsilon) → Gaussian DP noise on gradients         │
│   Called by FederationBridge before aggregation in Mode B                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. K-SHIELD Layer — Bridge and Adapter Interactions

```
kshiked/ui/institution/backend/
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  AutoPipeline.run(df)  ← triggered when spoke uploads CSV                  │
│   Step 1  Normalize  → numeric_df, fill NaN with column median             │
│   Step 2  Anomaly    → ScarcityBridge (legacy) → EventBus.publish          │
│                          "data_window" → OnlineAnomalyDetector             │
│                          → "scarcity.anomaly_detected" → result captured   │
│                          → anomaly_scores[], peak_score, peak_index        │
│   Step 3  Discovery  → kshiked.core.ScarcityBridge                        │
│                          → OnlineDiscoveryEngine.process_row() per row     │
│                          → get_knowledge_graph() → relationships[]         │
│   Step 4  Granger    → statsmodels.grangercausalitytests() on column pairs │
│                          → p-values → confirmed temporal lead-lag pairs     │
│   Step 5  Causal DAG → run_causal(specs) → effect_size, DAG dot string    │
│   Step 6  SFC Sim    → SFCEconomy(SFCConfig).run(steps)                   │
│                          → trajectory[gdp_growth, inflation, welfare]      │
│   Step 7  Forecast   → PredictiveForecaster GARCH-VARX                    │
│                          → forecast_matrix, variance_matrix                 │
│   Step 8  Risk Prop  → 2-hop cascade from knowledge graph                  │
│   Step 9  Threat Idx → ThreatIndexReport.compute_all() from Pulse indices  │
│   Step 10 Narrative  → _build_narrative() → plain-text report              │
│   Returns: PipelineResult {anomaly_scores, relationships, sfc_trajectory,  │
│                             forecast_matrix, threat_level, narrative}       │
│                                                                             │
│  ScarcityBridge (backend/scarcity_bridge.py)                               │
│   Owns: EventBus, HypergraphStore, OnlineAnomalyDetector,                  │
│          PredictiveForecaster                                               │
│   .process_dataframe(df) → async pipeline:                                 │
│      bus.subscribe("scarcity.anomaly_detected",  anomaly_handler)          │
│      bus.subscribe("scarcity.forecasted_trends", forecast_handler)         │
│      bus.subscribe("scarcity.drg_extension_profile", drg_handler)          │
│      bus.publish("data_window", {schema, data, timestamp})                 │
│      → triggers OnlineAnomalyDetector._handle_data_window()               │
│      → triggers PredictiveForecaster._handle_data_window()                 │
│      Returns: {anomalies, forecasts, drg_profiles}                         │
│                                                                             │
│  AnalyticsEngine (backend/analytics_engine.py)                             │
│   generate_inaction_projection(severity, shock_vector, projection_steps)   │
│      → converts severity (0-10) → supply_shock magnitude (max 8%)         │
│      → builds ResearchSFCConfig, injects shock_vector at step 1           │
│      → ResearchSFCEconomy.run(steps) → summary()                          │
│      → extracts: gdp_growth, unemployment, gini, reserves_months           │
│      → narrates in plain English for executive briefing                    │
│                                                                             │
│  ExecutiveBridge (backend/executive_bridge.py)                             │
│   _compute_garch_varx_forecast (from scarcity.engine.forecasting)          │
│   get_historical_context(risk_ids)                                          │
│      → DeltaSyncManager + ProjectManager → DB queries                      │
│      → compiles risk event timeline                                         │
│   build_county_convergence(signals) → geo heat-map data                    │
│   generate_recommendation(risk, history) → CoordinationRecommendation      │
│   compute_outcome_impact(project_id) → before/after severity comparison    │
│                                                                             │
│  FederationBridge (backend/federation_bridge.py)                           │
│   .aggregate_spoke_models(payloads, method)                                │
│      → FederatedAggregator(AggregationConfig)                              │
│      → method choices: FedAvg / TrimmedMean / Krum / Bulyan               │
│      → returns (global_weights: np.ndarray, metadata: dict)               │
│   .apply_differential_privacy(weights, epsilon)                            │
│      → PrivacyGuard(PrivacyConfig).add_dp_noise(weights, epsilon)         │
│   Full Mode B flow:                                                         │
│      spoke uploads local model gradients (not raw data)                    │
│      → apply DP noise per spoke                                             │
│      → aggregate via chosen method                                          │
│      → global weights distributed back (not visible in dashboard UI)       │
│                                                                             │
│  ReportNarrator (backend/report_narrator.py)                               │
│   narrate_risk_for_executive(anomaly_result, forecast_result, context)     │
│      → templates + LLM call → plain English risk narrative                 │
│      → used in Executive "National Briefing" and "Sector Reports"          │
│                                                                             │
│  LearningEngine (backend/learning_engine.py)                               │
│   .online_update(new_data_row) → incremental model update                 │
│   .get_model_weights() → current local gradient vector                     │
│      (this is what is extracted and federated in Mode B)                   │
└─────────────────────────────────────────────────────────────────────────────┘

kshiked/pulse/
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Scrapers (async, platform-specific)                                        │
│   XScraper, TelegramScraper, RedditScraper, FacebookScraper                │
│   JijiScraper, JumiaScraper (e-commerce price monitoring)                  │
│   → yield ScraperResult {platform, text, timestamp, geo}                  │
│            │                                                                │
│            ▼                                                                │
│  IngestionOrchestrator (ingestion/orchestrator.py)                          │
│   Schedules scraper runs, de-duplicates, stores raw posts in DB            │
│            │                                                                │
│            ▼                                                                │
│  PipelineIntegration.process_posts(posts)                                  │
│   For each post:                                                            │
│      LLMProvider (Gemini / Ollama) .analyze(text)                          │
│         → ThreatClassification {tier, confidence, topic, entities}        │
│      → DBSocialPost.save()  + LLMAnalysis.save()                          │
│      → extract KShield signals from classification                          │
│            │                                                                │
│            ▼                                                                │
│  PulseSensor / AsyncPulseSensor                                             │
│   .observe(posts) → PulseState                                              │
│      threat_score: float                                                    │
│      sentiment_index: float                                                 │
│      topic_distribution: Dict[str, float]                                  │
│      geo_hotspots: List[{county, intensity}]                               │
│            │                                                                │
│            ▼                                                                │
│  ThreatIndexReport.compute_all()                                            │
│      volatility_index, food_security_index, political_tension_index        │
│      economic_stress_index, overall_threat_level                           │
│            │                                                                │
│            ▼                                                                │
│  PulseConnector (ui/connector/pulse.py)                                    │
│   .get_threat_signals() → structured signal tiles for dashboard            │
│   .get_geo_data()       → county-level heat-map payloads                   │
└─────────────────────────────────────────────────────────────────────────────┘

kshiked/simulation/
┌─────────────────────────────────────────────────────────────────────────────┐
│  KenyaCalibration                                                           │
│   calibrate_from_data(df) → SFCConfig with Kenya-realistic parameters      │
│   Defaults: mpc=0.82, tax_rate=0.17, gov_spend_ratio=0.21, crr=0.0525     │
│                                                                             │
│  ScenarioTemplates                                                          │
│   get_shock_vectors("expansionary_fiscal")  → policy_vector arrays         │
│   get_shock_vectors("supply_shock_severe")  → shock_vector arrays          │
│   Available: no_intervention, expansionary_fiscal, monetary_tightening,    │
│              supply_shock_mild/severe, debt_restructuring                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Executive Dashboard — Full Data Flow

```
executive_dashboard.py  (Role: EXECUTIVE enforced by enforce_role())
│
├── Shared Sidebar (shared_sidebar.py)
│      render_shared_sidebar(nav_items, state_key="executive_nav")
│      → returns {active_key, changed, disconnect_clicked}
│
├── NATIONAL BRIEFING  ─────────────────────────────────────────────
│   Data sources:
│     DeltaSyncManager.get_recent_signals()     → anomaly signals
│     ProjectManager.get_active_projects()      → project count
│     SectorReportGenerator.aggregate()         → composite_scores
│   Computed:
│     strain_score = weighted(avg_impact, ops_pressure, sentiment_risk)
│     stability_score = 10 − (0.5×impact + 0.3×ops_pressure + 0.2×sentiment)
│     econ_status: "Stable" / "Watch" / "Stressed"
│   Renders:
│     KPI cards (strain, stability, econ status)
│     ReportNarrator.narrate_risk_for_executive() → narrative text
│     AnalyticsEngine.generate_inaction_projection() → SFC-backed forecast
│
├── SECTOR REPORTS  ─────────────────────────────────────────────────
│   SectorReportGenerator → per-basket summary
│     → composite scores (severity, trend, confidence)
│     → historical timeline
│     → county convergence from ExecutiveBridge.build_county_convergence()
│   render_sector_report() component
│
├── NATIONAL MAP  ───────────────────────────────────────────────────
│   PulseConnector.get_geo_data() → county heat-map
│   ExecutiveBridge.build_county_convergence() → risk density per county
│   Plotly choropleth over Kenya county boundaries
│
├── THREAT INTELLIGENCE  ────────────────────────────────────────────
│   DeltaSyncManager.get_recent_signals() → sorted by severity
│   ThreatIndexReport indices (food, political, economic, overall)
│   ExecutiveBridge.generate_recommendation() → coordination cards
│
├── SOCIAL SIGNALS  ─────────────────────────────────────────────────
│   PulseConnector.get_threat_signals() → live X/news tiles
│   PulseState: sentiment_index, topic_distribution, geo_hotspots
│
├── POLICY SIMULATOR  ───────────────────────────────────────────────
│   executive_simulator.py → render_executive_simulator()
│   │
│   ├── Mode A: Strategic Crisis Analysis
│   │     Inputs: scenario (expansionary / austerity / baseline)
│   │             shock intensity, policy mix, projection steps
│   │     Simulations (3 parallel):
│   │       no_intervention, selected_scenario, alternative_scenario
│   │       Each: ResearchSFCEconomy(ResearchSFCConfig)
│   │             .initialize() → .run(steps) → trajectory[]
│   │       Bootstrap (per scenario):
│   │         _run_bootstrap_bundle(cfg.sfc, shock_vecs, steps, n=12)
│   │           → 12× SFCEconomy with ±8% parameter jitter
│   │           → _extract_bands(bundles, dim) → p25, p75 per timepoint
│   │     Charts:
│   │       2×2 subplot: GDP Growth / Inflation / Unemployment / Welfare
│   │         1st pass: shaded p25-p75 band per scenario (behind lines)
│   │         2nd pass: deterministic trajectory lines on top
│   │         Legend note: "Shaded bands = 25th-75th percentile, 12 runs"
│   │     Comparison Table:
│   │       Final-period outcomes per scenario vs baseline delta
│   │     Distributional Panel (_render_distributional_panel):
│   │       Source data: trajectory[t]["inequality"]["quintile_incomes"]
│   │       Panel 1: Q1 vs Q5 income trajectories (all 3 scenarios)
│   │       Panel 2: Pro-poor growth bar chart
│   │                Q1 growth % vs Q5 growth % per scenario
│   │                green = pro-poor (Q1 > Q5), red = regressive
│   │       Panel 3: Gini trajectory (all 3 scenarios)
│   │                Kenya benchmark line at 0.408
│   │       Panel 4: Verdict cards (3 columns)
│   │                PRO-POOR / REGRESSIVE / NEUTRAL verdict
│   │                Q1 growth %, Q5 growth %, Gini direction (up/down/flat)
│   │     Stress Test:
│   │       ResearchSFCEconomy.stress_test(shock_scenarios)
│   │       → post-shock: trade_balance, reserve_adequacy, gdp_impact
│   │
│   └── Mode B: Econometric Research Workbench
│         ResearchSFCEconomy full research tabs
│         (inequality, IO structure, financial accelerator, open economy)
│
├── ACTIVE OPERATIONS  ──────────────────────────────────────────────
│   ProjectManager.get_projects(status="active")
│   SecureMessaging.get_unread_count()
│   render_collab_room() → real-time coordination
│
├── ARCHIVE  ────────────────────────────────────────────────────────
│   DeltaSyncManager.get_historical_signals(date_range)
│   ExecutiveBridge.get_historical_context(risk_ids)
│
└── COMMAND & CONTROL / SECTOR SUMMARIES / COLLABORATION ROOM
      ProjectManager, SecureMessaging, render_collab_room()
```

---

## 5. Admin (Sector Governance Console) — Full Data Flow

```
admin_governance.py  (Role: ADMIN enforced by enforce_role())
│
├── Shared Sidebar: 3 groups
│     OVERVIEW (red)    | SPOKES (green) | OPERATIONS (black)
│
├── SECTOR OVERVIEW  ────────────────────────────────────────────────
│   DB queries (get_connection()):
│     SELECT COUNT(*) FROM institutions WHERE basket_id = ?  → spoke count
│     SELECT * FROM signals WHERE basket_id = ? ORDER BY severity
│   DeltaSyncManager.get_recent_signals() → telemetry timeline
│   Plotly scatter: severity over time, colored by status
│
├── HISTORICAL ARCHIVE  ─────────────────────────────────────────────
│   DeltaSyncManager.get_historical_signals() → paginated signal history
│
├── SPOKE REPORTS  ──────────────────────────────────────────────────
│   DeltaSyncManager.get_pending_signals() → signals awaiting review
│   Admin actions: approve / escalate / archive per signal
│   SectorReportGenerator.generate(basket_id) → sector PDF/JSON report
│   render_sector_report(report) component
│
├── DATA SHARING  ───────────────────────────────────────────────────
│   DataSharingManager.get_sharing_agreements()
│   → which spokes have consented to what level of data sharing
│   Modes: Mode A (aggregate stats only) / Mode B (FL gradients only)
│
├── DATA GOVERNANCE & SCHEMAS  ──────────────────────────────────────
│   SchemaManager.get_schemas(basket_id) → sector expected format
│   Admin can define/update required column structure
│   Schema enforced when spokes upload in local_dashboard.py
│
├── OPERATIONAL PROJECTS  ───────────────────────────────────────────
│   ProjectManager.get_projects(basket_id)
│   → war room cards (active / pending / closed)
│   ProjectManager.create_project() / update_milestone()
│   compute_outcome_impact(project_id)
│     → DeltaSyncManager before/after comparison via ExecutiveBridge
│
├── RISK PROMOTION  ─────────────────────────────────────────────────
│   Admin reviews spoke anomaly signals
│   → decides to escalate to executive level
│   → DeltaSyncManager.promote_signal(signal_id, level="executive")
│
├── COMMUNICATIONS / COLLABORATION ROOM  ───────────────────────────
│   SecureMessaging (backend/messaging.py)
│   → send/receive between admin and spokes
│   render_collab_room() → shared workspace
│
└── FEDERATED LEARNING (MODE B)  ────────────────────────────────────
      fl_dashboard.py view
      FederationBridge.aggregate_spoke_models(payloads, method)
        ← payloads: list of spoke gradient submissions
        → FederatedAggregator.aggregate() → global_weights
        → FederationBridge.apply_differential_privacy(weights, epsilon)
              → PrivacyGuard.add_dp_noise()
      Shows: submission history, convergence metrics, privacy budget
```

---

## 6. Spoke (Local Institution) Dashboard — Full Data Flow

```
local_dashboard.py  (Role: INSTITUTION enforced by enforce_role())
│
├── DATA INTAKE  ────────────────────────────────────────────────────
│   st.file_uploader → CSV
│   OntologyEnforcer.validate(df, base_ontology)
│   SchemaManager.validate(df, basket_id) → sector custom schema
│   If valid: df stored in st.session_state["spoke_df"]
│
├── SIGNAL ANALYSIS  ────────────────────────────────────────────────
│   AutoPipeline.run(df)  →  full 10-step pipeline:
│   Step 1  Normalize columns
│   Step 2  ScarcityBridge → EventBus →
│             OnlineAnomalyDetector (RRCF)
│             → anomaly_scores[], peak_score, peak_index
│   Step 3  OnlineDiscoveryEngine.process_row() × N rows
│             → knowledge_graph (top-15 relationships)
│   Step 4  Granger causality → p-value matrix
│   Step 5  run_causal(specs) → DAG + effect estimates
│   Step 6  SFCEconomy.run() → macro projection
│   Step 7  PredictiveForecaster GARCH-VARX → forecasts
│   Step 8  Risk propagation → 2-hop cascade
│   Step 9  ThreatIndexReport → threat_level
│   Step 10 ReportNarrator → narrative text
│
│   Sensitivity level (UI dropdown):
│     Public:       aggregated stats transmitted to admin
│     Restricted:   composite scores only transmitted
│     Confidential: FL Mode B — no raw data transmitted
│
├── GRANGER CAUSALITY  ──────────────────────────────────────────────
│   From AutoPipeline.result.granger_pairs
│   _render_granger_section() → heatmap of p-values
│   Shows which indicators Granger-cause which others (lag 1-4 periods)
│
├── CAUSAL NETWORK  ─────────────────────────────────────────────────
│   causal_adapter runner → run_causal() → DAG dot string
│   kshiked.ui.kshield.causal.view → DoWhy/statsmodels OLS
│   Rendered: force-directed DAG, node = indicator, edge = causal effect
│   Edge weight = effect_size, colour = positive/negative
│
├── CROSS-CORRELATIONS  ─────────────────────────────────────────────
│   numpy/pandas pairwise Pearson on uploaded df
│   Plotly heatmap of correlation matrix
│
├── EFFECT ESTIMATION  ──────────────────────────────────────────────
│   User selects cause + effect variable
│   run_causal (single spec) → effect_size ± CI
│   Shows: point estimate, 95% CI, refutation test results
│
├── ACTIVE PROJECTS  ────────────────────────────────────────────────
│   ProjectManager.get_projects(institution_id=spoke_id)
│   → task cards, milestone status, shared with which peers
│
├── INBOX  ──────────────────────────────────────────────────────────
│   SecureMessaging.get_messages(institution_id)
│   → directives from admin, peer communications
│
├── COLLABORATION ROOM  ─────────────────────────────────────────────
│   render_collab_room() → shared document/planning workspace
│
├── MODEL CONFIGURATION  ────────────────────────────────────────────
│   Privacy level selector (Public / Restricted / Confidential)
│   FL mode toggle → switches AutoPipeline to Mode B path
│   LearningEngine config: learning_rate, regularization
│
└── FL TRAINING LOG  ────────────────────────────────────────────────
      DeltaSyncManager.get_fl_submissions(institution_id)
      → timestamp, gradient_norm, accepted/rejected by admin
      LearningEngine.get_model_weights() → current local state
```

---

## 7. K-SHIELD Module — Full Data Flow

```
kshiked/ui/kshield/page.py  (accessible to all roles)
│
├── TERRAIN ANALYSIS  ───────────────────────────────────────────────
│   terrain/view.py
│   Data: World Bank CSV (auto-discovered) or user upload
│   load_world_bank_data() → pd.DataFrame (year × indicator)
│   compute_policy_terrain_analytics(df, cause, effect, policy_var):
│     OLS regression → effect slope
│     Rolling correlation windows → stability map
│     Basin of stability (speed in state space) → safe/fragile zones
│     Pareto frontier → optimal policy combinations
│     Momentum field → direction of drift
│   Charts: terrain heatmap, Pareto curve, stability basin, momentum field
│
├── CAUSAL DISCOVERY  ───────────────────────────────────────────────
│   causal/view.py
│   Lazy imports: statsmodels (Granger, ADF), DoWhy, sklearn
│   _lazy_statsmodels() / _lazy_dowhy() / _lazy_sklearn()
│   World Bank data → _to_timeseries_dataframe() → cleaned panel
│   Tabs:
│     Granger: grangercausalitytests() on all column pairs
│     DoWhy:   CausalModel → .identify_effect() → .estimate_effect()
│     ADF:     adfuller() → stationarity, integration order
│     Network: force-directed graph of confirmed causal links
│
├── IMPACT ASSESSMENT  ──────────────────────────────────────────────
│   impact/view.py + components/
│     context.py   → macro context cards (GDP, inflation, reserves)
│     metrics.py   → impact KPI computation
│     live_policy.py → policy dial / slider → immediate SFC re-run
│     llm.py        → LLM narrative on policy impact (Gemini/Ollama)
│     layout.py     → assembles all components into page
│
└── SCENARIO RUNNER  ────────────────────────────────────────────────
    simulation/view.py  (router to sub-tabs)
    │
    ├── RUN TAB (run.py)
    │     KenyaCalibration.calibrate_from_data(df) → SFCConfig
    │     ScenarioTemplates.get_shock_vectors(scenario)
    │     SFCEconomy.run(steps)  OR  ResearchSFCEconomy.run(steps)
    │     → single trajectory chart (GDP, inflation, unemployment, welfare)
    │     → stores result in st.session_state["sim_trajectory"]
    │     → stores SFCConfig in st.session_state["sim_calibration"]
    │
    ├── COMPARE TAB (core_analysis.py)
    │     Reads: st.session_state["all_scenario_results"]
    │             (multiple named runs from Run tab)
    │     Dimension selector + Uncertainty bands checkbox
    │     If bands ON:
    │       band_cache_key = f"_compare_bands_{focus_dim}"
    │       if not cached:
    │         SFCConfig ← st.session_state["sim_calibration"].config
    │         12 × SFCEconomy(jittered_cfg).run(steps)
    │         np.percentile(arr, [25, 75], axis=0) → bands
    │         cached in st.session_state[band_cache_key]
    │       Render: band trace (fill=toself, rgba 0.13 opacity) then lines
    │
    ├── MONTE CARLO TAB (advanced.py)
    │     n_runs slider (default 20) × SFCEconomy jittered runs
    │     Percentile bands: (10, 90, opacity=0.15), (25, 75, opacity=0.25)
    │     Fan chart with fill='toself' concatenated upper+lower arrays
    │     Dimension selector → repeats for GDP / inflation / unemployment
    │
    ├── PARAMETER SURFACE TAB (param_surface.py)
    │     2 parameter dropdowns (e.g. mpc × tax_rate)
    │     Grid sweep: 8×8 = 64 SFCEconomy runs
    │     Result: 2D heatmap of selected outcome at final step
    │     Hover: exact parameter values and outcome
    │
    ├── RESEARCH ENGINE TAB (research.py)
    │     ResearchSFCEconomy.run(steps) → full frame[]
    │     Tabs within tab:
    │       Summary:     twin_deficit_analysis(), external_vulnerability_index()
    │       IO Structure: inter-sector demand flow Sankey
    │       Inequality:  Gini trajectory, Palma ratio, Theil index
    │                    Lorenz curve at selected step
    │       Open Economy: REER, trade balance, reserves_months timeline
    │       Financial:    leverage ratio, credit spread, NFC timeline
    │
    ├── INEQUALITY TAB (research.py)
    │     InequalityMetrics.gini_from_quintiles()
    │     Gauge chart: Gini (0-1) with Kenya benchmark 0.408
    │     Stacked area: Q1-Q5 income shares over time
    │     Pro-poor bar: Q1 vs Q5 growth per scenario
    │     Lorenz curve at selected step vs perfect equality line
    │
    └── WHAT-IF WORKBENCH (workbench/view.py)
          WhatIfManager.run_bootstrap(base_cfg, n=8, jitter_pct=8%)
          Returns: (lower_bound, upper_bound) per dimension
          Policy dial: adjusts one parameter → reruns → shows delta
          Scenario comparison: side-by-side outcome tables
```

---

## 8. Session State — Shared Data Contract

```
All components communicate through Streamlit session state:

st.session_state key              Set by              Read by
─────────────────────────────────────────────────────────────────────────
"sim_trajectory"                  Run tab             Compare, research tabs
"sim_calibration"                 Run tab             Compare (for band jitter)
"all_scenario_results"            Run tab (multi)     Compare tab
"spoke_df"                        Data Intake         AutoPipeline, all analysis tabs
"institution_node_id"             Login/signup        All spoke tabs
"basket_id"                       Login               Admin bridge, schema lookups
"executive_nav"                   Sidebar             Executive content router
"_compare_bands_{dim}"            Compare tab         Compare tab (cache)
"_compare_bands_dim"              Compare tab         Compare tab (stale check)
"force_causal_retrain"            Sidebar checkbox    Causal connector
"causal_data_source"              Sidebar             ScarcityConnector
```

---

## 9. End-to-End Request Flow Examples

### Example A: Spoke uploads CSV, runs Signal Analysis

```
Spoke uploads CSV
  → local_dashboard.py: OntologyEnforcer.validate()
  → SchemaManager.validate(basket_id)
  → AutoPipeline.run(df)
      → EventBus.publish("data_window")
          → OnlineAnomalyDetector._handle_data_window()
          ← bus: "scarcity.anomaly_detected" → anomaly_scores[]
          → PredictiveForecaster._handle_data_window()
          ← bus: "scarcity.forecasted_trends" → forecast_matrix[]
      → OnlineDiscoveryEngine.process_row() × N
          → HypothesisPool updates, AdaptiveGrouper clusters
          ← knowledge_graph top-15 relationships
      → SFCEconomy.run(20) → macro trajectory
      → ThreatIndexReport.compute_all() → threat_level
      → ReportNarrator → narrative
  → Display: anomaly chart, forecast chart, relationship graph, narrative
```

### Example B: Executive opens Policy Simulator, runs Strategic mode

```
Executive selects scenario "Expansionary Fiscal", steps=20
  → executive_simulator.py
  → KenyaCalibration.calibrate_from_data()  OR  defaults
  → ScenarioTemplates.get_shock_vectors("expansionary_fiscal")
  → 3 × ResearchSFCEconomy.run(20):
       no_intervention, selected, alternative
       Each: SFCEconomy.step() × 20
           + OpenEconomy.step() + Financial.step()
           + IO.step() + Heterogeneous.step()
           → trajectory[t] with full inequality frame
  → 3 × _run_bootstrap_bundle(cfg.sfc, shock_vecs, 20, n=12):
       12 × SFCEconomy(jittered SFCConfig).run(20) per scenario
       → _extract_bands() → p25[], p75[] per dimension
  → 2×2 Plotly subplot:
       band traces (p25-p75 filled) rendered first
       deterministic lines rendered on top
  → _render_distributional_panel():
       Q1/Q5 income trajectories from trajectory[t]["inequality"]
       Pro-poor growth bar
       Gini trajectory + Kenya 0.408 benchmark
       Verdict cards: PRO-POOR / REGRESSIVE / NEUTRAL
  → ResearchSFCEconomy.stress_test(shock_scenarios)
       → post-shock trade_balance, reserve_adequacy
```

### Example C: Admin runs Federated Learning Mode B aggregation

```
Spokes submit gradient payloads (not raw data)
  → admin_governance.py FL tab
  → FederationBridge.apply_differential_privacy(weights, epsilon=1.0)
      → PrivacyGuard.add_dp_noise(weights) per spoke
  → FederationBridge.aggregate_spoke_models(noisy_payloads, "trimmed_mean")
      → FederatedAggregator.aggregate(updates)
          → _trimmed_mean(array, alpha=0.1)  [removes top+bottom 10%]
          → FederatedAggregator.detect_outliers() → poisoning check
      → global_weights: np.ndarray
  → global weights stored → distributed back to spokes (backend only)
  → Admin UI shows: convergence curve, privacy budget spent, outlier flags
```

  ### Example D: Institution assurance snapshot with DRG explainability

  ```
  Executive or Developer opens "Model Assurance Snapshot"
    → build_quality_assurance_snapshot() in model_quality.py
    → Collect benchmark evidence:
      fl_model_accuracy_*.json
      meta_model_accuracy_*.json
      statistical_model_accuracy_*.json
      online_model_accuracy_*.json
    → Collect robustness/traceability/deployment evidence:
      benchmark freshness, drift sensitivity, fallback coverage,
      docs + artifact traceability, deployment and config indicators
    → Collect DRG evidence for deployment realism:
      scarcity/governor/drg_core.py presence
      scarcity bridge hooks for DRG events
      logs/drg/ runtime activity
      DRG documentation candidates
    → Compute criterion scores + formulas:
      metric_credibility_score
      robustness_score
      traceability_score
      deployment_realism_score (includes DRG contribution)
    → Compute overall assurance:
      weighted sum + traffic light band + rationale note
    → Return explainability payload:
      overall formula + weighted components
      criterion formulas + score_breakdown blocks
      transparency_breakdown + recent_override_samples
      dynamic_resource_allocator detail block
      → Compute delay economics (hybrid penalty model):
        do_nothing_loss_kes_b
        act_early_loss_kes_b
        late_penalty_kes_b
    → UI rendering (executive/developer dashboards):
      KPI row + DRG chip + "Why this score?" panel
      export buttons for JSON and CSV evidence packs
      → Unified report export (all institution dashboards):
        single .zip pack with plain-language summary, metrics CSV,
        structured JSON appendix, and optional table attachments
  ```

---

## 10. Meta-Learning System — Full Component Interaction Map

The meta-learning system sits above the discovery engine and federation layer.
It observes runtime telemetry, adapts hyperparameters across domains, and feeds
updated priors back into the engine's controller and evaluator — closing the
self-improving feedback loop.

### 10.1 Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tier 4: MetaLearningAgent  (scarcity/meta/meta_learning.py)               │
│   Orchestrates per-domain observation, cross-domain aggregation,            │
│   Reptile optimization, validation, storage, and telemetry.                 │
│                                                                             │
│   Subscribes (via EventBus):                                                │
│     "federation.policy_pack"   ← federated domain performance reports      │
│     "processing_metrics"       ← runtime throughput / latency / VRAM       │
│                                                                             │
│   Publishes (via EventBus):                                                 │
│     "meta_prior_update"        → engine controller + evaluator pick up     │
│     "meta_update"              → general listeners                         │
│     "meta_metrics"             → telemetry dashboard                       │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │  feeds updated global_prior
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tier 5: MetaSupervisor / MetaIntegrativeLayer                             │
│          (scarcity/meta/integrative_meta.py)                                │
│                                                                             │
│   Subscribes (via EventBus):                                                │
│     "processing_metrics"       ← same stream as Tier 4                     │
│     "meta_telemetry"           ← snapshot from MetaLearningAgent           │
│     "fmi.meta_prior_update"    ← from Federation-Meta Interface            │
│     "fmi.meta_policy_hint"     ← resource hints from FMI                  │
│     "fmi.warm_start_profile"   ← cross-node warm-start weights             │
│     "fmi.telemetry"            ← FMI operational telemetry                 │
│                                                                             │
│   Publishes (via EventBus):                                                 │
│     "meta_policy_update"       → engine controller + evaluator             │
│     "resource_profile_hint"    → DynamicResourceGovernor                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Internal Component Interactions

```
DOMAIN OBSERVATION
──────────────────
Each federated domain publishes a "federation.policy_pack" event containing:
  { domain_id, metrics: {meta_score, gain_p50, stability_avg},
    controller: {tau, gamma_diversity},
    evaluator:  {g_min, lambda_ci} }

MetaLearningAgent._handle_policy_pack(payload)
  └─► DomainMetaLearner.observe(domain_id, metrics, params)
            │
            │  Per domain, maintains DomainMetaState:
            │    ema_score        ← exponential moving average of meta_score
            │    confidence       ← decayed + boosted by sign agreement
            │    history[]        ← rolling window of score deltas (max 20)
            │
            │  Computes adaptive meta learning rate:
            │    meta_lr = lr_min + (lr_max - lr_min) × confidence
            │    (range: 0.05 to 0.20, driven by domain confidence)
            │
            └─► DomainMetaUpdate {
                  domain_id, vector (delta_params), keys,
                  confidence, timestamp, score_delta
                }
            → buffered in _pending_updates[domain_id]

CROSS-DOMAIN AGGREGATION (triggered by "processing_metrics" event)
──────────────────────────────────────────────────────────────────
MetaScheduler.should_update(metrics) → True if update_interval elapsed

CrossDomainMetaAggregator.aggregate(pending_updates.values())
  1. Filter: drop updates where confidence < min_confidence (0.05)
  2. Union all parameter keys across domains
  3. Stack update vectors into matrix [n_domains × n_params]
     (zero-fill for missing keys per domain)
  4. Apply trimmed_mean (alpha=0.1):
     sort each parameter column, drop top+bottom 10%, mean the rest
     OR median if configured
  Returns: (aggregated_vector: np.ndarray, keys: List[str],
            meta: {participants, confidence_mean, method})

REPTILE OPTIMIZATION
────────────────────
OnlineReptileOptimizer.apply(aggregated_vector, keys, reward, drg_profile)
  1. _update_beta(drg_profile):
       if VRAM high OR latency high:  beta *= 0.8  (slow down learning)
       if bandwidth free:             beta *= 1.1  (speed up)
       clamp to [beta_init×0.5, beta_max] = [0.05, 0.30]
  2. _record_history():  save current prior (up to 10 backup versions)
  3. Update:  prior[key] += beta × aggregated_vector[key]
              (Reptile: move global prior toward task-average parameters)
  4. _update_reward(reward): reward_ema ← EMA of meta_score

  should_rollback(reward):
       if reward_ema - reward > rollback_delta (0.1):
           rollback() → restore prior from most recent history backup

VALIDATION GATE
───────────────
MetaPacketValidator.validate_update(DomainMetaUpdate) → bool
  Checks: vector not empty, confidence in range, keys present
  Invalid updates are silently dropped — never reach aggregation

PERSISTENCE
───────────
MetaStorageManager
  .save_prior(global_prior)   → JSON file on disk
  .load_prior()               → restored on agent restart (warm start)
  .save_domain_vectors()      → per-domain parameter history
  Storage root: configurable (default: ./meta_storage/)

TELEMETRY BROADCAST
───────────────────
build_meta_metrics_snapshot(reward, update_rate, gain, confidence,
                             drift_score, latency_ms, storage_mb)
  → snapshot dict with all meta health metrics

publish_meta_metrics(bus, snapshot) → "meta_metrics" topic
  → consumed by admin FL dashboard for convergence display

global_prior published to:
  "meta_prior_update"  → engine.py subscribes → _handle_meta_policy_update()
  "meta_update"        → general listeners
```

### 10.3 How Meta Prior Updates Feed Back Into the Engine

```
EventBus topic: "meta_prior_update"
    { prior: {tau, gamma_diversity, g_min, lambda_ci, ...}, meta: {...} }
             │
             ▼
scarcity/engine/engine.py  _handle_meta_policy_update(topic, payload)
    │
    ├─► controller.apply_meta_update(tau, gamma_diversity)
    │      BanditRouter.apply_meta_update():
    │        self.config.tau           = new tau
    │        self.config.gamma_diversity = new gamma_diversity
    │        Effect: changes exploration-exploitation balance in
    │                hypothesis candidate selection (Thompson / UCB / ε-greedy)
    │
    └─► evaluator.apply_meta_update(g_min, lambda_ci)
           Evaluator.apply_meta_update():
             self.drg["g_min"]     = new g_min
             self.drg["lambda_ci"] = new lambda_ci
             Effect: changes minimum gain threshold and confidence interval
                     weight used when scoring hypothesis candidates

What each meta parameter controls:
  tau            — temperature of BanditRouter exploration
                   (higher = more random exploration of hypotheses)
  gamma_diversity — diversity bonus weight in candidate selection
                   (higher = prefers hypotheses from underexplored groups)
  g_min          — minimum gain for an evaluator to accept a hypothesis
                   (higher = stricter quality gate)
  lambda_ci      — weight on confidence interval width in eval scoring
                   (higher = penalises uncertain candidates more)
```

### 10.4 MetaIntegrativeLayer (Tier 5) — Rule-Based Governance

```
MetaIntegrativeLayer.update(telemetry)  called by MetaSupervisor on each
                                         "processing_metrics" event
  │
  ├── _compute_reward(telemetry):
  │     reward = w_accept × accept_rate
  │            + w_stability × stability_avg
  │            + w_contrast × rcl_contrast
  │            - p_latency × (latency_ms / 120)
  │            - p_vram × vram_util
  │            - p_oom × oom_flag
  │     clipped to [-1.0, 1.0]
  │
  ├── _update_ema(reward):
  │     ema_reward = 0.7 × ema_reward + 0.3 × reward
  │
  ├── _apply_policies(telemetry, reward, ema_reward):
  │     Adjusts knobs (with cooldown guard per knob):
  │       if accept_rate < 0.06 and EMA flat:
  │           tau ↑ 0.1, gamma_diversity ↑ 0.05  (explore more)
  │       if stability_avg > threshold and reward improving:
  │           tau ↓ 0.05                          (exploit more)
  │       if gain_p50 < g_min_floor:
  │           g_min ↓ slightly                    (relax quality gate)
  │       if latency_ms > target:
  │           tier3_topk ↓                        (fewer candidates)
  │     Returns: policy_update dict, list of changed knobs
  │
  ├── _resource_policy(telemetry):
  │     if vram_util > 0.85:  suggest reducing batch / topk
  │     if latency_ms > 200:  suggest disabling tier2
  │     Returns: resource_profile_hint
  │
  └── _safety_checks(reward, ema_reward, prev_snapshot, changed_knobs):
        if reward drops more than safety_delta from EMA:
            _rollback_previous(prev_snapshot)  → restore all knobs
            rollback_count++
        Returns: rollback_triggered bool

  Output dict published to "meta_policy_update":
    { meta_policy_update: {tau, gamma, g_min, lambda_ci, tier3_topk, ...},
      resource_profile_hint: {...},
      meta_score, meta_score_avg,
      meta_telemetry: {decision_count, rollback_count, success_rate, ...} }
```

### 10.5 Where Meta-Learning Is Wired Into K-SHIELD

```
kshiked/core/scarcity_bridge.py  ScarcityBridge._init_subsystems()
  │
  ├── Tier 0-1: EconomicDiscoveryEngine   (discovery + hypothesis learning)
  ├── Tier 4:   MetaLearningAgent         (cross-domain Reptile optimizer)
  │               bus.subscribe("federation.policy_pack", ...)
  │               bus.subscribe("processing_metrics", ...)
  ├── Tier 5:   MetaSupervisor            (integrative governance layer)
  │               bus.subscribe("processing_metrics", ...)
  │               bus.subscribe("meta_telemetry", ...)
  └── Governor: DynamicResourceGovernor  (hardware resource control loop)
                  bus.subscribe("resource_profile", ...)
                  bus.publish("resource_profile", metrics) each tick

  All four are wired to the same shared EventBus instance.
  ScarcityBridge.meta_agent property → direct access for inspection.
  ScarcityBridge.governor property   → direct access for inspection.

  The meta_agent is started when AutoPipeline.run() calls
  kshiked.core.ScarcityBridge in step 3 (Discovery Engine).
  It listens passively and updates the global prior whenever enough
  domain policy packs accumulate and processing_metrics fires.

  Updated priors flow back to the OnlineDiscoveryEngine's BanditRouter
  (hypothesis selection) and Evaluator (hypothesis scoring) — making
  the system self-calibrate across institutions over time without
  any explicit retraining step.
```

### 10.6 Full Meta-Learning Event Flow

```
[Spoke institution uploads CSV]
       │
       ▼
AutoPipeline.run(df)
  → OnlineDiscoveryEngine.process_row() × N
  → engine publishes "processing_metrics" every K rows
       │
       ▼
MetaScheduler.record_window()
  if should_update():
       │
       ▼
  DomainMetaLearner.observe(domain_id, metrics, params)
  → DomainMetaUpdate{vector, confidence, score_delta}
  → MetaPacketValidator.validate_update() → True
  → buffered in _pending_updates
       │
       ▼
  CrossDomainMetaAggregator.aggregate()
  → trimmed_mean across all pending domain vectors
  → (aggregated_vector, keys, meta)
       │
       ▼
  OnlineReptileOptimizer.apply(aggregated_vector, reward, drg_profile)
  → beta adjusted for resource pressure
  → prior[key] += beta × aggregated_vector[key]
  → should_rollback()? → restore backup prior if reward degraded
       │
       ├──► MetaStorageManager.save_prior()  [persists to disk]
       │
       ├──► build_meta_metrics_snapshot() → publish_meta_metrics()
       │       → "meta_metrics" topic → admin FL dashboard display
       │
       └──► bus.publish("meta_prior_update", {prior, meta})
                 │
                 ├──► engine._handle_meta_policy_update()
                 │         → BanditRouter.apply_meta_update(tau, gamma)
                 │         → Evaluator.apply_meta_update(g_min, lambda_ci)
                 │
                 └──► MetaSupervisor._handle_meta_policy_update()
                           → MetaIntegrativeLayer.update(telemetry)
                           → _apply_policies() → adjusted knobs
                           → _safety_checks() → rollback if needed
                           → publish "meta_policy_update" (refined)
                                     → engine picks up final adjusted values
```

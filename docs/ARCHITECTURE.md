# System Architecture — K-Scarcity / K-SHIELD / Institution Dashboards

---

## 1. Full System Data Flow

```mermaid
flowchart LR
    subgraph RAW["Raw Data Sources"]
        R1[Twitter / X]
        R2[Facebook / Telegram]
        R3[News Feeds]
        R4[World Bank / KNBS]
        R5[Institution CSVs]
    end

    subgraph PULSE["Pulse Engine"]
        P1[Scrapers]
        P2[NLP Pipeline]
        P3[15 Signal Detectors]
        P4[PulseState\nScarcityVector · ActorStress · BondStrength]
        P5[8 Threat Indices\nPI · LEI · MRS · ECI · IWI · SFI · ECR · ETM]
        P6[ShockGenerator\nGDP · Inflation · Trade · FX · Confidence]
    end

    subgraph SCARCITY["Scarcity Engine"]
        S1[Online Discovery Engine\n15 Hypotheses]
        S2[LearnedSFCEconomy]
        S3[Meta-Learning Agent\nReptile Optimizer]
        S4[DRG — Dynamic Resource Governor]
        S5[RRCF Anomaly Detector]
        S6[Bayes VARX Forecaster]
    end

    subgraph SIM["Simulation Layer"]
        M1[SFCEconomy\nBase Macro]
        M2[ResearchSFCEconomy\nHeterogeneous · Open · Financial · IO]
        M3[SectorSimulator\n6 Sectors × 20+ Indicators]
        M4[Shock Templates\n380+]
        M5[5–10 Year Projections]
    end

    subgraph FED["Aegis Federation"]
        F1[Institution Nodes]
        F2[Gossip Consensus]
        F3[Global Meta-Aggregation]
    end

    subgraph UI["Dashboards"]
        U1[K-SHIELD]
        U2[Institution Portal]
        U3[SENTINEL]
    end

    R1 & R2 & R3 --> P1
    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    P6 --> M3
    P5 --> U3

    R4 & R5 --> S1
    S1 --> S2
    S2 --> M1 & M2
    S3 --> S2
    S4 --> S2
    S5 --> S6

    M2 --> M3 --> M4 --> M5
    M5 --> U1 & U2

    F1 --> F2 --> F3 --> S3
    R5 --> F1
```

---

## 2. Pulse Engine — Signal Detection Pipeline

```mermaid
flowchart TD
    A[Raw Social Text] --> B[NLP Tokenization\n& Entity Recognition]
    B --> C{15 Signal Detectors}
    C --> D1[Distress Signals\nFood · Water · Healthcare]
    C --> D2[Anger Signals\nRage · Dehumanization]
    C --> D3[Institutional Signals\nLegitimacy Rejection]
    C --> D4[Identity Signals\nEthno-Regional Framing]
    C --> D5[Info Warfare Signals\nRumors · Conspiracy]

    D1 & D2 & D3 & D4 & D5 --> E[PulseState Update\nScarcityVector · ActorStress · BondStrength]

    E --> F[8 Threat Indices]
    F --> F1[PI — Polarization\nLanguage extremity · Bond fracture]
    F --> F2[LEI — Legitimacy Erosion\nAuthority rejection trajectory]
    F --> F3[MRS — Mobilization Readiness\nProtest / violence probability]
    F --> F4[ECI — Elite Cohesion\nLeadership fracture signals]
    F --> F5[IWI — Information Warfare\nMisinformation intensity]
    F --> F6[SFI — Security Friction\nStability erosion]
    F --> F7[ECR — Economic Cascade Risk\nShock propagation]
    F --> F8[ETM — Ethnic Tension Matrix\n12 Kenya ethnic groups]

    F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 --> G[SimulationShockGenerator]
    G --> H1[GDP Shock]
    G --> H2[Inflation Shock]
    G --> H3[Trade Shock]
    G --> H4[FX / Currency Shock]
    G --> H5[Confidence Shock]

    H1 & H2 & H3 & H4 & H5 --> I[SFC Simulation Engine]
```

---

## 3. Scarcity Engine — Online Learning Architecture

```mermaid
flowchart TD
    subgraph Preprocessing["Pre-Processing"]
        A1[Raw Stream] --> A2[Online Winsorization\n5th–95th percentile clipping]
        A2 --> A3[Online MAD\nMedian Absolute Deviation]
        A3 --> A4[Huber Loss\nGradient Clipping]
    end

    subgraph Encoding["Sketching & Encoding"]
        A4 --> B1[CountSketch + FFT\nPolynomial Approximation]
        B1 --> B2[Tensor Sketch\nKronecker Product Compression]
        B2 --> B3[Top-K Sparse Attention\nFP16 Transformer-style]
        B3 --> B4[Lag Positional Encodings]
    end

    subgraph Hypotheses["15 Competing Hypotheses (Online)"]
        B4 --> C1[Causal — Granger Augmented Ridge]
        B4 --> C2[Correlational — Welford Pearson]
        B4 --> C3[Temporal — Recursive Least Squares VAR-p]
        B4 --> C4[Functional — Online Polynomial RLS]
        B4 --> C5[Equilibrium — Kalman Mean Reversion]
        B4 --> C6[Compositional — Sum Constraints MAE]
        B4 --> C7[Competitive — CV Zero-Sum Detection]
        B4 --> C8[Synergistic — Interaction Term Regression]
        B4 --> C9[Probabilistic — Cohen's d Distribution Shift]
        B4 --> C10[Structural — ICC Hierarchical]
        B4 --> C11[Mediating — Baron-Kenny]
        B4 --> C12[Moderating — Conditional Effects]
        B4 --> C13[Graph — Network Density]
        B4 --> C14[Similarity — Online K-Means]
        B4 --> C15[Logical — Boolean Gate Rules]
    end

    subgraph Arbitration["Arbitration & Validation"]
        C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9 & C10 & C11 & C12 & C13 & C14 & C15 --> D1[HypothesisArbiter\nParsimony + Conflict Resolution]
        D1 --> D2[Page-Hinkley\nConcept Drift Detection]
        D2 --> D3[Bootstrap CI\nConfidence Intervals]
        D3 --> D4[Spearman Concordance\nSign-Agreement Validation]
    end

    subgraph Output["Knowledge Output"]
        D4 --> E1[Causal Knowledge Graph\nEdge strengths + confidence]
        D4 --> E2[RRCF Anomaly Detector]
        D4 --> E3[Bayes VARX Forecaster]
        E1 --> E4[LearnedSFCEconomy]
        E2 & E3 --> E5[DRG — Assurance Levels\nHIGH · MEDIUM · LOW · FALLBACK]
    end
```

---

## 4. SFC Economy — Simulation Architecture

```mermaid
flowchart TD
    subgraph SFCConfig["SFCConfig (Input)"]
        I1[shock_vectors\nGDP · Inflation · Trade · FX · Confidence]
        I2[policy_vectors\nMonetary · Fiscal · Sectoral]
        I3[parameters\nMPC · CRR · tax_rate · gov_spend_ratio]
    end

    subgraph Base["SFCEconomy — 5 Sectors"]
        I1 & I2 & I3 --> B1[Households\nConsumption = MPC × income + wealth_effect]
        B1 --> B2[Firms\nInvestment = acc × ΔGDP − credit_cost × Δr]
        B2 --> B3[Banks\nLending = deposits × 1-CRR × multiplier]
        B3 --> B4[Government\nTax = tax_rate × GDP\nSpend = gov_ratio × GDP]
        B4 --> B5[Foreign\nNet Exports = CA adjustment]
        B5 --> B6[GDP Frame\ngdp_growth · inflation · unemployment\nhousehold_welfare · sector_balances]
    end

    subgraph Research["ResearchSFCEconomy — Extensions"]
        B6 --> R1[HeterogeneousHouseholds\nQ1-Q5 agents · Gini · Palma ratio]
        B6 --> R2[OpenEconomyModule\nREER · reserves · trade balance]
        B6 --> R3[FinancialAcceleratorModule\nCredit cycles · LTV · leverage]
        B6 --> R4[IOStructureModule\nAgriculture · Manufacturing · Services · Finance]
        B6 --> R5[BayesianBeliefUpdater\nShock probability distributions]
        R1 & R2 & R3 & R4 & R5 --> R6[Unified Frame\noutcomes · inequality · sector_balances · flows]
    end

    subgraph Sectors["SectorSimulator — Post-Processing"]
        R6 --> S1[Economics/Finance]
        R6 --> S2[Healthcare]
        R6 --> S3[Environment/Water]
        R6 --> S4[Social Cohesion]
        R6 --> S5[Education/Labor]
        R6 --> S6[Security]
        S1 & S2 & S3 & S4 & S5 & S6 --> S7[5–10 Year Projections\nper sector × 20+ indicators]
    end
```

---

## 5. Aegis Federation Protocol

```mermaid
sequenceDiagram
    participant N1 as Institution Node A
    participant N2 as Institution Node B
    participant G as Gossip Layer
    participant AGG as Global Aggregator
    participant META as Meta-Learning Agent

    N1->>N1: Local training on sector data
    N2->>N2: Local training on sector data

    N1->>N1: Apply Local DP (Laplace noise)
    N2->>N2: Apply Local DP (Laplace noise)

    N1->>N1: Q8 quantize update
    N2->>N2: Q8 quantize update

    N1->>G: Gossip broadcast (HKDF-SHA256 masked)
    N2->>G: Gossip broadcast (HKDF-SHA256 masked)

    G->>G: Exponential time-decay merge\nAge-weighted aggregation

    G->>AGG: Basket-level consensus update

    AGG->>AGG: Byzantine defense\nKrum → Multi-Krum → Bulyan
    AGG->>AGG: Trimmed-Mean OR Element-wise Median
    AGG->>AGG: Apply Central DP (Gaussian noise)
    AGG->>AGG: Update trust scores\n(Agreement 60% / Compliance 30% / Impact 10%)

    AGG->>META: Aggregated gradient
    META->>META: Reptile optimizer step\nGlobal Prior update

    META->>N1: Broadcast new global prior
    META->>N2: Broadcast new global prior
```

---

## 6. Institution Dashboard — Navigation Structure

```mermaid
flowchart TD
    L[Landing Page\n5 Ws — Who · What · When · Where · Why] --> A

    A[Institution Portal Login\nSector + Invite Code + Password] --> B{Role}

    B --> C[Executive Dashboard]
    B --> D[Admin Governance Console]
    B --> E[Developer Dashboard]
    B --> F[Local / Spoke Dashboard]

    C --> C1[National Briefing\nThreat Intelligence · Social Signals]
    C --> C2[Sector Reports\n7 Sectors — status grid always visible]
    C --> C3[Command & Control\nActive Operations]
    C --> C4[Policy Simulator\nScenario design + projections]
    C --> C5[Collaboration Room\nCross-institution messaging]
    C --> C6[Archive\nHistorical reports]
    C --> C7[Analytics Pillars\nSO WHAT · COMPARED TO WHAT · WHERE EXACTLY\nWHAT SHOULD I DO · DID IT WORK]

    D --> D1[Pending Approvals\nInstitution registration review]
    D --> D2[Audit Logs\nFull approval audit trail]
    D --> D3[Topology Injection\nLevel 1/2 agency hierarchy]
    D --> D4[FL Dashboard\nFederated learning round management]
    D --> D5[Admin Data Schemas\nStructured project tracking]

    E --> E1[Model Quality\nDRG assurance levels]
    E --> E2[Causal Adapter\nDiscovery engine inspection]
    E --> E3[Technical Metrics\nLatency · throughput · hypothesis counts]

    F --> F1[County Analytics\nLocalized indicators]
    F --> F2[Cost of Delay\nKES billions — Do Nothing · Act Early · Price of Late]
    F --> F3[Data Upload\nCSV → FL training trigger]
    F --> F4[Report Export\nPDF · ZIP · CSV]
```

---

## 7. K-SHIELD — Module Architecture

```mermaid
flowchart LR
    HUB[KShieldHub\nSingleton Orchestrator]

    HUB --> K1[Causal Relationships]
    HUB --> K2[Policy Terrain]
    HUB --> K3[Simulations]
    HUB --> K4[Policy Impact]

    K1 --> K1a[OnlineDiscoveryEngine\nForce-directed graph]
    K1 --> K1b[Granger causality results\nConfidence rankings]
    K1 --> K1c[Top-K relationship list\nEdge weight visualization]

    K2 --> K2a[3D Stability Landscape\nInflation × Unemployment → Instability]
    K2 --> K2b[Phase space mapping\nCurrent economy position]
    K2 --> K2c[Stability heatmap\nPolicy corridor visualization]

    K3 --> K3a[SFC + ResearchSFC\n5–10 year forward projection]
    K3 --> K3b[Shock Scenario Designer\n380+ templates]
    K3 --> K3c[Policy Constraint Editor\nMonetary · Fiscal · Sectoral]
    K3 --> K3d[4D State Cube\nGDP · Inflation · Unemployment · Welfare]
    K3 --> K3e[Scenario Library\nSave · Load · Reproduce]

    K4 --> K4a[Public Sentiment\nPolicy satisfaction by domain]
    K4 --> K4b[ScarcityVector\nFinance · Healthcare · Security · Agriculture]
    K4 --> K4c[ActorStress\nCivil society · Business · Security]
    K4 --> K4d[Social Cohesion\nTrust bonds · Institutional · Intra-group]
```

---

## 8. Report Export Pipeline

```mermaid
flowchart TD
    A[Any Institution Dashboard\nExecutive · Admin · Developer · Spoke] --> B[UnifiedReportExport]

    B --> C1[ReportNarrator\nPlain-language narrative generation]
    B --> C2[MetricsExtractor\nHeadline indicator values]
    B --> C3[StructuredAppendix\nTechnical JSON payload]

    C1 --> D[report_summary.txt\nNon-technical audience]
    C2 --> E[metrics.csv\nHeadline values]
    C3 --> F[report_payload.json\nStructured technical appendix]
    B --> G[Optional table CSVs]

    D & E & F & G --> H[ZIP Archive]
    H --> I[PDF Export\nPrimary format with instant-analysis interpretation]

    style I fill:#1a6b3c,color:#fff
```

---

## 9. Cost of Delay Engine

```mermaid
flowchart LR
    A[Threat Index\nSeverity Score 0–1] --> B[Response Window\nDays remaining]

    B --> C{Delay Model}
    C --> C1[Linear Component\nBaseRate × Severity × Days]
    C --> C2[Staged Component\nStep-change at thresholds]
    C --> C3[Exponential Component\ne^rate × Days — compounding]

    C1 & C2 & C3 --> D[Blended Loss Function]

    D --> E1[Do Nothing Loss\nKES billions — full inaction trajectory]
    D --> E2[Act Early Loss\nKES billions — cost at t=0 intervention]
    D --> E3[Price of Being Late\nE1 − E2 — marginal delay cost]

    E1 & E2 & E3 --> F[Executive Display\nWhole-number KES billions]

    style E3 fill:#b5290e,color:#fff
```

---

## 10. DRG Assurance Levels

```mermaid
flowchart TD
    A[Projection Request] --> B{Discovery Confidence}

    B -->|≥ 0.85 + recent data| C[HIGH\nReliable for policy decisions]
    B -->|0.65–0.85| D[MEDIUM\nDirectionally correct\nQuantitative uncertainty]
    B -->|< 0.65 or stale| E[LOW\nIndicative only\nManual review recommended]
    B -->|Discovery failed| F[FALLBACK\nHardcoded SFC baselines]

    C --> G[LearnedSFCEconomy\nFull discovered relationships]
    D --> H[Blended: Learned + Hardcoded\nWeighted by confidence]
    E --> I[LearnedSFCEconomy\nwith wide confidence bands]
    F --> J[BaselineSFCEconomy\nStatic Kenya 2022 calibration]

    style C fill:#1a6b3c,color:#fff
    style D fill:#b8860b,color:#fff
    style E fill:#b5290e,color:#fff
    style F fill:#555,color:#fff
```

---

## 11. Component Interaction Map (Low-Level)

```
scarcity/engine/
┌──────────────────────────────────────────────────────────────────────┐
│  EventBus (runtime/bus.py)  — async pub/sub backbone                 │
│   "data_window"                ← new data row arrives                │
│   "scarcity.anomaly_detected"  → RRCF result                         │
│   "scarcity.forecasted_trends" → Bayes VARX result                   │
│   "scarcity.drg_extension_profile" → DRG risk profile                │
│                                                                      │
│  OnlineAnomalyDetector  (RRCF — streaming, no training phase)        │
│   Output: {anomaly_score: float, is_anomaly: bool, context: dict}    │
│                                                                      │
│  PredictiveForecaster  (GARCH-VARX — multi-variate + exogenous)      │
│   Output: {forecasts: List[float], variances, horizon}               │
│                                                                      │
│  OnlineDiscoveryEngine (engine_v2.py)                                 │
│   HypothesisPool → AdaptiveGrouper → HypothesisArbiter → MetaCtrl   │
│   .process_row(row) → update all hypotheses → arbitrate → promote    │
│   .get_knowledge_graph() → top-K confirmed relationships (JSON)      │
└──────────────────────────────────────────────────────────────────────┘

scarcity/simulation/
┌──────────────────────────────────────────────────────────────────────┐
│  SFCEconomy                                                          │
│   .step() → Consumption · Investment · Tax · Gov Spend · Net Exports │
│   .run(steps) → List[frame]                                          │
│   .apply_shock(type, magnitude)                                       │
│                                                                      │
│  ResearchSFCEconomy (wraps SFCEconomy)                               │
│   + HeterogeneousHouseholdEconomy (Q1–Q5 income quintiles)           │
│   + OpenEconomyModule (REER, reserves, trade balance)                │
│   + FinancialAcceleratorModule (credit cycles, LTV, leverage)        │
│   + IOStructureModule (agriculture, manufacturing, services, finance)│
│   + BayesianBeliefUpdater (shock probability distributions)          │
│   .stress_test(shocks) → shocked scenario outcomes                   │
│   .twin_deficit_analysis() → fiscal + current account positions      │
│   .external_vulnerability_index() → 0–1 reserve adequacy            │
│   .financial_stability_index() → 0–1 leverage + credit health       │
│                                                                      │
│  WhatIfManager                                                        │
│   .run_bootstrap(base_cfg, n=8, jitter_pct=8%)                       │
│   → (mean−std, mean+std) confidence interval tuple per dimension     │
└──────────────────────────────────────────────────────────────────────┘

kshiked/core/
┌──────────────────────────────────────────────────────────────────────┐
│  ScarcityBridge                                                       │
│   .train(data_path) → 306+ causal hypotheses from World Bank data    │
│   .create_learned_economy() → SFC with discovered relationships       │
│   .get_top_relationships(k) → ranked causal chains                   │
│   .get_confidence_map() → per-variable confidence scores (0–1)       │
│   .validate() → historical accuracy score + replay validation        │
│                                                                      │
│  EconomicGovernor                                                     │
│   Enforces resource stability constraints                            │
│   Transmits monetary/fiscal policy to SFC engine                     │
│                                                                      │
│  Shocks (Phase 4–5 Stochastic)                                        │
│   ImpulseShock      → exponential decay impulse                      │
│   OUProcessShock    → Ornstein-Uhlenbeck mean reversion              │
│   BrownianShock     → Geometric Brownian Motion                      │
│   MarkovSwitchingShock → Hamilton regime-switching                   │
│   JumpDiffusionShock → Poisson jump process                          │
│   StudentTShock     → fat-tailed shocks                              │
└──────────────────────────────────────────────────────────────────────┘

kshiked/federation/  (Aegis Protocol)
┌──────────────────────────────────────────────────────────────────────┐
│  AegisNode (extends FederationClientAgent)                           │
│   Security lattice: UNCLASSIFIED / RESTRICTED / SECRET / TOP_SECRET  │
│   Trust scoring per incoming packet                                   │
│   Graph merging from external nodes                                   │
│   CryptoSigner (Ed25519 signatures)                                   │
│                                                                      │
│  Cryptographic Secure Aggregation                                     │
│   Ed25519 long-term identity + X25519 ephemeral keys                 │
│   HKDF-SHA256 pairwise masking → summation cancellation              │
│   Q8 quantization before broadcast                                    │
│                                                                      │
│  Byzantine Defense Stack                                              │
│   1. Krum — reject outlier models by pairwise Euclidean distance      │
│   2. Multi-Krum — select k safest models                              │
│   3. Bulyan — Krum survivors → Trimmed-Mean (most hardened)          │
│   4. Coordinate-wise Trimmed Mean (top 10% + bottom 10% discarded)   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 12. Security Architecture

```mermaid
flowchart TD
    subgraph Clearance["Security Lattice"]
        L4[TOP_SECRET]
        L3[SECRET]
        L2[RESTRICTED]
        L1[UNCLASSIFIED]
        L4 --> L3 --> L2 --> L1
    end

    subgraph Auth["Authentication Layers"]
        A1[Institution: PBKDF2-SHA256\n200,000 iterations]
        A2[Module Access: SHA256 gate codes]
        A3[Federation: Ed25519 signatures]
        A4[Pairwise: HKDF-SHA256 masking]
    end

    subgraph Privacy["Privacy Guarantees"]
        P1[Local DP: Laplace noise on weights]
        P2[Central DP: Gaussian noise on aggregate\nσ = sensitivity × √2ln(1.25/δ) / ε]
        P3[Q8 quantization: economic precision preserved]
        P4[L2 materiality check: suppress trivial updates]
    end

    subgraph Trust["Trust Scoring"]
        T1[Agreement score: 60% weight]
        T2[Compliance score: 30% weight]
        T3[Impact score: 10% weight]
        T1 & T2 & T3 --> T4{Trust < 0.2?}
        T4 -->|Yes| T5[Sandboxed: packets accepted\nbut silently discarded]
        T4 -->|No| T6[Normal aggregation]
    end
```

---

## 13. Kenya Economic Baselines (KNBS / World Bank 2022)

| Sector | Indicator | Baseline |
|--------|-----------|---------|
| **Economics** | GDP Growth | 5.3% |
| **Economics** | Inflation | 7.6% |
| **Economics** | Unemployment | 5.5% |
| **Healthcare** | Capacity Utilization | 72% |
| **Healthcare** | Vaccination Coverage | 68% |
| **Healthcare** | Mortality Risk | 22% |
| **Environment** | Water Access | 62% |
| **Environment** | Drought Severity | 22% |
| **Environment** | Food Security | 68% |
| **Social** | Poverty Headcount | 36.5% |
| **Social** | Inequality (Gini-equivalent) | 38.6% |
| **Social** | Cohesion Index | 54% |
| **Education** | School Attendance | 83% |
| **Education** | Labor Productivity | 1.0 (index) |
| **Security** | Stability Index | 61% |
| **Security** | Conflict Risk | 28% |
| **Security** | Institutional Trust | 42% |

**Heterogeneous Household Calibration (Q1–Q5):**

| Quintile | Income Share | MPC | Formal Employment |
|----------|-------------|-----|------------------|
| Q1 (bottom 20%) | 4% | 0.95 | 10% |
| Q2 | 8% | 0.90 | 25% |
| Q3 | 12% | 0.85 | 45% |
| Q4 | 20% | 0.75 | 70% |
| Q5 (top 20%) | 56% | 0.60 | 90% |

---

## 14. Pulse Architecture Deep Dive — How It Calculates Everything

This section describes the Pulse system as an operational analytics pipeline: what it ingests, how it transforms raw signals, how each score is calculated, how uncertainty is handled, and how the final outputs are consumed by K-SHIELD and executive dashboards.

### 14.1 Design Goals

1. Convert noisy social and open-source signals into stable, decision-grade risk metrics.
2. Separate short-lived noise from persistent structural tension.
3. Produce interpretable indices that map to policy decisions and simulation shocks.
4. Keep the system robust to source outages, spam bursts, and coordinated manipulation.

### 14.2 End-to-End Pulse Computation Graph

```mermaid
flowchart TD
    A[Platform Scrapers\nX · Telegram · Reddit · Facebook · News] --> B[Ingestion Orchestrator\nDeduplicate · Timestamp normalization · Geo hints]
    B --> C[Content Canonicalizer\nLanguage normalize · Entity extraction · Topic tagging]
    C --> D[LLM + Rule Hybrid Classifier\nThreat tier · Intent · Confidence]
    D --> E[Signal Scoring Layer\n15 detector families]
    E --> F[PulseState Builder\nScarcityVector · ActorStress · BondStrength]
    F --> G[Index Engine\n8 composite indices]
    G --> H[ThreatIndexReport\nOverall level + rationale]
    H --> I[Shock Mapping\nGDP · Inflation · Trade · FX · Confidence]
    I --> J[SFC / Research SFC Simulations]
    H --> K[Executive Dashboard tiles]
    H --> L[Admin escalation queues]
    H --> M[Archive + trend storage]
```

### 14.3 Runtime Component Interactions

```mermaid
sequenceDiagram
    participant SCR as Scrapers
    participant ORCH as IngestionOrchestrator
    participant NLP as PipelineIntegration
    participant SENSOR as AsyncPulseSensor
    participant IDX as ThreatIndexReport
    participant UI as PulseConnector

    SCR->>ORCH: ScraperResult{platform,text,timestamp,geo_hint}
    ORCH->>ORCH: De-dup + source reliability tagging
    ORCH->>NLP: Batch of normalized posts
    NLP->>NLP: LLM + rules classification
    NLP->>SENSOR: Structured signals per post
    SENSOR->>SENSOR: Update rolling windows and EWMA states
    SENSOR->>IDX: PulseState snapshot
    IDX->>IDX: Compute 8 indices + overall threat level
    IDX->>UI: Threat tiles + county intensities + rationale
```

### 14.4 Data Contracts in Pulse

Incoming normalized signal unit:

1. Source metadata: platform, timestamp_utc, language, reliability weight.
2. Content fields: text, entities, topics, sentiment proxy, urgency markers.
3. Classification fields: threat category, confidence, intent label, geo scope.

PulseState snapshot:

1. ScarcityVector: stress by economic and social scarcity dimensions.
2. ActorStress: pressure estimates for state, market, civil, and local actors.
3. BondStrength: institutional trust and social cohesion proxies.
4. Velocity terms: first derivative of key tensions over recent windows.
5. Stability terms: rolling variance and agreement consistency across sources.

### 14.5 Core Calculation Stages

#### Stage A — Ingestion Quality Control

Each raw event is assigned a quality weight $w_q$ based on:

1. Source reliability history.
2. Duplicate density in the current time bucket.
3. Parsing completeness.
4. Geo confidence.

Effective signal contribution is weighted by $w_q \in [0,1]$ before any detector scoring.

#### Stage B — Detector Scoring (15 Families)

The detector layer transforms each normalized post into detector intensities $d_i$.

1. Distress family: food/water/health stress markers.
2. Anger and escalation family: aggression, mobilization language, urgency verbs.
3. Institutional legitimacy family: governance rejection and trust erosion signals.
4. Identity polarization family: group-framing and exclusion language.
5. Information warfare family: rumor, contradiction, synthetic amplification patterns.

Per detector family, Pulse uses a hybrid score:

$$
d_i = \alpha_i \cdot s_{rule} + \beta_i \cdot s_{ml} + \gamma_i \cdot s_{context}
$$

where:

1. $s_{rule}$ is deterministic keyword/pattern evidence.
2. $s_{ml}$ is model-derived class probability.
3. $s_{context}$ captures historical alignment with known risk trajectories.
4. Coefficients are calibrated to sum to 1 for interpretability.

#### Stage C — Temporal Smoothing and Burst Control

To avoid overreaction to one-off spikes, each detector stream is smoothed with exponentially weighted moving averages and burst clamps:

$$
	ilde{d}_{i,t} = \lambda \cdot d_{i,t} + (1-\lambda)\tilde{d}_{i,t-1}
$$

Burst clamp limits extreme short-window jumps when cross-source corroboration is weak.

#### Stage D — PulseState Assembly

PulseState is assembled from grouped detector vectors:

1. ScarcityVector from distress, access, service-breakdown indicators.
2. ActorStress from institutional conflict, mobilization, and pressure cues.
3. BondStrength from trust and cohesion evidence (inverted for risk).

This converts post-level events into state-level situational vectors.

### 14.6 How the 8 Threat Indices Are Computed

All indices are normalized to $[0,1]$, then scaled for dashboard display.

```mermaid
flowchart LR
    A[Smoothed Detector Streams] --> B[Domain Aggregators]
    B --> PI[PI Polarization]
    B --> LEI[LEI Legitimacy Erosion]
    B --> MRS[MRS Mobilization Readiness]
    B --> ECI[ECI Elite Cohesion]
    B --> IWI[IWI Information Warfare]
    B --> SFI[SFI Security Friction]
    B --> ECR[ECR Economic Cascade Risk]
    B --> ETM[ETM Ethnic Tension Matrix]
    PI & LEI & MRS & ECI & IWI & SFI & ECR & ETM --> C[Overall Threat Synthesizer]
```

Index construction principles:

1. PI: weights identity-framing intensity, antagonistic sentiment, and cross-group hostility transitions.
2. LEI: weights anti-institution narratives, compliance refusal, and trust-drop velocity.
3. MRS: weights coordination language, action imperatives, and temporal urgency concentration.
4. ECI: measures fragmentation among leadership and elite communication clusters.
5. IWI: combines contradiction density, rumor propagation velocity, and source anomaly patterns.
6. SFI: combines incident pressure, protective response strain, and security narrative volatility.
7. ECR: maps scarcity and instability cues into economic stress propagation potential.
8. ETM: matrix score across protected identity dimensions and regional overlap stress.

### 14.7 Overall Threat Level Synthesis

The final threat score is a weighted composition plus guardrails:

$$
T = \sum_{k=1}^{8} \omega_k I_k + \phi \cdot V - \psi \cdot C
$$

where:

1. $I_k$ are the eight index values.
2. $V$ is volatility pressure from recent variance and acceleration.
3. $C$ is cross-source consensus confidence (higher confidence reduces false alarms).
4. $\omega_k$ are domain weights tuned for policy relevance.

Threat level mapping:

```mermaid
stateDiagram-v2
    [*] --> LOW
    LOW --> GUARDED: T >= t1
    GUARDED --> ELEVATED: T >= t2
    ELEVATED --> HIGH: T >= t3
    HIGH --> CRITICAL: T >= t4

    CRITICAL --> HIGH: T < t4 - hysteresis
    HIGH --> ELEVATED: T < t3 - hysteresis
    ELEVATED --> GUARDED: T < t2 - hysteresis
    GUARDED --> LOW: T < t1 - hysteresis
```

Hysteresis prevents rapid level-flipping when scores hover near thresholds.

### 14.8 Geo Computation for County Heatmaps

County intensity is computed as a blended geo score:

1. Explicit geo mentions in post text.
2. Source metadata geo hints.
3. Topic-to-county priors from historical event distributions.

Each county score is then time-decayed and normalized across the active window.

### 14.9 Calibration, Drift, and Reliability Controls

Pulse includes ongoing calibration loops:

1. Baseline calibration to prevent level inflation during normal high-volume periods.
2. Seasonal adjustment for recurring event cycles.
3. Drift detection on detector distributions.
4. Reliability down-weighting for sources that diverge from corroborated outcomes.
5. Confidence floors before promoting signals to executive critical tiers.

### 14.10 How Pulse Feeds the Simulation Layer

Pulse outputs are transformed into scenario-ready shock vectors:

```mermaid
flowchart TD
    A[ThreatIndexReport] --> B[Shock Mapper]
    B --> C1[Demand confidence shock]
    B --> C2[Supply disruption shock]
    B --> C3[Inflation pressure shock]
    B --> C4[Trade/external balance shock]
    B --> C5[Institutional friction shock]
    C1 & C2 & C3 & C4 & C5 --> D[SFCConfig shock_vectors]
    D --> E[SFCEconomy / ResearchSFCEconomy projections]
    E --> F[Policy recommendation views]
```

This is the key bridge from social signal intelligence to macroeconomic and sectoral simulation outcomes.

### 14.11 Output Products Produced by Pulse

1. Threat tiles for Executive and Admin dashboards.
2. Geo hotspot payloads for county maps.
3. Trend vectors for archive and historical comparison.
4. Structured rationale payload for explainability and reporting.
5. Shock vectors consumed by simulation engines.

### 14.12 Operational Interpretation Guide

1. Rising PI + LEI with stable MRS: growing social fracture, but low immediate mobilization risk.
2. Rising MRS + SFI together: prioritize near-term coordination and operational readiness.
3. High IWI with low consensus confidence: monitor closely; avoid overreaction to likely manipulation.
4. Rising ECR with moderate social indices: economic interventions may reduce escalation before security hardening is needed.

This Pulse design ensures that high-volume signal streams are converted into interpretable, calibrated, and policy-actionable intelligence rather than raw sentiment noise.

---

## 15. Collaboration, Data Sharing, Privacy, and Encryption Architecture

This section documents how institutions collaborate through the dashboard while preserving privacy, enforcing consent boundaries, and protecting data in transit and at rest.

### 15.1 Collaboration Design Goals

1. Enable cross-institution coordination without forcing raw data exposure.
2. Support role-specific visibility across Executive, Admin, and Local views.
3. Provide verifiable audit trails for every promotion, share, and action.
4. Preserve analytical utility while minimizing privacy leakage risk.
5. Enforce secure transport and cryptographic integrity end to end.

### 15.2 Collaboration Architecture (Dashboard + Backend)

```mermaid
flowchart LR
    subgraph Local[Local Institution Dashboard]
        L1[Signal Analysis]
        L2[Inbox]
        L3[Collaboration Room]
        L4[Data Sharing Preferences]
    end

    subgraph Admin[Admin Governance Dashboard]
        A1[Spoke Reports]
        A2[Data Sharing Manager]
        A3[Escalation Controls]
        A4[Operational Projects]
    end

    subgraph Exec[Executive Dashboard]
        E1[Threat Intelligence]
        E2[National Briefing]
        E3[Coordination Recommendations]
    end

    subgraph Services[Collaboration Services]
        S1[SecureMessaging]
        S2[DeltaSyncManager]
        S3[ProjectManager]
        S4[Audit Logger]
    end

    subgraph Privacy[Privacy + Federation]
        P1[Policy/Consent Gate]
        P2[Differential Privacy Layer]
        P3[Federated Aggregator]
    end

    L1 --> S2
    L2 --> S1
    L3 --> S1
    L4 --> P1
    P1 --> A2
    A1 --> A3
    A3 --> S2
    S2 --> E1
    S2 --> E2
    E3 --> S1

    L1 --> P2 --> P3
    P3 --> A4

    S1 --> S4
    S2 --> S4
    P1 --> S4
```

### 15.3 Data Sharing Contract by Mode

The platform supports explicit sharing modes tied to governance policy and institution consent.

1. Mode A (Aggregated Analytics): shares aggregate signals and summary indicators only.
2. Mode B (Federated Learning): shares model updates or gradient-like artifacts, not raw source records.
3. Restricted/Confidential paths: tighten fields and resolution based on policy and role.

```mermaid
flowchart TD
    D0[Local Dataset / Events] --> G{Sharing Mode}
    G -->|Mode A| M1[Aggregate Stats\ncounts, trends, summary risk]
    G -->|Mode B| M2[Model Update Artifacts\nweights/gradients/metadata]

    M1 --> V1[Schema + Consent Validation]
    M2 --> V2[Schema + Consent Validation]

    V1 --> R1[Admin Review / Sector Views]
    V2 --> R2[Federated Aggregation Pipeline]

    R1 --> X[Executive Summaries]
    R2 --> X
```

### 15.4 Differential Privacy Pipeline

The privacy pipeline reduces re-identification risk before collaborative learning or cross-node aggregation.

```mermaid
flowchart LR
    I[Local Model Update] --> LDP[Local DP Noise]
    LDP --> Q[Quantization / Normalization]
    Q --> SEC[Secure Aggregation Exchange]
    SEC --> AGG[Robust Aggregation]
    AGG --> CDP[Central DP Noise]
    CDP --> OUT[Global Update + Metadata]
```

DP control principles:

1. Local perturbation is applied before leaving the institution boundary.
2. Robust aggregation mitigates poisoning and outlier updates.
3. Central perturbation protects aggregate release surfaces.
4. Privacy budget and contribution policy are tracked for governance visibility.

### 15.5 Encryption and Integrity Model

```mermaid
sequenceDiagram
    participant Node as Institution Node
    participant Mesh as Collaboration/Federation Mesh
    participant Agg as Aggregator Service
    participant Dash as Dashboard Services

    Node->>Node: Sign payload (Ed25519 identity)
    Node->>Mesh: Encrypt exchange channel (ephemeral key agreement)
    Mesh->>Agg: Forward masked contribution
    Agg->>Agg: Verify signature + trust policy
    Agg->>Dash: Publish accepted aggregate metadata
    Dash->>Dash: Persist audit record with immutable event id
```

Security guarantees in collaborative paths:

1. Authenticity: signed contributions and actor identity checks.
2. Confidentiality: encrypted transport for inter-node exchange.
3. Integrity: signature verification and tamper-evident logs.
4. Non-repudiation: auditable event trail for governance actions.

### 15.6 Dashboard-Level Collaboration Controls

Control surfaces by role:

1. Local dashboard:
   selects sharing mode, reviews outgoing scope, monitors inbox and directives.
2. Admin dashboard:
   approves/escalates signals, manages sharing agreements, governs schemas, monitors federated rounds.
3. Executive dashboard:
   consumes promoted intelligence, issues coordination directives, tracks operation status.

### 15.7 Collaboration Room and Messaging Flow

```mermaid
flowchart TD
    C1[Institution User Action] --> M1[SecureMessaging]
    M1 --> M2[Role + Scope Authorization]
    M2 --> M3[Delivery Queue]
    M3 --> M4[Recipient Inbox]
    M4 --> M5[Acknowledgement / Follow-up]
    M5 --> AUD[Audit Trail]
```

Messaging design properties:

1. Role-scoped delivery prevents unauthorized cross-sector visibility.
2. Message lifecycle events are auditable.
3. Collaboration threads can be linked to projects and escalation events.

### 15.8 Governance, Audit, and Explainability

Every collaboration-critical operation produces an auditable event:

1. sharing mode changes,
2. schema/consent validation outcomes,
3. promotion/escalation decisions,
4. federated submission acceptance/rejection,
5. directive issuance and acknowledgement.

This ensures policy compliance, post-incident traceability, and explainable collaboration outcomes across institutions.

### 15.9 Public vs Internal Disclosure Guidance

Safe to publish externally:

1. collaboration architecture concepts,
2. sharing-mode semantics,
3. high-level DP and encryption posture,
4. governance and audit principles.

Keep internal only:

1. exact privacy budget values and per-round thresholds,
2. key rotation schedules and key management internals,
3. trust cutoffs, rejection heuristics, and abuse-defense tuning,
4. infrastructure topology and operational endpoint details.

This design allows secure multi-institution collaboration while minimizing sensitive-data exposure and preserving accountability.

---

## 16. Scarcity Core Folder Architecture

This section maps the `scarcity/` package into execution layers, shows how modules connect at runtime, and explains how discovery, simulation, federation, and governance compose into one operating core.

### 16.1 Package Layer Map

```mermaid
flowchart TD
    subgraph IO[Ingestion and Streams]
        S1[stream/]
        S2[synthetic/]
        S3[runtime/]
    end

    subgraph Intelligence[Intelligence and Discovery]
        I1[engine/]
        I2[causal/]
        I3[analytics/]
    end

    subgraph Decision[Simulation and Decisioning]
        D1[simulation/]
        D2[governor/]
        D3[economic_config.py]
    end

    subgraph Distributed[Federated and Meta]
        F1[federation/]
        F2[meta/]
        F3[fmi/]
    end

    subgraph Products[Delivery Surfaces]
        P1[dashboard/]
        P2[tests/]
    end

    S1 --> I1
    S2 --> I1
    S3 --> I1
    I1 --> I2
    I1 --> I3
    I2 --> D1
    I3 --> D1
    D1 --> D2
    D2 --> P1
    I1 --> F1
    F1 --> F2
    F2 --> D1
    F3 --> D1
    D1 --> P2
```

### 16.2 Runtime Backbone and Event Flow

The `runtime/` layer is the orchestration plane that keeps stream processing, model updates, and simulation triggers synchronized.

```mermaid
sequenceDiagram
    participant SRC as stream/ source
    participant RT as runtime/ orchestrator
    participant ENG as engine/ discovery
    participant ANA as analytics/causal
    participant SIM as simulation/
    participant GOV as governor/
    participant UI as dashboard/

    SRC->>RT: normalized event window
    RT->>ENG: process_row / process_batch
    ENG->>ANA: candidate relationships + confidence
    ANA->>SIM: calibrated constraints and shock hints
    SIM->>GOV: projected trajectories and risk envelopes
    GOV->>UI: policy-safe outputs + assurance level
```

Design properties:

1. Runtime sequencing is deterministic for replayability.
2. Discovery updates and simulation runs are loosely coupled by typed handoff payloads.
3. Governor checks are final-stage gates before dashboard publication.

### 16.3 Discovery Core (`engine/` + `causal/`)

The discovery core runs continuous hypothesis competition and promotion under uncertainty.

```mermaid
flowchart LR
    A[Incoming features] --> B[Hypothesis pool update]
    B --> C[Scoring and arbitration]
    C --> D{promotion threshold met?}
    D -->|yes| E[relationship accepted]
    D -->|no| F[relationship retained as tentative]
    E --> G[causal/ validation]
    F --> B
    G --> H[confidence-weighted graph]
    H --> I[simulation-ready linkage map]
```

Operational intent:

1. Keep multiple structural explanations active instead of committing too early.
2. Promote only relationships that remain stable under rolling updates.
3. Preserve confidence metadata so downstream modules can adapt aggressiveness.

### 16.4 Simulation Core (`simulation/`)

The simulation layer is the state-transition engine for policy exploration and stress testing.

```mermaid
flowchart TD
    C0[Config and priors] --> C1[Baseline state initialization]
    C1 --> C2[Apply shocks and policy vectors]
    C2 --> C3[Advance sector transitions]
    C3 --> C4[Compute macro and welfare outputs]
    C4 --> C5[Uncertainty envelopes / scenario set]
    C5 --> C6[Decision payloads]
```

Core guarantees:

1. Same config and seed produce reproducible trajectories.
2. Discovery confidence can scale scenario width and caution flags.
3. Outputs are shaped for both analyst depth and executive summaries.

### 16.5 Governance Core (`governor/`)

The governor is the policy-safety and quality-control layer between model output and operational use.

```mermaid
flowchart LR
    A[Simulation outputs] --> B[Constraint checks]
    B --> C[Stability and feasibility tests]
    C --> D{pass?}
    D -->|yes| E[release with assurance label]
    D -->|no| F[fallback / conservative profile]
    E --> G[dashboard + export]
    F --> G
```

Governor responsibilities:

1. Enforce hard constraints and risk boundaries.
2. Convert technical uncertainty into explicit assurance levels.
3. Trigger safer fallback profiles when data quality or model stability drops.

### 16.6 Federated and Meta Layer (`federation/` + `meta/` + `fmi/`)

The distributed layer improves generalization across institutions while preserving local data boundaries.

```mermaid
flowchart TD
    N1[Institution node A] --> F[federation/ secure exchange]
    N2[Institution node B] --> F
    N3[Institution node C] --> F
    F --> M[meta/ aggregation and prior update]
    M --> X[fmi/ and simulation adapters]
    X --> ENG[engine/]
    X --> SIM[simulation/]
```

Composition rules:

1. Local nodes contribute update artifacts, not raw operational records.
2. Meta updates feed both discovery priors and simulation parameter adaptation.
3. FMI adapters keep cross-model coupling explicit and controlled.

### 16.7 Stream, Synthetic, and Testing Scaffolds

Supporting packages ensure safe iteration and reliable deployment behavior:

1. `stream/` provides live ingestion connectors and event normalization interfaces.
2. `synthetic/` provides controlled stress and scenario generation for calibration and demos.
3. `tests/` anchors regression checks across discovery, simulation, federation, and governance boundaries.

### 16.8 Why This Folder Design Scales

The `scarcity/` layout is intentionally layered so the platform can evolve without tight coupling:

1. Ingestion and runtime concerns are separated from model logic.
2. Discovery and simulation can iterate independently but exchange typed contracts.
3. Federation and meta-learning remain optional accelerators, not hard runtime dependencies.
4. Governance remains a mandatory release gate for operational safety.

This is the core Scarcity design: continuous discovery, guarded simulation, and secure distributed learning assembled as a modular architecture rather than a monolithic pipeline.

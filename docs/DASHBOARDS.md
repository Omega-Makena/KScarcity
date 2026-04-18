# Dashboard Guide — K-Scarcity

Four Streamlit dashboards serve different audiences and purposes.

---

## 1. Institution Portal
**Entry point:** `kshiked/ui/institution/page.py`  
**Default port:** 8506  
**Audience:** Institutional users (executives, analysts, admins, field officers)

### Onboarding Flow
```
Register (sector + invite code) → Admin Approval → Node Provisioned → Upload Data → FL Round
```

### Role-Based Sub-Dashboards

#### Executive Dashboard
Navigation items:

| Section | Sub-Items |
|---------|-----------|
| **Intelligence** | National Briefing, Threat Intelligence, Social Signals, National Map |
| **Sectors** | Sector Reports (7 sectors, always visible), Sector Summaries |
| **Command** | Active Operations, Command & Control, Policy Simulator, Collaboration Room, Archive |

**Analytics Pillars** (always displayed):

| Pillar | Question Answered |
|--------|------------------|
| SO WHAT | Why does this matter right now? |
| COMPARED TO WHAT | How does this compare to baseline / peers? |
| WHERE EXACTLY | Which county / sector / group is affected? |
| WHAT SHOULD I DO | Recommended interventions with cost estimates |
| DID IT WORK | Historical outcome tracking for previous decisions |

**Cost of Delay Panel:**
- Do Nothing Loss (KES billions)
- Act Early Loss (KES billions)
- Price of Being Late (marginal delay cost)

---

#### Admin Governance Console

| Feature | Description |
|---------|-------------|
| Pending Approvals | Review institution registrations, approve/reject with audit trail |
| Topology Injection | Assign Level 1 / Level 2 hierarchy roles |
| FL Dashboard | View federated learning rounds, model registry |
| Admin Schemas | Structured project tracking across sectors |
| Security Lattice | Assign clearance levels to approved nodes |

---

#### Developer Dashboard

| Feature | Description |
|---------|-------------|
| DRG Assurance | View confidence levels per projection (HIGH / MEDIUM / LOW / FALLBACK) |
| Causal Adapter | Inspect discovered causal graph, top relationships |
| Technical Metrics | Latency, throughput, hypothesis pool sizes |
| Model Quality | QA snapshots, drift alerts, validation scores |

---

#### Local / Spoke Dashboard

| Feature | Description |
|---------|-------------|
| County Analytics | Localized economic and social indicators |
| Cost of Delay | County-level loss projections in KES billions |
| Data Upload | CSV upload → triggers FL training round (lookback 168h) |
| Report Export | PDF / ZIP / CSV per county |

---

### Report Export
All institution dashboards expose a unified export:

```
.zip
├── report_summary.txt      ← plain-language narrative
├── report_payload.json     ← structured technical appendix
├── metrics.csv             ← headline indicator values
└── [optional table CSVs]
```

PDF export is the primary format with enriched instant-analysis interpretation.

---

## 2. K-SHIELD Command & Control
**Entry point:** `kshiked/ui/kshield/page.py`  
**Default port:** 8505  
**Audience:** Policy analysts, economists, defence planners

Four sub-modules, pre-warmed in background threads on startup:

### Causal Relationships
- Force-directed network graph of all discovered economic relationships
- Edge thickness = Granger confidence weight
- Node colour = sector (Finance / Health / Agriculture / Security / etc.)
- Top-K ranked relationships panel with causal direction and confidence score
- Powered by `OnlineDiscoveryEngine` trained on World Bank Kenya data

### Policy Terrain
- 3D surface: inflation axis × unemployment axis → instability score (Z)
- Current economy position marked on surface
- Phase space trajectory showing historical path
- Stability corridor shading
- Identifies which policy combinations shift economy toward stable attractor

### Simulations
- Full `ResearchSFCEconomy` runs (5–10 year horizon)
- 380+ shock templates (drought, cholera, insurgency, FX crisis, etc.)
- Policy constraint editor (monetary, fiscal, sectoral)
- 4D State Cube: GDP growth × Inflation × Unemployment × Household Welfare
- Scenario library: save / load / reproduce named scenarios
- Execution modes: `SINGLE_SECTOR`, `MULTI_SECTOR`, `FULL_SIMULATION`

### Policy Impact
- Public sentiment on active policies (scraped + modelled)
- ScarcityVector by domain: Finance, Healthcare, Security, Agriculture, Water
- ActorStress levels: Civil Society, Business, Security apparatus
- Social Cohesion breakdown: Trust bonds, Institutional, Intra-group
- Powered by live PulseState primitives

---

## 3. SENTINEL — Live Threat Command Center
**Entry point:** `kshiked/ui/sentinel_dashboard.py`  
**Default port:** 8507  
**Audience:** Operational security and intelligence teams

| Tab | Description |
|-----|-------------|
| **Home** | System status, active alerts summary |
| **Live Map** | Real-time Kenya county threat map (choropleth by threat index) |
| **Federation** | Multi-node gossip topology graph, node trust scores |
| **Signal Analysis** | Deep-dive into individual signal detections with NLP spans |
| **Policy Chat** | Natural-language policy recommendation chatbot |
| **Causal Sim** | Interactive causal path testing (tweak variable → see downstream effects) |
| **Operations** | Active incident tracking, escalation queue |
| **Executive View** | C-level briefings, summarised from operational data |

---

## 4. Home / Landing Page
**Entry point:** `kshiked/ui/home/page.py`  
**Audience:** All users — system orientation

3×3 CSS grid with the **5 Ws** orientation cards:

| Card | Content |
|------|---------|
| **Who** | Which institutions and actors are in the system |
| **What** | What the system monitors and models |
| **When** | Data freshness, last FL round, last causal run |
| **Where** | Geographic coverage (counties, sectors) |
| **Why** | Strategic rationale and mission framing |

Navigation links to all four dashboards.

---

## Running All Dashboards

```bash
# Institution Portal (primary)
streamlit run kshiked/ui/institution/page.py --server.port 8506

# K-SHIELD analytical module
streamlit run kshiked/ui/kshield/page.py --server.port 8505

# SENTINEL operational
streamlit run kshiked/ui/sentinel_dashboard.py --server.port 8507

# Home / landing
streamlit run kshiked/ui/home/page.py --server.port 8504
```

---

## Authentication Summary

| Dashboard | Method | Notes |
|-----------|--------|-------|
| Institution Portal | Institution ID + PBKDF2-SHA256 password | Demo mode: username only |
| Admin Console | Clearance level ≥ RESTRICTED | Set in federation node provisioning |
| K-SHIELD | SHA256 module access code | Set via `config/access_codes.json` |
| SENTINEL | Federation node trust score | Aegis Protocol clearance |

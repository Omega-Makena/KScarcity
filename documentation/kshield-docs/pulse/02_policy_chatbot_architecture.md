# Policy Impact Chatbot — Architecture Document

**Version**: 1.0  
**Date**: February 18, 2026  
**Status**: Design — Ready for Implementation

---

## 1. Vision

An AI-powered policy intelligence chatbot that allows analysts to upload Kenyan parliamentary bills and get immediate, data-backed impact predictions by cross-referencing against historical social media reactions, economic indicators, and news streams.

**User Story**: *"As a SENTINEL analyst, I paste a new Finance Bill section and the system tells me: which demographics will react, how severely, in which counties, and when — backed by historical pattern matching from the 2024 Finance Bill protests."*

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  POLICY IMPACT CHATBOT                                               │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ INPUT LAYER                                                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │  │
│  │  │ Paste    │  │ PDF      │  │ URL      │  │ Chat Follow  │ │  │
│  │  │ Text     │  │ Upload   │  │ Scraper  │  │ Up Questions │ │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬────────┘ │  │
│  │       └──────────────┴────────────┴───────────────┘          │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ EXTRACTION ENGINE (qwen2.5:3b)                                 │  │
│  │  • Bill section parser → provisions, tax changes, levies       │  │
│  │  • Affected sector identification (12 PolicySector tags)       │  │
│  │  • Target demographic extraction (youth, traders, counties)    │  │
│  │  • Monetary impact values (% change, KES amounts)              │  │
│  │  • Severity scoring (0-1 scale)                                │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ SEARCH & MATCH ENGINE                                          │  │
│  │                                                                │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │ Embedding   │  │ Historical   │  │ Economic             │ │  │
│  │  │ Search      │  │ Bill Matcher │  │ Indicators           │ │  │
│  │  │ (nomic)     │  │ (12 events)  │  │ (KNBS/CBK)           │ │  │
│  │  │             │  │              │  │                      │ │  │
│  │  │ Tweets →    │  │ Match to:    │  │ Inflation rate       │ │  │
│  │  │ similarity  │  │ Finance Bill │  │ Fuel prices          │ │  │
│  │  │ to bill     │  │ Housing Levy │  │ Unemployment         │ │  │
│  │  │ provisions  │  │ SHIF, etc.   │  │ M-Pesa volumes      │ │  │
│  │  └──────┬──────┘  └──────┬───────┘  └──────────┬───────────┘ │  │
│  │         │                │                      │             │  │
│  │  ┌──────┴──────┐  ┌──────┴───────┐                           │  │
│  │  │ News Cache  │  │ Incident     │                           │  │
│  │  │ (14 topics) │  │ History      │                           │  │
│  │  │ policies.   │  │ (2000-2026)  │                           │  │
│  │  │ json, etc.  │  │ CSV data     │                           │  │
│  │  └─────────────┘  └──────────────┘                           │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ PREDICTION ENGINE (qwen2.5:3b)                                 │  │
│  │  • Mobilization probability per provision (0-1)                │  │
│  │  • County risk heatmap (47 counties scored)                    │  │
│  │  • Timeline prediction (when would protests peak?)             │  │
│  │  • Historical comparison ("similar to Finance Bill 2024")      │  │
│  │  • Narrative forecast (which archetypes will emerge?)          │  │
│  │  • Role prediction (who mobilizes: youth, unions, politicians?)│  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ MONITORING ENGINE (continuous)                                  │  │
│  │  • Real-time stream watch on bill keywords + hashtags          │  │
│  │  • Sentiment shift tracking (before/after announcement)        │  │
│  │  • Escalation alerts when thresholds crossed                   │  │
│  │  • Policy phase tracking (LEAK→ANNOUNCE→REACT→MOBILIZE→...)   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ UI: Streamlit Chat Panel                                       │  │
│  │  • Chat history with context memory                            │  │
│  │  • Inline visualizations (risk heatmap, timeline, clusters)    │  │
│  │  • Source citations (which tweets/articles support predictions) │  │
│  │  • Export: PDF report, CSV data, JSON API                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Breakdown

### 3.1 New Modules to Build

```
kshiked/pulse/llm/
├── policy_chatbot.py        # Core chatbot engine (conversation + orchestration)
├── policy_extractor.py      # Bill document parsing + provision extraction
├── policy_search.py         # Multi-source search engine (tweets, news, econ)
├── policy_predictor.py      # Impact prediction + historical matching
└── policy_monitor.py        # Real-time stream monitoring + alerts

kshiked/ui/sentinel/
└── policy_chat.py           # Streamlit chat UI (embedded in SENTINEL)

kshiked/pulse/llm/
└── policy_chatbot_standalone.py  # Standalone Streamlit app (runs independently)
```

### 3.2 Existing Modules to Leverage

| Module | What We Use |
|--------|-------------|
| `prompts_kenya.py` | `KENYA_POLICY_SYSTEM`, `SHENG_GLOSSARY`, `KENYA_COUNTIES`, `KENYA_POLITICAL_CONTEXT_2025` |
| `ollama.py` | `OllamaProvider._generate_json()`, `_generate_text()`, `analyze_policy_impact()` |
| `embeddings.py` | `find_similar()`, `cluster_texts()`, `similarity_matrix()` |
| `signals.py` | `KShieldSignal`, `ThreatSignal`, `ContextAnalysis` |
| `analyzer.py` | `KShieldAnalyzer.analyze()` for deep-dive on flagged content |
| `batch_processor.py` | Scan full tweet CSV for bill-related content |
| `config.py` | `OllamaConfig`, `AnalysisTask` |
| `policy_events.py` | `PolicyEvent`, `PolicyPhase`, `build_kenya_2026_events()`, `PolicySector` |

### 3.3 Data Sources

| Source | Path | Content | Use |
|--------|------|---------|-----|
| Policy tweets | `data/synthetic_kenya_policy/tweets.csv` | Tweets with `policy_event_id`, `policy_phase`, `stance_score` | Primary tweet corpus for historical matching |
| General tweets | `data/synthetic_kenya/tweets.csv` | Tweets with threat/escalation scores | Broader social signal context |
| News cache | `data/news_cache/policies.json` | Real Kenyan policy news articles | Background context for prediction |
| News (all) | `data/news_cache/*.json` | 14 topic-specific news caches | Cross-domain impact analysis |
| Incidents | `data/kshield_kenya_unified_incidents_*.csv` | Historical incidents 2000-2026 | Violence/protest pattern matching |
| Policy scenarios | `scarcity/synthetic/policy_events.py` | 12+ fully defined Kenya policy events | Template for new bill classification |
| Economic data | KNBS/CBK (external) | Inflation, GDP, unemployment, M-Pesa volumes | Economic context for impact scoring |
| Kenya Gazette | `kenyagazette.go.ke` (scrape) | Official bill publications | Auto-ingest new bills |
| Parliament | `parliament.go.ke` (scrape) | Bill tracker, committee reports | Legislative stage tracking |

---

## 4. Conversation Flow

### 4.1 First Contact — Bill Ingestion

```
User: "Analyze the Finance Bill 2026"
                    │
                    ▼
        ┌─────────────────────────┐
        │ How was it provided?     │
        ├─────────────────────────┤
        │ • Pasted text → direct  │
        │ • PDF upload → extract  │
        │ • URL → scrape + parse  │
        │ • Name only → search    │
        │   Kenya Gazette / news  │
        └─────────┬───────────────┘
                  ▼
    PolicyExtractor.extract_provisions()
                  │
                  ▼
    Returns: BillAnalysis {
        title: "Finance Bill 2026"
        sector: TAXATION
        provisions: [
            {clause: "Section 12", description: "VAT on bread 16%",
             affected: ["low_income", "traders"], 
             monetary_impact: "+16% bread cost",
             severity: 0.85},
            {clause: "Section 23", description: "Digital services tax 3%",
             affected: ["tech_companies", "M-Pesa_users"],
             monetary_impact: "3% on transactions >KES 500",
             severity: 0.70},
            ...
        ]
        total_severity: 0.88
        keywords_en: ["vat", "bread", "digital tax", "finance bill"]
        keywords_sw: ["kodi", "mkate", "ushuru", "bill ya fedha"]
        hashtags: ["#RejectFinanceBill2026", "#KenyaTaxRevolt"]
    }
```

### 4.2 Analysis Response

```
Bot: "📋 Finance Bill 2026 — Initial Analysis

    I found 8 provisions. Here are the 3 highest-impact:

    1. VAT on Bread (16%) — Severity: 0.85
       ⚠️ CRITICAL: This mirrors Finance Bill 2024 Section 9 
       which triggered nationwide Gen Z protests.
       Historical mobilization rate: 78%
       Most affected counties: Nairobi, Mombasa, Kisumu

    2. Digital Tax on M-Pesa (3%) — Severity: 0.70
       🟡 HIGH: Similar to 2024 digital tax proposal.
       73% of Kenyans use M-Pesa daily — broad impact.
       Youth sentiment: Highly negative (0.82 anger score)

    3. Housing Levy Increase (3%) — Severity: 0.65
       🟠 MODERATE: Continuation of contested policy.
       Legal challenges still pending from 2024 version.

    📊 Overall Assessment:
    • Mobilization Probability: 0.82 (HIGH)
    • Predicted Phase: REACT within 48h of announcement
    • Most Likely Hotspots: Nairobi CBD, Mombasa, Eldoret
    • Historical Match: 94% similarity to Finance Bill 2024"

User: "What happened with the 2024 Finance Bill protests?"

Bot: "The 2024 Finance Bill protests (June-July 2024) followed this pattern:
     
     Phase 1 — LEAK (June 1-10): Bill text leaked on Twitter
     Phase 2 — REACT (June 10-18): #RejectFinanceBill trending
     Phase 3 — MOBILIZE (June 18-25): Gen Z organized via TikTok/Twitter
     Phase 4 — IMPLEMENT (June 25): Parliament stormed, 39 killed
     Phase 5 — SETTLE (July): Bill withdrawn by President Ruto
     
     📈 Matching tweets from our corpus:
     • 1,247 tweets with >0.7 similarity to current bill provisions
     • Top narrative: 'Serikali vs Mwananchi' (87% of discourse)
     • Dominant Sheng: 'Tutatoboa' (we will pierce through)"

User: "Which counties should we watch?"

Bot: [Displays interactive heatmap of 47 counties with risk scores]
```

### 4.3 Follow-Up Questions the Bot Supports

| Category | Example Questions |
|----------|-------------------|
| **Drill-down** | "What about Section 23 specifically?" |
| **Comparison** | "How does this compare to the Housing Levy rollout?" |
| **Geographic** | "What's the risk for Mombasa county?" |
| **Temporal** | "When would protests likely start?" |
| **Actor** | "Who are the likely mobilizers?" |
| **Stream** | "Show me related tweets from the last 30 days" |
| **Narrative** | "What narratives will emerge around this bill?" |
| **Economic** | "How will this affect inflation projections?" |
| **Advice** | "What counter-narratives could reduce mobilization?" |
| **Monitor** | "Set up alerts for this bill" |

---

## 5. Data Models

### 5.1 Core Data Classes

```python
@dataclass
class BillProvision:
    """Single provision/clause extracted from a bill."""
    clause_id: str              # "Section 12" or "Clause 23(a)"
    description: str            # Plain-language summary
    sector: PolicySector        # TAXATION, HEALTH, etc.
    affected_groups: List[str]  # ["low_income", "youth", "traders"]
    affected_counties: List[str]# ["Nairobi", "Mombasa"] or ["nationwide"]
    monetary_impact: str        # "+16% VAT on bread"
    severity: float             # 0-1 scale
    keywords_en: List[str]      # English search keywords
    keywords_sw: List[str]      # Swahili/Sheng search keywords

@dataclass
class BillAnalysis:
    """Complete bill analysis output."""
    title: str
    source_type: str            # "paste", "pdf", "url"
    raw_text: str               # Original bill text
    sectors: List[PolicySector]
    provisions: List[BillProvision]
    total_severity: float
    hashtags: List[str]
    keywords_en: List[str]
    keywords_sw: List[str]
    matched_historical_event: Optional[str]  # PolicyEvent ID
    match_similarity: float

@dataclass
class ImpactPrediction:
    """Prediction for a single provision or full bill."""
    provision_id: str
    mobilization_probability: float   # 0-1
    predicted_timeline: str           # "24h", "72h", "14d"
    risk_counties: Dict[str, float]   # county → risk score
    narrative_archetypes: List[str]   # from KENYA_NARRATIVE_SYSTEM
    likely_mobilizers: List[str]      # role types
    historical_match: str             # closest PolicyEvent
    historical_similarity: float      # 0-1
    supporting_evidence: List[Dict]   # tweets, news, incidents
    counter_narrative_suggestions: List[str]

@dataclass
class ChatMessage:
    """Single message in the conversation."""
    role: str                   # "user" or "assistant"
    content: str                # Display text
    timestamp: str
    metadata: Dict[str, Any]    # Visualizations, data refs, etc.
    sources: List[Dict]         # Citation links

@dataclass 
class PolicyChatSession:
    """Full conversation session with context memory."""
    session_id: str
    bill: Optional[BillAnalysis]
    predictions: List[ImpactPrediction]
    messages: List[ChatMessage]
    active_monitors: List[str]  # Keywords being monitored
    created_at: str
    last_active: str
```

### 5.2 Policy Phase Lifecycle (from existing policy_events.py)

```
LEAK ──► ANNOUNCE ──► REACT ──► MOBILIZE ──► IMPLEMENT ──► IMPACT ──► SETTLE
 │          │           │          │            │            │          │
 │          │           │          │            │            │          │
 Low        Medium      High       CRITICAL     High         Medium     Low
 Intensity  Intensity   Intensity  Intensity    Intensity    Intensity  Settle
 
 tweet_int  tweet_int   tweet_int  tweet_int    tweet_int    tweet_int  tweet_int
 0.2        0.6         1.3        1.8          1.0          0.8        0.3
```

---

## 6. Key Algorithms

### 6.1 Historical Bill Matching

```python
async def match_historical_bill(
    bill: BillAnalysis,
    events: List[PolicyEvent],
    embeddings: OllamaEmbeddings,
) -> Tuple[PolicyEvent, float]:
    """
    Match a new bill against the 12 historical policy events.
    
    Strategy:
    1. Embed bill provisions + historical event descriptions
    2. Cosine similarity between bill → each event
    3. Boost score for matching sectors and keywords
    4. Return best match + similarity score
    """
```

### 6.2 Tweet Corpus Search

```python
async def search_tweet_corpus(
    bill: BillAnalysis,
    corpus_path: str,
    embeddings: OllamaEmbeddings,
    top_k: int = 50,
) -> List[Dict]:
    """
    Find tweets most relevant to bill provisions.
    
    Strategy:
    1. Load tweets from CSV (policy_event_id column for filtering)
    2. Fast keyword pre-filter (reduce 100K → ~5K candidates)
    3. Embed candidates via nomic-embed-text
    4. Cosine similarity search against provision embeddings
    5. Return top-k with metadata (stance, phase, county)
    """
```

### 6.3 Impact Prediction

```python
async def predict_impact(
    bill: BillAnalysis,
    historical_match: PolicyEvent,
    matching_tweets: List[Dict],
    economic_context: Dict,
    provider: OllamaProvider,
) -> ImpactPrediction:
    """
    Predict social impact of bill provisions.
    
    Uses LLM with rich context:
    - Bill provisions + severity scores
    - Historical match + what happened then
    - Current tweet sentiment landscape
    - Economic indicators (inflation, unemployment)
    - Sheng/Swahili narrative patterns
    
    Returns structured prediction with county risk map.
    """
```

---

## 7. UI Design

### 7.1 SENTINEL Dashboard Integration

New sidebar entry: **"Policy Intelligence"** in `NAV_OPTIONS` dict.

```python
# kshiked/ui/sentinel/router.py — add to NAV_OPTIONS
NAV_OPTIONS = {
    ...
    "Policy Intelligence": "POLICY_CHAT",
    ...
}
```

### 7.2 Chat Panel Layout

```
┌──────────────────────────────────────────────────────────────┐
│  📋 Policy Intelligence                                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────── SIDEBAR ─────────────────────────┐  │
│  │ Bill Input:                                             │  │
│  │ ○ Paste Text  ○ Upload PDF  ○ Enter URL                │  │
│  │                                                         │  │
│  │ ┌─────────────────────────────────────────────────────┐│  │
│  │ │ [Text area / File uploader / URL input]             ││  │
│  │ └─────────────────────────────────────────────────────┘│  │
│  │ [Analyze Bill ▶]                                        │  │
│  │                                                         │  │
│  │ ─── Active Monitors ───                                 │  │
│  │ 🔴 Finance Bill 2026 (3 alerts)                         │  │
│  │ 🟡 SHIF Phase 2 (1 alert)                               │  │
│  │                                                         │  │
│  │ ─── Quick Actions ───                                   │  │
│  │ [📥 Export PDF Report]                                   │  │
│  │ [📊 Export CSV Data]                                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────── CHAT AREA ───────────────────────────┐  │
│  │                                                         │  │
│  │  🤖 Welcome to Policy Intelligence. Upload a bill or    │  │
│  │     ask about any Kenyan policy.                        │  │
│  │                                                         │  │
│  │  👤 Analyze the Finance Bill 2026 — here's the text:    │  │
│  │     [pasted bill text...]                               │  │
│  │                                                         │  │
│  │  🤖 📋 Finance Bill 2026 — Initial Analysis             │  │
│  │     [Risk summary cards]                                │  │
│  │     [County heatmap visualization]                      │  │
│  │     [Historical comparison chart]                       │  │
│  │                                                         │  │
│  │  👤 Which counties should we watch?                      │  │
│  │                                                         │  │
│  │  🤖 Based on historical patterns and current sentiment:  │  │
│  │     [Interactive county risk table]                      │  │
│  │     [Embedded map with risk shading]                    │  │
│  │                                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────── INPUT ──────────────────────────────┐  │
│  │ Ask about this bill...                            [Send]│  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 Standalone Mode

Same chat interface but running as `streamlit run policy_chatbot_standalone.py` on a separate port. Share the same backend modules.

---

## 8. URL Scraping & PDF Extraction

### 8.1 Kenya Gazette Integration

```python
class KenyaGazetteScraper:
    """Scrape bills from kenyagazette.go.ke and parliament.go.ke."""
    
    BASE_URLS = {
        "gazette": "http://kenyagazette.go.ke",
        "parliament": "http://parliament.go.ke/the-national-assembly/house-business/bills",
    }
    
    async def fetch_bill(self, url: str) -> str:
        """Download bill PDF/HTML and extract text."""
    
    async def search_bills(self, query: str) -> List[Dict]:
        """Search for bills by keyword (e.g., 'Finance Bill 2026')."""
    
    async def get_latest_bills(self, limit: int = 10) -> List[Dict]:
        """Get most recent bills from parliament tracker."""
```

### 8.2 PDF Extraction

```python
class BillPDFExtractor:
    """Extract structured text from Kenyan bill PDFs."""
    
    def extract(self, pdf_path: str) -> str:
        """Extract full text from PDF (PyMuPDF/pdfplumber)."""
    
    def extract_sections(self, text: str) -> List[Dict]:
        """Parse bill text into numbered sections/clauses."""
    
    def extract_monetary_values(self, text: str) -> List[Dict]:
        """Find KES amounts, percentages, tax rates."""
```

---

## 9. Implementation Phases

### Phase 1 — Core Chatbot (Week 1)
- [ ] `policy_extractor.py` — Bill text parsing + provision extraction via LLM
- [ ] `policy_search.py` — Tweet corpus search with embeddings (keyword pre-filter + semantic)
- [ ] `policy_chatbot.py` — Core conversation engine with context memory
- [ ] `policy_chat.py` (UI) — Streamlit chat panel with `st.chat_input`
- [ ] Wire into SENTINEL dashboard as "Policy Intelligence" sidebar entry
- [ ] Support paste-text input mode
- **Deliverable**: Working chatbot that takes pasted bill text and returns analysis

### Phase 2 — Historical Matching (Week 2)
- [ ] `policy_predictor.py` — Impact prediction engine
- [ ] Historical bill matcher using `policy_events.py` (12 events)
- [ ] County risk scoring based on historical incident data
- [ ] Timeline prediction using PolicyPhase lifecycle
- [ ] Inline visualizations (risk cards, timeline chart)
- **Deliverable**: Chatbot shows "this is 94% similar to Finance Bill 2024"

### Phase 3 — Document Input (Week 3)
- [ ] PDF upload + extraction (PyMuPDF or pdfplumber)
- [ ] URL scraping for Kenya Gazette / Parliament
- [ ] Section-by-section drill-down in chat
- [ ] Standalone app mode (`policy_chatbot_standalone.py`)
- **Deliverable**: Analyst uploads a PDF bill and gets full report

### Phase 4 — Monitoring & Streams (Week 4)
- [ ] `policy_monitor.py` — Real-time keyword/hashtag monitoring
- [ ] Integration with live Twitter/X API (if keys available)
- [ ] Sentiment shift tracking (before/after announcement)
- [ ] Alert system when escalation thresholds crossed
- [ ] Kenya Gazette auto-ingest for new bills
- **Deliverable**: Continuous monitoring with push alerts

### Phase 5 — Polish & Export (Week 5)
- [ ] PDF report generation (analyst-ready briefing)
- [ ] CSV export of predictions and supporting evidence
- [ ] Counter-narrative suggestions
- [ ] Multi-session support (analyze multiple bills concurrently)
- [ ] Performance optimization (caching, pre-computed embeddings)
- **Deliverable**: Production-ready tool for analyst workflow

---

## 10. Dependencies

### New Python Packages

| Package | Purpose | Install |
|---------|---------|---------|
| `PyMuPDF` (fitz) | PDF text extraction | `pip install PyMuPDF` |
| `pdfplumber` | Alternative PDF parser (tables) | `pip install pdfplumber` |
| `beautifulsoup4` | HTML scraping (Gazette/Parliament) | Already in project |
| `aiohttp` | Async HTTP (Ollama + scraping) | Already in project |
| `streamlit` | Chat UI | Already in project |

### External Services (Optional)

| Service | Purpose | Required? |
|---------|---------|-----------|
| Ollama (localhost:11434) | LLM inference | Yes |
| Twitter/X API | Live stream monitoring | Phase 4 only |
| Kenya Gazette website | Bill auto-ingest | Phase 3+ |
| KNBS API / CBK data | Economic indicators | Phase 2+ |

---

## 11. Prompt Strategy

### Bill Extraction Prompt

```
SYSTEM: You are a Kenyan legislative analyst. Extract structured provisions
from parliamentary bills. For each provision identify:
- Clause reference (Section/Part number)
- Plain language description
- Sector (TAXATION/HEALTH/HOUSING/FUEL_ENERGY/EDUCATION/DIGITAL/SECURITY/
  AGRICULTURE/DEVOLUTION/CONSTITUTIONAL/TRANSPORT/EMPLOYMENT)
- Affected demographics
- Affected counties (or "nationwide")
- Monetary impact (percentages, KES amounts, rate changes)
- Severity (0-1): How much social disruption will this cause?

Return JSON array of provisions.
```

### Impact Prediction Prompt

```
SYSTEM: You are a Kenyan political analyst predicting social response to
new legislation. You have access to:
- Historical patterns from {matched_event} which had {outcome}
- {n_tweets} related social media posts showing current sentiment
- Economic context: inflation={rate}%, unemployment={rate}%

For each provision, predict:
- mobilization_probability (0-1)
- timeline (when would protests peak?)
- risk_counties (which of 47 counties, with scores)
- narrative_archetypes (which narratives will dominate?)
- counter_narratives (what could reduce mobilization?)

Consider Sheng/Swahili framing and Kenya's political context 2025-2026.
```

---

## 12. Integration Points

### With Existing SENTINEL Systems

```python
# From KShieldAnalyzer — deep-dive on flagged content
report = await analyzer.analyze(flagged_tweet)

# From PolicyEventInjector — match new bill to event templates
events = build_kenya_2026_events()
matched = match_to_event(bill, events)

# From PulseSensor — signal detection on bill keywords
sensor = PulseSensor(use_nlp=True)
detections = sensor.process_text(bill_provision_text)

# From OllamaEmbeddings — semantic search
similar = await embeddings.find_similar(provision_text, tweet_corpus, top_k=50)
clusters = await embeddings.cluster_texts(similar_tweets, n_clusters=5)
```

---

## 13. Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Bill parsing accuracy | >90% provisions extracted | Manual review of 10 bills |
| Historical match quality | Top-1 match correct >80% | Compare to known bill outcomes |
| County risk accuracy | >70% overlap with actual protest locations | Backtest against 2024 data |
| Response time | <30s for initial analysis | Measure end-to-end latency |
| Analyst satisfaction | Useful for daily workflow | Qualitative feedback |

---

## Notes

- **Privacy**: All inference is local (Ollama). No bill text leaves the machine.
- **Sheng**: The chatbot inherits the full Sheng glossary from `prompts_kenya.py` — critical for understanding tweet reactions.
- **Policy phases**: The 7-phase lifecycle (LEAK→SETTLE) from `policy_events.py` is the temporal backbone for predictions.
- **County context**: All 47 counties are pre-categorized by risk profile in `prompts_kenya.py`.

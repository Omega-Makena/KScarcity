# KShield Pulse — TODO

Tracking upcoming work for the KShield Pulse LLM pipeline.

**Last Updated**: February 18, 2026

---

## Priority: Sheng Glossary & Model Improvement

### 1. Expand Sheng/Swahili Glossary
- [ ] Collect comprehensive Sheng-to-English mappings from:
  - Urban slang databases (Nairobi, Mombasa, Kisumu)
  - Twitter/X trending Sheng terms (2024-2026)
  - Academic Sheng lexicons (University of Nairobi linguistics dept)
  - Community-sourced urban dictionaries (Sheng Nation, UrbanSheng)
- [ ] Add threat-specific Sheng terms that carry implicit violence:
  - "Choma" (burn), "Piga" (hit/beat), "Maliza" (finish/eliminate)
  - "Rudi nyumbani" (go home — ethnic cleansing euphemism)
  - "Vita" (war), "Mchanga" (sand — burial reference)
  - County-specific slang for different regions
- [ ] Add coded/emerging slang that evades keyword filters:
  - Finance Bill era terms ("Maandamano" = protests)
  - Green/emoji codes used on Kenyan Twitter
  - Religious coded language (cross-faith mobilization)
- [ ] Structure glossary as JSON for easy updates:
  ```json
  {
    "term": "choma",
    "english": "burn",
    "threat_weight": 0.7,
    "context": "Can be literal (BBQ) or threatening (arson)",
    "region": "nationwide",
    "examples": ["Tutachoma hii jiji", "Choma ile kitu"]
  }
  ```
- [ ] Validate glossary with native Sheng speakers for accuracy
- [ ] Add glossary versioning so changes are tracked

### 2. Improve Model Quality for Sheng Detection
- [ ] **Prompt engineering** — Restructure THREAT_TAXONOMY_SYSTEM prompt to:
  - Include Sheng translation examples inline
  - Explicitly list Sheng threat phrases with expected classifications
  - Add few-shot examples of Sheng → correct Tier mapping
- [ ] **Test prompt variants** — A/B test different prompt structures on labeled Sheng tweets:
  - Current generic prompt vs Sheng-enhanced prompt
  - Measure classification accuracy on 50+ labeled Sheng threat tweets
- [ ] **Evaluate larger models** when hardware allows:
  - `qwen2.5:7b` (4.7GB) — Better multilingual, needs 8GB VRAM
  - `llama3.1:8b` (4.7GB) — Strong reasoning
  - `mistral:7b` (4.1GB) — Fast, good instruction following
- [ ] **Fine-tuning path** (future):
  - Collect 500+ labeled Sheng threat/non-threat pairs
  - Use `fine_tuning.py` LoRA adapter for qwen2.5:3b
  - Create Ollama Modelfile with custom system prompt
  - Benchmark fine-tuned vs base model accuracy

### 3. Build Labeled Sheng Dataset
- [ ] Source tweets from `data/kshield_kenya_unified_incidents_*.csv`
- [ ] Filter for Swahili/Sheng content using `detect_language()`
- [ ] Manual annotation: Tier 1-5 labels by native speakers
- [ ] Target: 500 labeled examples minimum
- [ ] Split: 400 train / 50 validation / 50 test
- [ ] Store in `data/sheng_labeled/` with provenance metadata

---

## Other TODO Items

### Pipeline Improvements
- [ ] Add streaming support (`stream: true`) for dashboard real-time display
- [ ] Implement response caching (avoid re-analyzing identical tweets)
- [ ] Add confidence calibration — track model accuracy over time
- [ ] Parallel execution for ≥8GB VRAM hardware (re-enable asyncio.gather)

### Dashboard Integration
- [ ] Wire KShieldAnalyzer into SENTINEL Command Center
- [ ] Add real-time threat feed panel
- [ ] Show Sheng detection indicators in tweet analysis view
- [ ] Display embedding similarity clusters as interactive scatter plot

---

## Priority: Policy Impact Chatbot

See full architecture: [`documentation/kshield-docs/pulse/02_policy_chatbot_architecture.md`](documentation/kshield-docs/pulse/02_policy_chatbot_architecture.md)

### Phase 1 — Core Chatbot (Week 1)
- [ ] `policy_extractor.py` — Bill text parsing + provision extraction via LLM
- [ ] `policy_search.py` — Tweet corpus search with embeddings (keyword pre-filter + semantic)
- [ ] `policy_chatbot.py` — Core conversation engine with context memory
- [ ] `policy_chat.py` (UI) — Streamlit chat panel with `st.chat_input`
- [ ] Wire into SENTINEL dashboard as "Policy Intelligence" sidebar entry
- [ ] Support paste-text input mode

### Phase 2 — Historical Matching (Week 2)
- [ ] `policy_predictor.py` — Impact prediction engine
- [ ] Historical bill matcher using 12 PolicyEvents from `policy_events.py`
- [ ] County risk scoring from historical incident data
- [ ] Timeline prediction using 7-phase PolicyPhase lifecycle
- [ ] Inline visualizations (risk cards, timeline chart)

### Phase 3 — Document Input (Week 3)
- [ ] PDF upload + text extraction (PyMuPDF / pdfplumber)
- [ ] URL scraping for Kenya Gazette / Parliament websites
- [ ] Section-by-section drill-down in chat
- [ ] Standalone app mode (`policy_chatbot_standalone.py`)

### Phase 4 — Monitoring & Streams (Week 4)
- [ ] `policy_monitor.py` — Real-time keyword/hashtag monitoring
- [ ] Integration with live Twitter/X API
- [ ] Sentiment shift tracking (before/after bill announcement)
- [ ] Alert system when escalation thresholds crossed
- [ ] Kenya Gazette auto-ingest for new bills

### Phase 5 — Polish & Export (Week 5)
- [ ] PDF report generation (analyst-ready briefing)
- [ ] CSV export of predictions + supporting evidence
- [ ] Counter-narrative suggestions
- [ ] Multi-session support (concurrent bill analysis)
- [ ] Performance optimization (caching, pre-computed embeddings)

### Testing
- [ ] Write pytest suite for all 7 LLM modules
- [ ] Add integration tests with Ollama mock server
- [ ] Benchmark inference latency across model sizes
- [ ] Test batch processor with full 100K+ tweet dataset

### Documentation
- [ ] Add API reference for each module (docstring → Sphinx)
- [ ] Create Sheng glossary contribution guide
- [ ] Document Ollama model evaluation criteria
- [ ] Add troubleshooting guide for common Ollama issues

---

## Notes

**Current Hardware**: NVIDIA GTX 1650 (4GB VRAM)  
**Current Models**: qwen2.5:3b (analysis) + nomic-embed-text (embeddings)  
**Known Gap**: Sheng threat detection accuracy — "Tutachoma hii jiji" (burn this city) misclassified as Tier 5  
**Root Cause**: 3B model has limited Sheng/Swahili training data; prompt doesn't provide enough in-context examples

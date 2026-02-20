# KShield Pulse LLM Architecture — Ollama Integration

**Version**: 3.0  
**Last Updated**: February 18, 2026  
**Status**: Production — Tested on GTX 1650 (4GB VRAM)

---

## Overview

The LLM subsystem provides AI-powered threat intelligence analysis for the KShield Pulse engine. It uses **Ollama** for local inference with no cloud dependencies, ensuring data sovereignty for sensitive Kenyan security data.

### Design Goals

1. **Local-only inference** — No data leaves the machine
2. **Hardware-aware** — Auto-configures for available VRAM (tested on 4GB)
3. **Kenya-specific** — Sheng/Swahili language detection, county awareness, policy context
4. **V3 Signal Architecture** — 14-category ranked threat taxonomy with Dual-Layer Risk (BaseRisk × CSM)
5. **Production-grade** — Retry logic, checkpointing, batch processing for 100K+ tweets

---

## Module Map

```
kshiked/pulse/llm/
├── config.py            # Model registry, task routing, hardware profiles
├── ollama.py            # Core Ollama provider (inference, retries, V3 pipeline)
├── prompts.py           # V3 threat taxonomy prompts, scoring rules
├── prompts_kenya.py     # Kenya-specific: Sheng glossary, county context, policy
├── signals.py           # V3 data models: ThreatSignal, ContextAnalysis, etc.
├── embeddings.py        # Semantic embeddings via nomic-embed-text
├── batch_processor.py   # CSV processing with checkpointing (100K+ tweets)
├── analyzer.py          # End-to-end orchestrator (KShieldAnalyzer)
├── models.py            # Ollama model lifecycle management + CLI
├── base.py              # Abstract LLMProvider base class
├── gemini.py            # Google Gemini provider (alternative)
├── fine_tuning.py       # LoRA fine-tuning utilities
└── __init__.py          # Public API exports
```

---

## Models

### Current Setup (4GB VRAM)

| Model | Size | Role | Status |
|-------|------|------|--------|
| `qwen2.5:3b` | 1.9GB | All analysis tasks | Active |
| `nomic-embed-text` | 274MB | Semantic embeddings (768-dim) | Active |

### Why qwen2.5:3b

- **Best JSON output** among 3B-class models — critical for structured threat taxonomy extraction
- **Strong multilingual** — Better Swahili/Sheng fragment handling than Phi-3 or Llama 3.2
- **Fits in 4GB VRAM** with ~2GB headroom for context and batching
- **Fast inference** — ~10-30s per analysis on GTX 1650

### Available Profiles (config.py)

| Profile | VRAM | Use When |
|---------|------|----------|
| `qwen2.5:3b` | 1.9GB | Default — constrained hardware |
| `llama3.1:8b` | 4.7GB | ≥8GB VRAM available |
| `mistral:7b` | 4.1GB | Fast classification focus |
| `qwen2.5:7b` | 4.7GB | Maximum multilingual quality |

---

## Architecture

### Analysis Pipeline (5 Steps)

```
Input Text
    │
    ▼
┌─────────────────────┐
│ 1. Language Detect   │ → Sheng/Swahili/English classification
│    (detect_language)  │   + automatic translation if needed
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Dual-Layer Scan   │ → ThreatSignal (BaseRisk) + ContextAnalysis (CSM)
│    (parallel)         │   14-category taxonomy + E0-E4/S0-S4 scales
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. Time-To-Action    │ → IMMEDIATE_24H / NEAR_TERM_72H / CHRONIC_14D
│    (analyze_tta)      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. Role Classify     │ → ideologue / mobilizer / broker / op_signaler /
│    (analyze_role_v3)  │   unwitting_amplifier / observer
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. Policy Impact     │ → Finance Bill, housing levy, digital tax context
│    (analyze_policy)   │   + mobilization potential score
└──────────┬──────────┘
           ▼
    AnalysisReport
    ├── risk_level: CRITICAL / HIGH / MODERATE / LOW / MINIMAL
    ├── gating_status: IMMEDIATE_ESCALATION / ACTIVE_MONITORING / etc.
    └── is_actionable: bool
```

### Dual-Layer Risk Formula

```
AdjustedRisk = BaseRisk × CSM (Context Stress Multiplier)

BaseRisk = weighted(intent, capability, specificity, reach, trajectory)
CSM = f(economic_grievance, social_grievance, shock_marker, polarization)
```

### 14-Category Threat Taxonomy

| Tier | Categories | Risk Range |
|------|-----------|------------|
| **Tier 1: Existential** | Mass violence, terrorism, infra sabotage | 95-100 |
| **Tier 2: Severe** | Insurrection, election subversion, official threats | 75-95 |
| **Tier 3: High-Risk** | Ethnic mobilization, disinfo campaigns, financial warfare | 60-80 |
| **Tier 4: Emerging** | Radicalization, hate networks, foreign influence | 40-65 |
| **Tier 5: Non-Threat** | Political criticism, satire/protest (protected speech) | 0-20 |

---

## Task Routing

The config supports assigning different models to different tasks:

```python
from kshiked.pulse.llm.config import OllamaConfig, AnalysisTask

# Auto-configured (all tasks → qwen2.5:3b, embeddings → nomic-embed-text)
config = OllamaConfig()

# Single-model mode
config = OllamaConfig.single_model("qwen2.5:3b")

# Hardware-aware
config = OllamaConfig.for_hardware(vram_gb=4.0)

# Get model for a specific task
model = config.get_model_for_task(AnalysisTask.THREAT_CLASSIFICATION)
```

### All Tasks

| Task | Default Model | Description |
|------|---------------|-------------|
| `THREAT_CLASSIFICATION` | qwen2.5:3b | 14-category taxonomy classification |
| `CONTEXT_ANALYSIS` | qwen2.5:3b | Economic/social grievance scoring |
| `INDICES_EXTRACTION` | qwen2.5:3b | LEI, SI, MS, AA indices |
| `TIME_TO_ACTION` | qwen2.5:3b | Temporal urgency assessment |
| `RESILIENCE_ANALYSIS` | qwen2.5:3b | Counter-narrative strength |
| `ROLE_CLASSIFICATION` | qwen2.5:3b | Actor role identification |
| `NARRATIVE_ANALYSIS` | qwen2.5:3b | Narrative framing analysis |
| `BATCH_CLASSIFICATION` | qwen2.5:3b | High-throughput batch mode |
| `EMBEDDING` | nomic-embed-text | 768-dim semantic vectors |
| `SHENG_TRANSLATION` | qwen2.5:3b | Sheng/Swahili → English |
| `POLICY_IMPACT` | qwen2.5:3b | Kenya policy context analysis |
| `SUMMARY` | qwen2.5:3b | General summarization |

---

## Usage Examples

### Quick Analysis

```python
import asyncio
from kshiked.pulse.llm.analyzer import KShieldAnalyzer
from kshiked.pulse.llm.config import OllamaConfig

async def analyze_tweet():
    config = OllamaConfig()
    async with KShieldAnalyzer(config) as analyzer:
        report = await analyzer.analyze(
            "If Ruto signs this Finance Bill, Kenyans will march to State House."
        )
        print(report.summary())
        # [CRITICAL] Tier 3: High-Risk | Risk: 96 | Role: mobilizer

asyncio.run(analyze_tweet())
```

### Embeddings & Similarity

```python
from kshiked.pulse.llm.embeddings import OllamaEmbeddings
from kshiked.pulse.llm.config import OllamaConfig

async def search():
    config = OllamaConfig()
    emb = OllamaEmbeddings(config)
    
    corpus = ["Finance Bill protest", "GDP growth report", "Ethnic violence threat"]
    similar = await emb.find_similar("reject finance bill", corpus, top_k=2)
    for idx, score, text in similar:
        print(f"{score:.3f}  {text}")
    
    await emb.close()
```

### Batch Processing (100K+ tweets)

```python
from kshiked.pulse.llm.batch_processor import BatchProcessor, ProcessingMode
from kshiked.pulse.llm.config import OllamaConfig

async def batch():
    config = OllamaConfig()
    processor = BatchProcessor(config)
    
    await processor.process_csv(
        input_path="data/kenya_tweets.csv",
        output_path="results/analyzed.csv",
        text_column="tweet_text",
        mode=ProcessingMode.STANDARD,
        checkpoint_every=100,  # Resume-safe
    )
```

### Model Management (CLI)

```bash
# Check system status
python -m kshiked.pulse.llm.models status

# Pull required models
python -m kshiked.pulse.llm.models setup

# Verify a specific model
python -m kshiked.pulse.llm.models verify qwen2.5:3b
```

---

## Robust Enum Parsing

The LLM output doesn't always match exact enum values. The provider includes fuzzy-matching parsers:

- `_parse_threat_category()` — Maps LLM strings like "TIER_5_PROTECTED" → `CAT_13_POLITICAL_CRITICISM`
- `_parse_threat_tier()` — Handles "Tier 3", "TIER_3_HIGH_RISK", "high risk" → `TIER_3`
- `_parse_economic_grievance()` — Maps "E2" → `E2_MOBILIZATION`
- `_parse_social_grievance()` — Maps "S1" → `S1_POLARIZATION`

---

## Known Limitations

1. **Sheng Threat Detection** — `qwen2.5:3b` sometimes under-classifies Sheng threats (e.g., "Tutachoma hii jiji" = "We'll burn this city" classified as Tier 5). Needs prompt tuning and expanded Sheng glossary.

2. **Sequential Execution** — Supplementary analyses (TTA, role, policy) run sequentially to prevent Ollama crashes on 4GB VRAM. This adds ~30-60s per full analysis.

3. **First-Load Latency** — First inference after model cold start takes ~30-60s as the model loads into VRAM. Subsequent calls are much faster.

4. **No Streaming** — All responses use `stream: false`. Streaming could improve perceived latency for dashboard integration.

---

## Configuration Reference

### OllamaConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `default_model` | `None` | Override all task routing |
| `max_retries` | `3` | Retry attempts per call |
| `retry_delay` | `1.0` | Base retry delay (seconds) |
| `retry_backoff` | `2.0` | Exponential backoff multiplier |
| `connect_timeout` | `10.0` | Connection timeout (seconds) |
| `read_timeout` | `300.0` | Inference timeout (seconds) |
| `batch_size` | `10` | Texts per batch |
| `max_concurrent` | `2` | Max parallel Ollama requests |
| `embedding_model` | `nomic-embed-text` | Embedding model |
| `embedding_dim` | `768` | Embedding dimensions |
| `enable_sheng_detection` | `true` | Detect Sheng/Swahili |
| `enable_policy_context` | `true` | Kenya policy analysis |
| `log_prompts` | `false` | Debug: log full prompts |
| `track_latency` | `true` | Track inference metrics |

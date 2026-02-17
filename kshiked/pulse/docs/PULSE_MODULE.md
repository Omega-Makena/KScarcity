# KShield Pulse Module Documentation

> **Package**: `kshiked.pulse`  
> **Purpose**: Social media monitoring, threat detection, and news ingestion for Kenya

---

## Table of Contents

1. [Database Layer (db)](#1-database-layer)
2. [Ingestion Pipeline](#2-ingestion-pipeline)
3. [LLM Integration](#3-llm-integration)
4. [News Module](#4-news-module)

---

# 1. Database Layer

**Package**: `kshiked.pulse.db`

## Models (`models.py`)

### Enums

| Enum | Values |
|------|--------|
| `Platform` | twitter, telegram, facebook, instagram, reddit |
| `ThreatTier` | tier_0 through tier_5 |
| `RoleType` | ideologue, mobilizer, amplifier, broker, legitimizer, gatekeeper |
| `EdgeType` | reply, retweet, quote, mention, cooccurrence |

### Core Tables

| Model | Purpose |
|-------|---------|
| `Author` | Social media profiles with suspicion scores |
| `SocialPost` | Raw posts from any platform |
| `ProcessedSignal` | Signal detections with intensity/confidence |
| `LLMAnalysis` | Gemini classification results |
| `NetworkEdge` | Graph relationships for network analysis |
| `PriceSnapshot` | E-commerce price tracking |

### Usage
```python
from kshiked.pulse.db import Database
db = Database()
await db.connect()
```

---

## Database Class (`database.py`)

Async SQLAlchemy 2.0 database manager.

| Method | Purpose |
|--------|---------|
| `connect()` | Initialize connection and create tables |
| `add()` / `add_all()` | Insert single/multiple objects |
| `batch_insert()` | Efficient bulk inserts (1000/batch) |
| `upsert_author()` | Insert or update author |
| `get_unprocessed_posts()` | Fetch posts not yet analyzed |
| `get_stats()` | Database statistics |

---

# 2. Ingestion Pipeline

**Package**: `kshiked.pulse.ingestion`

## Orchestrator (`orchestrator.py`)

Coordinates all scraping operations.

```python
async with IngestionOrchestrator(config) as orch:
    await orch.scrape_social_media()
    await orch.scrape_ecommerce()
```

### Managed Scrapers

| Type | Platforms |
|------|-----------|
| Social Media | X/Twitter, Reddit, Telegram, Instagram, Facebook |
| E-commerce | Jiji, Jumia, Kilimall |

---

## Pipeline Integration (`pipeline.py`)

Bridges scraped data to PulseSensor.

```
Scrapers → LLM Classification → PulseSensor → Database
```

| Method | Purpose |
|--------|---------|
| `process_posts()` | Run posts through full pipeline |
| `get_threat_summary()` | Current threat breakdown by tier |
| `get_state()` | PulseSensor state |

---

## Scheduler (`scheduler.py`)

Asyncio-based job scheduling.

| Job | Interval |
|-----|----------|
| `social_media` | Every 30 minutes |
| `ecommerce` | Every 6 hours |
| `processing` | Every 1 hour |

```python
scheduler = IngestionScheduler(orchestrator)
await scheduler.start()
await scheduler.run_now("social_media")
```

---

# 3. LLM Integration

**Package**: `kshiked.pulse.llm`

## GeminiProvider (`gemini.py`)

Google Gemini API wrapper for threat analysis.

```python
provider = create_gemini_provider(api_key="...")
result = await provider.classify_threat(text)
```

| Method | Returns |
|--------|---------|
| `classify_threat()` | `ThreatClassification` with tier 0-5 |
| `identify_role()` | `RoleIdentification` (mobilizer, broker, etc.) |
| `analyze_narrative()` | `NarrativeAnalysis` with themes |
| `batch_classify()` | Multiple texts in one call |

---

## Prompts (`prompts.py`)

Prompt templates for Kenya-specific threat taxonomy.

| Template | Purpose |
|----------|---------|
| `THREAT_CLASSIFICATION_PROMPT` | Classify post into Tier 0-5 |
| `NARRATIVE_ANALYSIS_PROMPT` | Identify themes across posts |
| `SYSTEM_PROMPT` | Kenya context (ethnic tensions, maandamano) |

### Threat Tiers

| Tier | Description |
|------|-------------|
| 0 | Benign content |
| 1 | Low-risk complaints |
| 2 | Moderate grievances |
| 3 | High-risk mobilization |
| 4 | Direct incitement |
| 5 | Immediate threat, violence |

---

# 4. News Module

**Module**: `kshiked.pulse.news` + `kshiked.pulse.db.news_db`

## NewsIngestor

Daily news fetching from NewsAPI.org.

```python
from kshiked.pulse.news import get_news_ingestor
ingestor = get_news_ingestor()
articles = ingestor.fetch_all()
```

| Method | Purpose |
|--------|---------|
| `fetch_pipeline(category)` | Fetch single category |
| `fetch_all()` | Fetch all 13 categories |

---

## Categories

| Category | Keywords |
|----------|----------|
| `economics` | inflation, debt, imf, shilling |
| `politics` | ruto, raila, parliament |
| `security` | police, dci, al-shabaab |
| `agriculture` | farming, maize, drought |
| `health` | hospital, nhif, sha |
| `education` | school, knut, helb |

---

## Trusted Domains

```
nation.africa, standardmedia.co.ke, the-star.co.ke
businessdailyafrica.com, citizen.digital, kbc.co.ke
```

---

## NewsDatabase (`news_db.py`)

SQLite storage for news history.

```python
from kshiked.pulse.db.news_db import NewsDatabase
db = NewsDatabase()
history = db.get_history("politics", limit=50)
```

---

## Caching

| Behavior | Description |
|----------|-------------|
| 24-hour TTL | Reset at midnight EAT |
| Stale-while-revalidate | Serve cache on API errors |
| Deduplication | `INSERT OR IGNORE` by URL |

**Paths**: `data/news_cache/`, `data/news_db.sqlite`

---

# Configuration

All modules load from environment variables:

| Variable | Module |
|----------|--------|
| `NEWSAPI_KEY` | NewsIngestor |
| `GEMINI_API_KEY` | GeminiProvider |
| `DATABASE_URL` | Database |
| `X_API_KEY` | Twitter scraper |
| `REDDIT_CLIENT_ID` | Reddit scraper |

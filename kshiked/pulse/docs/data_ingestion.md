# KShield Pulse - Data Ingestion Pipeline

## Overview

Complete production-quality data ingestion pipeline for the KShield Pulse national threat detection system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KSHIELD PULSE PIPELINE                        │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   SCRAPING      │   ANALYSIS      │   OUTPUT                     │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ Social Media    │ Gemini LLM      │ Threat Summary              │
│ - X/Twitter     │ - Tier 0-5      │ - Crisis Probability        │
│ - Reddit        │ - Role ID       │ - Instability Index         │
│ - Telegram      │ - Narratives    │                             │
│ - Instagram     │                 │                             │
│ - Facebook      │ PulseSensor     │ Shock Vectors               │
│                 │ - 15 Signals    │ - For simulation            │
│ E-Commerce      │                 │                             │
│ - Jiji          │ ESI Calculator  │ Inflation Reports           │
│ - Jumia         │ - Price indices │ - Economic stress           │
│ - Kilimall      │ - Domain agg    │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```powershell
pip install sqlalchemy[asyncio] aiosqlite httpx beautifulsoup4 lxml google-generativeai
pip install twscrape ntscraper praw telethon instaloader facebook-scraper
```

### 2. Configure Environment

Create `.env` file in `pulse/` directory:

```bash
# Database
DATABASE_URL=sqlite+aiosqlite:///pulse.db

# Gemini LLM
GEMINI_API_KEY=your-api-key

# Reddit (from reddit.com/prefs/apps)
REDDIT_CLIENT_ID=xxx
REDDIT_CLIENT_SECRET=xxx

# Telegram (from my.telegram.org)
TELEGRAM_API_ID=12345
TELEGRAM_API_HASH=xxx

# Optional
INSTAGRAM_USERNAME=xxx
INSTAGRAM_PASSWORD=xxx
FACEBOOK_EMAIL=xxx
FACEBOOK_PASSWORD=xxx
```

### 3. Run Pipeline

```python
from kshiked.pulse.ingestion import run_full_pipeline

result = await run_full_pipeline(
    search_terms=["Kenya", "Ruto", "maandamano"],
    gemini_api_key="your-api-key"
)

print(result["threat_summary"])
```

## Components

### Social Media Scrapers

| Platform | Library | Auth Required |
|----------|---------|---------------|
| X/Twitter | twscrape + ntscraper | Optional (Nitter fallback) |
| Reddit | PRAW | Yes (free API) |
| Telegram | Telethon | Yes (API ID/Hash) |
| Instagram | Instaloader | Optional |
| Facebook | facebook-scraper | Optional |

### E-Commerce Scrapers

| Site | Method | Purpose |
|------|--------|---------|
| Jiji | httpx + BS4 | C2C prices, rentals |
| Jumia | httpx + BS4 | Retail prices, groceries |
| Kilimall | httpx + BS4 | Electronics, fashion |

### LLM Integration

Uses Gemini with Kenya-specific threat prompts:

| Tier | Level | Example |
|------|-------|---------|
| TIER_1 | Existential | Violence incitement |
| TIER_2 | Severe | Dehumanization |
| TIER_3 | High-Risk | Mobilization calls |
| TIER_4 | Emerging | Economic grievance |
| TIER_5 | Non-threat | Normal discourse |
| TIER_0 | Protected | Legitimate criticism |

## File Structure

```
pulse/
├── db/
│   ├── models.py      # 9 SQLAlchemy models
│   └── database.py    # Async connection layer
├── scrapers/
│   ├── base.py        # Abstract scraper
│   ├── x_scraper.py
│   ├── reddit_scraper.py
│   ├── telegram_scraper.py
│   ├── instagram_scraper.py
│   ├── facebook_scraper.py
│   └── ecommerce/
│       ├── jiji_scraper.py
│       ├── jumia_scraper.py
│       ├── kilimall_scraper.py
│       └── price_aggregator.py
├── llm/
│   ├── base.py        # LLMProvider interface
│   ├── gemini.py      # Gemini implementation
│   ├── prompts.py     # Kenya threat prompts
│   └── fine_tuning.py # Training data prep
├── filters/
│   └── kenya_keywords.py  # 200+ keywords
├── ingestion/
│   ├── orchestrator.py    # Main coordinator
│   ├── scheduler.py       # Periodic jobs
│   └── pipeline.py        # PulseSensor bridge
└── config.py              # Unified configuration
```

## API Reference

### IngestionOrchestrator

```python
config = IngestionConfig.from_env()

async with IngestionOrchestrator(config) as orchestrator:
    # Scrape social media
    posts = await orchestrator.scrape_social_media(["Kenya"])
    
    # Scrape prices
    prices = await orchestrator.scrape_ecommerce()
    
    # Run continuously
    await orchestrator.run(duration_hours=24)
```

### PipelineIntegration

```python
pipeline = PipelineIntegration(
    sensor=PulseSensor(use_nlp=True),
    llm_provider=GeminiProvider(...),
    database=Database(...),
)

results = await pipeline.process_posts(scraped_posts)
summary = pipeline.get_threat_summary()
```

### GeminiProvider

```python
provider = create_gemini_provider(api_key="...")

classification = await provider.classify_threat(
    "Rise up Kenya!",
    context={"platform": "twitter", "followers": 10000}
)

print(classification.tier)  # ThreatTier.TIER_3
```

## Kenya-Specific Keywords

```python
from kshiked.pulse.filters import (
    KENYA_POLITICAL,      # 48 keywords
    KENYA_ECONOMIC,       # 40 keywords
    KENYA_THREAT_SIGNALS, # 35 keywords
    KENYA_LOCATIONS,      # 50+ locations
    is_threat_related,    # Detection function
)
```

## Threat Tier Classification

Gemini classifies posts according to KShield taxonomy:

- **Intent Score**: Clear intent to cause harm (0-1)
- **Capability Score**: Author's reach/influence (0-1)
- **Specificity Score**: Named targets, times, locations (0-1)
- **Reach Score**: Potential audience size (0-1)

Combined into **Base Risk = 0.35×Intent + 0.20×Capability + 0.30×Specificity + 0.15×Reach**

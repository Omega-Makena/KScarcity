# KSHIELD AUDIT — BATCH MODE (kshiked/)

Role: forensic auditor. Scope: ONLY `kshiked/` (all `kshiked/**/*.py`). Evidence-only; no code changes performed.

## 1) Coverage Summary

- Files audited (Python): 88
- Missing files: 0
- Provided-order files: 70
- Extra `kshiked/**/*.py` files not in provided order list: 18

Quick prevalence counts (pattern-based, not a complete risk measure):
- UTF-16 encoded `__init__.py` under `kshiked/`: 4 (breaks Python imports; see P0)
- Files with absolute Windows path / OneDrive references: 29
- Files with bare `except:`: 6

## 2) Findings Grouped by Severity

### P0 — Crash / Import Breakage / Verification Failure

P0.1) `kshiked` cannot be imported because multiple package `__init__.py` files are UTF-16 encoded (null bytes).

**Evidence:** `kshiked/core/__init__.py:L1-L3`
```python
"""
Core KShield logic.
"""
```

**Evidence:** `kshiked/analysis/__init__.py:L1-L3`
```python
"""
Analysis modules for KShield.
"""
```

**Evidence:** `kshiked/tests/__init__.py:L1-L3`
```python
"""
Tests for KShield.
"""
```

**Evidence (command output):**
```text
kshiked/analysis/__init__.py: Python script, Unicode text, UTF-16, little-endian text executable, with CRLF line terminators
kshiked/core/__init__.py:     Python script, Unicode text, UTF-16, little-endian text executable, with CRLF line terminators
kshiked/sim/__init__.py:      Unicode text, UTF-16, little-endian text, with CRLF line terminators
kshiked/tests/__init__.py:    Python script, Unicode text, UTF-16, little-endian text executable, with CRLF line terminators
```

**Evidence (command output):**
```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/kshiked/__init__.py", line 8, in <module>
    from .core.governance import (
SyntaxError: source code string cannot contain null bytes
```

P0.2) `kshiked/__init__.py` imports `ShockType` from `kshiked/core/shocks.py`, but `ShockType` is NOT FOUND in that module (would raise ImportError even if UTF-16 issue were fixed).

**Evidence:** `kshiked/__init__.py:L7-L20`
```python
# Core Governance Modules
from .core.governance import (
    EconomicGovernor, 
    EconomicGovernorConfig
)
from .core.policies import (
    default_economic_policies,
    EconomicPolicy
)
from .core.shocks import (
    Shock, 
    ShockManager, 
    ShockType
)

```

**Evidence (command output):**
```text
NOT FOUND
```

**Evidence:** `kshiked/core/shocks.py:L1-L25`
```python
"""
Exogenous Shock System for Kshield.
Phase 4: Adversarial Stress Test (Stochastic Processes)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import random

@dataclass
class Shock:
    """Base Shock Class"""
    name: str
    target_metric: str
    active: bool = True

    def get_delta(self, current_val: float) -> float:
        return 0.0

    def step(self):
        pass

@dataclass
class ImpulseShock(Shock):
    """Classic 'Hit' shock"""
```

P0.3) `kshiked/sim/run_governance.py` imports non-existent modules `kshiked.governance` and `kshiked.shocks`.

**Evidence:** `kshiked/sim/run_governance.py:L24-L30`
```python
except ImportError as e:
    logger.error(f"Failed to import scarcity simulation components: {e}")
    sys.exit(1)

from kshiked.governance import EconomicGovernor, EconomicGovernorConfig
from kshiked.shocks import ShockManager, OUProcessShock

async def load_and_train_graph(csv_path: str):
```

**Evidence (command output):**
```text
MISSING
MISSING
```

P0.4) Mandatory test verification is currently FAILING (pytest capture crash AND import-time errors).

See “Verification Report” for verbatim outputs.

### P1 — Wrong Results / Silent Bug

P1.1) Multiple scripts hardcode an absolute Windows/OneDrive dataset path; portability and reproducibility are brittle (will fail on non-Windows or different user paths).

**Evidence:** `kshiked/sim/run_economic_simulation.py:L18-L25`
```python
    # 1. load data
    logger.info("loading kenya dataset...")
    df = pd.read_csv("C:/Users/omegam/OneDrive - Innova Limited/scace4/API_KEN_DS2_en_csv_v2_14659.csv", skiprows=4)
    
    # pivot logic
    pivoted = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                      var_name='Year', value_name='Value')
    pivoted['Year'] = pd.to_numeric(pivoted['Year'], errors='coerce')
```

**Evidence:** `kshiked/verify_names.py:L1-L12`
```python

import pandas as pd

codes = [
    "FR.INR.RINR", 
    "GC.XPN.TOTL.GD.ZS"
]

csv_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv"
df = pd.read_csv(csv_path, skiprows=4)
for code in codes:
    match = df[df['Indicator Code'] == code]['Indicator Name'].values
```

P1.2) Broad exception handling defaults centrality computations to zeros, risking silent corruption of network metrics on failure.

**Evidence:** `kshiked/pulse/network.py:L246-L263`
```python
        out_degree = dict(self.graph.out_degree())
        
        # Betweenness centrality (expensive, sample for large graphs)
        try:
            if len(self.graph) > 1000:
                betweenness = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
            else:
                betweenness = nx.betweenness_centrality(self.graph)
        except:
            betweenness = {n: 0.0 for n in self.graph.nodes()}
        
        # PageRank
        try:
            pagerank = nx.pagerank(self.graph, weight="weight")
        except:
            pagerank = {n: 0.0 for n in self.graph.nodes()}
        
```

P1.3) X client tweet parsing swallows timestamp parse errors with bare `except:` and silently falls back to `datetime.now()`.

**Evidence:** `kshiked/pulse/x_client.py:L439-L450`
```python
    def _parse_tweet(self, data: Dict[str, Any]) -> SocialPost:
        """Parse X API response into SocialPost."""
        metrics = data.get("public_metrics", {})
        
        created_at = datetime.now()
        if "created_at" in data:
            try:
                created_at = datetime.fromisoformat(
                    data["created_at"].replace("Z", "+00:00")
                )
            except:
                pass
        
```

### P2 — Reliability / Edge Cases

P2.1) `BaseScraper.stream()` runs indefinitely (`while True`) with no stop condition other than cancellation, and uses an unbounded `seen_ids` set (memory growth risk).

**Evidence:** `kshiked/pulse/scrapers/base.py:L537-L566`
```python
    async def stream(
        self,
        query: str,
        batch_size: int = 100,
    ) -> AsyncIterator[ScraperResult]:
        """
        Stream results for continuous monitoring.
        
        Yields results in batches, continuing until stopped.
        
        Args:
            query: Search query.
            batch_size: Results per batch.
            
        Yields:
            Individual ScraperResult objects.
        """
        seen_ids = set()
        
        while True:
            results = await self.scrape(query, limit=batch_size)
            
            for result in results:
                if result.platform_id not in seen_ids:
                    seen_ids.add(result.platform_id)
                    yield result
            
            # Wait before next batch
            await asyncio.sleep(60)  # 1 minute between batches
    
```

P2.2) `universal_downloader.py` embeds a very large Base64 string (repo/runtime bloat risk).

**Evidence (command output):**
```text
kshiked/pulse/diagrams/universal_downloader.py bytes=12866963
```

**Evidence (command output):** (CONTENT length without printing it)
```text
CONTENT_line 5
CONTENT_chars 12866160
file_bytes 12866963
```

P2.3) Gemini LLM response parsing uses a permissive regex-based fallback; malformed/partial model outputs can be coerced into a dict (classification quality + safety risk).

**Evidence:** `kshiked/pulse/llm/gemini.py:L184-L207`
```python
    def _parse_json(self, text: str) -> Dict:
        """Extract and parse JSON from response text."""
        # Try to find JSON in the response
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object or array
            match = re.search(r'(\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"Failed to parse JSON from response: {text[:200]}...")
            return {}
    
```

### P3 — Maintainability / Hygiene

P3.1) Multiple scripts modify `sys.path` at runtime (packaging/import hygiene risk).

**Evidence:** `kshiked/pulse/demo.py:L14-L22`
```python
import sys
sys.path.insert(0, '.')

import time
from datetime import datetime, timedelta
from kshiked.pulse import (
    PulseSensor, PulseState, SignalID, RiskScorer,
    create_kshield_bridge, SocialPost, Platform,
)
```

## 3) Per-file Mini Summary Table

Columns: file | purpose | overall risk | # findings (coarse) | key evidence pointer

| file | purpose | overall risk | # findings | key evidence pointers |
|---|---|---:|---:|---|
| kshiked/core/__init__.py | Core KShield logic. | CRITICAL | 0 | kshiked/core/__init__.py:L1-L5 |
| kshiked/core/governance.py | Governance logic for Kshield Economic Simulation (V4). Phase 4: Tensor Engine & Event Bus Logic. | LOW | 0 | kshiked/core/governance.py:L1-L5 |
| kshiked/core/policies.py | Enhanced policy definition for Economic Governance (V3). Now supports PID Control and Crisis Management. | LOW | 0 | kshiked/core/policies.py:L1-L5 |
| kshiked/core/shocks.py | Exogenous Shock System for Kshield. Phase 4: Adversarial Stress Test (Stochastic Processes) | LOW | 0 | kshiked/core/shocks.py:L1-L5 |
| kshiked/core/tensor_policies.py | Vectorized Policy Evaluation Engine (V4). Compiles N individual policies into Matrices for O(1) Batch Evaluation. | LOW | 0 | kshiked/core/tensor_policies.py:L1-L5 |
| kshiked/pulse/__init__.py | SIGINT layer for detecting social signals and mapping them to economic primitives. The Pulse Engine provides: - 15 intelligence signal de... | LOW | 0 | kshiked/pulse/__init__.py:L1-L5 |
| kshiked/pulse/bridge.py | KShield Simulation Integration — Bridge between Pulse and Economic Simulation Provides: - Clean interface for Pulse -> Simulation shock t... | LOW | 1 | kshiked/pulse/bridge.py:L1-L5 |
| kshiked/pulse/cli.py | KShield Pulse CLI - Command Line Interface Usage: python -m kshiked.pulse.cli analyze "text to analyze" python -m kshiked.pulse.cli repor... | LOW | 0 | kshiked/pulse/cli.py:L1-L5 |
| kshiked/pulse/config.py | Pulse Engine Configuration Loads API credentials and configuration from environment variables or .env files. NEVER hardcode credentials i... | MEDIUM | 2 | kshiked/pulse/config.py:L248-L248 |
| kshiked/pulse/cooccurrence.py | Time-Weighted Co-Occurrence Analysis Provides: - Temporal decay functions for signal weighting - Rolling time windows for signal aggregat... | LOW | 0 | kshiked/pulse/cooccurrence.py:L1-L5 |
| kshiked/pulse/dashboard.py | Pulse Engine Dashboard Components Provides Streamlit-based visualization for: - Real-time signal intensity - Signal co-occurrence heatmap... | LOW | 0 | kshiked/pulse/dashboard.py:L1-L5 |
| kshiked/pulse/demo.py | Pulse Engine Demo — Showcases the full pipeline with sample data This demo uses realistic sample social media posts to demonstrate: 1. Si... | LOW | 1 | kshiked/pulse/demo.py:L15-L15 |
| kshiked/pulse/demo_ingestion.py | KShield Pulse - Demo Script Demonstrates the full data ingestion pipeline. Usage: python demo_ingestion.py --test # Run with test data on... | HIGH | 3 | kshiked/pulse/demo_ingestion.py:L93-L93 |
| kshiked/pulse/detectors.py | NLP Signal Detectors — Enhanced Detectors using NLP Pipeline Replaces simple keyword matching with: - Sentiment-weighted detection - Enti... | LOW | 0 | kshiked/pulse/detectors.py:L1-L5 |
| kshiked/pulse/indices.py | KShield Pulse - Comprehensive Threat Indices Computes all threat-related indices from signal data: Phase 1 (HIGH Priority): - Polarizatio... | LOW | 0 | kshiked/pulse/indices.py:L1-L5 |
| kshiked/pulse/mapper.py | Signal Mapper — Maps 15 Intelligence Signals to Pulse Primitives This module defines the mapping logic between detected social signals an... | LOW | 1 | kshiked/pulse/mapper.py:L1-L5 |
| kshiked/pulse/network.py | Network Analysis Module for KShield Pulse Provides: - Actor role detection (Mobilizer, Broker, Ideologue, etc.) - Community detection (cl... | MEDIUM | 2 | kshiked/pulse/network.py:L255-L255 |
| kshiked/pulse/nlp.py | NLP Utilities — Core NLP Processing for Signal Detection Provides: - Sentiment Analysis (VADER-based for social media) - Named Entity Rec... | LOW | 0 | kshiked/pulse/nlp.py:L1-L5 |
| kshiked/pulse/primitives.py | Pulse Engine Primitives — Core Data Models Defines the four fundamental primitives that social signals map to: 1. ScarcityVector: Resourc... | LOW | 0 | kshiked/pulse/primitives.py:L1-L5 |
| kshiked/pulse/sensor.py | Pulse Sensor — Main Orchestrator for Signal Detection and Primitive Updates The PulseSensor is the central coordinator that: 1. Maintains... | LOW | 1 | kshiked/pulse/sensor.py:L1-L5 |
| kshiked/pulse/simulation_connector.py | KShield Simulation Connector Connects Pulse Engine signals to the Economic Simulation Engine. This module: 1. Translates Pulse threat ind... | LOW | 0 | kshiked/pulse/simulation_connector.py:L1-L5 |
| kshiked/pulse/social.py | Social Media API Client Framework Provides unified interfaces for: - Twitter/X API (v2) - TikTok API - Instagram API All clients implemen... | HIGH | 2 | kshiked/pulse/social.py:L1-L5 |
| kshiked/pulse/unified_dashboard.py | KShield Unified Dashboard Comprehensive visualization dashboard combining: 1. Pulse Engine - Signal detection, threat indices 2. Simulati... | LOW | 0 | kshiked/pulse/unified_dashboard.py:L1-L5 |
| kshiked/pulse/visualization.py | KShield Pulse - Visualization Module Provides interactive visualizations for threat analysis: - Kenya threat heatmap (geographic) - Threa... | MEDIUM | 3 | kshiked/pulse/visualization.py:L563-L563 |
| kshiked/pulse/x_client.py | X (Twitter) API v2 Client — Real Implementation Implements actual X API calls with: - Strict rate limiting for Free tier (100 posts/month... | HIGH | 6 | kshiked/pulse/x_client.py:L271-L271 |
| kshiked/pulse/db/__init__.py | Database layer for KShield Pulse data ingestion pipeline. Provides: - SQLAlchemy 2.0 async models - Connection pooling and transaction ma... | LOW | 0 | kshiked/pulse/db/__init__.py:L1-L5 |
| kshiked/pulse/db/database.py | Database Connection Layer for KShield Pulse Provides: - Async/sync database connections with SQLAlchemy 2.0 - Connection pooling for prod... | HIGH | 2 | kshiked/pulse/db/database.py:L13-L13 |
| kshiked/pulse/db/models.py | Database Models for KShield Pulse SQLAlchemy 2.0 models for storing: - Social media posts from all platforms - Processed signal detection... | LOW | 0 | kshiked/pulse/db/models.py:L1-L5 |
| kshiked/pulse/filters/__init__.py | Kenya-focused content filters for KShield Pulse. Provides keyword lists and filtering utilities for detecting Kenya-related content acros... | LOW | 0 | kshiked/pulse/filters/__init__.py:L1-L5 |
| kshiked/pulse/filters/kenya_keywords.py | Kenya-Focused Keywords for KShield Pulse Collections of keywords for filtering and prioritizing Kenya-related content across all platform... | LOW | 0 | kshiked/pulse/filters/kenya_keywords.py:L1-L5 |
| kshiked/pulse/ingestion/__init__.py | Ingestion orchestration for KShield Pulse. Provides: - IngestionOrchestrator: Main coordinator for all scrapers - IngestionScheduler: APS... | LOW | 0 | kshiked/pulse/ingestion/__init__.py:L1-L5 |
| kshiked/pulse/ingestion/orchestrator.py | Ingestion Orchestrator for KShield Pulse Main coordinator for the data ingestion pipeline. Architecture: - Manages all scraper instances ... | HIGH | 4 | kshiked/pulse/ingestion/orchestrator.py:L66-L66 |
| kshiked/pulse/ingestion/pipeline.py | Pipeline Integration Bridge for KShield Pulse Connects the data ingestion pipeline to the existing PulseSensor. This bridge: 1. Takes scr... | MEDIUM | 1 | kshiked/pulse/ingestion/pipeline.py:L1-L5 |
| kshiked/pulse/ingestion/scheduler.py | Scheduler for KShield Pulse Ingestion APScheduler-based job scheduling for: - Social media scraping (every 30 minutes) - E-commerce price... | LOW | 1 | kshiked/pulse/ingestion/scheduler.py:L1-L5 |
| kshiked/pulse/llm/__init__.py | LLM integration layer for KShield Pulse. Provides swappable LLM provider interface with: - Gemini implementation (default) - Threat tier ... | LOW | 0 | kshiked/pulse/llm/__init__.py:L1-L5 |
| kshiked/pulse/llm/base.py | Base LLM Provider Interface for KShield Pulse Provides: - Abstract LLMProvider interface (swappable) - Standard classification result typ... | LOW | 1 | kshiked/pulse/llm/base.py:L1-L5 |
| kshiked/pulse/llm/fine_tuning.py | Fine-Tuning Infrastructure for KShield Pulse Prepares training data for fine-tuning LLMs on Kenya-specific threat detection. Workflow: 1.... | LOW | 2 | kshiked/pulse/llm/fine_tuning.py:L292-L292 |
| kshiked/pulse/llm/gemini.py | Gemini LLM Provider for KShield Pulse Google Gemini implementation of the LLMProvider interface. Usage: provider = GeminiProvider(api_key... | LOW | 2 | kshiked/pulse/llm/gemini.py:L1-L5 |
| kshiked/pulse/llm/prompts.py | Prompt Templates for KShield Pulse LLM Contains carefully crafted prompts for: - Threat tier classification (Tier 0-5) - Role identificat... | LOW | 0 | kshiked/pulse/llm/prompts.py:L1-L5 |
| kshiked/pulse/scrapers/__init__.py | Social media and e-commerce scrapers for data ingestion. Provides unified scraper interface with dual architecture: - Primary: Scraping (... | LOW | 1 | kshiked/pulse/scrapers/__init__.py:L1-L5 |
| kshiked/pulse/scrapers/base.py | Base Scraper Classes for KShield Pulse Provides: - Abstract base class with dual scrape/API pattern - Common result types and error handl... | LOW | 1 | kshiked/pulse/scrapers/base.py:L1-L5 |
| kshiked/pulse/scrapers/facebook_scraper.py | Facebook Scraper for KShield Pulse Monitors Kenya-related public pages using Playwright and facebook-scraper. Usage: config = FacebookScr... | HIGH | 5 | kshiked/pulse/scrapers/facebook_scraper.py:L153-L153 |
| kshiked/pulse/scrapers/instagram_scraper.py | Instagram Scraper for KShield Pulse Monitors Kenya-related hashtags and public profiles using Instaloader. Usage: config = InstagramScrap... | HIGH | 3 | kshiked/pulse/scrapers/instagram_scraper.py:L1-L5 |
| kshiked/pulse/scrapers/reddit_scraper.py | Reddit Scraper for KShield Pulse Monitors Kenya-related subreddits and discussions using PRAW. Usage: config = RedditScraperConfig( clien... | HIGH | 4 | kshiked/pulse/scrapers/reddit_scraper.py:L17-L17 |
| kshiked/pulse/scrapers/telegram_scraper.py | Telegram Scraper for KShield Pulse Monitors Kenya-related public Telegram channels and groups using Telethon. Usage: config = TelegramScr... | HIGH | 4 | kshiked/pulse/scrapers/telegram_scraper.py:L17-L17 |
| kshiked/pulse/scrapers/x_scraper.py | X (Twitter) Scraper for KShield Pulse Multi-strategy X scraper: 1. Official API via existing x_client.py (if bearer token configured) 2. ... | HIGH | 7 | kshiked/pulse/scrapers/x_scraper.py:L16-L16 |
| kshiked/sim/__init__.py | NEEDS VERIFICATION | CRITICAL | 0 | kshiked/sim/__init__.py:L1-L5 |
| kshiked/sim/backtest_prediction.py | NEEDS VERIFICATION | HIGH | 3 | kshiked/sim/backtest_prediction.py:L252-L252 |
| kshiked/sim/demo_economic_simulation.py | NEEDS VERIFICATION | MEDIUM | 3 | kshiked/sim/demo_economic_simulation.py:L29-L29 |
| kshiked/sim/run_economic_simulation.py | run economic simulation: fiscal multiplier. scenario: does increasing 'military_exp_gdp' affect 'gdp_growth'? we compare a baseline simul... | HIGH | 2 | kshiked/sim/run_economic_simulation.py:L21-L21 |
| kshiked/sim/run_governance.py | Loads data and runs MPIE (or mock) to discover the causal graph. Returns (AgentRegistry, variable_names, df_pivot) | CRITICAL | 2 | kshiked/sim/run_governance.py:L94-L94 |
| kshiked/__init__.py | KShield - National Threat Detection & Economic Governance System Exposes core components for the unified dashboard. | CRITICAL | 0 | kshiked/__init__.py:L1-L5 |
| kshiked/hub.py | KShield Hub - Central Orchestrator Unified interface that brings together: 1. Pulse Engine (Threat Detection) 2. Scarcity Engine (Economi... | LOW | 1 | kshiked/hub.py:L1-L5 |
| kshiked/verify_names.py | NEEDS VERIFICATION | MEDIUM | 1 | kshiked/verify_names.py:L9-L9 |
| kshiked/tests/__init__.py | Tests for KShield. | CRITICAL | 0 | kshiked/tests/__init__.py:L1-L5 |
| kshiked/tests/benchmark_architecture.py | Benchmark: OOP vs Vectorized Architecture. Determines if we need GPU or just Vectorized CPU for 1M hypotheses. | LOW | 1 | kshiked/tests/benchmark_architecture.py:L9-L9 |
| kshiked/tests/debug_growth.py | NEEDS VERIFICATION | LOW | 1 | kshiked/tests/debug_growth.py:L7-L7 |
| kshiked/tests/prove_scarcity_v2.py | Proof of Concept: Online Relationship Discovery on Kenya Data. Demonstrates the new `OnlineDiscoveryEngine` finding relationships (Compos... | HIGH | 4 | kshiked/tests/prove_scarcity_v2.py:L27-L27 |
| kshiked/tests/sanity_check.py | NEEDS VERIFICATION | LOW | 2 | kshiked/tests/sanity_check.py:L95-L95 |
| kshiked/tests/sanity_check_sync.py | NEEDS VERIFICATION | LOW | 0 | kshiked/tests/sanity_check_sync.py:L1-L5 |
| kshiked/tests/test_ingestion.py | Tests for KShield Pulse Data Ingestion Pipeline Tests: - Database models and operations - Scraper initialization - LLM provider - Pipelin... | HIGH | 2 | kshiked/tests/test_ingestion.py:L49-L49 |
| kshiked/tests/test_integration.py | Integration Tests for KShield Pulse End-to-end tests for the complete pipeline: - Signal detection → Threat classification → Index comput... | LOW | 0 | kshiked/tests/test_integration.py:L1-L5 |
| kshiked/tests/test_pulse.py | Comprehensive Pulse Engine Tests Tests for: - Primitives (ScarcityVector, ActorStress, BondStrength, ShockPropagation) - NLP (SentimentAn... | LOW | 0 | kshiked/tests/test_pulse.py:L1-L5 |
| kshiked/tests/torture_test.py | Torture Test for Online Discovery Engine. Feeds the engine with: - NaNs and Nones - Infinities - Strings and Garbled text - Unexpected Ty... | HIGH | 1 | kshiked/tests/torture_test.py:L1-L5 |
| kshiked/tests/verify_features.py | Verification script for TerrainGenerator features (Opacity/Risk). | MEDIUM | 2 | kshiked/tests/verify_features.py:L10-L10 |
| kshiked/tests/verify_simulation.py | Verification Script for Policy Simulation. | HIGH | 2 | kshiked/tests/verify_simulation.py:L18-L18 |
| kshiked/tests/verify_tab2.py | Verification script for Tab 2 "Time-Manifold" logic. Mimics the loop in dashboard.py to ensure dimensions align. | MEDIUM | 2 | kshiked/tests/verify_tab2.py:L11-L11 |
| kshiked/tests/verify_terrain.py | Verification script for TerrainGenerator. Ensures that the policy-response surface generation works correctly. | MEDIUM | 2 | kshiked/tests/verify_terrain.py:L11-L11 |
| kshiked/tests/verify_trajectory.py | Verification script for TerrainGenerator with Trajectory. | MEDIUM | 2 | kshiked/tests/verify_trajectory.py:L10-L10 |
| kshiked/tests/verify_vectorized_engine.py | Verification Script for Vectorized Engine. | LOW | 0 | kshiked/tests/verify_vectorized_engine.py:L1-L5 |
| kshiked/analysis/__init__.py | Analysis modules for KShield. | CRITICAL | 0 | kshiked/analysis/__init__.py:L1-L5 |
| kshiked/analysis/analyze_data_quality.py | Analyze Dataset Suitability. Checks if N=65 is enough for our 1400 variables. | HIGH | 2 | kshiked/analysis/analyze_data_quality.py:L11-L11 |
| kshiked/analysis/find_crash.py | NEEDS VERIFICATION | MEDIUM | 2 | kshiked/analysis/find_crash.py:L4-L4 |
| kshiked/pulse/create_merged_pdf.py | Merge all documents from pulse folder into a single PDF file. Includes: text files, images, and PDFs. Excludes: HTML files | HIGH | 2 | kshiked/pulse/create_merged_pdf.py:L232-L232 |
| kshiked/pulse/diagrams/big_document.py | NEEDS VERIFICATION | LOW | 2 | kshiked/pulse/diagrams/big_document.py:L10-L10 |
| kshiked/pulse/diagrams/deepseek_python_20260126_45b585.py | NEEDS VERIFICATION | LOW | 0 | kshiked/pulse/diagrams/deepseek_python_20260126_45b585.py:L1-L5 |
| kshiked/pulse/diagrams/merge_all_with_images.py | Merge ALL documents (text + images) into a single HTML file. Images are embedded as base64. Excludes original HTML files. | HIGH | 3 | kshiked/pulse/diagrams/merge_all_with_images.py:L149-L149 |
| kshiked/pulse/diagrams/merge_documents.py | Script to merge all documents from a directory (excluding HTML files) and save them to a specified output location. | MEDIUM | 2 | kshiked/pulse/diagrams/merge_documents.py:L114-L114 |
| kshiked/pulse/diagrams/merge_to_single_file.py | Merge all text-based documents into a single file. Excludes HTML files and binary files (like images). | MEDIUM | 3 | kshiked/pulse/diagrams/merge_to_single_file.py:L86-L86 |
| kshiked/pulse/diagrams/universal_downloader.py | NEEDS VERIFICATION | HIGH | 3 | kshiked/pulse/diagrams/universal_downloader.py:L13-L13 |
| kshiked/pulse/merge_all_complete.py | Merge ALL documents from pulse folder (including subdirectories). Includes: text files, images, and PDFs (embedded as links or base64). E... | HIGH | 3 | kshiked/pulse/merge_all_complete.py:L203-L203 |
| kshiked/pulse/merge_docs_only.py | Merge only documents (images + PDFs) into a single PDF. NO code files included. | MEDIUM | 2 | kshiked/pulse/merge_docs_only.py:L96-L96 |
| kshiked/pulse/scrapers/ecommerce/__init__.py | E-commerce price scrapers for inflation monitoring. Tracks prices from Kenya's major e-commerce platforms: - Jiji Kenya (jiji.co.ke) - C2... | LOW | 0 | kshiked/pulse/scrapers/ecommerce/__init__.py:L1-L5 |
| kshiked/pulse/scrapers/ecommerce/base.py | Base E-Commerce Scraper for KShield Pulse Provides: - Abstract base class for e-commerce scrapers - Standard price data format - Category... | LOW | 1 | kshiked/pulse/scrapers/ecommerce/base.py:L1-L5 |
| kshiked/pulse/scrapers/ecommerce/jiji_scraper.py | Jiji Kenya Scraper for KShield Pulse Scrapes Jiji Kenya (jiji.co.ke) for: - Product prices and listings - Property rentals (housing costs... | HIGH | 4 | kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:L46-L46 |
| kshiked/pulse/scrapers/ecommerce/jumia_scraper.py | Jumia Kenya Scraper for KShield Pulse Scrapes Jumia Kenya (jumia.co.ke) for: - Grocery and daily essentials prices - Electronics prices -... | HIGH | 5 | kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:L45-L45 |
| kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py | Kilimall Kenya Scraper for KShield Pulse Scrapes Kilimall (kilimall.co.ke) for: - Electronics prices - Fashion and clothing - Home applia... | HIGH | 4 | kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:L45-L45 |
| kshiked/pulse/scrapers/ecommerce/price_aggregator.py | Price Aggregator for KShield Pulse Computes inflation indices from e-commerce price data: - Daily price indices by category - Week-over-w... | LOW | 1 | kshiked/pulse/scrapers/ecommerce/price_aggregator.py:L1-L5 |

## 4) Hardcoding Ledger (>=30 parameters)

Evidence blocks below show where parameters are defined. Ledger rows reference those definitions; “Where used” is limited to in-file usage unless explicitly evidenced elsewhere.

**Evidence:** `kshiked/core/policies.py:L5-L23`
```python
@dataclass
class EconomicPolicy(PolicyRule):
    """
    Enhanced policy definition for Economic Governance (V3).
    Now supports PID Control and Crisis Management.
    """
    authority: str = "Central Bank"
    cooldown: int = 5
    temporal_lag: int = 0
    uncertainty_tolerance: float = 0.2
    
    # PID Control Parameters
    kp: float = 0.0          # Proportional Gain (replaces 'factor')
    ki: float = 0.0          # Integral Gain
    kd: float = 0.0          # Derivative Gain
    
    # Crisis Management
    crisis_threshold: float = 999.0 # Value which triggers Crisis Mode
    crisis_multiplier: float = 5.0  # Weight multiplier during crisis
```

**Evidence:** `kshiked/core/tensor_policies.py:L15-L38`
```python
@dataclass
class TensorEngineConfig:
    """
    Configuration for PolicyTensorEngine.
    
    Allows runtime tuning of crisis response and other parameters
    without modifying individual policies.
    """
    # Crisis response configuration
    crisis_multiplier: float = 5.0  # Weight multiplier during crisis (was hardcoded 5.0)
    normal_weight: float = 1.0      # Weight multiplier during normal conditions
    
    # PID tuning overrides (applied globally if > 0)
    global_kp_scale: float = 1.0    # Scale factor for proportional gains
    global_ki_scale: float = 1.0    # Scale factor for integral gains
    global_kd_scale: float = 1.0    # Scale factor for derivative gains
    
    # Integral windup prevention
    integral_max: float = 100.0     # Maximum integral accumulation
    integral_min: float = -100.0    # Minimum integral accumulation
    
    # Output limiting
    max_magnitude: float = 10.0     # Maximum action magnitude per policy
    min_action_threshold: float = 1e-6  # Minimum magnitude to report
```

**Evidence:** `kshiked/pulse/sensor.py:L259-L274`
```python
@dataclass
class PulseSensorConfig:
    """Configuration for the Pulse Sensor."""
    # Detection thresholds
    min_intensity_threshold: float = 0.1
    min_confidence_threshold: float = 0.3
    
    # Time decay for signal aggregation (lambda for exponential decay)
    time_decay_lambda: float = 0.01  # per second
    
    # Aggregation window (seconds)
    aggregation_window: float = 3600.0  # 1 hour
    
    # Update throttling (minimum seconds between state updates)
    update_interval: float = 60.0

```

**Evidence:** `kshiked/pulse/config.py:L115-L150`
```python
@dataclass  
class PulseConfig:
    """
    Master configuration for Pulse Engine.
    
    Usage:
        config = PulseConfig.from_env()
        if config.x_api.is_configured():
            client = TwitterClient(config.x_api)
    """
    x_api: XAPIConfig = field(default_factory=XAPIConfig)
    tiktok_api: TikTokAPIConfig = field(default_factory=TikTokAPIConfig)
    instagram_api: InstagramAPIConfig = field(default_factory=InstagramAPIConfig)
    
    # Sensor settings
    use_nlp: bool = True
    min_intensity_threshold: float = 0.1
    min_confidence_threshold: float = 0.3
    
    # Bridge settings
    shock_interval_seconds: float = 300
    enable_probabilistic_shocks: bool = True
    
    @classmethod
    def from_env(cls) -> "PulseConfig":
        """Load all configuration from environment."""
        return cls(
            x_api=XAPIConfig.from_env(),
            tiktok_api=TikTokAPIConfig.from_env(),
            instagram_api=InstagramAPIConfig.from_env(),
            use_nlp=os.getenv("PULSE_USE_NLP", "true").lower() == "true",
            min_intensity_threshold=float(os.getenv("PULSE_MIN_INTENSITY", "0.1")),
            min_confidence_threshold=float(os.getenv("PULSE_MIN_CONFIDENCE", "0.3")),
            shock_interval_seconds=float(os.getenv("PULSE_SHOCK_INTERVAL", "300")),
            enable_probabilistic_shocks=os.getenv("PULSE_PROBABILISTIC", "true").lower() == "true",
        )
```

**Evidence:** `kshiked/pulse/config.py:L238-L297`
```python
@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = ""
    pool_size: int = 5
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        load_env_file()
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///pulse.db"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        )


@dataclass
class ScraperConfig:
    """
    Unified configuration for all scrapers.
    
    Usage:
        config = ScraperConfig.from_env()
        if config.reddit.is_configured():
            scraper = RedditScraper(config.reddit)
    """
    # Social media
    x_api: XAPIConfig = field(default_factory=XAPIConfig)
    reddit: RedditAPIConfig = field(default_factory=RedditAPIConfig)
    telegram: TelegramAPIConfig = field(default_factory=TelegramAPIConfig)
    instagram: InstagramAPIConfig = field(default_factory=InstagramAPIConfig)
    facebook: FacebookAPIConfig = field(default_factory=FacebookAPIConfig)
    
    # LLM
    gemini: GeminiAPIConfig = field(default_factory=GeminiAPIConfig)
    
    # Database
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Scraping settings
    social_scrape_interval_minutes: int = 30
    ecommerce_scrape_interval_hours: int = 6
    max_posts_per_scrape: int = 100
    max_products_per_category: int = 50
    
    @classmethod
    def from_env(cls) -> "ScraperConfig":
        """Load all scraper configuration from environment."""
        return cls(
            x_api=XAPIConfig.from_env(),
            reddit=RedditAPIConfig.from_env(),
            telegram=TelegramAPIConfig.from_env(),
            instagram=InstagramAPIConfig.from_env(),
            facebook=FacebookAPIConfig.from_env(),
            gemini=GeminiAPIConfig.from_env(),
            database=DatabaseConfig.from_env(),
            social_scrape_interval_minutes=int(os.getenv("SCRAPE_INTERVAL_MINUTES", "30")),
            ecommerce_scrape_interval_hours=int(os.getenv("ECOMMERCE_INTERVAL_HOURS", "6")),
            max_posts_per_scrape=int(os.getenv("MAX_POSTS_PER_SCRAPE", "100")),
            max_products_per_category=int(os.getenv("MAX_PRODUCTS_PER_CATEGORY", "50")),
        )
```

**Evidence:** `kshiked/pulse/social.py:L120-L139`
```python
class StreamConfig:
    """Configuration for streaming API connections."""
    # Filter terms
    keywords: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Rate limiting
    max_posts_per_minute: int = 100
    backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    
    # Deduplication
    dedupe_window_seconds: int = 3600
    
    # Reconnection
    max_reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 5.0

```

**Evidence:** `kshiked/pulse/bridge.py:L216-L223`
```python
@dataclass
class SchedulerConfig:
    """Configuration for shock scheduling."""
    min_interval_seconds: float = 300       # Min time between shocks
    max_pending_shocks: int = 10            # Max shocks in queue
    probability_scale: float = 1.0          # Scale for probabilistic triggering
    enable_probabilistic: bool = True       # Use probabilistic triggering

```

**Evidence:** `kshiked/pulse/cooccurrence.py:L305-L312`
```python
    # Category weights for risk computation
    CATEGORY_WEIGHTS = {
        SignalCategory.DISTRESS: 0.15,
        SignalCategory.ANGER: 0.25,
        SignalCategory.INSTITUTIONAL: 0.25,
        SignalCategory.IDENTITY: 0.20,
        SignalCategory.INFORMATION: 0.15,
    }
```

**Evidence:** `kshiked/pulse/simulation_connector.py:L43-L69`
```python
@dataclass
class ShockConfig:
    """Configuration for signal → shock mapping."""
    
    # Thresholds for shock generation
    min_instability: float = 0.25
    min_crisis_probability: float = 0.20
    min_index_value: float = 0.30
    
    # Magnitude scaling
    gdp_coefficient: float = -0.05  # Negative = GDP decrease
    inflation_coefficient: float = 0.10
    trade_coefficient: float = -0.08
    currency_coefficient: float = -0.06
    confidence_coefficient: float = -0.10
    
    # Timing
    shock_duration_steps: int = 4
    decay_rate: float = 0.5
    
    # Simulation variable names (match your economic model)
    gdp_variable: str = "GDP (current US$)"
    inflation_variable: str = "Inflation, consumer prices (annual %)"
    trade_variable: str = "Exports of goods and services (BoP, current US$)"
    currency_variable: str = "Official exchange rate (LCU per US$, period average)"
    confidence_variable: str = "Consumer confidence index"

```

**Evidence:** `kshiked/pulse/ingestion/orchestrator.py:L53-L123`
```python
@dataclass
class IngestionConfig:
    """
    Configuration for the ingestion orchestrator.
    
    Loads from environment variables or uses defaults.
    """
    # Database
    database_url: Optional[str] = None
    
    # X/Twitter
    x_accounts: List[Dict[str, str]] = field(default_factory=list)
    x_nitter_instances: List[str] = field(default_factory=lambda: [
        "https://nitter.net",
    ])
    x_bearer_token: str = ""
    
    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "Kenya", "NairobiCity", "africa",
    ])
    
    # Telegram
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_channels: List[str] = field(default_factory=list)
    
    # Instagram
    instagram_username: str = ""
    instagram_password: str = ""
    instagram_hashtags: List[str] = field(default_factory=lambda: [
        "Kenya", "Nairobi", "KenyaNews",
    ])
    
    # Facebook
    facebook_email: str = ""
    facebook_password: str = ""
    facebook_pages: List[str] = field(default_factory=list)
    
    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    
    # E-commerce
    ecommerce_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        "jiji": ["vehicles", "property", "electronics"],
        "jumia": ["groceries", "phones", "health-beauty"],
        "kilimall": ["electronics", "fashion", "home-living"],
    })
    
    # Scraping schedule
    social_scrape_interval_minutes: int = 30
    ecommerce_scrape_interval_hours: int = 6
    
    # Limits
    max_posts_per_scrape: int = 100
    max_products_per_category: int = 50
    
    # Kenya-focused search terms
    kenya_search_terms: List[str] = field(default_factory=lambda: [
        "Kenya",
        "Nairobi",
        "Ruto",
        "Raila",
        "maandamano",
        "cost of living Kenya",
        "unga prices",
    ])
    
```

**Evidence:** `kshiked/pulse/x_client.py:L234-L256`
```python
@dataclass
class XClientConfig:
    """Configuration for X client."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    bearer_token: str = ""
    
    # Rate limiting
    min_request_interval: float = 2.0  # seconds between requests
    
    @classmethod
    def from_env(cls) -> "XClientConfig":
        """Load from environment."""
        config = XAPIConfig.from_env()
        return cls(
            api_key=config.api_key,
            api_secret=config.api_secret,
            access_token=config.access_token,
            access_secret=config.access_secret,
            bearer_token=config.bearer_token,
        )
```

| Parameter | Where defined | Where used | Hardcoded/Learned/External | Overridable? | Risk |
|---|---|---|---|---|---|
| EconomicPolicy.cooldown | `kshiked/core/policies.py:L11-L23` | NEEDS VERIFICATION | Hardcoded default | Config default | Governance response cadence |
| EconomicPolicy.temporal_lag | `kshiked/core/policies.py:L11-L23` | NEEDS VERIFICATION | Hardcoded default | Config default | Lag assumptions affect control |
| EconomicPolicy.uncertainty_tolerance | `kshiked/core/policies.py:L11-L23` | NEEDS VERIFICATION | Hardcoded default | Config default | Tolerance controls sensitivity |
| EconomicPolicy.kp | `kshiked/core/policies.py:L16-L23` | Used by PolicyTensorEngine PID | Hardcoded default | Config default | Control gain stability |
| EconomicPolicy.ki | `kshiked/core/policies.py:L16-L23` | Used by PolicyTensorEngine PID | Hardcoded default | Config default | Integral windup risk |
| EconomicPolicy.kd | `kshiked/core/policies.py:L16-L23` | Used by PolicyTensorEngine PID | Hardcoded default | Config default | Derivative noise sensitivity |
| EconomicPolicy.crisis_threshold | `kshiked/core/policies.py:L21-L23` | Used by PolicyTensorEngine crisis check | Hardcoded default | Config default | Crisis trigger |
| EconomicPolicy.crisis_multiplier | `kshiked/core/policies.py:L21-L23` | NEEDS VERIFICATION | Hardcoded default | Config default | Crisis weight |
| TensorEngineConfig.crisis_multiplier | `kshiked/core/tensor_policies.py:L15-L38` | Used in np.where(is_crisis, ...) | Hardcoded default | Constructor arg | Amplifies actions in crisis |
| TensorEngineConfig.normal_weight | `kshiked/core/tensor_policies.py:L15-L38` | Used in np.where(is_crisis, ...) | Hardcoded default | Constructor arg | Baseline action weight |
| TensorEngineConfig.global_kp_scale | `kshiked/core/tensor_policies.py:L15-L38` | Used in p_term scaling | Hardcoded default | Constructor arg | Global gain override |
| TensorEngineConfig.global_ki_scale | `kshiked/core/tensor_policies.py:L15-L38` | Used in i_term scaling | Hardcoded default | Constructor arg | Global gain override |
| TensorEngineConfig.global_kd_scale | `kshiked/core/tensor_policies.py:L15-L38` | Used in d_term scaling | Hardcoded default | Constructor arg | Global gain override |
| TensorEngineConfig.integral_max | `kshiked/core/tensor_policies.py:L15-L38` | Used in np.clip(integrals) | Hardcoded default | Constructor arg | Windup clamp |
| TensorEngineConfig.integral_min | `kshiked/core/tensor_policies.py:L15-L38` | Used in np.clip(integrals) | Hardcoded default | Constructor arg | Windup clamp |
| TensorEngineConfig.max_magnitude | `kshiked/core/tensor_policies.py:L15-L38` | Used in np.clip(magnitudes) | Hardcoded default | Constructor arg | Action clamp |
| TensorEngineConfig.min_action_threshold | `kshiked/core/tensor_policies.py:L15-L38` | Used to filter results | Hardcoded default | Constructor arg | Suppress small actions |
| PulseSensorConfig.min_intensity_threshold | `kshiked/pulse/sensor.py:L259-L274` | Used by _passes_thresholds() | Hardcoded default | Constructor arg | Signal gating |
| PulseSensorConfig.min_confidence_threshold | `kshiked/pulse/sensor.py:L259-L274` | Used by _passes_thresholds() | Hardcoded default | Constructor arg | Signal gating |
| PulseSensorConfig.time_decay_lambda | `kshiked/pulse/sensor.py:L259-L274` | Used for decay weighting | Hardcoded default | Constructor arg | Aggregation dynamics |
| PulseSensorConfig.aggregation_window | `kshiked/pulse/sensor.py:L259-L274` | Used to prune history | Hardcoded default | Constructor arg | Memory/time horizon |
| PulseSensorConfig.update_interval | `kshiked/pulse/sensor.py:L259-L274` | Used to throttle updates | Hardcoded default | Constructor arg | Update cadence |
| PulseConfig.min_intensity_threshold | `kshiked/pulse/config.py:L129-L150` | Loaded from env PULSE_MIN_INTENSITY | Hardcoded default | Env var | Threshold config drift |
| PulseConfig.min_confidence_threshold | `kshiked/pulse/config.py:L129-L150` | Loaded from env PULSE_MIN_CONFIDENCE | Hardcoded default | Env var | Threshold config drift |
| PulseConfig.shock_interval_seconds | `kshiked/pulse/config.py:L134-L150` | Loaded from env PULSE_SHOCK_INTERVAL | Hardcoded default | Env var | Shock cadence |
| DatabaseConfig.url | `kshiked/pulse/config.py:L238-L250` | Loaded from env DATABASE_URL | Default hardcoded | Env var | DB location/coupling |
| DatabaseConfig.pool_size | `kshiked/pulse/config.py:L238-L250` | Loaded from env DB_POOL_SIZE | Hardcoded default | Env var | Connection resources |
| ScraperConfig.social_scrape_interval_minutes | `kshiked/pulse/config.py:L276-L297` | Loaded from env SCRAPE_INTERVAL_MINUTES | Hardcoded default | Env var | Scrape cadence |
| ScraperConfig.ecommerce_scrape_interval_hours | `kshiked/pulse/config.py:L276-L297` | Loaded from env ECOMMERCE_INTERVAL_HOURS | Hardcoded default | Env var | Scrape cadence |
| ScraperConfig.max_posts_per_scrape | `kshiked/pulse/config.py:L276-L297` | Loaded from env MAX_POSTS_PER_SCRAPE | Hardcoded default | Env var | Volume limits |

## 5) Claims vs Reality Tables (evidence-backed where possible)

### NLP/LLM claims vs reality

| Claim | Evidence | Reality | Evidence |
|---|---|---|---|
| Prompts require “Return ONLY JSON” | `kshiked/pulse/llm/prompts.py:L165-L189` | Parser accepts non-JSON (regex fallback) | `kshiked/pulse/llm/gemini.py:L184-L207` |

### Social/graph mapping claims vs reality

| Claim | Evidence | Reality | Evidence |
|---|---|---|---|
| Centrality computed; betweenness is sampled when graph is large | `kshiked/pulse/network.py:L249-L253` | Exceptions fall back to zeros without surfacing the root cause | `kshiked/pulse/network.py:L250-L262` |

### Simulation/economy claims vs reality

| Claim | Evidence | Reality | Evidence |
|---|---|---|---|
| Simulation scripts load Kenya dataset | `kshiked/sim/demo_economic_simulation.py:L27-L33` | Path is hardcoded to a specific Windows user directory | `kshiked/sim/demo_economic_simulation.py:L27-L33` |

## 6) Verification Report (verbatim outputs)

### 6.1 pytest --collect-only -vv (verbatim)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /mnt/c/Users/omegam/OneDrive - Innova Limited/scace4
configfile: pyproject.toml
plugins: anyio-4.11.0, asyncio-1.3.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.12/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
                     ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/config/__init__.py", line 221, in console_main
    code = main()
           ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/config/__init__.py", line 197, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/main.py", line 360, in wrap_session
    config._ensure_unconfigure()
  File "/usr/local/lib/python3.12/dist-packages/_pytest/config/__init__.py", line 1140, in _ensure_unconfigure
    self._cleanup_stack.close()
  File "/usr/lib/python3.12/contextlib.py", line 618, in close
    self.__exit__(None, None, None)
  File "/usr/lib/python3.12/contextlib.py", line 610, in __exit__
    raise exc_details[1]
  File "/usr/lib/python3.12/contextlib.py", line 595, in __exit__
    if cb(*exc_details):
       ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 478, in _exit_wrapper
    callback(*args, **kwds)
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 778, in stop_global_capturing
    self._global_capturing.pop_outerr_to_orig()
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 659, in pop_outerr_to_orig
    out, err = self.readouterr()
               ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 706, in readouterr
    out = self.out.snap() if self.out else ""
          ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 594, in snap
    self.tmpfile.truncate()
FileNotFoundError: [Errno 2] No such file or directory
collected 0 items

========================= no tests collected in 1.72s ==========================
```

### 6.2 pytest -q (verbatim)
```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.12/dist-packages/pytest/__main__.py", line 9, in <module>
    raise SystemExit(pytest.console_main())
                     ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/config/__init__.py", line 221, in console_main
    code = main()
           ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/config/__init__.py", line 197, in main
    ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/usr/local/lib/python3.12/dist-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/main.py", line 365, in pytest_cmdline_main
    return wrap_session(config, _main)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/main.py", line 360, in wrap_session
    config._ensure_unconfigure()
  File "/usr/local/lib/python3.12/dist-packages/_pytest/config/__init__.py", line 1140, in _ensure_unconfigure
    self._cleanup_stack.close()
  File "/usr/lib/python3.12/contextlib.py", line 618, in close
    self.__exit__(None, None, None)
  File "/usr/lib/python3.12/contextlib.py", line 610, in __exit__
    raise exc_details[1]
  File "/usr/lib/python3.12/contextlib.py", line 595, in __exit__
    if cb(*exc_details):
       ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 478, in _exit_wrapper
    callback(*args, **kwds)
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 778, in stop_global_capturing
    self._global_capturing.pop_outerr_to_orig()
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 659, in pop_outerr_to_orig
    out, err = self.readouterr()
               ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 706, in readouterr
    out = self.out.snap() if self.out else ""
          ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/_pytest/capture.py", line 594, in snap
    self.tmpfile.truncate()
FileNotFoundError: [Errno 2] No such file or directory

no tests ran in 1.55s
```

### 6.3 Additional verification runs (capture disabled)
These were run to surface collection/import errors after the capture error. Outputs are included verbatim for reproducibility.

#### pytest --collect-only -vv --capture=no (verbatim)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /mnt/c/Users/omegam/OneDrive - Innova Limited/scace4
configfile: pyproject.toml
plugins: anyio-4.11.0, asyncio-1.3.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... Results written to shock_results.txt
collected 114 items / 8 errors / 1 skipped

<Dir scace4>
  <Dir backend>
    <Dir tests>
      <Module test_api_endpoints.py>
        Test v2 API endpoints with actual HTTP requests.
        <Function test_endpoints>
          Test all v2 API endpoints.
  <Package scarcity>
    <Package tests>
      <Module test_audit_federation_aggregation.py>
        <Function test_federated_aggregation_weighted>
        <Function test_federated_aggregation_adaptive>
      <Module test_audit_fmi_aggregation.py>
        <Function test_fmi_aggregation_applies_dp_noise>
      <Module test_audit_fmi_emitter.py>
        <Function test_fmi_emitter_bridges_meta_prior>
      <Module test_audit_fmi_validator.py>
        <Function test_fmi_validator_requires_dp_when_configured>
      <Module test_audit_granger.py>
        <Function test_granger_step_bounds>
      <Module test_audit_hypotheses_types.py>
        <Function test_engine_v2_initializes_all_hypothesis_types>
      <Module test_audit_meta_update.py>
        <Function test_meta_prior_update_applies_policy>
      <Module test_audit_online_algorithms.py>
        <Function test_rls_kalman_updates_finite>
      <Module test_audit_privacy_guard.py>
        <Function test_privacy_guard_noise_applied>
      <Module test_audit_secure_agg.py>
        <Function test_secure_aggregation_dropout_unmask>
      <Module test_audit_smoke.py>
        <Function test_audit_engine_v2_smoke>
        <Function test_audit_store_roundtrip>
      <Module test_audit_telemetry.py>
        <Function test_telemetry_snapshot_has_aliases>
      <Module test_audit_transport.py>
        <Function test_build_transport_selects_protocol>
      <Module test_audit_winsorizer.py>
        <Function test_evaluate_does_not_update_winsorizer>
      <Module test_engine_integration.py>
        Test: OnlineDiscoveryEngine with New Hypothesis Classes
        
        Validates that the V2 initialization works and processes data correctly.
        <Class TestEngineV2Integration>
          <Function test_initialize_v2_creates_hypotheses>
            initialize_v2 should populate the hypothesis pool.
          <Function test_process_rows_updates_hypotheses>
            process_row should update hypothesis metrics.
          <Function test_get_knowledge_graph>
            Should return learned relationships.
      <Module test_federation.py>
        Test: Federation Layer
        
        Validates Byzantine-robust aggregation methods work correctly.
        <Class TestFederatedAggregatorTrimmedMean>
          <Function test_rejects_outliers>
            Trimmed mean should ignore outlier updates.
        <Class TestFederatedAggregatorKrum>
          <Function test_krum_selects_honest>
            Krum should select the honest client closest to majority.
        <Class TestFederatedAggregatorBulyan>
          <Function test_bulyan_robust>
            Bulyan combines Krum selection with trimmed mean.
        <Class TestDetectOutliers>
          <Function test_identifies_outliers>
            detect_outliers should flag divergent updates.
        <Class TestFederationVarianceReduction>
          <Function test_aggregation_reduces_variance>
            Aggregation should produce lower variance than individual clients.
      <Module test_hierarchical_federation.py>
        Test: Hierarchical Federation
        
        Validates the hierarchical federated learning implementation including:
        - Domain basket management
        - Gossip protocol with local DP
        - Memory buffer with staleness handling
        - Two-layer aggregation with secure agg + central DP
        - End-to-end integration
        <Class TestBasketManager>
          Tests for domain basket management.
          <Function test_register_client_creates_basket>
            Registering a client should create a basket for the domain.
          <Function test_same_domain_same_basket>
            Clients in the same domain should be in the same basket.
          <Function test_different_domain_different_basket>
            Clients in different domains should be in different baskets.
          <Function test_basket_status_forming_until_min_size>
            Basket should be FORMING until it reaches minimum size.
          <Function test_get_basket_peers>
            Should return all clients in a basket.
          <Function test_unregister_client>
            Should remove client from basket.
        <Class TestLocalDPMechanism>
          Tests for local differential privacy.
          <Function test_noise_is_added>
            Noise should be added to vectors.
          <Function test_clip_enforces_norm>
            Clipping should enforce L2 norm bound.
          <Function test_higher_epsilon_less_noise>
            Higher epsilon should mean less noise (lower sigma).
        <Class TestGossipProtocol>
          Tests for gossip protocol.
          <Function test_create_message_applies_dp>
            Created messages should have DP noise applied.
          <Function test_message_budget_enforced>
            Message count per day should be limited.
          <Function test_pull_round_samples_peers>
            Pull round should sample k peers from basket.
        <Class TestMaterialityDetector>
          Tests for materiality detection.
          <Function test_first_update_is_material>
            First update should always be material.
          <Function test_small_change_not_material>
            Small changes should not trigger push.
          <Function test_large_change_is_material>
            Large changes should trigger push.
        <Class TestUpdateBuffer>
          Tests for update buffer.
          <Function test_add_update>
            Should store updates.
          <Function test_replay_detection>
            Should reject replay attacks.
          <Function test_weighted_aggregate>
            Should compute decay-weighted aggregate.
        <Class TestPrivacyAccountant>
          Tests for privacy budget tracking.
          <Function test_spend_budget>
            Should track spent budget.
          <Function test_budget_exhaustion>
            Should reject when budget exhausted.
          <Function test_can_release>
            Should check if release is allowed.
        <Class TestTriggerEngine>
          Tests for aggregation triggers.
          <Function test_count_trigger>
            Should trigger after count threshold.
          <Function test_time_trigger>
            Should trigger after time interval.
        <Class TestLayerAggregation>
          Tests for two-layer aggregation.
          <Function test_layer1_aggregates_basket>
            Layer 1 should aggregate within a basket.
          <Function test_layer2_bounded_influence>
            Layer 2 should clip basket contributions.
          <Function test_layer2_minimum_support>
            Layer 2 should require minimum basket support.
        <Class TestSecureAggregator>
          Tests for secure aggregation.
          <Function test_requires_min_participants>
            Should require minimum participants.
          <Function test_computes_sum>
            Should compute sum of shares.
        <Class TestHierarchicalFederation>
          End-to-end tests for hierarchical federation.
          <Function test_register_and_submit>
            Should allow registering clients and submitting updates.
          <Function test_multiple_baskets>
            Should handle multiple domain baskets.
          <Function test_end_to_end_aggregation>
            Should aggregate updates through both layers.
          <Function test_privacy_budget_tracking>
            Should track privacy budget across rounds.
        <Class TestConvergence>
          Tests for learning convergence.
          <Function test_gossip_converges_within_basket>
            Gossip should lead to convergence within a basket.
      <Module test_meta.py>
        Test: Meta-Learning Layer
        
        Validates cross-domain transfer and meta-update generation.
        <Class TestDomainMetaLearner>
          <Function test_observe_generates_update>
            observe() should generate a meta-update.
          <Function test_confidence_increases_with_good_performance>
            Confidence should increase with consistent improvements.
          <Function test_state_persistence>
            State should persist across observations.
        <Class TestCrossDomainTransfer>
          <Function test_multiple_domains_tracked>
            Learner should track multiple domains independently.
          <Function test_domain_updates_independent>
            Domain updates should be independent.
        <Class TestMetaLearningAdaptation>
          <Function test_adaptive_learning_rate>
            Meta learning rate should adapt based on confidence.
      <Module test_online_learning.py>
        Test: Online Learning and Real-time Updates
        
        Validates that the engine converges on streaming data and detects regime changes.
        <Class TestOnlineLearningConvergence>
          Test that online learning converges on stationary data.
          <Function test_mse_decreases_over_time>
            Prediction error should decrease as more data is seen.
          <Function test_convergence_on_functional_relationship>
            Functional hypothesis should converge to true coefficients.
        <Class TestRegimeChangeDetection>
          Test detection of regime changes in data.
          <Function test_equilibrium_detects_mean_shift>
            EquilibriumHypothesis should detect when mean shifts.
        <Class TestLatencyRequirements>
          Test that processing is fast enough for real-time.
          <Function test_process_row_under_50ms>
            Each row should process in under 50ms.
        <Class TestIncrementalUpdates>
          Test that model updates are truly incremental.
          <Function test_bounded_memory>
            Memory should not grow unboundedly with more data.
          <Function test_incremental_update>
            Each fit_step should be O(1), not O(n).
      <Module test_relationships.py>
        Test: Relationship Hypothesis Classes
        
        Tests that each hypothesis type correctly identifies its relationship type
        using the synthetic data generators.
        <Class TestCausalHypothesis>
          <Function test_detects_causal_relationship>
            Should detect X → Y causality.
        <Class TestCorrelationalHypothesis>
          <Function test_detects_correlation>
            Should detect correlation.
        <Class TestTemporalHypothesis>
          <Function test_detects_autocorrelation>
            Should detect autoregressive structure.
        <Class TestFunctionalHypothesis>
          <Function test_detects_functional_relationship>
            Should detect deterministic Y = f(X).
        <Class TestEquilibriumHypothesis>
          <Function test_detects_mean_reversion>
            Should detect mean-reverting process.
        <Class TestCompositionalHypothesis>
          <Function test_detects_sum_constraint>
            Should detect A + B + C = Total.
        <Class TestCompetitiveHypothesis>
          <Function test_detects_trade_off>
            Should detect X + Y = constant.
        <Class TestSynergisticHypothesis>
          <Function test_detects_interaction>
            Should detect significant X1*X2 interaction.
        <Class TestProbabilisticHypothesis>
          <Function test_detects_distribution_shift>
            Should detect X shifts distribution of Y.
        <Class TestStructuralHypothesis>
          <Function test_detects_hierarchy>
            Should detect group structure.
        <Class TestMediatingHypothesis>
          <Function test_detects_mediation>
            Should detect X → M → Y path.
        <Class TestModeratingHypothesis>
          <Function test_detects_moderation>
            Should detect Z moderates X→Y.
        <Class TestGraphHypothesis>
          <Function test_detects_graph_structure>
            Should track graph edges.
        <Class TestSimilarityHypothesis>
          <Function test_detects_clusters>
            Should detect cluster structure.
        <Class TestLogicalHypothesis>
          <Function test_detects_boolean_rule>
            Should detect Z = X AND Y.
      <Module test_sfc.py>
        Test: Stock-Flow Consistent Economic Simulation
        
        Validates SFC economy maintains consistency and produces sensible dynamics.
        <Class TestSectorBalanceSheet>
          <Function test_balance_sheet_identity>
            Assets = Liabilities + Net Worth.
          <Function test_net_lending>
            Net lending = Income - Expenses.
        <Class TestSFCEconomyInitialization>
          <Function test_initialize_creates_consistent_state>
            Initialization should create internally consistent state.
        <Class TestSFCEconomyDynamics>
          <Function test_gdp_follows_demand>
            GDP should adjust toward aggregate demand.
          <Function test_taylor_rule_responds_to_inflation>
            Interest rate should be higher with higher inflation.
          <Function test_unemployment_follows_okun>
            Unemployment should fall when GDP grows.
          <Function test_fiscal_deficit_increases_debt>
            Government deficit should increase debt.
        <Class TestSFCShocks>
          <Function test_demand_shock>
            Demand shock should increase GDP.
          <Function test_monetary_shock>
            Monetary shock should change interest rate.
        <Class TestSFCConsistency>
          <Function test_validate_sfc_economy>
            Full validation should pass.
          <Function test_history_recorded>
            History should be recorded at each step.
          <Function test_no_explosive_behavior>
            Economy should not explode over reasonable horizon.
      <Module test_synthetic.py>
        Test: Synthetic Data Generators
        
        Validates that all 15 synthetic data generators produce valid data.
        <Class TestCausalGenerator>
          <Function test_creates_valid_data>
          <Function test_lag_relationship>
        <Class TestCorrelationalGenerator>
          <Function test_creates_spurious_correlation>
        <Class TestStructuralGenerator>
          <Function test_creates_hierarchical_data>
        <Class TestTemporalGenerator>
          <Function test_creates_ar_process>
        <Class TestFunctionalGenerator>
          <Function test_exact_relationship>
        <Class TestProbabilisticGenerator>
          <Function test_distribution_shift>
        <Class TestCompositionalGenerator>
          <Function test_sum_constraint>
        <Class TestCompetitiveGenerator>
          <Function test_zero_sum>
        <Class TestSynergisticGenerator>
          <Function test_interaction_term>
        <Class TestMediatingGenerator>
          <Function test_mediation_path>
        <Class TestModeratingGenerator>
          <Function test_conditional_effect>
        <Class TestGraphGenerator>
          <Function test_creates_edges>
        <Class TestSimilarityGenerator>
          <Function test_creates_clusters>
        <Class TestEquilibriumGenerator>
          <Function test_mean_reversion>
        <Class TestLogicalGenerator>
          <Function test_boolean_rule>
        <Class TestGenerateAll>
          <Function test_generates_all_15>

==================================== ERRORS ====================================
_____________ ERROR collecting backend/tests/test_v2_endpoints.py ______________
ImportError while importing test module '/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/backend/tests/test_v2_endpoints.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
/usr/local/lib/python3.12/dist-packages/_pytest/assertion/rewrite.py:197: in exec_module
    exec(co, module.__dict__)
backend/tests/test_v2_endpoints.py:11: in <module>
    from app.core.scarcity_manager import ScarcityCoreManager
backend/app/core/scarcity_manager.py:31: in <module>
    from app.core.config import Settings, get_settings
backend/app/core/config.py:9: in <module>
    from pydantic_settings import BaseSettings, SettingsConfigDict
E   ModuleNotFoundError: No module named 'pydantic_settings'
_______________ ERROR collecting kshiked/tests/test_ingestion.py _______________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
______________ ERROR collecting kshiked/tests/test_integration.py ______________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
_________________ ERROR collecting kshiked/tests/test_pulse.py _________________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
________________ ERROR collecting kshiked/tests/torture_test.py ________________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
_______________________ ERROR collecting manual_test.py ________________________
ImportError while importing test module '/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/manual_test.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
/usr/local/lib/python3.12/dist-packages/_pytest/assertion/rewrite.py:197: in exec_module
    exec(co, module.__dict__)
manual_test.py:3: in <module>
    from dowhy import datasets
E   ModuleNotFoundError: No module named 'dowhy'
_______________________ ERROR collecting test_output.txt _______________________
/usr/lib/python3.12/pathlib.py:1030: in read_text
    return f.read()
           ^^^^^^^^
<frozen codecs>:322: in decode
    ???
E   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
__________________ ERROR collecting tests/test_diagnostics.py __________________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
/usr/local/lib/python3.12/dist-packages/_pytest/assertion/rewrite.py:197: in exec_module
    exec(co, module.__dict__)
tests/test_diagnostics.py:12: in <module>
    from kshiked.core.governance import EconomicGovernor, EconomicGovernorConfig
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
=============================== warnings summary ===============================
scarcity/tests/test_audit_secure_agg.py:7
  /mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/scarcity/tests/test_audit_secure_agg.py:7: PytestUnknownMarkWarning: Unknown pytest.mark.slow - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.slow

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
ERROR backend/tests/test_v2_endpoints.py
ERROR kshiked/tests/test_ingestion.py
ERROR kshiked/tests/test_integration.py
ERROR kshiked/tests/test_pulse.py
ERROR kshiked/tests/torture_test.py
ERROR manual_test.py
ERROR test_output.txt - UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
ERROR tests/test_diagnostics.py
!!!!!!!!!!!!!!!!!!! Interrupted: 8 errors during collection !!!!!!!!!!!!!!!!!!!!
=================== 114 tests collected, 8 errors in 11.71s ====================
```

#### pytest -q --capture=no (verbatim)
```text
Results written to shock_results.txt

==================================== ERRORS ====================================
_____________ ERROR collecting backend/tests/test_v2_endpoints.py ______________
ImportError while importing test module '/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/backend/tests/test_v2_endpoints.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
backend/tests/test_v2_endpoints.py:11: in <module>
    from app.core.scarcity_manager import ScarcityCoreManager
backend/app/core/scarcity_manager.py:31: in <module>
    from app.core.config import Settings, get_settings
backend/app/core/config.py:9: in <module>
    from pydantic_settings import BaseSettings, SettingsConfigDict
E   ModuleNotFoundError: No module named 'pydantic_settings'
_______________ ERROR collecting kshiked/tests/test_ingestion.py _______________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
______________ ERROR collecting kshiked/tests/test_integration.py ______________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
_________________ ERROR collecting kshiked/tests/test_pulse.py _________________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
________________ ERROR collecting kshiked/tests/torture_test.py ________________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
<frozen importlib._bootstrap_external>:995: in exec_module
    ???
<frozen importlib._bootstrap>:488: in _call_with_frames_removed
    ???
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
_______________________ ERROR collecting manual_test.py ________________________
ImportError while importing test module '/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/manual_test.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
manual_test.py:3: in <module>
    from dowhy import datasets
E   ModuleNotFoundError: No module named 'dowhy'
_______________________ ERROR collecting test_output.txt _______________________
/usr/lib/python3.12/pathlib.py:1030: in read_text
    return f.read()
           ^^^^^^^^
<frozen codecs>:322: in decode
    ???
E   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
__________________ ERROR collecting tests/test_diagnostics.py __________________
/usr/local/lib/python3.12/dist-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/usr/local/lib/python3.12/dist-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
/usr/local/lib/python3.12/dist-packages/_pytest/assertion/rewrite.py:197: in exec_module
    exec(co, module.__dict__)
tests/test_diagnostics.py:12: in <module>
    from kshiked.core.governance import EconomicGovernor, EconomicGovernorConfig
kshiked/__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes
=============================== warnings summary ===============================
scarcity/tests/test_audit_secure_agg.py:7
  /mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/scarcity/tests/test_audit_secure_agg.py:7: PytestUnknownMarkWarning: Unknown pytest.mark.slow - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.slow

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
ERROR backend/tests/test_v2_endpoints.py
ERROR kshiked/tests/test_ingestion.py
ERROR kshiked/tests/test_integration.py
ERROR kshiked/tests/test_pulse.py
ERROR kshiked/tests/torture_test.py
ERROR manual_test.py
ERROR test_output.txt - UnicodeDecodeError: 'utf-8' codec can't decode byte 0...
ERROR tests/test_diagnostics.py
!!!!!!!!!!!!!!!!!!! Interrupted: 8 errors during collection !!!!!!!!!!!!!!!!!!!!
1 skipped, 1 warning, 8 errors in 8.76s
```

### 6.4 Error Ledger

| # | Area | Exception | Evidence | Fix status |
|---:|---|---|---|---|
| 1 | pytest capture (default) | FileNotFoundError in pytest capture teardown | See 6.1 / 6.2 | NOT FIXED (audit-only) |
| 2 | kshiked import during test collection | SyntaxError: source code string cannot contain null bytes | See 6.3 output (kshiked/__init__.py import chain) + UTF-16 file evidence in Section 2 | NOT FIXED (audit-only) |
| 3 | kshiked package API | ImportError for ShockType (NOT FOUND) | Section 2 (P0.2) | NOT FIXED (audit-only) |
| 4 | sim demo runner | Imports missing kshiked.governance / kshiked.shocks | Section 2 (P0.3) | NOT FIXED (audit-only) |

## 7) NOT VERIFIED / NOT FOUND List

- NOT FOUND: `ShockType` definition in `kshiked/core/shocks.py` (see P0.2).
- NOT VERIFIED: External API behavior (X/Facebook/Telegram/Reddit/Instagram) — network calls require credentials and were not executed in this audit.
- NOT VERIFIED: Runtime behavior of simulation components (`scarcity.*`) — outside `kshiked/` scope.

## Appendix A) 50-item checklist scan (verbatim command outputs)
```text
## 01 eval
$ rg -n --glob 'kshiked/**/*.py' '\beval\b' kshiked || echo NOT FOUND
NOT FOUND

## 02 exec
$ rg -n --glob 'kshiked/**/*.py' '\bexec\b' kshiked || echo NOT FOUND
NOT FOUND

## 03 pickle
$ rg -n --glob 'kshiked/**/*.py' 'pickle\.(load|loads)' kshiked || echo NOT FOUND
NOT FOUND

## 04 yaml.load
$ rg -n --glob 'kshiked/**/*.py' 'yaml\.load\b' kshiked || echo NOT FOUND
NOT FOUND

## 05 subprocess
$ rg -n --glob 'kshiked/**/*.py' 'subprocess\.' kshiked || echo NOT FOUND
NOT FOUND

## 06 os.system
$ rg -n --glob 'kshiked/**/*.py' 'os\.system\b' kshiked || echo NOT FOUND
NOT FOUND

## 07 importlib dynamic
$ rg -n --glob 'kshiked/**/*.py' 'importlib\.(import_module|reload)' kshiked || echo NOT FOUND
NOT FOUND

## 08 requests/httpx/aiohttp
$ rg -n --glob 'kshiked/**/*.py' '\b(requests|httpx|aiohttp)\b' kshiked || echo NOT FOUND
kshiked/pulse/x_client.py:32:# Use httpx for async HTTP (fall back to requests if not available)
kshiked/pulse/x_client.py:34:    import httpx
kshiked/pulse/x_client.py:39:        import requests
kshiked/pulse/x_client.py:149:    """Generate OAuth 1.0a signatures for X API requests."""
kshiked/pulse/x_client.py:244:    min_request_interval: float = 2.0  # seconds between requests
kshiked/pulse/x_client.py:338:                async with httpx.AsyncClient() as client:
kshiked/pulse/x_client.py:350:                logger.error("httpx not installed. Run: pip install httpx")
kshiked/pulse/social.py:147:    Token bucket rate limiter for API requests.
kshiked/pulse/social.py:151:        self.rate = requests_per_minute / 60.0  # requests per second
kshiked/pulse/social.py:274:    the actual Twitter API using tweepy or httpx.
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:103:            import httpx
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:105:            self._session = httpx.AsyncClient(
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:118:            logger.error("httpx not installed. Run: pip install httpx")
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:109:            import httpx
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:112:            self._session = httpx.AsyncClient(
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:125:            logger.error(f"Missing dependencies: {e}. Run: pip install httpx beautifulsoup4")
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:108:            import httpx
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:110:            self._session = httpx.AsyncClient(
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:123:            logger.error("httpx not installed. Run: pip install httpx")

## 09 playwright
$ rg -n --glob 'kshiked/**/*.py' '\bplaywright\b' kshiked || echo NOT FOUND
kshiked/pulse/scrapers/facebook_scraper.py:123:            from playwright.async_api import async_playwright
kshiked/pulse/scrapers/facebook_scraper.py:125:            playwright = await async_playwright().start()
kshiked/pulse/scrapers/facebook_scraper.py:126:            self._browser = await playwright.chromium.launch(
kshiked/pulse/scrapers/facebook_scraper.py:142:                "Playwright not installed. Run: pip install playwright && playwright install chromium"
kshiked/pulse/scrapers/facebook_scraper.py:334:                            raw_data={"source": "playwright"},

## 10 telethon
$ rg -n --glob 'kshiked/**/*.py' '\btelethon\b' kshiked || echo NOT FOUND
kshiked/pulse/scrapers/telegram_scraper.py:115:            from telethon import TelegramClient
kshiked/pulse/scrapers/telegram_scraper.py:116:            from telethon.sessions import StringSession
kshiked/pulse/scrapers/telegram_scraper.py:138:            logger.warning("Telethon not installed. Run: pip install telethon")
kshiked/pulse/scrapers/telegram_scraper.py:166:            from telethon.tl.functions.messages import SearchRequest
kshiked/pulse/scrapers/telegram_scraper.py:167:            from telethon.tl.types import InputMessagesFilterEmpty
kshiked/pulse/scrapers/telegram_scraper.py:275:            from telethon import events
kshiked/pulse/scrapers/telegram_scraper.py:347:                    "source": "telethon",

## 11 hardcoded windows paths
$ rg -n --glob 'kshiked/**/*.py' '[A-Za-z]:(\\\\|/)' kshiked || echo NOT FOUND
kshiked/analysis/analyze_data_quality.py:11:    df = pd.read_csv("C:/Users/omegam/OneDrive - Innova Limited/scace4/API_KEN_DS2_en_csv_v2_14659.csv", skiprows=4)
kshiked/pulse/x_client.py:271:    BASE_URL = "https://api.twitter.com/2"
kshiked/tests/prove_scarcity_v2.py:27:    data_path = "c:\\Users\\omegam\\OneDrive - Innova Limited\\scace4\\API_KEN_DS2_en_csv_v2_14659.csv"
kshiked/sim/run_economic_simulation.py:21:    df = pd.read_csv("C:/Users/omegam/OneDrive - Innova Limited/scace4/API_KEN_DS2_en_csv_v2_14659.csv", skiprows=4)
kshiked/tests/test_ingestion.py:49:            product_url="https://jumia.co.ke/product/123",
kshiked/tests/test_ingestion.py:231:            product_url="https://jumia.co.ke/123",
kshiked/pulse/visualization.py:563:        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
kshiked/pulse/ingestion/orchestrator.py:66:        "https://nitter.net",
kshiked/tests/verify_simulation.py:18:    df = pd.read_csv("C:/Users/omegam/OneDrive - Innova Limited/scace4/API_KEN_DS2_en_csv_v2_14659.csv", skiprows=4)
kshiked/pulse/config.py:248:            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///pulse.db"),
kshiked/pulse/db/database.py:13:    db = Database("sqlite+aiosqlite:///pulse.db")
kshiked/pulse/db/database.py:80:            url = f"sqlite+aiosqlite:///{db_path}"
kshiked/pulse/demo_ingestion.py:93:    config = DatabaseConfig(url="sqlite+aiosqlite:///test_demo.db")
kshiked/pulse/scrapers/facebook_scraper.py:153:            await self._page.goto("https://www.facebook.com/login")
kshiked/pulse/scrapers/facebook_scraper.py:312:            url = f"https://www.facebook.com/search/posts?q={query}"
kshiked/pulse/scrapers/telegram_scraper.py:17:    Requires Telegram API credentials from https://my.telegram.org
kshiked/pulse/scrapers/telegram_scraper.py:46:    # Telegram API credentials (from https://my.telegram.org)
kshiked/pulse/scrapers/telegram_scraper.py:110:                "Get them from https://my.telegram.org"
kshiked/pulse/scrapers/reddit_scraper.py:17:    Reddit API requires registration at https://www.reddit.com/prefs/apps
kshiked/pulse/scrapers/reddit_scraper.py:46:    # Reddit API credentials (from https://www.reddit.com/prefs/apps)
kshiked/pulse/scrapers/x_scraper.py:16:        nitter_instances=["https://nitter.net"],
kshiked/pulse/scrapers/x_scraper.py:79:        "https://nitter.net",
kshiked/pulse/scrapers/x_scraper.py:80:        "https://nitter.privacydev.net",
kshiked/pulse/scrapers/x_scraper.py:81:        "https://nitter.poast.org",
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:46:    base_url: str = "https://jiji.co.ke"
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:45:    base_url: str = "https://www.kilimall.co.ke"
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:45:    base_url: str = "https://www.jumia.co.ke"

## 12 OneDrive
$ rg -n --glob 'kshiked/**/*.py' 'OneDrive' kshiked || echo NOT FOUND
kshiked/sim/run_economic_simulation.py:21:    df = pd.read_csv("C:/Users/omegam/OneDrive - Innova Limited/scace4/API_KEN_DS2_en_csv_v2_14659.csv", skiprows=4)
kshiked/verify_names.py:9:csv_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv"
kshiked/sim/demo_economic_simulation.py:29:    csv_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv"
kshiked/sim/run_governance.py:94:    csv_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv"
kshiked/analysis/find_crash.py:4:csv_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv"
kshiked/analysis/analyze_data_quality.py:11:    df = pd.read_csv("C:/Users/omegam/OneDrive - Innova Limited/scace4/API_KEN_DS2_en_csv_v2_14659.csv", skiprows=4)
kshiked/sim/backtest_prediction.py:252:    csv_path=r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv",
kshiked/tests/prove_scarcity_v2.py:27:    data_path = "c:\\Users\\omegam\\OneDrive - Innova Limited\\scace4\\API_KEN_DS2_en_csv_v2_14659.csv"
kshiked/tests/verify_simulation.py:18:    df = pd.read_csv("C:/Users/omegam/OneDrive - Innova Limited/scace4/API_KEN_DS2_en_csv_v2_14659.csv", skiprows=4)
kshiked/pulse/merge_docs_only.py:96:        r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse",
kshiked/pulse/merge_all_complete.py:203:    SOURCE_DIR = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse"
kshiked/pulse/create_merged_pdf.py:232:    SOURCE_DIR = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse"
kshiked/pulse/diagrams/merge_all_with_images.py:149:    SOURCE_DIR = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse\diagrams"
kshiked/pulse/diagrams/merge_documents.py:114:    SOURCE_DIR = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse\diagrams"
kshiked/pulse/diagrams/merge_to_single_file.py:86:    SOURCE_DIR = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse\diagrams"

## 13 hardcoded urls
$ rg -n --glob 'kshiked/**/*.py' 'https?://' kshiked || echo NOT FOUND
kshiked/pulse/x_client.py:271:    BASE_URL = "https://api.twitter.com/2"
kshiked/pulse/visualization.py:563:        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
kshiked/tests/test_ingestion.py:49:            product_url="https://jumia.co.ke/product/123",
kshiked/tests/test_ingestion.py:231:            product_url="https://jumia.co.ke/123",
kshiked/pulse/ingestion/orchestrator.py:66:        "https://nitter.net",
kshiked/pulse/scrapers/telegram_scraper.py:17:    Requires Telegram API credentials from https://my.telegram.org
kshiked/pulse/scrapers/telegram_scraper.py:46:    # Telegram API credentials (from https://my.telegram.org)
kshiked/pulse/scrapers/telegram_scraper.py:110:                "Get them from https://my.telegram.org"
kshiked/pulse/scrapers/reddit_scraper.py:17:    Reddit API requires registration at https://www.reddit.com/prefs/apps
kshiked/pulse/scrapers/reddit_scraper.py:46:    # Reddit API credentials (from https://www.reddit.com/prefs/apps)
kshiked/pulse/scrapers/facebook_scraper.py:153:            await self._page.goto("https://www.facebook.com/login")
kshiked/pulse/scrapers/facebook_scraper.py:312:            url = f"https://www.facebook.com/search/posts?q={query}"
kshiked/pulse/scrapers/x_scraper.py:16:        nitter_instances=["https://nitter.net"],
kshiked/pulse/scrapers/x_scraper.py:79:        "https://nitter.net",
kshiked/pulse/scrapers/x_scraper.py:80:        "https://nitter.privacydev.net",
kshiked/pulse/scrapers/x_scraper.py:81:        "https://nitter.poast.org",
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:45:    base_url: str = "https://www.jumia.co.ke"
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:45:    base_url: str = "https://www.kilimall.co.ke"
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:46:    base_url: str = "https://jiji.co.ke"

## 14 api_key/token/password literal
$ rg -n --glob 'kshiked/**/*.py' '\b(api[_-]?key|api_secret|token|password|bearer_token)\b\s*=\s*"[^"]+"' kshiked || echo NOT FOUND
kshiked/pulse/llm/gemini.py:7:    provider = GeminiProvider(api_key="your-api-key")
kshiked/pulse/llm/base.py:16:    provider = GeminiProvider(api_key="...")
kshiked/pulse/scrapers/instagram_scraper.py:9:        password="your_password",
kshiked/pulse/scrapers/x_scraper.py:13:            XAccount(username="account1", password="pass1", email="email1@example.com"),

## 15 env var access
$ rg -n --glob 'kshiked/**/*.py' 'os\.getenv\(' kshiked || echo NOT FOUND
kshiked/pulse/config.py:69:            api_key=os.getenv("X_API_KEY", ""),
kshiked/pulse/config.py:70:            api_secret=os.getenv("X_API_SECRET", ""),
kshiked/pulse/config.py:71:            access_token=os.getenv("X_ACCESS_TOKEN", ""),
kshiked/pulse/config.py:72:            access_secret=os.getenv("X_ACCESS_SECRET", ""),
kshiked/pulse/config.py:73:            bearer_token=os.getenv("X_BEARER_TOKEN", ""),
kshiked/pulse/config.py:92:            client_key=os.getenv("TIKTOK_CLIENT_KEY", ""),
kshiked/pulse/config.py:93:            client_secret=os.getenv("TIKTOK_CLIENT_SECRET", ""),
kshiked/pulse/config.py:94:            access_token=os.getenv("TIKTOK_ACCESS_TOKEN", ""),
kshiked/pulse/config.py:109:            access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN", ""),
kshiked/pulse/config.py:110:            app_id=os.getenv("INSTAGRAM_APP_ID", ""),
kshiked/pulse/config.py:111:            app_secret=os.getenv("INSTAGRAM_APP_SECRET", ""),
kshiked/pulse/config.py:145:            use_nlp=os.getenv("PULSE_USE_NLP", "true").lower() == "true",
kshiked/pulse/config.py:146:            min_intensity_threshold=float(os.getenv("PULSE_MIN_INTENSITY", "0.1")),
kshiked/pulse/config.py:147:            min_confidence_threshold=float(os.getenv("PULSE_MIN_CONFIDENCE", "0.3")),
kshiked/pulse/config.py:148:            shock_interval_seconds=float(os.getenv("PULSE_SHOCK_INTERVAL", "300")),
kshiked/pulse/config.py:149:            enable_probabilistic_shocks=os.getenv("PULSE_PROBABILISTIC", "true").lower() == "true",
kshiked/pulse/config.py:169:            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
kshiked/pulse/config.py:170:            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
kshiked/pulse/config.py:171:            user_agent=os.getenv("REDDIT_USER_AGENT", "KShieldPulse/1.0"),
kshiked/pulse/config.py:190:            api_id=int(os.getenv("TELEGRAM_API_ID", "0")),
kshiked/pulse/config.py:191:            api_hash=os.getenv("TELEGRAM_API_HASH", ""),
kshiked/pulse/config.py:192:            phone=os.getenv("TELEGRAM_PHONE", ""),
kshiked/pulse/config.py:210:            email=os.getenv("FACEBOOK_EMAIL", ""),
kshiked/pulse/config.py:211:            password=os.getenv("FACEBOOK_PASSWORD", ""),
kshiked/pulse/config.py:229:            api_key=os.getenv("GEMINI_API_KEY", ""),
kshiked/pulse/config.py:230:            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
kshiked/pulse/config.py:231:            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
kshiked/pulse/config.py:248:            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///pulse.db"),
kshiked/pulse/config.py:249:            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
kshiked/pulse/config.py:293:            social_scrape_interval_minutes=int(os.getenv("SCRAPE_INTERVAL_MINUTES", "30")),
kshiked/pulse/config.py:294:            ecommerce_scrape_interval_hours=int(os.getenv("ECOMMERCE_INTERVAL_HOURS", "6")),
kshiked/pulse/config.py:295:            max_posts_per_scrape=int(os.getenv("MAX_POSTS_PER_SCRAPE", "100")),
kshiked/pulse/config.py:296:            max_products_per_category=int(os.getenv("MAX_PRODUCTS_PER_CATEGORY", "50")),
kshiked/pulse/db/database.py:93:            url=os.getenv("DATABASE_URL"),
kshiked/pulse/db/database.py:94:            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
kshiked/pulse/db/database.py:95:            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "10")),
kshiked/pulse/db/database.py:96:            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
kshiked/pulse/ingestion/orchestrator.py:128:            database_url=os.getenv("DATABASE_URL"),
kshiked/pulse/ingestion/orchestrator.py:129:            x_bearer_token=os.getenv("X_BEARER_TOKEN", ""),
kshiked/pulse/ingestion/orchestrator.py:130:            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
kshiked/pulse/ingestion/orchestrator.py:131:            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
kshiked/pulse/ingestion/orchestrator.py:132:            telegram_api_id=int(os.getenv("TELEGRAM_API_ID", "0")),
kshiked/pulse/ingestion/orchestrator.py:133:            telegram_api_hash=os.getenv("TELEGRAM_API_HASH", ""),
kshiked/pulse/ingestion/orchestrator.py:134:            instagram_username=os.getenv("INSTAGRAM_USERNAME", ""),
kshiked/pulse/ingestion/orchestrator.py:135:            instagram_password=os.getenv("INSTAGRAM_PASSWORD", ""),
kshiked/pulse/ingestion/orchestrator.py:136:            facebook_email=os.getenv("FACEBOOK_EMAIL", ""),
kshiked/pulse/ingestion/orchestrator.py:137:            facebook_password=os.getenv("FACEBOOK_PASSWORD", ""),
kshiked/pulse/ingestion/orchestrator.py:138:            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),

## 16 bare except
$ rg -n --glob 'kshiked/**/*.py' '^\s*except\s*:\s*$' kshiked || echo NOT FOUND
kshiked/pulse/x_client.py:449:            except:
kshiked/pulse/ingestion/orchestrator.py:430:            except:
kshiked/pulse/ingestion/orchestrator.py:436:            except:
kshiked/pulse/network.py:255:        except:
kshiked/pulse/network.py:261:        except:
kshiked/pulse/scrapers/facebook_scraper.py:273:                except:
kshiked/pulse/scrapers/facebook_scraper.py:336:                except:
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:232:                except:
kshiked/pulse/scrapers/x_scraper.py:353:            except:
kshiked/pulse/scrapers/x_scraper.py:391:        except:

## 17 except Exception
$ rg -n --glob 'kshiked/**/*.py' '^\s*except\s+Exception\b' kshiked || echo NOT FOUND
kshiked/hub.py:94:        except Exception as e:
kshiked/sim/demo_economic_simulation.py:39:    except Exception as e:
kshiked/analysis/find_crash.py:35:    except Exception as e:
kshiked/sim/run_governance.py:44:    except Exception as e:
kshiked/sim/run_governance.py:79:        except Exception as e:
kshiked/tests/verify_tab2.py:71:    except Exception as e:
kshiked/tests/verify_features.py:56:    except Exception as e:
kshiked/tests/verify_trajectory.py:71:    except Exception as e:
kshiked/tests/verify_terrain.py:94:    except Exception as e:
kshiked/tests/torture_test.py:41:    except Exception as e:
kshiked/tests/torture_test.py:48:    except Exception as e:
kshiked/tests/torture_test.py:55:    except Exception as e:
kshiked/tests/torture_test.py:62:    except Exception as e:
kshiked/tests/torture_test.py:69:    except Exception as e:
kshiked/tests/torture_test.py:76:    except Exception as e:
kshiked/tests/prove_scarcity_v2.py:33:    except Exception:
kshiked/pulse/config.py:51:    except Exception as e:
kshiked/tests/sanity_check.py:70:            except Exception as e:
kshiked/pulse/bridge.py:390:            except Exception as e:
kshiked/pulse/bridge.py:425:            except Exception as e:
kshiked/pulse/create_merged_pdf.py:163:            except Exception as e:
kshiked/pulse/create_merged_pdf.py:187:            except Exception as e:
kshiked/pulse/create_merged_pdf.py:209:            except Exception as e:
kshiked/pulse/x_client.py:103:            except Exception as e:
kshiked/pulse/x_client.py:117:        except Exception as e:
kshiked/pulse/x_client.py:353:        except Exception as e:
kshiked/pulse/db/database.py:230:            except Exception:
kshiked/pulse/ingestion/scheduler.py:153:            except Exception as e:
kshiked/pulse/demo_ingestion.py:144:        except Exception as e:
kshiked/pulse/ingestion/orchestrator.py:296:                    except Exception as e:
kshiked/pulse/ingestion/orchestrator.py:301:            except Exception as e:
kshiked/pulse/ingestion/orchestrator.py:350:            except Exception as e:
kshiked/pulse/ingestion/orchestrator.py:396:                    except Exception as e:
kshiked/pulse/ingestion/orchestrator.py:407:                    except Exception as e:
kshiked/pulse/ingestion/pipeline.py:120:            except Exception as e:
kshiked/pulse/ingestion/pipeline.py:137:            except Exception as e:
kshiked/pulse/ingestion/pipeline.py:234:        except Exception as e:
kshiked/pulse/merge_all_complete.py:140:            except Exception as e:
kshiked/pulse/merge_all_complete.py:159:            except Exception as e:
kshiked/pulse/merge_all_complete.py:179:            except Exception as e:
kshiked/pulse/network.py:303:        except Exception as e:
kshiked/pulse/merge_docs_only.py:69:        except Exception as e:
kshiked/pulse/social.py:530:            except Exception as e:
kshiked/pulse/social.py:598:            except Exception as e:
kshiked/pulse/social.py:722:                except Exception as e:
kshiked/pulse/mapper.py:633:        except Exception as e:
kshiked/pulse/diagrams/merge_all_with_images.py:108:            except Exception as e:
kshiked/pulse/diagrams/merge_all_with_images.py:127:            except Exception as e:
kshiked/pulse/sensor.py:364:            except Exception as e:
kshiked/pulse/sensor.py:568:            except Exception as e:
kshiked/pulse/scrapers/reddit_scraper.py:189:        except Exception as e:
kshiked/pulse/scrapers/reddit_scraper.py:249:            except Exception as e:
kshiked/pulse/scrapers/reddit_scraper.py:308:        except Exception as e:
kshiked/pulse/scrapers/instagram_scraper.py:136:                except Exception as e:
kshiked/pulse/scrapers/instagram_scraper.py:161:                except Exception as e:
kshiked/pulse/scrapers/instagram_scraper.py:213:        except Exception as e:
kshiked/pulse/scrapers/instagram_scraper.py:253:        except Exception as e:
kshiked/pulse/scrapers/instagram_scraper.py:287:        except Exception as e:
kshiked/pulse/scrapers/instagram_scraper.py:335:        except Exception as e:
kshiked/pulse/scrapers/instagram_scraper.py:386:            except Exception as e:
kshiked/pulse/diagrams/merge_to_single_file.py:67:            except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:144:        except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:169:        except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:190:            except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:197:            except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:241:                    except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:255:        except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:294:        except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:341:        except Exception as e:
kshiked/pulse/scrapers/facebook_scraper.py:386:            except Exception as e:
kshiked/pulse/llm/fine_tuning.py:347:        except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:167:        except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:182:        except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:206:            except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:213:            except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:250:        except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:290:        except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:331:        except Exception as e:
kshiked/pulse/scrapers/x_scraper.py:373:        except Exception as e:
kshiked/pulse/llm/gemini.py:115:        except Exception as e:
kshiked/pulse/llm/gemini.py:175:            except Exception as e:
kshiked/pulse/scrapers/telegram_scraper.py:139:        except Exception as e:
kshiked/pulse/scrapers/telegram_scraper.py:192:                except Exception as e:
kshiked/pulse/scrapers/telegram_scraper.py:202:        except Exception as e:
kshiked/pulse/scrapers/telegram_scraper.py:254:            except Exception as e:
kshiked/pulse/scrapers/telegram_scraper.py:288:        except Exception as e:
kshiked/pulse/scrapers/telegram_scraper.py:352:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:168:                    except Exception as e:
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:175:            except Exception as e:
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:255:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:324:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:356:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/base.py:281:            except Exception as e:
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:165:                    except Exception as e:
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:172:            except Exception as e:
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:242:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:303:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:172:                    except Exception as e:
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:179:            except Exception as e:
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:250:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:331:        except Exception as e:
kshiked/pulse/scrapers/ecommerce/price_aggregator.py:248:        except Exception as e:
kshiked/pulse/diagrams/big_document.py:13:        except Exception as e:
kshiked/pulse/diagrams/universal_downloader.py:24:    except Exception as e:

## 18 except then pass
$ rg -n --glob 'kshiked/**/*.py' -U '^\s*except\s*:\s*\n\s*pass\s*$' kshiked || echo NOT FOUND
kshiked/pulse/x_client.py:449:            except:
kshiked/pulse/x_client.py:450:                pass
kshiked/pulse/x_client.py:451:        
kshiked/pulse/ingestion/orchestrator.py:430:            except:
kshiked/pulse/ingestion/orchestrator.py:431:                pass
kshiked/pulse/ingestion/orchestrator.py:432:        
kshiked/pulse/ingestion/orchestrator.py:436:            except:
kshiked/pulse/ingestion/orchestrator.py:437:                pass
kshiked/pulse/ingestion/orchestrator.py:438:        
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:232:                except:
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:233:                    pass
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:234:            

## 19 sys.path modification
$ rg -n --glob 'kshiked/**/*.py' 'sys\.path\.(append|insert)\(' kshiked || echo NOT FOUND
kshiked/sim/backtest_prediction.py:15:    sys.path.append(project_root)
kshiked/tests/prove_scarcity_v2.py:16:sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
kshiked/tests/verify_trajectory.py:10:sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
kshiked/tests/debug_growth.py:7:sys.path.append(os.getcwd())
kshiked/tests/verify_features.py:10:sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
kshiked/tests/verify_terrain.py:11:sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
kshiked/tests/benchmark_architecture.py:9:sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
kshiked/tests/verify_tab2.py:11:sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
kshiked/pulse/demo.py:15:sys.path.insert(0, '.')
kshiked/pulse/demo_ingestion.py:21:sys.path.insert(0, str(project_root))

## 20 TODO/FIXME
$ rg -n --glob 'kshiked/**/*.py' '\b(TODO|FIXME)\b' kshiked || echo NOT FOUND
kshiked/pulse/scrapers/facebook_scraper.py:356:        # TODO: Implement Meta Graph API when approved
kshiked/pulse/ingestion/scheduler.py:191:        # TODO: Integrate with GeminiProvider for batch classification
kshiked/pulse/scrapers/instagram_scraper.py:351:        # TODO: Implement Meta Graph API when credentials available

## 21 open write
$ rg -n --glob 'kshiked/**/*.py' 'open\([^\n]*,\s*["\']w' kshiked || echo NOT FOUND
/tmp/kshield_checklist_scan.sh: eval: line 9: unexpected EOF while looking for matching `''

## 22 os.remove
$ rg -n --glob 'kshiked/**/*.py' '\bos\.remove\(' kshiked || echo NOT FOUND
kshiked/tests/test_integration.py:280:                os.remove(output_path)
kshiked/pulse/demo_ingestion.py:117:        os.remove("test_demo.db")

## 23 shutil
$ rg -n --glob 'kshiked/**/*.py' '\bshutil\.' kshiked || echo NOT FOUND
kshiked/pulse/diagrams/merge_documents.py:53:                shutil.copy2(file_path, dest_file)

## 24 Path writes
$ rg -n --glob 'kshiked/**/*.py' '\bPath\([^\n]*\)\s*/\s*f' kshiked || echo NOT FOUND
kshiked/pulse/create_merged_pdf.py:154:                temp_path = Path(output_path.parent) / f'temp_{img_path.name}'

## 25 tempfile
$ rg -n --glob 'kshiked/**/*.py' '\btempfile\b' kshiked || echo NOT FOUND
kshiked/tests/test_integration.py:14:import tempfile
kshiked/tests/test_integration.py:267:        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:

## 26 while True
$ rg -n --glob 'kshiked/**/*.py' '^\s*while\s+True\s*:' kshiked || echo NOT FOUND
kshiked/pulse/scrapers/base.py:240:            while True:
kshiked/pulse/scrapers/base.py:556:        while True:

## 27 asyncio.sleep
$ rg -n --glob 'kshiked/**/*.py' 'await\s+asyncio\.sleep\(' kshiked || echo NOT FOUND
kshiked/pulse/x_client.py:317:            await asyncio.sleep(self.config.min_request_interval - elapsed)
kshiked/pulse/bridge.py:428:            await asyncio.sleep(interval_seconds)
kshiked/pulse/ingestion/scheduler.py:129:            await asyncio.sleep(1)
kshiked/pulse/ingestion/scheduler.py:166:                await asyncio.sleep(wait_time)
kshiked/sim/demo_economic_simulation.py:127:    await asyncio.sleep(1)
kshiked/pulse/social.py:167:                await asyncio.sleep(wait_time)
kshiked/pulse/social.py:725:            await asyncio.sleep(interval_seconds)
kshiked/pulse/ingestion/orchestrator.py:294:                        await asyncio.sleep(2)  # Rate limiting between terms
kshiked/pulse/ingestion/orchestrator.py:411:                await asyncio.sleep(60)
kshiked/pulse/simulation_connector.py:440:                await asyncio.sleep(interval_seconds)
kshiked/pulse/scrapers/instagram_scraper.py:381:                await asyncio.sleep(5)
kshiked/pulse/scrapers/facebook_scraper.py:154:            await asyncio.sleep(2)
kshiked/pulse/scrapers/facebook_scraper.py:161:            await asyncio.sleep(5)
kshiked/pulse/scrapers/facebook_scraper.py:239:                        await asyncio.sleep(2)  # Delay between pages
kshiked/pulse/scrapers/facebook_scraper.py:314:            await asyncio.sleep(3)
kshiked/pulse/scrapers/facebook_scraper.py:319:                await asyncio.sleep(2)
kshiked/pulse/scrapers/facebook_scraper.py:381:                await asyncio.sleep(10)  # Long delay between pages
kshiked/pulse/llm/gemini.py:132:                await asyncio.sleep(wait_time)
kshiked/pulse/llm/gemini.py:178:                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
kshiked/pulse/scrapers/ecommerce/jiji_scraper.py:177:                await asyncio.sleep(1)  # Rate limiting
kshiked/pulse/scrapers/ecommerce/kilimall_scraper.py:170:                await asyncio.sleep(1)
kshiked/pulse/scrapers/base.py:257:                await asyncio.sleep(wait_time)
kshiked/pulse/scrapers/base.py:332:            await asyncio.sleep(delay)
kshiked/pulse/scrapers/base.py:345:                await asyncio.sleep(delay)
kshiked/pulse/scrapers/base.py:565:            await asyncio.sleep(60)  # 1 minute between batches
kshiked/pulse/scrapers/ecommerce/base.py:279:                await asyncio.sleep(2)
kshiked/pulse/scrapers/ecommerce/jumia_scraper.py:173:                await asyncio.sleep(1)

## 28 numpy random
$ rg -n --glob 'kshiked/**/*.py' 'np\.random\.' kshiked || echo NOT FOUND
kshiked/pulse/bridge.py:238:        self._rng = np.random.default_rng()
kshiked/core/shocks.py:63:        dw = np.random.normal(0, np.sqrt(self.dt))
kshiked/core/shocks.py:83:        epsilon = np.random.normal(0, 1)
kshiked/tests/sanity_check_sync.py:20:    rng = np.random.default_rng(42)
kshiked/tests/sanity_check.py:39:    rng = np.random.default_rng(42)
kshiked/tests/benchmark_architecture.py:27:    x = np.random.randn(2)
kshiked/tests/benchmark_architecture.py:50:    X = np.random.randn(N_HYPOTHESES, N_FEATURES).astype(np.float32)
kshiked/tests/benchmark_architecture.py:51:    Y = np.random.randn(N_HYPOTHESES).astype(np.float32)
kshiked/pulse/primitives.py:290:    def should_trigger(self, aggregate_intensity: float, rng: np.random.Generator = None) -> bool:
kshiked/pulse/primitives.py:306:            rng = np.random.default_rng()

## 29 random module
$ rg -n --glob 'kshiked/**/*.py' '^\s*import\s+random\b' kshiked || echo NOT FOUND
kshiked/pulse/visualization.py:466:    import random
kshiked/core/shocks.py:8:import random
kshiked/pulse/scrapers/base.py:288:        import random

## 30 default_rng
$ rg -n --glob 'kshiked/**/*.py' 'default_rng\(' kshiked || echo NOT FOUND
kshiked/tests/sanity_check_sync.py:20:    rng = np.random.default_rng(42)
kshiked/tests/sanity_check.py:39:    rng = np.random.default_rng(42)
kshiked/pulse/bridge.py:238:        self._rng = np.random.default_rng()
kshiked/pulse/primitives.py:306:            rng = np.random.default_rng()

## 31 SQLAlchemy
$ rg -n --glob 'kshiked/**/*.py' '\bsqlalchemy\b' kshiked || echo NOT FOUND
kshiked/pulse/db/models.py:26:from sqlalchemy import (
kshiked/pulse/db/models.py:31:from sqlalchemy.orm import (
kshiked/pulse/db/database.py:34:from sqlalchemy import create_engine, text, event
kshiked/pulse/db/database.py:35:from sqlalchemy.ext.asyncio import (
kshiked/pulse/db/database.py:38:from sqlalchemy.orm import sessionmaker, Session
kshiked/pulse/db/database.py:39:from sqlalchemy.pool import StaticPool, QueuePool
kshiked/pulse/db/database.py:309:        from sqlalchemy import select
kshiked/pulse/db/database.py:342:        from sqlalchemy import select, func
kshiked/pulse/db/database.py:359:        from sqlalchemy import select
kshiked/pulse/db/database.py:360:        from sqlalchemy.orm import selectinload
kshiked/pulse/db/database.py:379:        from sqlalchemy import update
kshiked/pulse/db/database.py:401:        from sqlalchemy import select
kshiked/pulse/db/database.py:422:        from sqlalchemy import select, func
kshiked/pulse/llm/fine_tuning.py:140:        from sqlalchemy import select
kshiked/pulse/scrapers/ecommerce/price_aggregator.py:159:        from sqlalchemy import select, func
kshiked/pulse/scrapers/ecommerce/price_aggregator.py:203:        from sqlalchemy import select, func

## 32 aiosqlite
$ rg -n --glob 'kshiked/**/*.py' '\baiosqlite\b' kshiked || echo NOT FOUND
kshiked/pulse/config.py:248:            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///pulse.db"),
kshiked/pulse/db/database.py:13:    db = Database("sqlite+aiosqlite:///pulse.db")
kshiked/pulse/db/database.py:80:            url = f"sqlite+aiosqlite:///{db_path}"
kshiked/pulse/db/database.py:112:        if "aiosqlite" in self.url:
kshiked/pulse/db/database.py:113:            return self.url.replace("sqlite+aiosqlite", "sqlite")
kshiked/pulse/demo_ingestion.py:93:    config = DatabaseConfig(url="sqlite+aiosqlite:///test_demo.db")

## 33 sqlite URLs
$ rg -n --glob 'kshiked/**/*.py' 'sqlite\+aiosqlite:///' kshiked || echo NOT FOUND
kshiked/pulse/demo_ingestion.py:93:    config = DatabaseConfig(url="sqlite+aiosqlite:///test_demo.db")
kshiked/pulse/db/database.py:13:    db = Database("sqlite+aiosqlite:///pulse.db")
kshiked/pulse/db/database.py:80:            url = f"sqlite+aiosqlite:///{db_path}"
kshiked/pulse/config.py:248:            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///pulse.db"),

## 34 JSON parsing
$ rg -n --glob 'kshiked/**/*.py' 'json\.loads\(' kshiked || echo NOT FOUND
kshiked/pulse/llm/gemini.py:195:            return json.loads(text)
kshiked/pulse/llm/gemini.py:201:                    return json.loads(match.group(1))

## 35 regex JSON extraction
$ rg -n --glob 'kshiked/**/*.py' 'Failed to parse JSON|parse_json' kshiked || echo NOT FOUND
kshiked/pulse/llm/gemini.py:184:    def _parse_json(self, text: str) -> Dict:
kshiked/pulse/llm/gemini.py:205:            logger.warning(f"Failed to parse JSON from response: {text[:200]}...")
kshiked/pulse/llm/gemini.py:230:        data = self._parse_json(response_text)
kshiked/pulse/llm/gemini.py:293:        data = self._parse_json(response_text)
kshiked/pulse/llm/gemini.py:333:        data = self._parse_json(response_text)
kshiked/pulse/llm/gemini.py:389:        data = self._parse_json(response_text)

## 36 LLM prompt templates
$ rg -n --glob 'kshiked/**/*.py' 'THREAT_CLASSIFIER_SYSTEM|ROLE_CLASSIFIER_SYSTEM|NARRATIVE_ANALYZER_SYSTEM' kshiked || echo NOT FOUND
kshiked/pulse/llm/prompts.py:22:THREAT_CLASSIFIER_SYSTEM = """You are an expert analyst for the KShield national threat detection system in Kenya. Your role is to classify social media posts according to their potential threat to national stability.
kshiked/pulse/llm/prompts.py:82:ROLE_CLASSIFIER_SYSTEM = """You are analyzing social media accounts to identify their role in potential threat networks. Based on posting patterns, identify the actor type.
kshiked/pulse/llm/prompts.py:133:NARRATIVE_ANALYZER_SYSTEM = """You are analyzing collections of social media posts to identify emerging narrative patterns. Look for:
kshiked/pulse/llm/gemini.py:36:    THREAT_CLASSIFIER_SYSTEM, ROLE_CLASSIFIER_SYSTEM,
kshiked/pulse/llm/gemini.py:37:    NARRATIVE_ANALYZER_SYSTEM, format_threat_prompt,
kshiked/pulse/llm/gemini.py:226:            system_prompt=THREAT_CLASSIFIER_SYSTEM,
kshiked/pulse/llm/gemini.py:290:            system_prompt=ROLE_CLASSIFIER_SYSTEM,
kshiked/pulse/llm/gemini.py:330:            system_prompt=NARRATIVE_ANALYZER_SYSTEM,
kshiked/pulse/llm/gemini.py:385:            system_prompt=THREAT_CLASSIFIER_SYSTEM,

## 37 Gemini provider
$ rg -n --glob 'kshiked/**/*.py' 'GeminiProvider|google-generativeai|genai\.' kshiked || echo NOT FOUND
kshiked/tests/test_ingestion.py:285:            LLMProvider, ThreatClassification, GeminiProvider, ThreatTier
kshiked/pulse/ingestion/scheduler.py:191:        # TODO: Integrate with GeminiProvider for batch classification
kshiked/pulse/ingestion/pipeline.py:15:        llm_provider=GeminiProvider(...),
kshiked/pulse/llm/gemini.py:7:    provider = GeminiProvider(api_key="your-api-key")
kshiked/pulse/llm/gemini.py:74:class GeminiProvider(LLMProvider):
kshiked/pulse/llm/gemini.py:78:    Uses google-generativeai SDK for API calls.
kshiked/pulse/llm/gemini.py:96:            genai.configure(api_key=self.config.api_key)
kshiked/pulse/llm/gemini.py:99:            self._model = genai.GenerativeModel(
kshiked/pulse/llm/gemini.py:113:            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
kshiked/pulse/llm/gemini.py:430:) -> GeminiProvider:
kshiked/pulse/llm/gemini.py:440:        Configured GeminiProvider instance.
kshiked/pulse/llm/gemini.py:448:    return GeminiProvider(config)
kshiked/pulse/llm/__init__.py:17:from .gemini import GeminiProvider, create_gemini_provider
kshiked/pulse/llm/__init__.py:26:    "GeminiProvider",
kshiked/pulse/llm/base.py:10:- GeminiProvider (default, using your API key)
kshiked/pulse/llm/base.py:16:    provider = GeminiProvider(api_key="...")
kshiked/pulse/llm/base.py:184:    - GeminiProvider (default)
kshiked/pulse/llm/fine_tuning.py:324:        Requires google-generativeai >= 0.3.0 with tuning support.
kshiked/pulse/llm/fine_tuning.py:333:            training_dataset = genai.upload_file(training_file)
kshiked/pulse/llm/fine_tuning.py:345:            logger.error("google-generativeai not installed or outdated")

## 38 LLM batch classify
$ rg -n --glob 'kshiked/**/*.py' 'batch_classify\(' kshiked || echo NOT FOUND
kshiked/pulse/ingestion/pipeline.py:109:                    classifications = await self.llm.batch_classify(texts)
kshiked/pulse/llm/gemini.py:355:    async def batch_classify(
kshiked/pulse/llm/base.py:245:    async def batch_classify(

## 39 model outputs used as control signals
$ rg -n --glob 'kshiked/**/*.py' 'classify_threat\(|tier\b|ThreatTier' kshiked || echo NOT FOUND
kshiked/tests/test_integration.py:190:        tracker.record_activity("nairobi", is_threat=True, threat_tier=3)
kshiked/tests/test_integration.py:205:            tracker.record_activity("nairobi", is_threat=True, threat_tier=2)
kshiked/tests/test_integration.py:207:            tracker.record_activity("mombasa", is_threat=True, threat_tier=3)
kshiked/pulse/x_client.py:5:- Strict rate limiting for Free tier (100 posts/month)
kshiked/pulse/x_client.py:263:    Uses the Free tier with strict 100 posts/month limit.
kshiked/pulse/x_client.py:305:        # For Free tier with just API key/secret, we're limited
kshiked/pulse/x_client.py:307:        logger.info("X client initialized (Free tier - 100 posts/month)")
kshiked/pulse/x_client.py:366:        Note: Free tier has very limited search access.
kshiked/pulse/x_client.py:384:            "max_results": str(min(max_results, 10)),  # Free tier max is 10
kshiked/pulse/x_client.py:430:        Streaming not available on Free tier.
kshiked/pulse/x_client.py:434:        logger.warning("Streaming not available on X Free tier. Use search() instead.")
kshiked/tests/test_ingestion.py:68:            tier="tier_3",
kshiked/tests/test_ingestion.py:76:        assert analysis.tier == "tier_3"
kshiked/tests/test_ingestion.py:172:        """Test ThreatTier enum."""
kshiked/tests/test_ingestion.py:173:        from kshiked.pulse.llm import ThreatTier
kshiked/tests/test_ingestion.py:175:        assert ThreatTier.TIER_1.value == "tier_1"
kshiked/tests/test_ingestion.py:176:        assert ThreatTier.TIER_1.severity == 1
kshiked/tests/test_ingestion.py:177:        assert ThreatTier.TIER_5.severity == 5
kshiked/tests/test_ingestion.py:178:        assert ThreatTier.TIER_0.severity == 0
kshiked/tests/test_ingestion.py:189:        from kshiked.pulse.llm.base import ThreatClassification, ThreatTier
kshiked/tests/test_ingestion.py:192:            tier=ThreatTier.TIER_3,
kshiked/tests/test_ingestion.py:198:        assert classification.tier == ThreatTier.TIER_3
kshiked/tests/test_ingestion.py:204:            tier=ThreatTier.TIER_1,
kshiked/tests/test_ingestion.py:285:            LLMProvider, ThreatClassification, GeminiProvider, ThreatTier
kshiked/pulse/unified_dashboard.py:770:    location_tracker.record_activity("nairobi", is_threat=True, threat_tier=2)
kshiked/pulse/unified_dashboard.py:771:    location_tracker.record_activity("mombasa", is_threat=True, threat_tier=3)
kshiked/pulse/ingestion/pipeline.py:38:from ..llm.base import LLMProvider, ThreatClassification, ThreatTier
kshiked/pulse/ingestion/pipeline.py:112:                        cls = await self.llm.classify_threat(
kshiked/pulse/ingestion/pipeline.py:156:            "tier": None,
kshiked/pulse/ingestion/pipeline.py:162:            result["tier"] = classification.tier.value
kshiked/pulse/ingestion/pipeline.py:174:            "threat_tier": classification.tier.value if classification else None,
kshiked/pulse/ingestion/pipeline.py:211:                    tier=classification.tier.value,
kshiked/pulse/db/models.py:50:class ThreatTier(str, Enum):
kshiked/pulse/db/models.py:288:    Stores threat tier classification, role identification,
kshiked/pulse/db/models.py:308:    threat_tier: Mapped[str] = mapped_column(SQLEnum(ThreatTier), nullable=False)
kshiked/pulse/db/models.py:340:        Index('ix_llm_threat_tier', 'threat_tier'),
kshiked/pulse/demo_ingestion.py:140:            result = await provider.classify_threat(text)
kshiked/pulse/demo_ingestion.py:142:            print(f"     Tier: {result.tier.value}, Confidence: {result.confidence:.2f}")
kshiked/pulse/network.py:131:        # IDEOLOGUE: High original content, threat-tier posts
kshiked/pulse/network.py:133:        high_tier = self.threat_tier_counts.get("tier_1", 0) + self.threat_tier_counts.get("tier_2", 0)
kshiked/pulse/network.py:134:        if high_tier > 3:
kshiked/pulse/network.py:429:    avg_threat_tier: float = 5.0  # Lower = more severe
kshiked/pulse/network.py:443:        tier_factor = (5 - self.avg_threat_tier) / 4  # 0-1
kshiked/pulse/network.py:474:        threat_tier: Optional[int] = None,
kshiked/pulse/network.py:491:        if threat_tier is not None:
kshiked/pulse/network.py:494:            loc.avg_threat_tier = ((n - 1) * loc.avg_threat_tier + threat_tier) / n
kshiked/pulse/llm/gemini.py:9:    result = await provider.classify_threat(
kshiked/pulse/llm/gemini.py:14:    print(result.tier)  # ThreatTier.TIER_3
kshiked/pulse/llm/gemini.py:33:    NarrativeAnalysis, ThreatTier, RoleType, NarrativeMaturity,
kshiked/pulse/llm/gemini.py:208:    async def classify_threat(
kshiked/pulse/llm/gemini.py:213:        """Classify a post's threat tier using Gemini."""
kshiked/pulse/llm/gemini.py:232:        # Map tier string to enum
kshiked/pulse/llm/gemini.py:233:        tier_str = data.get("tier", "TIER_5").upper().replace("-", "_")
kshiked/pulse/llm/gemini.py:235:            tier = ThreatTier(tier_str.lower())
kshiked/pulse/llm/gemini.py:237:            tier = ThreatTier.TIER_5
kshiked/pulse/llm/gemini.py:240:            tier=tier,
kshiked/pulse/llm/gemini.py:369:                result = await self.classify_threat(text, ctx)
kshiked/pulse/llm/gemini.py:394:                tier_str = item.get("tier", "TIER_5").upper().replace("-", "_")
kshiked/pulse/llm/gemini.py:396:                    tier = ThreatTier(tier_str.lower())
kshiked/pulse/llm/gemini.py:398:                    tier = ThreatTier.TIER_5
kshiked/pulse/llm/gemini.py:401:                    tier=tier,
kshiked/pulse/llm/gemini.py:413:                tier=ThreatTier.TIER_5,
kshiked/pulse/llm/base.py:18:    result = await provider.classify_threat(
kshiked/pulse/llm/base.py:23:    print(result.tier)  # ThreatTier.TIER_3
kshiked/pulse/llm/base.py:39:class ThreatTier(str, Enum):
kshiked/pulse/llm/base.py:60:        if self == ThreatTier.TIER_0:
kshiked/pulse/llm/base.py:92:    Result of threat tier classification.
kshiked/pulse/llm/base.py:94:    Contains the tier, confidence, and supporting details.
kshiked/pulse/llm/base.py:97:    tier: ThreatTier
kshiked/pulse/llm/base.py:120:        return self.tier not in [ThreatTier.TIER_0, ThreatTier.TIER_5]
kshiked/pulse/llm/base.py:125:        return self.tier in [ThreatTier.TIER_1, ThreatTier.TIER_2]
kshiked/pulse/llm/base.py:191:    async def classify_threat(
kshiked/pulse/llm/base.py:197:        Classify a post's threat tier.
kshiked/pulse/llm/base.py:204:            ThreatClassification with tier and confidence.
kshiked/pulse/llm/fine_tuning.py:43:    tier: str
kshiked/pulse/llm/fine_tuning.py:87:Respond with JSON: {"tier": "TIER_X", "confidence": 0.0-1.0, "reasoning": "..."}"""
kshiked/pulse/llm/fine_tuning.py:158:                        "tier": analysis.tier.value,
kshiked/pulse/llm/fine_tuning.py:162:                    tier=analysis.tier.value,
kshiked/pulse/llm/fine_tuning.py:174:        tier: str,
kshiked/pulse/llm/fine_tuning.py:187:                "tier": tier,
kshiked/pulse/llm/fine_tuning.py:191:            tier=tier,
kshiked/pulse/llm/fine_tuning.py:199:        count_per_tier: int = 50,
kshiked/pulse/llm/fine_tuning.py:204:        Creates examples for each tier based on patterns.
kshiked/pulse/llm/fine_tuning.py:243:        for pattern in tier1_patterns[:count_per_tier]:
kshiked/pulse/llm/fine_tuning.py:248:        for pattern in tier3_patterns[:count_per_tier]:
kshiked/pulse/llm/fine_tuning.py:253:        for pattern in tier5_patterns[:count_per_tier]:
kshiked/pulse/llm/fine_tuning.py:257:        for pattern in tier0_patterns[:count_per_tier]:
kshiked/pulse/llm/fine_tuning.py:268:        min_examples_per_tier: int = 10,
kshiked/pulse/llm/fine_tuning.py:276:            min_examples_per_tier: Minimum examples required per tier.
kshiked/pulse/llm/fine_tuning.py:284:            tier_counts[ex.tier] = tier_counts.get(ex.tier, 0) + 1
kshiked/pulse/llm/fine_tuning.py:286:        logger.info(f"Examples per tier: {tier_counts}")
kshiked/pulse/llm/fine_tuning.py:306:            tier_counts[ex.tier] = tier_counts.get(ex.tier, 0) + 1
kshiked/pulse/llm/fine_tuning.py:311:            "by_tier": tier_counts,
kshiked/pulse/llm/__init__.py:7:- Threat tier classification
kshiked/pulse/llm/__init__.py:16:from .base import LLMProvider, ThreatClassification, RoleClassification, ThreatTier, RoleType
kshiked/pulse/llm/__init__.py:24:    "ThreatTier",
kshiked/pulse/llm/prompts.py:5:- Threat tier classification (Tier 0-5)
kshiked/pulse/llm/prompts.py:178:  "tier": "TIER_1" | "TIER_2" | "TIER_3" | "TIER_4" | "TIER_5" | "TIER_0",
kshiked/pulse/llm/prompts.py:249:    "tier": "TIER_X",

## 40 prompt injection guardrails
$ rg -n --glob 'kshiked/**/*.py' 'prompt injection|sanitize|escape' kshiked || echo NOT FOUND
kshiked/sim/run_economic_simulation.py:65:    last_state = engine.core._sanitize_row(stream_data[-1])
kshiked/pulse/nlp.py:335:            for match in re.finditer(re.escape(entity_text), text_lower):

## 41 graph centrality
$ rg -n --glob 'kshiked/**/*.py' 'betweenness_centrality|pagerank' kshiked || echo NOT FOUND
kshiked/pulse/network.py:252:                betweenness = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
kshiked/pulse/network.py:254:                betweenness = nx.betweenness_centrality(self.graph)
kshiked/pulse/network.py:260:            pagerank = nx.pagerank(self.graph, weight="weight")
kshiked/pulse/network.py:262:            pagerank = {n: 0.0 for n in self.graph.nodes()}
kshiked/pulse/network.py:269:                "pagerank": pagerank.get(node, 0.0),

## 42 networkx import
$ rg -n --glob 'kshiked/**/*.py' '\bimport\s+networkx\b|\bnetworkx\b' kshiked || echo NOT FOUND
kshiked/pulse/network.py:25:    import networkx as nx
kshiked/pulse/network.py:199:            raise ImportError("networkx required: pip install networkx")

## 43 streamlit
$ rg -n --glob 'kshiked/**/*.py' '\bstreamlit\b' kshiked || echo NOT FOUND
kshiked/pulse/__init__.py:103:# Dashboard is optional (requires streamlit)
kshiked/pulse/dashboard.py:12:    import streamlit as st
kshiked/pulse/dashboard.py:27:    import streamlit as st
kshiked/pulse/dashboard.py:426:        print("Streamlit not installed. Run: pip install streamlit")
kshiked/pulse/unified_dashboard.py:25:    import streamlit as st
kshiked/pulse/unified_dashboard.py:184:        logger.error("Streamlit not installed: pip install streamlit")
kshiked/pulse/unified_dashboard.py:743:        streamlit run unified_dashboard.py
kshiked/pulse/unified_dashboard.py:746:        print("Error: Streamlit required. Install with: pip install streamlit")
kshiked/pulse/unified_dashboard.py:782:# Entry point for streamlit run

## 44 plotly CDN
$ rg -n --glob 'kshiked/**/*.py' 'cdn\.plot\.ly' kshiked || echo NOT FOUND
kshiked/pulse/visualization.py:563:        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

## 45 base64 huge content
$ python3 - <<'PY'
from pathlib import Path
p=Path('kshiked/pulse/diagrams/universal_downloader.py')
print(p, 'bytes', p.stat().st_size)
PY
kshiked/pulse/diagrams/universal_downloader.py bytes 12866963

## 46 UTF-16 python sources
$ file kshiked/**/*.py | rg 'UTF-16' || echo NOT FOUND
kshiked/analysis/__init__.py:              Python script, Unicode text, UTF-16, little-endian text executable, with CRLF line terminators
kshiked/core/__init__.py:                  Python script, Unicode text, UTF-16, little-endian text executable, with CRLF line terminators
kshiked/sim/__init__.py:                   Unicode text, UTF-16, little-endian text, with CRLF line terminators
kshiked/tests/__init__.py:                 Python script, Unicode text, UTF-16, little-endian text executable, with CRLF line terminators

## 47 import errors (kshiked)
$ python3 -c 'import kshiked' 2>&1 | head -n 50 || true
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/kshiked/__init__.py", line 8, in <module>
    from .core.governance import (
SyntaxError: source code string cannot contain null bytes

## 48 import errors (kshiked.pulse)
$ python3 -c 'import importlib; importlib.import_module("kshiked.pulse")' 2>&1 | head -n 50 || true
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/kshiked/__init__.py", line 8, in <module>
    from .core.governance import (
SyntaxError: source code string cannot contain null bytes

## 49 import errors (kshiked.core)
$ python3 -c 'import importlib; importlib.import_module("kshiked.core")' 2>&1 | head -n 50 || true
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/kshiked/__init__.py", line 8, in <module>
    from .core.governance import (
SyntaxError: source code string cannot contain null bytes

## 50 import errors (kshiked.tests)
$ python3 -c 'import importlib; importlib.import_module("kshiked.tests")' 2>&1 | head -n 50 || true
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/mnt/c/Users/omegam/OneDrive - Innova Limited/scace4/kshiked/__init__.py", line 8, in <module>
    from .core.governance import (
SyntaxError: source code string cannot contain null bytes
```

# Ingestion package
"""
Ingestion orchestration for KShield Pulse.

Provides:
- IngestionOrchestrator: Main coordinator for all scrapers
- IngestionScheduler: APScheduler-based job scheduling
- PipelineIntegration: Bridge to PulseSensor

Architecture:
┌──────────────────────────────────────────────────────┐
│                 HYBRID INGESTION                      │
├────────────────┬────────────────┬────────────────────┤
│  Always-On     │    Hourly      │       Daily        │
│  Streaming     │  Aggregation   │  Deep Analysis     │
├────────────────┼────────────────┼────────────────────┤
│ - Telegram     │ - Price index  │ - Full network     │
│ - X search     │ - Signal agg   │ - LLM batch        │
│ - Reddit hot   │ - Baseline     │ - Reports          │
└────────────────┴────────────────┴────────────────────┘
"""

from .orchestrator import IngestionOrchestrator, IngestionConfig
from .scheduler import IngestionScheduler
from .pipeline import PipelineIntegration, create_pipeline, run_full_pipeline

__all__ = [
    "IngestionOrchestrator",
    "IngestionConfig",
    "IngestionScheduler",
    "PipelineIntegration",
    "create_pipeline",
    "run_full_pipeline",
]


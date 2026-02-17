# Database package for KShield Pulse
"""
Database layer for KShield Pulse data ingestion pipeline.

Provides:
- SQLAlchemy 2.0 async models
- Connection pooling and transaction management
- Support for SQLite (dev) and PostgreSQL (production)
"""

from .models import (
    Base,
    SocialPost,
    ProcessedSignal,
    LLMAnalysis,
    Author,
    NetworkEdge,
    PriceSnapshot,
    ProductCategory,
)
from .database import Database, DatabaseConfig

__all__ = [
    "Base",
    "SocialPost",
    "ProcessedSignal",
    "LLMAnalysis",
    "Author",
    "NetworkEdge",
    "PriceSnapshot",
    "ProductCategory",
    "Database",
    "DatabaseConfig",
]


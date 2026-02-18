# Database package for KShield Pulse
"""
Database layer for KShield Pulse data ingestion pipeline.

Provides:
- SQLAlchemy 2.0 async models
- Connection pooling and transaction management
- Support for SQLite (dev) and PostgreSQL (production)
"""

try:
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
except Exception:  # pragma: no cover - optional SQLAlchemy runtime
    Base = None
    SocialPost = None
    ProcessedSignal = None
    LLMAnalysis = None
    Author = None
    NetworkEdge = None
    PriceSnapshot = None
    ProductCategory = None

try:
    from .database import Database, DatabaseConfig
except Exception:  # pragma: no cover - optional SQLAlchemy runtime
    Database = None
    DatabaseConfig = None

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


"""
Database Connection Layer for KShield Pulse

Provides:
- Async/sync database connections with SQLAlchemy 2.0
- Connection pooling for production use
- SQLite for development, PostgreSQL ready for production
- Batch insert optimization for high-volume ingestion
- Transaction management

Usage:
    # Async usage
    db = Database("sqlite+aiosqlite:///pulse.db")
    await db.connect()
    
    async with db.session() as session:
        post = SocialPost(...)
        session.add(post)
        await session.commit()
    
    # Or use the batch insert for high volume
    await db.batch_insert(posts)  # 1000+ rows efficiently
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, List, Any, TypeVar, Type, Sequence
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, AsyncEngine, async_sessionmaker
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool

from .models import Base, SocialPost, Author, PriceSnapshot

logger = logging.getLogger("kshield.pulse.db")

T = TypeVar('T', bound=Base)


# =============================================================================
# Configuration
# =============================================================================

class DatabaseConfig:
    """Database configuration."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
    ):
        """
        Initialize database configuration.
        
        Args:
            url: Database URL. If None, uses SQLite in pulse directory.
            echo: Echo SQL statements (for debugging).
            pool_size: Connection pool size (PostgreSQL only).
            max_overflow: Max overflow connections (PostgreSQL only).
            pool_timeout: Pool timeout in seconds.
            pool_recycle: Connection recycle time in seconds.
        """
        if url is None:
            # Default to SQLite in the pulse directory
            pulse_dir = Path(__file__).parent.parent
            db_path = pulse_dir / "data" / "pulse.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite+aiosqlite:///{db_path}"
        
        self.url = url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            url=os.getenv("DATABASE_URL"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
        )
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return "sqlite" in self.url.lower()
    
    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return "postgresql" in self.url.lower() or "postgres" in self.url.lower()
    
    @property
    def sync_url(self) -> str:
        """Get synchronous database URL."""
        if "aiosqlite" in self.url:
            return self.url.replace("sqlite+aiosqlite", "sqlite")
        elif "asyncpg" in self.url:
            return self.url.replace("postgresql+asyncpg", "postgresql")
        return self.url


# =============================================================================
# Async Database Class
# =============================================================================

class Database:
    """
    Async database connection manager.
    
    Handles:
    - Connection pooling
    - Session management
    - Batch inserts
    - Schema creation
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration. Uses defaults if None.
        """
        self.config = config or DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._sync_engine = None
        self._connected = False
    
    async def connect(self) -> None:
        """
        Establish database connection and create tables.
        
        Creates the async engine, session factory, and ensures
        all tables exist.
        """
        if self._connected:
            return
        
        logger.info(f"Connecting to database: {self._safe_url()}")
        
        # Create async engine with appropriate settings
        engine_kwargs = {
            "echo": self.config.echo,
        }
        
        if self.config.is_sqlite:
            # SQLite-specific settings
            engine_kwargs["poolclass"] = StaticPool
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            # PostgreSQL-specific settings
            engine_kwargs["pool_size"] = self.config.pool_size
            engine_kwargs["max_overflow"] = self.config.max_overflow
            engine_kwargs["pool_timeout"] = self.config.pool_timeout
            engine_kwargs["pool_recycle"] = self.config.pool_recycle
            engine_kwargs["poolclass"] = QueuePool
        
        self._engine = create_async_engine(self.config.url, **engine_kwargs)
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Create all tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._connected = True
        logger.info("Database connection established, tables created")
    
    async def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._connected = False
            logger.info("Database connection closed")
    
    def _safe_url(self) -> str:
        """Get URL safe for logging (no passwords)."""
        url = self.config.url
        if "@" in url:
            # Hide password
            parts = url.split("@")
            pre_at = parts[0]
            if ":" in pre_at:
                # Hide password portion
                scheme_user = pre_at.rsplit(":", 1)[0]
                return f"{scheme_user}:***@{parts[1]}"
        return url
    
    @asynccontextmanager
    async def session(self):
        """
        Get an async session context manager.
        
        Usage:
            async with db.session() as session:
                session.add(post)
                await session.commit()
        """
        if not self._session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
    async def add(self, obj: Base) -> Base:
        """Add a single object to the database."""
        async with self.session() as session:
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            return obj
    
    async def add_all(self, objects: List[Base]) -> List[Base]:
        """Add multiple objects to the database."""
        async with self.session() as session:
            session.add_all(objects)
            await session.commit()
            for obj in objects:
                await session.refresh(obj)
            return objects
    
    async def get(
        self, 
        model: Type[T], 
        id: str,
    ) -> Optional[T]:
        """Get an object by ID."""
        async with self.session() as session:
            return await session.get(model, id)
    
    async def batch_insert(
        self,
        objects: Sequence[Base],
        batch_size: int = 1000,
    ) -> int:
        """
        Efficiently insert large batches of objects.
        
        Uses raw INSERT for performance with batching
        to avoid memory issues.
        
        Args:
            objects: Objects to insert.
            batch_size: Number of objects per batch.
            
        Returns:
            Number of objects inserted.
        """
        if not objects:
            return 0
        
        total = 0
        async with self.session() as session:
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                session.add_all(batch)
                await session.flush()
                total += len(batch)
            
            await session.commit()
        
        logger.debug(f"Batch inserted {total} objects")
        return total
    
    # =========================================================================
    # Social Media Specific Operations
    # =========================================================================
    
    async def upsert_author(self, author: Author) -> Author:
        """
        Insert or update an author.
        
        If author with same platform+platform_id exists,
        updates the existing record.
        """
        from sqlalchemy import select
        
        async with self.session() as session:
            # Check for existing
            stmt = select(Author).where(
                Author.platform == author.platform,
                Author.platform_id == author.platform_id,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing
                existing.username = author.username or existing.username
                existing.display_name = author.display_name or existing.display_name
                existing.follower_count = author.follower_count or existing.follower_count
                existing.following_count = author.following_count or existing.following_count
                existing.bio = author.bio or existing.bio
                existing.location = author.location or existing.location
                existing.verified = author.verified
                existing.last_seen = datetime.utcnow()
                existing.post_count = existing.post_count + 1
                await session.commit()
                return existing
            else:
                # Insert new
                session.add(author)
                await session.commit()
                await session.refresh(author)
                return author
    
    async def post_exists(self, platform: str, platform_id: str) -> bool:
        """Check if a post already exists (for deduplication)."""
        from sqlalchemy import select, func
        
        async with self.session() as session:
            stmt = select(func.count()).select_from(SocialPost).where(
                SocialPost.platform == platform,
                SocialPost.platform_id == platform_id,
            )
            result = await session.execute(stmt)
            count = result.scalar()
            return count > 0

    async def get_post(self, platform: str, platform_id: str) -> Optional[SocialPost]:
        """Fetch a single post by (platform, platform_id)."""
        from sqlalchemy import select

        async with self.session() as session:
            stmt = select(SocialPost).where(
                SocialPost.platform == platform,
                SocialPost.platform_id == platform_id,
            ).limit(1)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_unprocessed_posts(
        self, 
        limit: int = 100,
        platform: Optional[str] = None,
    ) -> List[SocialPost]:
        """Get posts that haven't been processed by the Pulse engine."""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        async with self.session() as session:
            stmt = select(SocialPost).where(
                SocialPost.processed == False  # noqa: E712
            ).options(
                selectinload(SocialPost.author)
            ).order_by(
                SocialPost.scraped_at.desc()
            ).limit(limit)
            
            if platform:
                stmt = stmt.where(SocialPost.platform == platform)
            
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def mark_posts_processed(self, post_ids: List[str]) -> int:
        """Mark posts as processed."""
        from sqlalchemy import update
        
        async with self.session() as session:
            stmt = update(SocialPost).where(
                SocialPost.id.in_(post_ids)
            ).values(processed=True)
            
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
    
    # =========================================================================
    # Price Data Operations
    # =========================================================================
    
    async def get_price_history(
        self,
        product_id: str,
        source: str,
        days: int = 30,
    ) -> List[PriceSnapshot]:
        """Get price history for a product."""
        from sqlalchemy import select
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        async with self.session() as session:
            stmt = select(PriceSnapshot).where(
                PriceSnapshot.product_id == product_id,
                PriceSnapshot.source == source,
                PriceSnapshot.scraped_at >= cutoff,
            ).order_by(PriceSnapshot.scraped_at.desc())
            
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    async def get_stats(self) -> dict:
        """Get database statistics."""
        from sqlalchemy import select, func
        
        async with self.session() as session:
            post_count = await session.execute(
                select(func.count()).select_from(SocialPost)
            )
            author_count = await session.execute(
                select(func.count()).select_from(Author)
            )
            price_count = await session.execute(
                select(func.count()).select_from(PriceSnapshot)
            )
            
            return {
                "posts": post_count.scalar() or 0,
                "authors": author_count.scalar() or 0,
                "price_snapshots": price_count.scalar() or 0,
            }


# =============================================================================
# Global Database Instance
# =============================================================================

_database: Optional[Database] = None


async def get_database(config: Optional[DatabaseConfig] = None) -> Database:
    """
    Get the global database instance.
    
    Creates and connects if not already connected.
    """
    global _database
    
    if _database is None:
        _database = Database(config or DatabaseConfig.from_env())
        await _database.connect()
    
    return _database


async def close_database() -> None:
    """Close the global database instance."""
    global _database
    
    if _database:
        await _database.disconnect()
        _database = None

"""
Base Scraper Classes for KShield Pulse

Provides:
- Abstract base class with dual scrape/API pattern
- Common result types and error handling
- Rate limiting and retry logic
- Conversion to database models

Design Pattern:
    Every scraper implements both scrape() and scrape_via_api().
    The scrape() method works immediately via web scraping.
    The scrape_via_api() method is ready to use when API credentials
    are available - just add credentials and call use_api().

Usage:
    scraper = XScraper(config)
    
    # Scraping mode (default)
    posts = await scraper.scrape("Kenya politics", limit=100)
    
    # API mode (when credentials available)
    if scraper.has_api_credentials():
        posts = await scraper.scrape_via_api("Kenya politics", limit=100)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncIterator
from enum import Enum

logger = logging.getLogger("kshield.pulse.scrapers")


# =============================================================================
# Exceptions
# =============================================================================

class ScraperError(Exception):
    """Base exception for scraper errors."""
    pass


class RateLimitError(ScraperError):
    """Raised when rate limit is hit."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(ScraperError):
    """Raised when authentication fails."""
    pass


class NotFoundError(ScraperError):
    """Raised when resource is not found."""
    pass


class BlockedError(ScraperError):
    """Raised when scraper is blocked by the platform."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

class Platform(str, Enum):
    """Supported platforms."""
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    REDDIT = "reddit"


@dataclass
class ScraperResult:
    """
    Unified result from any scraper.
    
    Standardizes data format across all platforms for
    consistent database storage and processing.
    """
    # Identifiers
    platform: Platform
    platform_id: str
    
    # Content
    text: str
    language: str = "en"
    
    # Author info
    author_id: Optional[str] = None
    author_username: Optional[str] = None
    author_display_name: Optional[str] = None
    author_followers: Optional[int] = None
    author_verified: bool = False
    
    # Engagement metrics
    likes: int = 0
    shares: int = 0
    replies: int = 0
    views: Optional[int] = None
    
    # Context
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    
    # Location (Kenya-focused)
    geo_location: Optional[str] = None
    mentioned_locations: List[str] = field(default_factory=list)
    
    # Thread/conversation
    reply_to_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Timestamps
    posted_at: datetime = field(default_factory=datetime.utcnow)
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    
    # Raw data for debugging
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_social_post(self) -> "SocialPost":
        """Convert to database SocialPost model."""
        from ..db.models import SocialPost, Platform as DBPlatform
        
        return SocialPost(
            platform=DBPlatform(self.platform.value),
            platform_id=self.platform_id,
            text=self.text,
            language=self.language,
            likes=self.likes,
            shares=self.shares,
            replies=self.replies,
            views=self.views,
            hashtags=self.hashtags,
            mentions=self.mentions,
            urls=self.urls,
            media_urls=self.media_urls,
            geo_location=self.geo_location,
            mentioned_locations={"locations": self.mentioned_locations},
            reply_to_id=self.reply_to_id,
            conversation_id=self.conversation_id,
            posted_at=self.posted_at,
            scraped_at=self.scraped_at,
            raw_data=self.raw_data,
        )
    
    def to_author(self) -> Optional["Author"]:
        """Convert author info to database Author model."""
        if not self.author_id:
            return None
        
        from ..db.models import Author, Platform as DBPlatform
        
        return Author(
            platform=DBPlatform(self.platform.value),
            platform_id=self.author_id,
            username=self.author_username,
            display_name=self.author_display_name,
            follower_count=self.author_followers,
            verified=self.author_verified,
        )


@dataclass
class ScraperStats:
    """Statistics for a scraper session."""
    posts_scraped: int = 0
    errors: int = 0
    rate_limits_hit: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_scrape_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get duration of scraping session."""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    @property
    def posts_per_minute(self) -> float:
        """Get scraping rate."""
        minutes = self.duration_seconds / 60
        if minutes == 0:
            return 0
        return self.posts_scraped / minutes


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for controlling scrape frequency.
    
    Prevents hitting platform rate limits by controlling
    request frequency.
    """
    
    def __init__(
        self,
        requests_per_minute: float = 30,
        burst_size: Optional[int] = None,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Sustained request rate.
            burst_size: Max burst capacity. Defaults to 1.5x rate.
        """
        self.rate = requests_per_minute / 60.0  # Per second
        self.burst_size = burst_size or int(requests_per_minute * 1.5)
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens to acquire.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                
                # Add tokens based on time elapsed
                self.tokens = min(
                    self.burst_size,
                    self.tokens + elapsed * self.rate
                )
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                
                # Wait for tokens to accumulate
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
    
    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()


# =============================================================================
# Retry Logic
# =============================================================================

class RetryPolicy:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        import random
        
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay *= (0.5 + random.random())
        
        return delay


async def with_retry(
    func,
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[callable] = None,
):
    """
    Execute async function with retry logic.
    
    Args:
        func: Async function to execute.
        policy: Retry policy. Uses defaults if None.
        on_retry: Callback for retry events.
        
    Returns:
        Result of func.
        
    Raises:
        Last exception if all retries fail.
    """
    policy = policy or RetryPolicy()
    last_error = None
    
    for attempt in range(policy.max_retries + 1):
        try:
            return await func()
        except RateLimitError as e:
            last_error = e
            delay = e.retry_after or policy.get_delay(attempt)
            logger.warning(f"Rate limited, waiting {delay:.1f}s before retry")
            
            if on_retry:
                on_retry(attempt, e)
            
            await asyncio.sleep(delay)
        except (BlockedError, AuthenticationError):
            # Don't retry these
            raise
        except ScraperError as e:
            last_error = e
            if attempt < policy.max_retries:
                delay = policy.get_delay(attempt)
                logger.warning(f"Scraper error ({e}), retrying in {delay:.1f}s")
                
                if on_retry:
                    on_retry(attempt, e)
                
                await asyncio.sleep(delay)
    
    raise last_error or ScraperError("Unknown error after retries")


# =============================================================================
# Base Scraper Class
# =============================================================================

class BaseScraper(ABC):
    """
    Abstract base class for all scrapers.
    
    Implements the dual scrape/API pattern:
    - scrape(): Works immediately via web scraping
    - scrape_via_api(): Ready for when API credentials are available
    
    Subclasses must implement:
    - platform property
    - _scrape_impl()
    - _scrape_via_api_impl()
    - has_api_credentials()
    """
    
    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """
        Initialize base scraper.
        
        Args:
            rate_limiter: Rate limiter instance. Creates default if None.
            retry_policy: Retry policy. Creates default if None.
        """
        self._rate_limiter = rate_limiter or RateLimiter()
        self._retry_policy = retry_policy or RetryPolicy()
        self._stats = ScraperStats()
        self._initialized = False
    
    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The platform this scraper targets."""
        pass
    
    @abstractmethod
    async def _initialize(self) -> None:
        """
        Initialize scraper resources.
        
        Called once before first scrape. Override to set up
        HTTP clients, authenticate, etc.
        """
        pass
    
    @abstractmethod
    async def has_api_credentials(self) -> bool:
        """
        Check if API credentials are configured.
        
        Returns:
            True if API mode is available.
        """
        pass
    
    @abstractmethod
    async def _scrape_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Implementation of web scraping.
        
        Args:
            query: Search query or target to scrape.
            limit: Maximum results to return.
            since: Only return posts after this time.
            
        Returns:
            List of scraped results.
        """
        pass
    
    @abstractmethod
    async def _scrape_via_api_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Implementation of API-based scraping.
        
        This method should be ready to use when API credentials
        are configured. It will raise NotImplementedError if
        credentials are not available.
        
        Args:
            query: Search query.
            limit: Maximum results.
            since: Only return posts after this time.
            
        Returns:
            List of results from API.
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize scraper if not already done."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
    
    async def scrape(
        self,
        query: str,
        limit: int = 100,
        since: Optional[datetime] = None,
        prefer_api: bool = False,
    ) -> List[ScraperResult]:
        """
        Scrape for posts matching query.
        
        Uses web scraping by default. If prefer_api is True
        and API credentials are available, uses API instead.
        
        Args:
            query: Search query or target.
            limit: Maximum results.
            since: Only posts after this time.
            prefer_api: Prefer API if available.
            
        Returns:
            List of scraped results.
        """
        await self.initialize()
        
        # Choose method
        if prefer_api and await self.has_api_credentials():
            impl = lambda: self._scrape_via_api_impl(query, limit, since)
        else:
            impl = lambda: self._scrape_impl(query, limit, since)
        
        # Execute with rate limiting and retry
        await self._rate_limiter.acquire()
        
        try:
            results = await with_retry(impl, self._retry_policy)
            self._stats.posts_scraped += len(results)
            self._stats.last_scrape_time = datetime.utcnow()
            return results
        except ScraperError:
            self._stats.errors += 1
            raise
    
    async def scrape_via_api(
        self,
        query: str,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Explicitly use API for scraping.
        
        Raises:
            AuthenticationError: If API credentials not configured.
        """
        await self.initialize()
        
        if not await self.has_api_credentials():
            raise AuthenticationError(
                f"API credentials not configured for {self.platform.value}"
            )
        
        await self._rate_limiter.acquire()
        
        try:
            results = await with_retry(
                lambda: self._scrape_via_api_impl(query, limit, since),
                self._retry_policy
            )
            self._stats.posts_scraped += len(results)
            self._stats.last_scrape_time = datetime.utcnow()
            return results
        except ScraperError:
            self._stats.errors += 1
            raise
    
    async def stream(
        self,
        query: str,
        batch_size: int = 100,
        max_seen_ids: int = 50000,
    ) -> AsyncIterator[ScraperResult]:
        """
        Stream results for continuous monitoring.
        
        Yields results in batches, continuing until stopped.
        
        Args:
            query: Search query.
            batch_size: Results per batch.
            max_seen_ids: Maximum IDs to track for deduplication (LRU eviction).
            
        Yields:
            Individual ScraperResult objects.
        """
        from collections import OrderedDict
        seen_ids: OrderedDict = OrderedDict()
        
        while True:
            results = await self.scrape(query, limit=batch_size)
            
            for result in results:
                if result.platform_id not in seen_ids:
                    # LRU eviction: remove oldest if at capacity
                    if len(seen_ids) >= max_seen_ids:
                        seen_ids.popitem(last=False)
                    seen_ids[result.platform_id] = True
                    yield result
            
            # Wait before next batch
            await asyncio.sleep(60)  # 1 minute between batches
    
    def get_stats(self) -> ScraperStats:
        """Get scraping statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = ScraperStats()
    
    async def close(self) -> None:
        """
        Close scraper resources.
        
        Override to clean up HTTP clients, sessions, etc.
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

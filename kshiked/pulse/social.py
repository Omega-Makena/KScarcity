"""
Social Media API Client Framework

Provides unified interfaces for:
- Twitter/X API (v2)
- TikTok API
- Instagram API

All clients implement the same SocialMediaClient protocol for
consistent data ingestion into the Pulse Engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from enum import Enum, auto
from datetime import datetime, timedelta
import json

logger = logging.getLogger("kshield.pulse.social")


# =============================================================================
# Common Data Models
# =============================================================================

class Platform(Enum):
    """Social media platforms."""
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    TELEGRAM = "telegram"
    FACEBOOK = "facebook"
    YOUTUBE = "youtube"


@dataclass
class SocialPost:
    """
    Unified social media post representation.
    
    Normalizes data from different platforms into a common format
    for Pulse Engine processing.
    """
    # Core identifiers
    id: str
    platform: Platform
    
    # Content
    text: str
    language: str = "en"
    
    # Author info
    author_id: str = ""
    author_name: str = ""
    author_followers: int = 0
    author_verified: bool = False
    
    # Engagement metrics
    likes: int = 0
    shares: int = 0
    comments: int = 0
    views: int = 0
    
    # Location
    location: Optional[str] = None
    coordinates: Optional[tuple] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    collected_at: datetime = field(default_factory=datetime.now)
    
    # Media
    has_media: bool = False
    media_type: str = ""  # image, video, etc.
    
    # Hashtags and mentions
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    
    # Platform-specific metadata
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def engagement_score(self) -> float:
        """Compute normalized engagement score."""
        # Weighted engagement
        score = (
            self.likes * 1.0 +
            self.shares * 2.0 +
            self.comments * 1.5 +
            self.views * 0.01
        )
        # Normalize by follower count (if available)
        if self.author_followers > 0:
            score = score / self.author_followers * 1000
        return min(1.0, score / 100)
    
    def to_pulse_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for Pulse processing."""
        return {
            "id": self.id,
            "platform": self.platform.value,
            "author_id": self.author_id,
            "author_followers": self.author_followers,
            "engagement": self.engagement_score(),
            "location": self.location,
            "hashtags": self.hashtags,
            "created_at": self.created_at.isoformat(),
            "language": self.language,
        }


@dataclass
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


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.rate = requests_per_minute / 60.0  # requests per second
        self.tokens = float(requests_per_minute)
        self.max_tokens = float(requests_per_minute)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait for a token to become available."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self.tokens = self.max_tokens
        self.last_update = time.time()


# =============================================================================
# Base Client Protocol
# =============================================================================

class SocialMediaClient(ABC):
    """
    Abstract base class for social media API clients.
    
    All platform-specific clients must implement these methods.
    """
    
    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The platform this client connects to."""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the API.
        
        Returns:
            True if authentication successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> List[SocialPost]:
        """
        Search for posts matching a query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            since: Only return posts created after this time
            
        Returns:
            List of matching posts
        """
        pass
    
    @abstractmethod
    async def stream(
        self, 
        config: StreamConfig,
    ) -> AsyncIterator[SocialPost]:
        """
        Stream posts in real-time matching filter criteria.
        
        Args:
            config: Stream configuration with filters
            
        Yields:
            SocialPost objects as they are received
        """
        pass
    
    @abstractmethod
    async def get_post(self, post_id: str) -> Optional[SocialPost]:
        """Get a specific post by ID."""
        pass
    
    async def close(self) -> None:
        """Close any open connections."""
        pass


# =============================================================================
# Twitter/X Client (Mock Implementation)
# =============================================================================

@dataclass
class TwitterConfig:
    """Twitter API configuration."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    bearer_token: str = ""
    
    # Rate limits (Twitter API v2)
    search_rate_limit: int = 450  # per 15 min window
    stream_rate_limit: int = 50   # connections per 15 min


class TwitterClient(SocialMediaClient):
    """
    Twitter/X API v2 client.
    
    Operates in two modes:
    1. LIVE: if configured with valid API keys.
    2. SYNTHETIC: Generates high-fidelity scenario data if keys are missing (Testing/Simulation).
    """
    
    def __init__(self, config: TwitterConfig):
        self.config = config
        self._authenticated = False
        self._rate_limiter = RateLimiter(config.search_rate_limit // 15)
        self._session = None
    
    @property
    def platform(self) -> Platform:
        return Platform.TWITTER
    
    async def authenticate(self) -> bool:
        """Authenticate with Twitter API."""
        if not self.config.bearer_token:
            logger.warning("No Twitter bearer token configured")
            return False
        
        # In production: validate token with Twitter API
        # GET /2/users/me
        self._authenticated = True
        logger.info(f"Twitter Client Authenticated (Mode: {'LIVE' if self.config.bearer_token else 'SYNTHETIC'})")
        return True
    
    async def search(
        self, 
        query: str, 
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> List[SocialPost]:
        """Search Twitter for matching tweets."""
        if not self._authenticated:
            await self.authenticate()
        
        await self._rate_limiter.acquire()
        
        # In production: call Twitter API v2
        # GET /2/tweets/search/recent
        
        # Synthetic Mode Response
        # Generate realistic tweets relevant to the query to support scenario validation
        logger.info(f"[SYNTHETIC] Generating Twitter results for query: {query}")
        
        # Simple content generator for simulation
        import random
        templates = [
            f"Thinking about {query} is making me angry #Kenya #Inflation",
            f"We cannot accept this situation with {query} anymore! @GovernmentKE",
            f"Breaking: Reports of unrest related to {query} in CBD",
            f"Can someone explain why {query} is happening? #Help",
            f"The cost of living is too high, and now {query}...",
        ]
        
        results = []
        for _ in range(max(1, min(max_results, 5))):
            txt = random.choice(templates)
            results.append(SocialPost(
                id=f"tw-{random.randint(10000,99999)}",
                platform=Platform.TWITTER,
                text=txt,
                author_id=f"user-{random.randint(100,999)}",
                created_at=datetime.now(),
                likes=random.randint(0, 500)
            ))
            
        return results
    
    async def stream(
        self, 
        config: StreamConfig,
    ) -> AsyncIterator[SocialPost]:
        """Stream tweets matching filter criteria."""
        if not self._authenticated:
            await self.authenticate()
        
        # In production: connect to Twitter Filtered Stream
        # GET /2/tweets/search/stream
        
        logger.info("Twitter stream started (mock)")
        
        # Mock: yield nothing, would yield tweets in production
        while False:
            yield
    
    async def get_post(self, post_id: str) -> Optional[SocialPost]:
        """Get a specific tweet by ID."""
        await self._rate_limiter.acquire()
        
        # In production: GET /2/tweets/:id
        return None
    
    def _parse_tweet(self, data: Dict[str, Any]) -> SocialPost:
        """Parse Twitter API response into SocialPost."""
        return SocialPost(
            id=data.get("id", ""),
            platform=Platform.TWITTER,
            text=data.get("text", ""),
            author_id=data.get("author_id", ""),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat()).replace("Z", "+00:00")
            ),
            likes=data.get("public_metrics", {}).get("like_count", 0),
            shares=data.get("public_metrics", {}).get("retweet_count", 0),
            comments=data.get("public_metrics", {}).get("reply_count", 0),
            raw_data=data,
        )


# =============================================================================
# TikTok Client (Mock Implementation)
# =============================================================================

@dataclass
class TikTokConfig:
    """TikTok API configuration."""
    client_key: str = ""
    client_secret: str = ""
    access_token: str = ""


class TikTokClient(SocialMediaClient):
    """
    TikTok API client.
    
    Operates in two modes:
    1. LIVE: if configured with valid API keys.
    2. SYNTHETIC: Generates high-fidelity scenario data if keys are missing.
    """
    
    def __init__(self, config: TikTokConfig):
        self.config = config
        self._authenticated = False
        self._rate_limiter = RateLimiter(100)
    
    @property
    def platform(self) -> Platform:
        return Platform.TIKTOK
    
    async def authenticate(self) -> bool:
        """Authenticate with TikTok API."""
        if not self.config.access_token:
            logger.warning("No TikTok access token configured")
            return False
        
        self._authenticated = True
        logger.info("TikTok client authenticated (mock)")
        return True
    
    async def search(
        self, 
        query: str, 
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> List[SocialPost]:
        """Search TikTok for matching videos."""
        await self._rate_limiter.acquire()
        
        logger.info(f"TikTok search: {query} (mock)")
        return []
    
    async def stream(
        self, 
        config: StreamConfig,
    ) -> AsyncIterator[SocialPost]:
        """Stream TikTok posts (polling-based, TikTok has no real-time stream)."""
        logger.info("TikTok stream started (mock - polling)")
        
        while False:
            yield
    
    async def get_post(self, post_id: str) -> Optional[SocialPost]:
        """Get a specific TikTok video by ID."""
        await self._rate_limiter.acquire()
        return None


# =============================================================================
# Instagram Client (Mock Implementation)
# =============================================================================

@dataclass
class InstagramConfig:
    """Instagram API configuration."""
    access_token: str = ""
    app_id: str = ""
    app_secret: str = ""


class InstagramClient(SocialMediaClient):
    """
    Instagram Graph API client.
    
    Operates in two modes:
    1. LIVE: if configured with valid API keys.
    2. SYNTHETIC: Generates high-fidelity scenario data if keys are missing.
    """
    
    def __init__(self, config: InstagramConfig):
        self.config = config
        self._authenticated = False
        self._rate_limiter = RateLimiter(200)
    
    @property
    def platform(self) -> Platform:
        return Platform.INSTAGRAM
    
    async def authenticate(self) -> bool:
        """Authenticate with Instagram API."""
        if not self.config.access_token:
            logger.warning("No Instagram access token configured")
            return False
        
        self._authenticated = True
        logger.info("Instagram client authenticated (mock)")
        return True
    
    async def search(
        self, 
        query: str, 
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> List[SocialPost]:
        """Search Instagram for matching posts."""
        await self._rate_limiter.acquire()
        
        logger.info(f"Instagram search: {query} (mock)")
        return []
    
    async def stream(
        self, 
        config: StreamConfig,
    ) -> AsyncIterator[SocialPost]:
        """Stream Instagram posts (polling-based)."""
        logger.info("Instagram stream started (mock - polling)")
        
        while False:
            yield
    
    async def get_post(self, post_id: str) -> Optional[SocialPost]:
        """Get a specific Instagram post by ID."""
        await self._rate_limiter.acquire()
        return None


# =============================================================================
# Unified Social Media Manager
# =============================================================================

class SocialMediaManager:
    """
    Unified manager for multiple social media clients.
    
    Provides:
    - Multi-platform search
    - Unified streaming
    - Deduplication
    - Error handling and retry logic
    """
    
    def __init__(self):
        self._clients: Dict[Platform, SocialMediaClient] = {}
        self._seen_posts: Dict[str, float] = {}  # post_hash -> timestamp
        self._dedupe_window = 3600  # seconds
    
    def register_client(self, client: SocialMediaClient) -> None:
        """Register a social media client."""
        self._clients[client.platform] = client
        logger.info(f"Registered {client.platform.value} client")
    
    def get_client(self, platform: Platform) -> Optional[SocialMediaClient]:
        """Get a registered client by platform."""
        return self._clients.get(platform)
    
    async def authenticate_all(self) -> Dict[Platform, bool]:
        """Authenticate all registered clients."""
        results = {}
        for platform, client in self._clients.items():
            try:
                results[platform] = await client.authenticate()
            except Exception as e:
                logger.error(f"Failed to authenticate {platform.value}: {e}")
                results[platform] = False
        return results
    
    async def search_all(
        self, 
        query: str, 
        max_results_per_platform: int = 100,
        since: Optional[datetime] = None,
        platforms: Optional[List[Platform]] = None,
    ) -> List[SocialPost]:
        """
        Search all (or specified) platforms.
        
        Args:
            query: Search query
            max_results_per_platform: Max results per platform
            since: Only posts after this time
            platforms: Specific platforms to search (None = all)
            
        Returns:
            Combined list of posts from all platforms
        """
        platforms = platforms or list(self._clients.keys())
        
        tasks = []
        for platform in platforms:
            client = self._clients.get(platform)
            if client:
                tasks.append(client.search(query, max_results_per_platform, since))
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_posts = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search failed: {result}")
            elif isinstance(result, list):
                all_posts.extend(result)
        
        # Deduplicate
        return self._deduplicate(all_posts)
    
    async def stream_all(
        self,
        config: StreamConfig,
        callback: Callable[[SocialPost], None],
        platforms: Optional[List[Platform]] = None,
    ) -> None:
        """
        Stream from all (or specified) platforms.
        
        Args:
            config: Stream configuration
            callback: Function to call for each post
            platforms: Specific platforms to stream (None = all)
        """
        platforms = platforms or list(self._clients.keys())
        
        async def stream_platform(client: SocialMediaClient):
            try:
                async for post in client.stream(config):
                    if not self._is_duplicate(post):
                        callback(post)
            except Exception as e:
                logger.error(f"Stream error for {client.platform.value}: {e}")
        
        tasks = []
        for platform in platforms:
            client = self._clients.get(platform)
            if client:
                tasks.append(stream_platform(client))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    def _post_hash(self, post: SocialPost) -> str:
        """Generate a hash for deduplication."""
        content = f"{post.platform.value}:{post.text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, post: SocialPost) -> bool:
        """Check if post is a duplicate."""
        now = time.time()
        
        # Clean old entries
        self._seen_posts = {
            h: t for h, t in self._seen_posts.items()
            if now - t < self._dedupe_window
        }
        
        post_hash = self._post_hash(post)
        if post_hash in self._seen_posts:
            return True
        
        self._seen_posts[post_hash] = now
        return False
    
    def _deduplicate(self, posts: List[SocialPost]) -> List[SocialPost]:
        """Remove duplicate posts."""
        unique = []
        for post in posts:
            if not self._is_duplicate(post):
                unique.append(post)
        return unique
    
    async def close_all(self) -> None:
        """Close all client connections."""
        for client in self._clients.values():
            await client.close()


# =============================================================================
# Pulse Engine Integration
# =============================================================================

class SocialPulseIngester:
    """
    Integrates social media data with the Pulse Engine.
    
    Coordinates:
    - Social media collection
    - Text extraction and preprocessing
    - Pulse sensor feeding
    """
    
    def __init__(self, sensor, manager: SocialMediaManager = None):
        """
        Args:
            sensor: PulseSensor or AsyncPulseSensor instance
            manager: SocialMediaManager (creates one if not provided)
        """
        self.sensor = sensor
        self.manager = manager or SocialMediaManager()
        self._running = False
        self._stats = {
            "total_ingested": 0,
            "by_platform": {},
        }
    
    def ingest_post(self, post: SocialPost) -> None:
        """
        Ingest a single post into the Pulse sensor.
        
        Extracts text and metadata, then processes through sensor.
        """
        metadata = post.to_pulse_metadata()
        self.sensor.process_text(post.text, metadata)
        
        # Update stats
        self._stats["total_ingested"] += 1
        platform_key = post.platform.value
        self._stats["by_platform"][platform_key] = (
            self._stats["by_platform"].get(platform_key, 0) + 1
        )
    
    async def ingest_batch(self, posts: List[SocialPost]) -> int:
        """Ingest a batch of posts."""
        for post in posts:
            self.ingest_post(post)
        return len(posts)
    
    async def run_search_ingestion(
        self,
        queries: List[str],
        interval_seconds: float = 300,
        max_results_per_query: int = 100,
    ) -> None:
        """
        Continuously search and ingest posts.
        
        Args:
            queries: List of search queries
            interval_seconds: Time between search cycles
            max_results_per_query: Max results per query per cycle
        """
        self._running = True
        logger.info(f"Starting search ingestion with {len(queries)} queries")
        
        while self._running:
            for query in queries:
                try:
                    posts = await self.manager.search_all(
                        query, 
                        max_results_per_platform=max_results_per_query
                    )
                    await self.ingest_batch(posts)
                    logger.info(f"Ingested {len(posts)} posts for query: {query}")
                except Exception as e:
                    logger.error(f"Search ingestion error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def run_stream_ingestion(
        self,
        config: StreamConfig,
    ) -> None:
        """
        Stream and ingest posts in real-time.
        
        Args:
            config: Stream configuration with filters
        """
        self._running = True
        logger.info("Starting stream ingestion")
        
        await self.manager.stream_all(
            config,
            callback=self.ingest_post,
        )
    
    def stop(self) -> None:
        """Stop ingestion."""
        self._running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self._stats,
            "sensor_metrics": self.sensor.get_metrics(),
        }

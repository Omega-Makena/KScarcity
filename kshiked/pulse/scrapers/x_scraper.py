"""
X (Twitter) Scraper for KShield Pulse

Multi-strategy X scraper:
1. Twikit web scraper (`x_web_scraper.py`) with session/proxy rotation
2. twscrape - Uses X accounts for scraping (no API key needed)
3. ntscraper - Scrapes via Nitter instances (no auth required)
4. Official API via existing x_client.py (if bearer token configured)

Usage:
    config = XScraperConfig(
        backend_mode="web_primary",
        # For twscrape/legacy
        accounts=[
            XAccount(username="account1", password="pass1", email="email1@example.com"),
        ],
        # For ntscraper fallback
        nitter_instances=["https://nitter.net"],
    )
    
    async with XScraper(config) as scraper:
        posts = await scraper.scrape("Kenya politics", limit=100)

Note on Libraries:
    - twscrape: pip install twscrape (uses accounts, more reliable)
    - ntscraper: pip install ntscraper (uses Nitter, no auth but less reliable)
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from .base import (
    BaseScraper, ScraperResult, ScraperError, RateLimitError,
    AuthenticationError, Platform, RateLimiter, RetryPolicy,
)

logger = logging.getLogger("kshield.pulse.scrapers.x")

if TYPE_CHECKING:
    from .x_web_scraper import KenyaXScraper, ScrapedTweet


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class XAccount:
    """X account credentials for twscrape."""
    username: str
    password: str
    email: str
    email_password: Optional[str] = None


@dataclass
class XScraperConfig:
    """
    Configuration for X scraper.
    
    Backend modes:
    1. web_primary: Twikit first, fallback to legacy backends
    2. web_only: Twikit only (raise on web failure)
    3. legacy_default: Legacy first, fallback to Twikit
    4. legacy_only: Legacy only
    """
    # Backend routing mode
    backend_mode: str = "web_primary"

    # For API mode (plug in when available)
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    bearer_token: str = ""
    
    # For twscrape mode
    accounts: List[XAccount] = field(default_factory=list)
    
    # For ntscraper mode (fallback)
    nitter_instances: List[str] = field(default_factory=lambda: [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
    ])
    
    # Kenya-focused defaults
    kenya_keywords: List[str] = field(default_factory=lambda: [
        "Kenya", "Nairobi", "Ruto", "Raila", "Gachagua",
        "KenyaKwanza", "Azimio", "hustler", "maandamano",
        "unga prices", "fuel prices", "cost of living Kenya",
    ])
    
    # Rate limiting
    requests_per_minute: float = 20

    # Web backend auth/session/proxy/checkpoint controls
    web_username: str = ""
    web_password: str = ""
    web_email: str = ""
    web_cookie_path: str = ""
    web_output_dir: str = ""
    web_session_config_path: str = ""
    web_proxies: List[str] = field(default_factory=list)
    web_session_cookies: List[str] = field(default_factory=list)
    web_checkpoint_path: str = "data/pulse/x_scraper_checkpoint.json"
    web_enable_checkpoint: bool = True
    web_resume_from_checkpoint: bool = True
    web_checkpoint_every_pages: int = 1
    web_rotate_on_rate_limit: bool = True
    web_rotate_on_detection: bool = True
    web_request_delay_s: float = 1.2
    web_query_delay_s: float = 3.0
    web_request_jitter_s: float = 0.0
    web_query_jitter_s: float = 0.0
    web_detection_cooldown_hours: float = 24.0
    web_wait_if_cooldown_active: bool = False
    web_export_csv: bool = True


# =============================================================================
# X Scraper Implementation
# =============================================================================

class XScraper(BaseScraper):
    """
    X (Twitter) scraper with multiple fallback strategies.
    
    Strategy is controlled by `XScraperConfig.backend_mode`.
    """
    
    def __init__(
        self,
        config: Optional[XScraperConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        super().__init__(
            rate_limiter=rate_limiter or RateLimiter(
                requests_per_minute=config.requests_per_minute if config else 20
            ),
            retry_policy=retry_policy,
        )
        self.config = config or XScraperConfig()
        
        # Lazy-loaded clients
        self._twscrape_api = None
        self._ntscraper = None
        self._x_client = None
        self._web_scraper: Optional["KenyaXScraper"] = None
        self._web_export_cache: Dict[str, "ScrapedTweet"] = {}

    @staticmethod
    def _resolve_path(path_value: str) -> Optional[Path]:
        if not path_value:
            return None
        return Path(path_value).expanduser()

    @staticmethod
    def _split_csv_field(value: Optional[str]) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item and item.strip()]

    @property
    def platform(self) -> Platform:
        return Platform.TWITTER

    async def _initialize(self) -> None:
        """Initialize available scraping backends."""
        mode = (self.config.backend_mode or "web_primary").lower()
        web_enabled = mode in {"web_primary", "web_only", "legacy_default"}
        legacy_enabled = mode in {"web_primary", "legacy_default", "legacy_only"}

        if web_enabled:
            await self._init_web_scraper()

        if legacy_enabled:
            await self._init_legacy_backends()

        if mode == "web_only" and not self._web_scraper:
            raise ScraperError("web_only backend mode requested, but web backend failed to initialize.")

        if mode == "legacy_only" and not self._has_legacy_backend():
            raise ScraperError("legacy_only backend mode requested, but no legacy backend initialized.")

        if mode == "web_primary" and not (self._web_scraper or self._has_legacy_backend()):
            raise ScraperError("No X backend available (web and legacy both unavailable).")

        if mode == "legacy_default" and not (self._has_legacy_backend() or self._web_scraper):
            raise ScraperError("No X backend available (legacy and web both unavailable).")

        logger.info(
            "X scraper initialized: mode=%s web=%s twscrape=%s ntscraper=%s",
            mode,
            self._web_scraper is not None,
            self._twscrape_api is not None,
            self._ntscraper is not None,
        )

    async def _init_web_scraper(self) -> None:
        """Initialize Twikit-based web scraper backend."""
        try:
            from .x_web_scraper import (
                KenyaXScraper,
                _load_session_configs_from_file,
            )
        except Exception as e:
            if (self.config.backend_mode or "").lower() == "web_only":
                raise ScraperError(f"Failed to import web backend: {e}") from e
            logger.warning("Web backend unavailable (import failed): %s", e)
            return

        session_configs = None
        if self.config.web_session_config_path:
            config_path = self._resolve_path(self.config.web_session_config_path)
            if config_path:
                try:
                    base_cookie = self._resolve_path(self.config.web_cookie_path) or Path(
                        "data/pulse/.x_cookies.json"
                    )
                    session_configs = _load_session_configs_from_file(
                        path=config_path,
                        default_username=self.config.web_username,
                        default_password=self.config.web_password,
                        default_email=self.config.web_email,
                        base_cookie_path=base_cookie,
                    )
                except Exception as e:
                    if (self.config.backend_mode or "").lower() == "web_only":
                        raise ScraperError(f"Failed to load web session config: {e}") from e
                    logger.warning("Failed to load web session config '%s': %s", config_path, e)

        session_cookie_paths = [
            self._resolve_path(raw)
            for raw in self.config.web_session_cookies
            if self._resolve_path(raw) is not None
        ]

        try:
            self._web_scraper = KenyaXScraper(
                username=self.config.web_username,
                password=self.config.web_password,
                email=self.config.web_email,
                cookie_path=self._resolve_path(self.config.web_cookie_path),
                output_dir=self._resolve_path(self.config.web_output_dir),
                session_configs=session_configs,
                proxies=[p for p in self.config.web_proxies if p],
                session_cookie_paths=session_cookie_paths or None,
                checkpoint_path=self._resolve_path(self.config.web_checkpoint_path),
                enable_checkpoint=self.config.web_enable_checkpoint,
                resume_from_checkpoint=self.config.web_resume_from_checkpoint,
                checkpoint_every_pages=self.config.web_checkpoint_every_pages,
                rotate_on_rate_limit=self.config.web_rotate_on_rate_limit,
                rotate_on_detection=self.config.web_rotate_on_detection,
                request_delay_s=self.config.web_request_delay_s,
                query_delay_s=self.config.web_query_delay_s,
                request_jitter_s=self.config.web_request_jitter_s,
                query_jitter_s=self.config.web_query_jitter_s,
                detection_cooldown_hours=self.config.web_detection_cooldown_hours,
                wait_if_cooldown_active=self.config.web_wait_if_cooldown_active,
            )
            await self._web_scraper.initialize()
            logger.info("Web backend initialized")
        except Exception as e:
            self._web_scraper = None
            if (self.config.backend_mode or "").lower() == "web_only":
                raise ScraperError(f"Web backend initialization failed: {e}") from e
            logger.warning("Web backend initialization failed: %s", e)

    async def _init_legacy_backends(self) -> None:
        """Initialize legacy backends (twscrape + ntscraper)."""
        if self.config.accounts:
            await self._init_twscrape()
        self._init_ntscraper()

    def _has_legacy_backend(self) -> bool:
        return self._twscrape_api is not None or self._ntscraper is not None

    async def _init_twscrape(self) -> None:
        """Initialize twscrape with configured accounts."""
        try:
            from twscrape import API, AccountsPool
            
            # Create accounts pool
            pool = AccountsPool()
            
            for acc in self.config.accounts:
                await pool.add_account(
                    acc.username,
                    acc.password,
                    acc.email,
                    acc.email_password or acc.password,
                )
            
            # Login all accounts
            await pool.login_all()
            
            self._twscrape_api = API(pool)
            logger.info(f"twscrape initialized with {len(self.config.accounts)} accounts")
            
        except ImportError:
            logger.warning("twscrape not installed. Run: pip install twscrape")
        except Exception as e:
            logger.warning(f"Failed to initialize twscrape: {e}")
    
    def _init_ntscraper(self) -> None:
        """Initialize ntscraper (Nitter-based scraping)."""
        try:
            from ntscraper import Nitter
            
            # Use first available instance
            instance = self.config.nitter_instances[0] if self.config.nitter_instances else None
            self._ntscraper = Nitter(instance=instance)
            logger.info("ntscraper initialized")
            
        except ImportError:
            logger.warning("ntscraper not installed. Run: pip install ntscraper")
        except Exception as e:
            logger.warning(f"Failed to initialize ntscraper: {e}")
    
    async def has_api_credentials(self) -> bool:
        """Check if official API credentials are configured."""
        return bool(self.config.bearer_token)
    
    async def _scrape_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape X using configured backend routing.
        """
        mode = (self.config.backend_mode or "web_primary").lower()

        if mode == "web_only":
            return await self._scrape_web(query, limit, since)

        if mode == "legacy_only":
            return await self._scrape_legacy(query, limit, since)

        if mode == "legacy_default":
            try:
                return await self._scrape_legacy(query, limit, since)
            except Exception as e:
                logger.warning("Legacy backend failed: %s, trying web backend", e)
                return await self._scrape_web(query, limit, since)

        # default: web_primary
        try:
            return await self._scrape_web(query, limit, since)
        except Exception as e:
            logger.warning("Web backend failed: %s, falling back to legacy backend", e)
            return await self._scrape_legacy(query, limit, since)

    async def _scrape_web(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape using Twikit-based web backend."""
        if not self._web_scraper:
            raise ScraperError("Web scraper is not initialized")

        try:
            tweets = await self._web_scraper.search_tweets(query, limit=limit)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise RateLimitError(f"Web scraper rate limit hit: {e}") from e
            raise ScraperError(f"Web scraper error: {e}") from e

        results: List[ScraperResult] = []
        for tweet in tweets:
            mapped = self._parse_web_tweet(tweet)
            if not mapped:
                continue
            if since and mapped.posted_at < since:
                continue
            results.append(mapped)
            if len(results) >= limit:
                break

        if self.config.web_export_csv:
            try:
                for tweet in tweets:
                    self._web_export_cache[tweet.tweet_id] = tweet
                self._web_scraper.save_tweets_csv(tweets)
                self._web_scraper.save_accounts_csv(self._web_scraper.get_accounts())
                self._web_scraper.export_dashboard_csv(list(self._web_export_cache.values()))
            except Exception as e:
                logger.warning("Failed exporting web CSV artifacts: %s", e)

        logger.info("Web backend returned %s tweets for '%s'", len(results), query)
        return results

    def _parse_web_tweet(self, tweet: "ScrapedTweet") -> Optional[ScraperResult]:
        """Convert x_web_scraper ScrapedTweet to ScraperResult."""
        try:
            posted_at = tweet.created_at
            if not isinstance(posted_at, datetime):
                posted_at = datetime.utcnow()

            scraped_at = datetime.utcnow()
            if tweet.scraped_at:
                try:
                    scraped_at = datetime.fromisoformat(tweet.scraped_at.replace("Z", "+00:00"))
                except Exception:
                    scraped_at = datetime.utcnow()

            return ScraperResult(
                platform=Platform.TWITTER,
                platform_id=str(tweet.tweet_id),
                text=tweet.text,
                language=tweet.language or "en",
                author_id=tweet.author_id or None,
                author_username=tweet.author_username or None,
                author_display_name=tweet.author_display_name or None,
                author_followers=tweet.author_followers or None,
                author_verified=bool(tweet.author_verified),
                likes=int(tweet.like_count or 0),
                shares=int(tweet.retweet_count or 0),
                replies=int(tweet.reply_count or 0),
                views=int(tweet.view_count or 0),
                hashtags=self._split_csv_field(tweet.hashtags),
                mentions=self._split_csv_field(tweet.mentions),
                urls=self._split_csv_field(tweet.urls),
                media_urls=self._split_csv_field(tweet.media_urls),
                geo_location=tweet.location_county or None,
                mentioned_locations=self._split_csv_field(tweet.mentioned_counties),
                reply_to_id=tweet.reply_to_tweet_id or None,
                conversation_id=tweet.conversation_id or None,
                posted_at=posted_at,
                scraped_at=scraped_at,
                raw_data={
                    "source": tweet.source,
                    "reply_to_user": tweet.reply_to_user,
                    "is_retweet": bool(tweet.is_retweet),
                    "is_quote": bool(tweet.is_quote),
                    "latitude": tweet.latitude,
                    "longitude": tweet.longitude,
                    "author_location": tweet.author_location,
                    "author_bio": tweet.author_bio,
                },
            )
        except Exception as e:
            logger.warning("Failed to parse web tweet: %s", e)
            return None

    async def _scrape_legacy(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape using legacy backends (twscrape then ntscraper)."""
        if self._twscrape_api:
            try:
                return await self._scrape_twscrape(query, limit, since)
            except Exception as e:
                logger.warning("twscrape failed: %s, falling back to ntscraper", e)

        if self._ntscraper:
            try:
                return await self._scrape_ntscraper(query, limit, since)
            except Exception as e:
                raise ScraperError(f"Legacy X scraping failed: {e}") from e

        raise ScraperError("No legacy X backend available. Install twscrape or ntscraper.")
    
    async def _scrape_twscrape(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape using twscrape."""
        results = []
        
        # Build search query
        search_query = query
        if since:
            since_str = since.strftime("%Y-%m-%d")
            search_query = f"{query} since:{since_str}"
        
        # Add Kenya filter if not already present
        if "kenya" not in query.lower() and "nairobi" not in query.lower():
            search_query = f"{search_query} (Kenya OR Nairobi)"
        
        try:
            async for tweet in self._twscrape_api.search(search_query, limit=limit):
                result = self._parse_twscrape_tweet(tweet)
                if result:
                    results.append(result)
                    
                    if len(results) >= limit:
                        break
            
            logger.info(f"twscrape returned {len(results)} tweets for '{query}'")
            return results
            
        except Exception as e:
            if "rate" in str(e).lower():
                raise RateLimitError(f"X rate limit hit: {e}")
            raise ScraperError(f"twscrape error: {e}")
    
    def _parse_twscrape_tweet(self, tweet: Any) -> Optional[ScraperResult]:
        """Parse twscrape tweet object to ScraperResult."""
        try:
            # Extract hashtags and mentions from text
            text = tweet.rawContent if hasattr(tweet, 'rawContent') else str(tweet)
            hashtags = re.findall(r'#(\w+)', text)
            mentions = re.findall(r'@(\w+)', text)
            urls = re.findall(r'https?://\S+', text)
            
            # Get user info
            user = tweet.user if hasattr(tweet, 'user') else None
            
            return ScraperResult(
                platform=Platform.TWITTER,
                platform_id=str(tweet.id) if hasattr(tweet, 'id') else str(hash(text)),
                text=text,
                language=tweet.lang if hasattr(tweet, 'lang') else "en",
                author_id=str(user.id) if user and hasattr(user, 'id') else None,
                author_username=user.username if user and hasattr(user, 'username') else None,
                author_display_name=user.displayname if user and hasattr(user, 'displayname') else None,
                author_followers=user.followersCount if user and hasattr(user, 'followersCount') else None,
                author_verified=user.verified if user and hasattr(user, 'verified') else False,
                likes=tweet.likeCount if hasattr(tweet, 'likeCount') else 0,
                shares=tweet.retweetCount if hasattr(tweet, 'retweetCount') else 0,
                replies=tweet.replyCount if hasattr(tweet, 'replyCount') else 0,
                views=tweet.viewCount if hasattr(tweet, 'viewCount') else None,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                reply_to_id=str(tweet.inReplyToTweetId) if hasattr(tweet, 'inReplyToTweetId') and tweet.inReplyToTweetId else None,
                conversation_id=str(tweet.conversationId) if hasattr(tweet, 'conversationId') else None,
                posted_at=tweet.date if hasattr(tweet, 'date') else datetime.utcnow(),
                scraped_at=datetime.utcnow(),
                raw_data={"source": "twscrape"},
            )
        except Exception as e:
            logger.warning(f"Failed to parse tweet: {e}")
            return None
    
    async def _scrape_ntscraper(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape using ntscraper (Nitter)."""
        results = []
        
        try:
            # ntscraper is synchronous, run in executor
            loop = asyncio.get_event_loop()
            
            # Get tweets from search
            tweets = await loop.run_in_executor(
                None,
                lambda: self._ntscraper.get_tweets(query, mode='search', number=limit)
            )
            
            if not tweets or 'tweets' not in tweets:
                return results
            
            for tweet_data in tweets['tweets']:
                result = self._parse_ntscraper_tweet(tweet_data)
                if result:
                    # Filter by since if specified
                    if since and result.posted_at < since:
                        continue
                    
                    results.append(result)
                    
                    if len(results) >= limit:
                        break
            
            logger.info(f"ntscraper returned {len(results)} tweets for '{query}'")
            return results
            
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise RateLimitError(f"Nitter rate limit: {e}")
            raise ScraperError(f"ntscraper error: {e}")
    
    def _parse_ntscraper_tweet(self, tweet_data: Dict[str, Any]) -> Optional[ScraperResult]:
        """Parse ntscraper tweet dict to ScraperResult."""
        try:
            text = tweet_data.get('text', '')
            
            # Extract patterns
            hashtags = re.findall(r'#(\w+)', text)
            mentions = re.findall(r'@(\w+)', text)
            urls = re.findall(r'https?://\S+', text)
            
            # Parse stats
            stats = tweet_data.get('stats', {})
            
            # Parse timestamp
            date_str = tweet_data.get('date', '')
            try:
                posted_at = datetime.strptime(date_str, "%b %d, %Y · %I:%M %p %Z")
            except ValueError:
                posted_at = datetime.utcnow()
            
            return ScraperResult(
                platform=Platform.TWITTER,
                platform_id=tweet_data.get('link', '').split('/')[-1] or str(hash(text)),
                text=text,
                author_id=None,  # Not available from Nitter
                author_username=tweet_data.get('user', {}).get('username'),
                author_display_name=tweet_data.get('user', {}).get('name'),
                likes=self._parse_stat(stats.get('likes', '0')),
                shares=self._parse_stat(stats.get('retweets', '0')),
                replies=self._parse_stat(stats.get('comments', '0')),
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                posted_at=posted_at,
                scraped_at=datetime.utcnow(),
                raw_data={"source": "ntscraper", "original": tweet_data},
            )
        except Exception as e:
            logger.warning(f"Failed to parse ntscraper tweet: {e}")
            return None
    
    def _parse_stat(self, stat_str: str) -> int:
        """Parse stat string like '1.2K' to integer."""
        if not stat_str:
            return 0
        
        stat_str = str(stat_str).strip().upper()
        
        try:
            if 'K' in stat_str:
                return int(float(stat_str.replace('K', '')) * 1000)
            elif 'M' in stat_str:
                return int(float(stat_str.replace('M', '')) * 1000000)
            else:
                return int(stat_str.replace(',', ''))
        except ValueError:
            return 0
    
    async def _scrape_via_api_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape using official X API.
        
        Uses the existing x_client.py implementation.
        Ready to use when bearer_token is configured.
        """
        if not self.config.bearer_token:
            raise AuthenticationError("X API bearer token not configured")
        
        # Import existing X client
        try:
            from ..x_client import RealXClient, XClientConfig
        except ImportError:
            raise ScraperError("x_client.py not found")
        
        # Create client with configured credentials
        client_config = XClientConfig(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            access_token=self.config.access_token,
            access_secret=self.config.access_secret,
            bearer_token=self.config.bearer_token,
        )
        
        client = RealXClient(client_config)
        
        try:
            await client.authenticate()
            posts = await client.search(query, max_results=limit, since=since)
            
            # Convert to ScraperResult
            results = []
            for post in posts:
                result = ScraperResult(
                    platform=Platform.TWITTER,
                    platform_id=post.id,
                    text=post.text,
                    language=post.language,
                    author_id=post.author_id,
                    author_username=post.author_username,
                    likes=post.likes,
                    shares=post.shares,
                    replies=post.replies,
                    views=post.views,
                    hashtags=post.hashtags,
                    mentions=post.mentions,
                    posted_at=post.created_at,
                    scraped_at=datetime.utcnow(),
                    raw_data=post.raw_data,
                )
                results.append(result)
            
            return results
            
        finally:
            await client.close()
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._web_scraper:
            try:
                close_method = getattr(self._web_scraper, "close", None)
                if callable(close_method):
                    result = close_method()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                pass
            self._web_scraper = None

        if self._twscrape_api:
            # twscrape doesn't need explicit cleanup
            self._twscrape_api = None
        
        if self._ntscraper:
            self._ntscraper = None
        
        if self._x_client:
            await self._x_client.close()
            self._x_client = None


# =============================================================================
# Factory Function
# =============================================================================

def create_x_scraper(
    accounts: Optional[List[Dict[str, str]]] = None,
    nitter_instances: Optional[List[str]] = None,
    bearer_token: Optional[str] = None,
    backend_mode: str = "web_primary",
) -> XScraper:
    """
    Create X scraper with specified configuration.
    
    Args:
        accounts: List of account dicts with username, password, email.
        nitter_instances: List of Nitter instance URLs.
        bearer_token: Official API bearer token (if available).
        backend_mode: Backend routing mode for X scraping.
        
    Returns:
        Configured XScraper instance.
    """
    account_objs = []
    if accounts:
        account_objs = [
            XAccount(
                username=acc['username'],
                password=acc['password'],
                email=acc['email'],
                email_password=acc.get('email_password'),
            )
            for acc in accounts
        ]
    
    config = XScraperConfig(
        backend_mode=backend_mode,
        accounts=account_objs,
        nitter_instances=nitter_instances or XScraperConfig().nitter_instances,
        bearer_token=bearer_token or "",
    )
    
    return XScraper(config)

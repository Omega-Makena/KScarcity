"""
X (Twitter) Scraper for KShield Pulse

Multi-strategy X scraper:
1. Official API via existing x_client.py (if bearer token configured)
2. twscrape - Uses X accounts for scraping (no API key needed)
3. ntscraper - Scrapes via Nitter instances (no auth required)

Usage:
    config = XScraperConfig(
        # For twscrape (recommended)
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

from .base import (
    BaseScraper, ScraperResult, ScraperError, RateLimitError,
    BlockedError, AuthenticationError, Platform, RateLimiter, RetryPolicy,
)

logger = logging.getLogger("kshield.pulse.scrapers.x")


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
    
    Supports three modes:
    1. API mode: Uses bearer_token with existing x_client.py
    2. twscrape mode: Uses accounts list for authenticated scraping
    3. ntscraper mode: Uses Nitter instances (no auth required)
    """
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


# =============================================================================
# X Scraper Implementation
# =============================================================================

class XScraper(BaseScraper):
    """
    X (Twitter) scraper with multiple fallback strategies.
    
    Strategy order:
    1. If bearer_token configured → Use official API (x_client.py)
    2. If accounts configured → Use twscrape
    3. Fallback → Use ntscraper (Nitter)
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
    
    @property
    def platform(self) -> Platform:
        return Platform.TWITTER
    
    async def _initialize(self) -> None:
        """Initialize available scraping backends."""
        # Try to initialize twscrape if accounts configured
        if self.config.accounts:
            await self._init_twscrape()
        
        # ntscraper is always available as fallback
        self._init_ntscraper()
        
        logger.info(f"X scraper initialized. twscrape={self._twscrape_api is not None}, ntscraper={self._ntscraper is not None}")
    
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
        Scrape X using best available method.
        
        Tries in order:
        1. twscrape (if accounts configured)
        2. ntscraper (Nitter fallback)
        """
        # Try twscrape first
        if self._twscrape_api:
            try:
                return await self._scrape_twscrape(query, limit, since)
            except Exception as e:
                logger.warning(f"twscrape failed: {e}, falling back to ntscraper")
        
        # Fallback to ntscraper
        if self._ntscraper:
            try:
                return await self._scrape_ntscraper(query, limit, since)
            except Exception as e:
                logger.error(f"ntscraper also failed: {e}")
                raise ScraperError(f"All X scraping methods failed: {e}")
        
        raise ScraperError("No X scraping backend available. Install twscrape or ntscraper.")
    
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
) -> XScraper:
    """
    Create X scraper with specified configuration.
    
    Args:
        accounts: List of account dicts with username, password, email.
        nitter_instances: List of Nitter instance URLs.
        bearer_token: Official API bearer token (if available).
        
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
        accounts=account_objs,
        nitter_instances=nitter_instances or XScraperConfig().nitter_instances,
        bearer_token=bearer_token or "",
    )
    
    return XScraper(config)

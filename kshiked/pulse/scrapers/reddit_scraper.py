"""
Reddit Scraper for KShield Pulse

Monitors Kenya-related subreddits and discussions using PRAW.

Usage:
    config = RedditScraperConfig(
        client_id="your_client_id",
        client_secret="your_client_secret",
        user_agent="KShieldPulse/1.0",
    )
    
    async with RedditScraper(config) as scraper:
        posts = await scraper.scrape("Kenya politics", limit=100)

Note:
    Reddit API requires registration at https://www.reddit.com/prefs/apps
    PRAW handles rate limiting automatically.
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
    AuthenticationError, Platform, RateLimiter, RetryPolicy,
)

logger = logging.getLogger("kshield.pulse.scrapers.reddit")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RedditScraperConfig:
    """Configuration for Reddit scraper."""
    
    # Reddit API credentials (from https://www.reddit.com/prefs/apps)
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "KShieldPulse/1.0 (Kenya Threat Monitoring)"
    
    # Optional user login for more access
    username: str = ""
    password: str = ""
    
    # Kenya-focused subreddits
    subreddits: List[str] = field(default_factory=lambda: [
        "Kenya",
        "NairobiCity", 
        "africa",
        "worldnews",  # For Kenya-related international news
    ])
    
    # Kenya-focused keywords
    kenya_keywords: List[str] = field(default_factory=lambda: [
        "Kenya", "Nairobi", "Mombasa", "Ruto", "Raila",
        "Kenyan", "KenyaKwanza", "Azimio",
    ])
    
    # Rate limiting
    requests_per_minute: float = 30  # Reddit is generous


# =============================================================================
# Reddit Scraper Implementation
# =============================================================================

class RedditScraper(BaseScraper):
    """
    Reddit scraper using PRAW.
    
    Features:
    - Monitors specified subreddits
    - Searches across all of Reddit with Kenya filters
    - Handles both posts and comments
    - Respects Reddit's rate limits
    """
    
    def __init__(
        self,
        config: Optional[RedditScraperConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        super().__init__(
            rate_limiter=rate_limiter or RateLimiter(
                requests_per_minute=config.requests_per_minute if config else 30
            ),
            retry_policy=retry_policy,
        )
        self.config = config or RedditScraperConfig()
        self._reddit = None
    
    @property
    def platform(self) -> Platform:
        return Platform.REDDIT
    
    async def _initialize(self) -> None:
        """Initialize PRAW Reddit instance."""
        try:
            import praw
            
            # Create Reddit instance
            kwargs = {
                "client_id": self.config.client_id or "PLACEHOLDER",
                "client_secret": self.config.client_secret or "PLACEHOLDER",
                "user_agent": self.config.user_agent,
            }
            
            # Add login if available
            if self.config.username and self.config.password:
                kwargs["username"] = self.config.username
                kwargs["password"] = self.config.password
            
            self._reddit = praw.Reddit(**kwargs)
            
            # Test connection with read-only if no credentials
            if not self.config.client_id:
                self._reddit.read_only = True
            
            logger.info("Reddit scraper initialized")
            
        except ImportError:
            logger.warning("PRAW not installed. Run: pip install praw")
            raise ScraperError("PRAW not installed")
    
    async def has_api_credentials(self) -> bool:
        """Check if Reddit API credentials are configured."""
        return bool(self.config.client_id and self.config.client_secret)
    
    async def _scrape_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape Reddit for matching posts.
        
        Uses read-only mode if no credentials configured.
        """
        if not self._reddit:
            raise ScraperError("Reddit client not initialized")
        
        results = []
        
        # Run PRAW operations in thread executor (it's synchronous)
        loop = asyncio.get_event_loop()
        
        # Add Kenya context to query
        search_query = self._build_kenya_query(query)
        
        try:
            # Search across reddit
            search_results = await loop.run_in_executor(
                None,
                lambda: list(self._reddit.subreddit("all").search(
                    search_query,
                    limit=limit,
                    sort="new",
                    time_filter="week",
                ))
            )
            
            for submission in search_results:
                result = self._parse_submission(submission)
                if result:
                    # Filter by since if specified
                    if since and result.posted_at < since:
                        continue
                    
                    results.append(result)
                    
                    if len(results) >= limit:
                        break
            
            logger.info(f"Reddit returned {len(results)} posts for '{query}'")
            return results
            
        except Exception as e:
            if "rate" in str(e).lower():
                raise RateLimitError(f"Reddit rate limit: {e}")
            raise ScraperError(f"Reddit scraping error: {e}")
    
    async def _scrape_via_api_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape using Reddit API (same as scrape_impl with PRAW).
        
        PRAW IS the API, so this is the same implementation
        but with credential validation.
        """
        if not await self.has_api_credentials():
            raise AuthenticationError("Reddit API credentials not configured")
        
        return await self._scrape_impl(query, limit, since)
    
    async def scrape_subreddits(
        self,
        limit_per_sub: int = 50,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape new posts from configured Kenya subreddits.
        
        Returns:
            List of posts from all configured subreddits.
        """
        if not self._reddit:
            await self.initialize()
        
        results = []
        loop = asyncio.get_event_loop()
        
        for subreddit_name in self.config.subreddits:
            try:
                subreddit = await loop.run_in_executor(
                    None,
                    lambda name=subreddit_name: self._reddit.subreddit(name)
                )
                
                submissions = await loop.run_in_executor(
                    None,
                    lambda sub=subreddit: list(sub.new(limit=limit_per_sub))
                )
                
                for submission in submissions:
                    result = self._parse_submission(submission)
                    if result:
                        if since and result.posted_at < since:
                            continue
                        results.append(result)
                
                logger.debug(f"Got {len(submissions)} posts from r/{subreddit_name}")
                
            except Exception as e:
                logger.warning(f"Failed to scrape r/{subreddit_name}: {e}")
                continue
        
        return results
    
    def _build_kenya_query(self, query: str) -> str:
        """Add Kenya context to search query."""
        query_lower = query.lower()
        
        # Check if already has Kenya context
        has_kenya = any(
            kw.lower() in query_lower 
            for kw in self.config.kenya_keywords
        )
        
        if has_kenya:
            return query
        
        # Add Kenya OR Nairobi to make it Kenya-focused
        return f"({query}) AND (Kenya OR Nairobi)"
    
    def _parse_submission(self, submission: Any) -> Optional[ScraperResult]:
        """Parse PRAW submission to ScraperResult."""
        try:
            # Combine title and selftext
            text = submission.title
            if hasattr(submission, 'selftext') and submission.selftext:
                text = f"{submission.title}\n\n{submission.selftext}"
            
            # Extract URLs
            urls = []
            if hasattr(submission, 'url') and submission.url:
                urls.append(submission.url)
            
            # Parse timestamp
            posted_at = datetime.utcfromtimestamp(submission.created_utc)
            
            return ScraperResult(
                platform=Platform.REDDIT,
                platform_id=submission.id,
                text=text,
                author_id=str(submission.author) if submission.author else None,
                author_username=str(submission.author) if submission.author else None,
                likes=submission.score,
                shares=0,  # Reddit doesn't have shares
                replies=submission.num_comments,
                urls=urls,
                posted_at=posted_at,
                scraped_at=datetime.utcnow(),
                raw_data={
                    "source": "praw",
                    "subreddit": str(submission.subreddit),
                    "permalink": submission.permalink,
                    "upvote_ratio": getattr(submission, 'upvote_ratio', None),
                    "is_self": getattr(submission, 'is_self', None),
                    "flair": getattr(submission, 'link_flair_text', None),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse Reddit submission: {e}")
            return None
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._reddit:
            # PRAW doesn't need explicit cleanup
            self._reddit = None


# =============================================================================
# Factory Function
# =============================================================================

def create_reddit_scraper(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    subreddits: Optional[List[str]] = None,
) -> RedditScraper:
    """
    Create Reddit scraper with specified configuration.
    
    Args:
        client_id: Reddit API client ID.
        client_secret: Reddit API client secret.
        subreddits: List of subreddits to monitor.
        
    Returns:
        Configured RedditScraper instance.
    """
    config = RedditScraperConfig(
        client_id=client_id or "",
        client_secret=client_secret or "",
        subreddits=subreddits or RedditScraperConfig().subreddits,
    )
    
    return RedditScraper(config)

"""
Instagram Scraper for KShield Pulse

Monitors Kenya-related hashtags and public profiles using Instaloader.

Usage:
    config = InstagramScraperConfig(
        username="your_username",
        password="your_password",
        hashtags=["KenyaNews", "Nairobi"],
    )
    
    async with InstagramScraper(config) as scraper:
        posts = await scraper.scrape("#KenyaNews", limit=100)

Note:
    - Login credentials recommended for better access
    - Instaloader saves sessions locally for persistence
    - Instagram aggressively rate limits - use sparingly
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import (
    BaseScraper, ScraperResult, ScraperError, RateLimitError,
    AuthenticationError, BlockedError, Platform, RateLimiter, RetryPolicy,
)

logger = logging.getLogger("kshield.pulse.scrapers.instagram")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InstagramScraperConfig:
    """Configuration for Instagram scraper."""
    
    # Login credentials (recommended for better access)
    username: str = ""
    password: str = ""
    
    # Session file location
    session_file: Optional[str] = None
    
    # Kenya-focused hashtags to monitor
    hashtags: List[str] = field(default_factory=lambda: [
        "Kenya", "Nairobi", "KenyaNews", "NairobiNews",
        "KenyaPolitics", "Kenyans", "KenyaOnlineCommunity",
    ])
    
    # Kenya-focused profiles to monitor
    profiles: List[str] = field(default_factory=list)
    
    # Rate limiting (Instagram is very aggressive)
    requests_per_minute: float = 5  # Very conservative
    
    # Max posts per hashtag to avoid detection
    max_posts_per_hashtag: int = 50


# =============================================================================
# Instagram Scraper Implementation
# =============================================================================

class InstagramScraper(BaseScraper):
    """
    Instagram scraper using Instaloader.
    
    Features:
    - Monitors hashtags
    - Scrapes public profiles
    - Handles session persistence
    - Conservative rate limiting
    
    Note: Instagram is very aggressive with anti-scraping measures.
    Use sparingly and consider proxies for production use.
    """
    
    def __init__(
        self,
        config: Optional[InstagramScraperConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        super().__init__(
            rate_limiter=rate_limiter or RateLimiter(
                requests_per_minute=config.requests_per_minute if config else 5
            ),
            retry_policy=retry_policy,
        )
        self.config = config or InstagramScraperConfig()
        self._loader = None
        self._logged_in = False
    
    @property
    def platform(self) -> Platform:
        return Platform.INSTAGRAM
    
    async def _initialize(self) -> None:
        """Initialize Instaloader instance."""
        try:
            import instaloader
            
            # Create loader with conservative settings
            self._loader = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_geotags=False,
                download_comments=False,
                save_metadata=False,
                compress_json=False,
                quiet=True,
                user_agent=None,  # Use default
            )
            
            # Try to load existing session
            if self.config.session_file and Path(self.config.session_file).exists():
                try:
                    self._loader.load_session_from_file(
                        self.config.username, 
                        self.config.session_file
                    )
                    self._logged_in = True
                    logger.info("Loaded existing Instagram session")
                except Exception as e:
                    logger.warning(f"Failed to load session: {e}")
            
            # Login if credentials provided and not already logged in
            if (self.config.username and 
                self.config.password and 
                not self._logged_in):
                
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: self._loader.login(
                            self.config.username,
                            self.config.password
                        )
                    )
                    self._logged_in = True
                    
                    # Save session for persistence
                    if self.config.session_file:
                        self._loader.save_session_to_file(self.config.session_file)
                    
                    logger.info("Instagram login successful")
                    
                except Exception as e:
                    logger.warning(f"Instagram login failed: {e}. Continuing without login.")
            
            logger.info(f"Instagram scraper initialized. logged_in={self._logged_in}")
            
        except ImportError:
            logger.warning("Instaloader not installed. Run: pip install instaloader")
    
    async def has_api_credentials(self) -> bool:
        """Check if login credentials are configured."""
        return bool(self.config.username and self.config.password)
    
    async def _scrape_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape Instagram for matching posts.
        
        Query can be:
        - Hashtag (with or without #)
        - Profile username
        """
        if not self._loader:
            raise ScraperError("Instaloader not initialized")
        
        loop = asyncio.get_event_loop()
        results = []
        
        try:
            # Determine if hashtag or profile
            query = query.strip()
            is_hashtag = query.startswith('#') or not query.startswith('@')
            
            if is_hashtag:
                hashtag = query.lstrip('#')
                results = await loop.run_in_executor(
                    None,
                    lambda: self._scrape_hashtag(hashtag, limit, since)
                )
            else:
                profile = query.lstrip('@')
                results = await loop.run_in_executor(
                    None,
                    lambda: self._scrape_profile(profile, limit, since)
                )
            
            logger.info(f"Instagram returned {len(results)} posts for '{query}'")
            return results
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "wait" in error_str:
                raise RateLimitError(f"Instagram rate limit: {e}")
            elif "login" in error_str or "auth" in error_str:
                raise AuthenticationError(f"Instagram auth error: {e}")
            elif "block" in error_str or "banned" in error_str:
                raise BlockedError(f"Instagram blocked: {e}")
            raise ScraperError(f"Instagram scraping error: {e}")
    
    def _scrape_hashtag(
        self,
        hashtag: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape posts from a hashtag (synchronous)."""
        import instaloader
        
        results = []
        
        try:
            posts = instaloader.Hashtag.from_name(self._loader.context, hashtag).get_posts()
            
            count = 0
            for post in posts:
                if count >= limit:
                    break
                
                # Filter by date
                if since and post.date < since:
                    continue
                
                result = self._parse_post(post)
                if result:
                    results.append(result)
                    count += 1
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to scrape hashtag #{hashtag}: {e}")
            raise
    
    def _scrape_profile(
        self,
        username: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape posts from a profile (synchronous)."""
        import instaloader
        
        results = []
        
        try:
            profile = instaloader.Profile.from_username(self._loader.context, username)
            
            count = 0
            for post in profile.get_posts():
                if count >= limit:
                    break
                
                # Filter by date
                if since and post.date < since:
                    continue
                
                result = self._parse_post(post)
                if result:
                    results.append(result)
                    count += 1
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to scrape profile @{username}: {e}")
            raise
    
    def _parse_post(self, post: Any) -> Optional[ScraperResult]:
        """Parse Instaloader post to ScraperResult."""
        try:
            # Get caption
            text = post.caption or ""
            
            # Extract hashtags from caption
            hashtags = re.findall(r'#(\w+)', text)
            
            # Extract mentions
            mentions = re.findall(r'@(\w+)', text)
            
            # Get media URLs
            media_urls = []
            if hasattr(post, 'url'):
                media_urls.append(post.url)
            
            # Get location
            location = None
            if hasattr(post, 'location') and post.location:
                location = str(post.location)
            
            return ScraperResult(
                platform=Platform.INSTAGRAM,
                platform_id=post.shortcode,
                text=text,
                author_id=str(post.owner_id) if hasattr(post, 'owner_id') else None,
                author_username=post.owner_username if hasattr(post, 'owner_username') else None,
                likes=post.likes if hasattr(post, 'likes') else 0,
                shares=0,  # Instagram doesn't expose shares
                replies=post.comments if hasattr(post, 'comments') else 0,
                views=post.video_view_count if hasattr(post, 'video_view_count') and post.is_video else None,
                hashtags=hashtags,
                mentions=mentions,
                media_urls=media_urls,
                geo_location=location,
                posted_at=post.date if hasattr(post, 'date') else datetime.utcnow(),
                scraped_at=datetime.utcnow(),
                raw_data={
                    "source": "instaloader",
                    "shortcode": post.shortcode,
                    "is_video": post.is_video if hasattr(post, 'is_video') else False,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse Instagram post: {e}")
            return None
    
    async def _scrape_via_api_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape using Meta Graph API.
        
        This is a placeholder for when official API access is available.
        The Graph API requires business account verification.
        """
        # TODO: Implement Meta Graph API when credentials available
        raise AuthenticationError(
            "Instagram Graph API not implemented. "
            "Using Instaloader scraping instead."
        )
    
    async def scrape_hashtags(
        self,
        limit_per_hashtag: int = 20,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape posts from all configured Kenya hashtags.
        
        Returns:
            List of posts from all hashtags.
        """
        if not self._loader:
            await self.initialize()
        
        results = []
        limit = min(limit_per_hashtag, self.config.max_posts_per_hashtag)
        
        for hashtag in self.config.hashtags:
            try:
                await self._rate_limiter.acquire()
                posts = await self.scrape(f"#{hashtag}", limit=limit, since=since)
                results.extend(posts)
                
                # Extra delay between hashtags
                await asyncio.sleep(5)
                
            except RateLimitError:
                logger.warning(f"Rate limited on #{hashtag}, stopping")
                break
            except Exception as e:
                logger.warning(f"Failed to scrape #{hashtag}: {e}")
                continue
        
        return results
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._loader:
            self._loader.close()
            self._loader = None


# =============================================================================
# Factory Function
# =============================================================================

def create_instagram_scraper(
    username: Optional[str] = None,
    password: Optional[str] = None,
    hashtags: Optional[List[str]] = None,
) -> InstagramScraper:
    """
    Create Instagram scraper with specified configuration.
    
    Args:
        username: Instagram login username.
        password: Instagram login password.
        hashtags: List of hashtags to monitor.
        
    Returns:
        Configured InstagramScraper instance.
    """
    config = InstagramScraperConfig(
        username=username or "",
        password=password or "",
        hashtags=hashtags or InstagramScraperConfig().hashtags,
    )
    
    return InstagramScraper(config)

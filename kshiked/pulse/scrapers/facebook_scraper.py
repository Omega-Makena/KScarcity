"""
Facebook Scraper for KShield Pulse

Monitors Kenya-related public pages using Playwright and facebook-scraper.

Usage:
    config = FacebookScraperConfig(
        pages=["KenyaNewsPage", "NairobiGossip"],
    )
    
    async with FacebookScraper(config) as scraper:
        posts = await scraper.scrape("Kenya news", limit=100)

Note:
    - Facebook is very aggressive with anti-scraping
    - Uses Playwright for browser automation
    - Rate limit heavily to avoid blocks
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from .base import (
    BaseScraper, ScraperResult, ScraperError, RateLimitError,
    AuthenticationError, BlockedError, Platform, RateLimiter, RetryPolicy,
)

logger = logging.getLogger("kshield.pulse.scrapers.facebook")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FacebookScraperConfig:
    """Configuration for Facebook scraper."""
    
    # Login credentials (optional but recommended)
    email: str = ""
    password: str = ""
    
    # Kenya-focused public pages to monitor
    pages: List[str] = field(default_factory=lambda: [
        # Add Kenya public pages here
        # Example: "KenyaNewsUpdates", "NairobiTrending"
    ])
    
    # Kenya-focused groups (public only)
    groups: List[str] = field(default_factory=list)
    
    # Rate limiting (Facebook is VERY aggressive)
    requests_per_minute: float = 3  # Very conservative
    
    # Browser settings
    headless: bool = True
    
    # Use facebook-scraper library (simpler but less reliable)
    use_facebook_scraper: bool = True


# =============================================================================
# Facebook Scraper Implementation
# =============================================================================

class FacebookScraper(BaseScraper):
    """
    Facebook scraper using facebook-scraper library and Playwright.
    
    Strategies:
    1. facebook-scraper: Simple library-based scraping (default)
    2. Playwright: Browser automation for more complex cases
    
    Note: Facebook is extremely aggressive with anti-scraping.
    Use very conservative rate limiting.
    """
    
    def __init__(
        self,
        config: Optional[FacebookScraperConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        super().__init__(
            rate_limiter=rate_limiter or RateLimiter(
                requests_per_minute=config.requests_per_minute if config else 3
            ),
            retry_policy=retry_policy,
        )
        self.config = config or FacebookScraperConfig()
        self._browser = None
        self._context = None
        self._page = None
        self._fb_scraper_available = False
    
    @property
    def platform(self) -> Platform:
        return Platform.FACEBOOK
    
    async def _initialize(self) -> None:
        """Initialize scraping backends."""
        # Try to import facebook-scraper
        try:
            import facebook_scraper
            self._fb_scraper_available = True
            logger.info("facebook-scraper library available")
        except ImportError:
            logger.warning("facebook-scraper not installed. Run: pip install facebook-scraper")
        
        # Initialize Playwright for fallback
        if not self._fb_scraper_available or not self.config.use_facebook_scraper:
            await self._init_playwright()
    
    async def _init_playwright(self) -> None:
        """Initialize Playwright browser."""
        try:
            from playwright.async_api import async_playwright
            
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(
                headless=self.config.headless
            )
            self._context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self._page = await self._context.new_page()
            
            # Login if credentials provided
            if self.config.email and self.config.password:
                await self._login()
            
            logger.info("Playwright initialized for Facebook scraping")
            
        except ImportError:
            logger.warning(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Playwright: {e}")
    
    async def _login(self) -> None:
        """Login to Facebook using Playwright."""
        if not self._page:
            return
        
        try:
            await self._page.goto("https://www.facebook.com/login")
            await asyncio.sleep(2)
            
            # Fill login form
            await self._page.fill('input[name="email"]', self.config.email)
            await self._page.fill('input[name="pass"]', self.config.password)
            await self._page.click('button[name="login"]')
            
            await asyncio.sleep(5)
            
            # Check if logged in
            if "login" not in self._page.url.lower():
                logger.info("Facebook login successful")
            else:
                logger.warning("Facebook login may have failed")
                
        except Exception as e:
            logger.warning(f"Facebook login error: {e}")
    
    async def has_api_credentials(self) -> bool:
        """Check if API credentials are configured."""
        # Facebook Graph API requires app approval, so we check for login
        return bool(self.config.email and self.config.password)
    
    async def _scrape_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape Facebook using best available method.
        """
        # Try facebook-scraper first
        if self._fb_scraper_available and self.config.use_facebook_scraper:
            try:
                return await self._scrape_fb_scraper(query, limit, since)
            except Exception as e:
                logger.warning(f"facebook-scraper failed: {e}")
        
        # Fallback to Playwright
        if self._page:
            try:
                return await self._scrape_playwright(query, limit, since)
            except Exception as e:
                logger.warning(f"Playwright scraping failed: {e}")
        
        raise ScraperError("No Facebook scraping method available")
    
    async def _scrape_fb_scraper(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape using facebook-scraper library."""
        import facebook_scraper as fb
        
        loop = asyncio.get_event_loop()
        results = []
        
        # Determine if query is a page name or search term
        is_page = not ' ' in query and len(self.config.pages) > 0
        
        try:
            if is_page or query in self.config.pages:
                # Scrape specific page
                posts = await loop.run_in_executor(
                    None,
                    lambda: list(fb.get_posts(query, pages=limit))
                )
            else:
                # Scrape all configured pages for posts containing query
                posts = []
                for page in self.config.pages[:3]:  # Limit pages to avoid detection
                    try:
                        page_posts = await loop.run_in_executor(
                            None,
                            lambda p=page: list(fb.get_posts(p, pages=min(limit, 5)))
                        )
                        # Filter by query
                        for post in page_posts:
                            text = post.get('text', '') or ''
                            if query.lower() in text.lower():
                                posts.append(post)
                        
                        await asyncio.sleep(2)  # Delay between pages
                        
                    except Exception as e:
                        logger.warning(f"Failed to scrape page {page}: {e}")
                        continue
            
            # Parse posts
            for post in posts[:limit]:
                result = self._parse_fb_post(post)
                if result:
                    if since and result.posted_at < since:
                        continue
                    results.append(result)
            
            return results
            
        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                raise RateLimitError(f"Facebook rate limit: {e}")
            raise ScraperError(f"facebook-scraper error: {e}")
    
    def _parse_fb_post(self, post: Dict[str, Any]) -> Optional[ScraperResult]:
        """Parse facebook-scraper post dict to ScraperResult."""
        try:
            text = post.get('text', '') or post.get('post_text', '') or ''
            
            if not text:
                return None
            
            # Extract data
            posted_at = post.get('time')
            if isinstance(posted_at, str):
                try:
                    posted_at = datetime.strptime(posted_at, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    posted_at = datetime.utcnow()
            elif not posted_at:
                posted_at = datetime.utcnow()
            
            return ScraperResult(
                platform=Platform.FACEBOOK,
                platform_id=str(post.get('post_id', hash(text))),
                text=text,
                author_id=str(post.get('user_id')) if post.get('user_id') else None,
                author_username=post.get('username'),
                author_display_name=post.get('user_url', '').split('/')[-1] if post.get('user_url') else None,
                likes=post.get('likes', 0) or 0,
                shares=post.get('shares', 0) or 0,
                replies=post.get('comments', 0) or 0,
                urls=post.get('link', []) if isinstance(post.get('link'), list) else [post.get('link')] if post.get('link') else [],
                media_urls=post.get('images', []) or [],
                posted_at=posted_at,
                scraped_at=datetime.utcnow(),
                raw_data={"source": "facebook-scraper"},
            )
        except Exception as e:
            logger.warning(f"Failed to parse Facebook post: {e}")
            return None
    
    async def _scrape_playwright(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """Scrape using Playwright browser automation."""
        if not self._page:
            raise ScraperError("Playwright not initialized")
        
        results = []
        
        try:
            # Navigate to page
            url = f"https://www.facebook.com/search/posts?q={query}"
            await self._page.goto(url)
            await asyncio.sleep(3)
            
            # Scroll to load more posts
            for _ in range(min(limit // 10, 5)):
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)
            
            # Extract posts (simplified)
            posts = await self._page.query_selector_all('[data-pagelet*="FeedUnit"]')
            
            for post_elem in posts[:limit]:
                try:
                    text = await post_elem.inner_text()
                    if text:
                        results.append(ScraperResult(
                            platform=Platform.FACEBOOK,
                            platform_id=str(hash(text[:100])),
                            text=text,
                            posted_at=datetime.utcnow(),
                            scraped_at=datetime.utcnow(),
                            raw_data={"source": "playwright"},
                        ))
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            raise ScraperError(f"Playwright scraping failed: {e}")
    
    async def _scrape_via_api_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape using Meta Graph API.
        
        Placeholder for when official API access is available.
        Graph API requires business verification.
        """
        # TODO: Implement Meta Graph API when approved
        raise AuthenticationError(
            "Facebook Graph API not implemented. "
            "Using facebook-scraper instead."
        )
    
    async def scrape_pages(
        self,
        limit_per_page: int = 10,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape posts from all configured pages.
        
        Returns:
            List of posts from all pages.
        """
        results = []
        
        for page in self.config.pages:
            try:
                await self._rate_limiter.acquire()
                posts = await self.scrape(page, limit=limit_per_page, since=since)
                results.extend(posts)
                
                await asyncio.sleep(10)  # Long delay between pages
                
            except RateLimitError:
                logger.warning(f"Rate limited on {page}, stopping")
                break
            except Exception as e:
                logger.warning(f"Failed to scrape {page}: {e}")
                continue
        
        return results
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        
        self._page = None
        self._context = None
        self._browser = None


# =============================================================================
# Factory Function
# =============================================================================

def create_facebook_scraper(
    email: Optional[str] = None,
    password: Optional[str] = None,
    pages: Optional[List[str]] = None,
) -> FacebookScraper:
    """
    Create Facebook scraper with specified configuration.
    
    Args:
        email: Facebook login email.
        password: Facebook login password.
        pages: List of public pages to monitor.
        
    Returns:
        Configured FacebookScraper instance.
    """
    config = FacebookScraperConfig(
        email=email or "",
        password=password or "",
        pages=pages or [],
    )
    
    return FacebookScraper(config)

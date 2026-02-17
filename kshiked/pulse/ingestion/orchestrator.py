"""
Ingestion Orchestrator for KShield Pulse

Main coordinator for the data ingestion pipeline.

Architecture:
- Manages all scraper instances
- Coordinates scraping schedules
- Routes data to database storage
- Triggers processing pipeline

Usage:
    config = IngestionConfig.from_env()
    
    async with IngestionOrchestrator(config) as orchestrator:
        # Run continuous ingestion
        await orchestrator.run()

    # Or run one-time scrape
    async with IngestionOrchestrator(config) as orchestrator:
        await orchestrator.scrape_social_media()
        await orchestrator.scrape_ecommerce()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable

from ..db import Database, DatabaseConfig, SocialPost, PriceSnapshot
from ..scrapers.base import BaseScraper, ScraperResult
from ..scrapers.x_scraper import XScraper, XScraperConfig
from ..scrapers.reddit_scraper import RedditScraper, RedditScraperConfig
from ..scrapers.telegram_scraper import TelegramScraper, TelegramScraperConfig
from ..scrapers.instagram_scraper import InstagramScraper, InstagramScraperConfig
from ..scrapers.facebook_scraper import FacebookScraper, FacebookScraperConfig
from ..scrapers.ecommerce.jiji_scraper import JijiScraper
from ..scrapers.ecommerce.jumia_scraper import JumiaScraper
from ..scrapers.ecommerce.kilimall_scraper import KilimallScraper
from ..scrapers.ecommerce.base import PriceData

logger = logging.getLogger("kshield.pulse.ingestion")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IngestionConfig:
    """
    Configuration for the ingestion orchestrator.
    
    Loads from environment variables or uses defaults.
    """
    # Database
    database_url: Optional[str] = None
    
    # X/Twitter
    x_accounts: List[Dict[str, str]] = field(default_factory=list)
    x_nitter_instances: List[str] = field(default_factory=lambda: [
        "https://nitter.net",
    ])
    x_bearer_token: str = ""
    
    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "Kenya", "NairobiCity", "africa",
    ])
    
    # Telegram
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_channels: List[str] = field(default_factory=list)
    
    # Instagram
    instagram_username: str = ""
    instagram_password: str = ""
    instagram_hashtags: List[str] = field(default_factory=lambda: [
        "Kenya", "Nairobi", "KenyaNews",
    ])
    
    # Facebook
    facebook_email: str = ""
    facebook_password: str = ""
    facebook_pages: List[str] = field(default_factory=list)
    
    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    
    # E-commerce
    ecommerce_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        "jiji": ["vehicles", "property", "electronics"],
        "jumia": ["groceries", "phones", "health-beauty"],
        "kilimall": ["electronics", "fashion", "home-living"],
    })
    
    # Scraping schedule
    social_scrape_interval_minutes: int = 30
    ecommerce_scrape_interval_hours: int = 6
    
    # Limits
    max_posts_per_scrape: int = 100
    max_products_per_category: int = 50
    
    # Kenya-focused search terms
    kenya_search_terms: List[str] = field(default_factory=lambda: [
        "Kenya",
        "Nairobi",
        "Ruto",
        "Raila",
        "maandamano",
        "cost of living Kenya",
        "unga prices",
    ])
    
    @classmethod
    def from_env(cls) -> "IngestionConfig":
        """Load configuration from environment variables."""
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            x_bearer_token=os.getenv("X_BEARER_TOKEN", ""),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            telegram_api_id=int(os.getenv("TELEGRAM_API_ID", "0")),
            telegram_api_hash=os.getenv("TELEGRAM_API_HASH", ""),
            instagram_username=os.getenv("INSTAGRAM_USERNAME", ""),
            instagram_password=os.getenv("INSTAGRAM_PASSWORD", ""),
            facebook_email=os.getenv("FACEBOOK_EMAIL", ""),
            facebook_password=os.getenv("FACEBOOK_PASSWORD", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        )


# =============================================================================
# Orchestrator
# =============================================================================

class IngestionOrchestrator:
    """
    Main coordinator for the data ingestion pipeline.
    
    Manages:
    - Social media scrapers (X, Reddit, Telegram, Instagram, Facebook)
    - E-commerce scrapers (Jiji, Jumia, Kilimall)
    - Database storage
    - Processing pipeline
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self._db: Optional[Database] = None
        self._scrapers: Dict[str, BaseScraper] = {}
        self._ecommerce_scrapers: Dict[str, Any] = {}
        self._running = False
        self._stats = {
            "posts_scraped": 0,
            "products_scraped": 0,
            "errors": 0,
            "last_scrape": None,
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize database and scrapers."""
        logger.info("Initializing ingestion orchestrator...")
        
        # Initialize database
        db_config = DatabaseConfig(url=self.config.database_url)
        self._db = Database(db_config)
        await self._db.connect()
        
        # Initialize social media scrapers
        await self._init_scrapers()
        
        # Initialize e-commerce scrapers
        await self._init_ecommerce_scrapers()
        
        logger.info("Ingestion orchestrator initialized")
    
    async def _init_scrapers(self) -> None:
        """Initialize social media scrapers."""
        # X/Twitter scraper
        x_config = XScraperConfig(
            bearer_token=self.config.x_bearer_token,
            nitter_instances=self.config.x_nitter_instances,
        )
        self._scrapers["x"] = XScraper(x_config)
        
        # Reddit scraper
        if self.config.reddit_client_id:
            reddit_config = RedditScraperConfig(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                subreddits=self.config.reddit_subreddits,
            )
            self._scrapers["reddit"] = RedditScraper(reddit_config)
        
        # Telegram scraper
        if self.config.telegram_api_id:
            telegram_config = TelegramScraperConfig(
                api_id=self.config.telegram_api_id,
                api_hash=self.config.telegram_api_hash,
                channels=self.config.telegram_channels,
            )
            self._scrapers["telegram"] = TelegramScraper(telegram_config)
        
        # Instagram scraper
        instagram_config = InstagramScraperConfig(
            username=self.config.instagram_username,
            password=self.config.instagram_password,
            hashtags=self.config.instagram_hashtags,
        )
        self._scrapers["instagram"] = InstagramScraper(instagram_config)
        
        # Facebook scraper
        facebook_config = FacebookScraperConfig(
            email=self.config.facebook_email,
            password=self.config.facebook_password,
            pages=self.config.facebook_pages,
        )
        self._scrapers["facebook"] = FacebookScraper(facebook_config)
        
        logger.info(f"Initialized {len(self._scrapers)} social media scrapers")
    
    async def _init_ecommerce_scrapers(self) -> None:
        """Initialize e-commerce scrapers."""
        self._ecommerce_scrapers["jiji"] = JijiScraper()
        self._ecommerce_scrapers["jumia"] = JumiaScraper()
        self._ecommerce_scrapers["kilimall"] = KilimallScraper()
        
        logger.info("Initialized 3 e-commerce scrapers")
    
    async def scrape_social_media(
        self,
        search_terms: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[SocialPost]:
        """
        Scrape all social media platforms.
        
        Args:
            search_terms: Search queries. Uses config defaults if None.
            limit: Max posts per platform.
            
        Returns:
            List of stored SocialPost objects.
        """
        search_terms = search_terms or self.config.kenya_search_terms
        limit = limit or self.config.max_posts_per_scrape
        
        all_posts = []
        
        for name, scraper in self._scrapers.items():
            try:
                logger.info(f"Scraping {name}...")
                
                for term in search_terms[:3]:  # Limit terms per run
                    try:
                        results = await scraper.scrape(
                            query=term,
                            limit=limit // len(search_terms),
                        )
                        
                        # Store results
                        for result in results:
                            # Check for duplicates
                            exists = await self._db.post_exists(
                                result.platform.value,
                                result.platform_id,
                            )
                            
                            if not exists:
                                post = result.to_social_post()
                                await self._db.add(post)
                                all_posts.append(post)
                                self._stats["posts_scraped"] += 1
                        
                        await asyncio.sleep(2)  # Rate limiting between terms
                        
                    except Exception as e:
                        logger.warning(f"Error scraping {name} for '{term}': {e}")
                        self._stats["errors"] += 1
                        continue
                
            except Exception as e:
                logger.error(f"Failed to scrape {name}: {e}")
                self._stats["errors"] += 1
                continue
        
        self._stats["last_scrape"] = datetime.utcnow()
        logger.info(f"Scraped {len(all_posts)} new posts from social media")
        
        return all_posts
    
    async def scrape_ecommerce(
        self,
        categories: Optional[Dict[str, List[str]]] = None,
        limit: Optional[int] = None,
    ) -> List[PriceSnapshot]:
        """
        Scrape all e-commerce platforms for prices.
        
        Args:
            categories: Categories per platform. Uses config if None.
            limit: Max products per category.
            
        Returns:
            List of stored PriceSnapshot objects.
        """
        categories = categories or self.config.ecommerce_categories
        limit = limit or self.config.max_products_per_category
        
        all_prices = []
        
        for name, scraper in self._ecommerce_scrapers.items():
            try:
                logger.info(f"Scraping {name} for prices...")
                
                site_categories = categories.get(name, [])
                
                async with scraper:
                    prices = await scraper.scrape_categories(
                        categories=site_categories,
                        limit_per_category=limit,
                    )
                    
                    # Store prices
                    for price_data in prices:
                        snapshot = price_data.to_price_snapshot()
                        await self._db.add(snapshot)
                        all_prices.append(snapshot)
                        self._stats["products_scraped"] += 1
                
            except Exception as e:
                logger.error(f"Failed to scrape {name}: {e}")
                self._stats["errors"] += 1
                continue
        
        logger.info(f"Scraped {len(all_prices)} price points from e-commerce")
        
        return all_prices
    
    async def run(
        self,
        duration_hours: Optional[float] = None,
    ) -> None:
        """
        Run continuous ingestion.
        
        Args:
            duration_hours: How long to run. None = indefinitely.
        """
        self._running = True
        start_time = datetime.utcnow()
        
        last_social_scrape = datetime.min
        last_ecommerce_scrape = datetime.min
        
        logger.info("Starting continuous ingestion...")
        
        try:
            while self._running:
                now = datetime.utcnow()
                
                # Check duration limit
                if duration_hours:
                    elapsed = (now - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        logger.info("Duration limit reached, stopping")
                        break
                
                # Social media scraping
                social_interval = timedelta(
                    minutes=self.config.social_scrape_interval_minutes
                )
                if now - last_social_scrape >= social_interval:
                    try:
                        await self.scrape_social_media()
                        last_social_scrape = now
                    except Exception as e:
                        logger.error(f"Social media scrape failed: {e}")
                
                # E-commerce scraping
                ecommerce_interval = timedelta(
                    hours=self.config.ecommerce_scrape_interval_hours
                )
                if now - last_ecommerce_scrape >= ecommerce_interval:
                    try:
                        await self.scrape_ecommerce()
                        last_ecommerce_scrape = now
                    except Exception as e:
                        logger.error(f"E-commerce scrape failed: {e}")
                
                # Wait before next check
                await asyncio.sleep(60)
                
        except asyncio.CancelledError:
            logger.info("Ingestion cancelled")
        finally:
            self._running = False
    
    async def stop(self) -> None:
        """Stop continuous ingestion."""
        self._running = False
    
    async def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing ingestion orchestrator...")
        
        # Close scrapers
        for scraper in self._scrapers.values():
            try:
                await scraper.close()
            except:
                pass
        
        for scraper in self._ecommerce_scrapers.values():
            try:
                await scraper.close()
            except:
                pass
        
        # Close database
        if self._db:
            await self._db.disconnect()
        
        logger.info("Ingestion orchestrator closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return dict(self._stats)


# =============================================================================
# Quick Run Functions
# =============================================================================

async def quick_scrape(
    search_terms: Optional[List[str]] = None,
    social: bool = True,
    ecommerce: bool = True,
) -> Dict[str, Any]:
    """
    Run a quick one-time scrape.
    
    Args:
        search_terms: Search queries for social media.
        social: Whether to scrape social media.
        ecommerce: Whether to scrape e-commerce.
        
    Returns:
        Statistics dict.
    """
    config = IngestionConfig.from_env()
    
    async with IngestionOrchestrator(config) as orchestrator:
        if social:
            await orchestrator.scrape_social_media(search_terms)
        
        if ecommerce:
            await orchestrator.scrape_ecommerce()
        
        return orchestrator.get_stats()


if __name__ == "__main__":
    # Quick test run
    import asyncio
    
    async def main():
        stats = await quick_scrape(
            search_terms=["Kenya"],
            social=True,
            ecommerce=False,
        )
        print(f"Stats: {stats}")
    
    asyncio.run(main())

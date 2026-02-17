# Scrapers package for KShield Pulse
"""
Social media and e-commerce scrapers for data ingestion.

Provides unified scraper interface with dual architecture:
- Primary: Scraping (works immediately)
- Secondary: API (ready to plug in when credentials available)

Platforms:
- X/Twitter: twscrape + ntscraper
- Telegram: Telethon
- Reddit: PRAW
- Instagram: Instaloader
- Facebook: Playwright

E-Commerce:
- Jiji Kenya
- Kilimall
- Jumia Kenya
"""

from .base import BaseScraper, ScraperResult, ScraperError

__all__ = [
    "BaseScraper",
    "ScraperResult",
    "ScraperError",
]

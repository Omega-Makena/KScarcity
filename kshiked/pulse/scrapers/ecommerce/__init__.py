# E-commerce scrapers package
"""
E-commerce price scrapers for inflation monitoring.

Tracks prices from Kenya's major e-commerce platforms:
- Jiji Kenya (jiji.co.ke) - C2C marketplace
- Kilimall - Electronics and consumer goods
- Jumia Kenya (jumia.co.ke) - General retail

Price data is mapped to KShield's ResourceDomain for
computing Economic Satisfaction Index (ESI).
"""

from .base import EcommerceScraper, PriceData, EcommerceScraperConfig, ResourceDomain
from .jiji_scraper import JijiScraper, create_jiji_scraper
from .jumia_scraper import JumiaScraper, create_jumia_scraper
from .kilimall_scraper import KilimallScraper, create_kilimall_scraper
from .price_aggregator import PriceAggregator, PriceIndex, EconomicSatisfactionScore

__all__ = [
    "EcommerceScraper",
    "PriceData", 
    "EcommerceScraperConfig",
    "ResourceDomain",
    "JijiScraper",
    "create_jiji_scraper",
    "JumiaScraper",
    "create_jumia_scraper",
    "KilimallScraper",
    "create_kilimall_scraper",
    "PriceAggregator",
    "PriceIndex",
    "EconomicSatisfactionScore",
]


"""
Base E-Commerce Scraper for KShield Pulse

Provides:
- Abstract base class for e-commerce scrapers
- Standard price data format
- Category mapping to economic domains
- Price change detection

All e-commerce scrapers inherit from EcommerceScraper
and map product categories to KShield's ResourceDomain
for inflation analysis.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger("kshield.pulse.scrapers.ecommerce")


# =============================================================================
# Enums
# =============================================================================

class ResourceDomain(str, Enum):
    """Economic resource domains matching KShield primitives."""
    FOOD = "food"
    FUEL = "fuel"
    HOUSING = "housing"
    TRANSPORT = "transport"
    HEALTHCARE = "healthcare"
    GENERAL = "general"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PriceData:
    """
    Standardized price data from e-commerce sites.
    
    Captures product pricing information for inflation tracking.
    """
    # Required fields first (no defaults)
    source: str  # jiji, jumia, kilimall
    product_url: str
    product_id: str
    product_name: str
    category: str
    price_kes: float
    
    # Optional fields with defaults
    subcategory: Optional[str] = None
    original_price_kes: Optional[float] = None  # Before discount
    discount_percent: Optional[float] = None
    economic_domain: ResourceDomain = ResourceDomain.GENERAL
    in_stock: bool = True
    stock_quantity: Optional[int] = None
    seller_name: Optional[str] = None
    seller_rating: Optional[float] = None
    seller_location: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_price_snapshot(self) -> "PriceSnapshot":
        """Convert to database PriceSnapshot model."""
        from ...db.models import PriceSnapshot
        
        return PriceSnapshot(
            source=self.source,
            product_url=self.product_url,
            product_id=self.product_id,
            product_name=self.product_name,
            price_kes=self.price_kes,
            original_price_kes=self.original_price_kes,
            discount_percent=self.discount_percent,
            in_stock=self.in_stock,
            stock_quantity=self.stock_quantity,
            seller_name=self.seller_name,
            seller_rating=self.seller_rating,
            scraped_at=self.scraped_at,
        )


@dataclass 
class EcommerceScraperConfig:
    """Base configuration for e-commerce scrapers."""
    
    # Rate limiting
    requests_per_minute: float = 10
    
    # Browser settings
    headless: bool = True
    
    # Categories to scrape
    categories: List[str] = field(default_factory=list)
    
    # Max products per category
    max_products_per_category: int = 50


# =============================================================================
# Category Mappings
# =============================================================================

# Maps e-commerce categories to ResourceDomain
CATEGORY_DOMAIN_MAP: Dict[str, ResourceDomain] = {
    # Food & Grocery
    "food": ResourceDomain.FOOD,
    "grocery": ResourceDomain.FOOD,
    "groceries": ResourceDomain.FOOD,
    "supermarket": ResourceDomain.FOOD,
    "food & beverage": ResourceDomain.FOOD,
    "food & drinks": ResourceDomain.FOOD,
    "beverages": ResourceDomain.FOOD,
    
    # Fuel & Energy
    "fuel": ResourceDomain.FUEL,
    "gas": ResourceDomain.FUEL,
    "petrol": ResourceDomain.FUEL,
    "diesel": ResourceDomain.FUEL,
    "lpg": ResourceDomain.FUEL,
    "cooking gas": ResourceDomain.FUEL,
    "energy": ResourceDomain.FUEL,
    
    # Housing
    "property": ResourceDomain.HOUSING,
    "real estate": ResourceDomain.HOUSING,
    "rent": ResourceDomain.HOUSING,
    "apartments": ResourceDomain.HOUSING,
    "houses": ResourceDomain.HOUSING,
    "land": ResourceDomain.HOUSING,
    "furniture": ResourceDomain.HOUSING,
    "home & garden": ResourceDomain.HOUSING,
    "home & furniture": ResourceDomain.HOUSING,
    
    # Transport
    "vehicles": ResourceDomain.TRANSPORT,
    "cars": ResourceDomain.TRANSPORT,
    "motorcycles": ResourceDomain.TRANSPORT,
    "bikes": ResourceDomain.TRANSPORT,
    "transport": ResourceDomain.TRANSPORT,
    "auto parts": ResourceDomain.TRANSPORT,
    "boda boda": ResourceDomain.TRANSPORT,
    
    # Healthcare
    "health": ResourceDomain.HEALTHCARE,
    "healthcare": ResourceDomain.HEALTHCARE,
    "pharmacy": ResourceDomain.HEALTHCARE,
    "medicine": ResourceDomain.HEALTHCARE,
    "medical": ResourceDomain.HEALTHCARE,
    "health & beauty": ResourceDomain.HEALTHCARE,
}


def map_category_to_domain(category: str) -> ResourceDomain:
    """Map e-commerce category to ResourceDomain."""
    category_lower = category.lower().strip()
    
    for key, domain in CATEGORY_DOMAIN_MAP.items():
        if key in category_lower:
            return domain
    
    return ResourceDomain.GENERAL


# =============================================================================
# Base E-Commerce Scraper
# =============================================================================

class EcommerceScraper(ABC):
    """
    Abstract base class for e-commerce scrapers.
    
    Subclasses must implement:
    - source property
    - _scrape_category()
    - _scrape_product()
    """
    
    def __init__(
        self,
        config: Optional[EcommerceScraperConfig] = None,
    ):
        self.config = config or EcommerceScraperConfig()
        self._browser = None
        self._page = None
        self._initialized = False
    
    @property
    @abstractmethod
    def source(self) -> str:
        """The source identifier (jiji, jumia, kilimall)."""
        pass
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize scraper resources (browser, etc)."""
        pass
    
    @abstractmethod
    async def _scrape_category(
        self,
        category: str,
        limit: int,
    ) -> List[PriceData]:
        """
        Scrape products from a category.
        
        Args:
            category: Category name or URL.
            limit: Maximum products to scrape.
            
        Returns:
            List of price data objects.
        """
        pass
    
    @abstractmethod
    async def _scrape_product(
        self,
        url: str,
    ) -> Optional[PriceData]:
        """
        Scrape a single product page.
        
        Args:
            url: Product URL.
            
        Returns:
            Price data or None if failed.
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize scraper if not already done."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
    
    async def scrape_categories(
        self,
        categories: Optional[List[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> List[PriceData]:
        """
        Scrape all configured categories.
        
        Args:
            categories: Categories to scrape. Uses config if None.
            limit_per_category: Max products per category.
            
        Returns:
            List of all price data.
        """
        await self.initialize()
        
        categories = categories or self.config.categories
        limit = limit_per_category or self.config.max_products_per_category
        
        all_prices = []
        
        for category in categories:
            try:
                logger.info(f"Scraping {self.source} category: {category}")
                prices = await self._scrape_category(category, limit)
                all_prices.extend(prices)
                
                # Delay between categories
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"Failed to scrape {category} from {self.source}: {e}")
                continue
        
        logger.info(f"Scraped {len(all_prices)} products from {self.source}")
        return all_prices
    
    async def scrape_product(self, url: str) -> Optional[PriceData]:
        """
        Scrape a single product.
        
        Args:
            url: Product URL.
            
        Returns:
            Price data or None.
        """
        await self.initialize()
        return await self._scrape_product(url)
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._page:
            await self._page.close()
        if self._browser:
            await self._browser.close()
        
        self._page = None
        self._browser = None
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

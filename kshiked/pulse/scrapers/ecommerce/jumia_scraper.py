"""
Jumia Kenya Scraper for KShield Pulse

Scrapes Jumia Kenya (jumia.co.ke) for:
- Grocery and daily essentials prices
- Electronics prices
- Fashion and clothing prices
- Flash sale data (economic sentiment)

Usage:
    async with JumiaScraper() as scraper:
        prices = await scraper.scrape_categories(["groceries", "phones"])

Note:
    Jumia is Kenya's largest online retailer with reliable pricing
    data for consumer goods and daily essentials.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin

from .base import (
    EcommerceScraper, EcommerceScraperConfig, PriceData,
    ResourceDomain, map_category_to_domain,
)

logger = logging.getLogger("kshield.pulse.scrapers.ecommerce.jumia")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JumiaScraperConfig(EcommerceScraperConfig):
    """Configuration for Jumia scraper."""
    
    base_url: str = "https://www.jumia.co.ke"
    
    # Default categories - focus on essentials
    categories: List[str] = field(default_factory=lambda: [
        "groceries",          # Food & daily essentials
        "phones",             # Mobile phones
        "computing",          # Laptops & accessories
        "health-beauty",      # Healthcare products
        "home-office",        # Furniture & appliances
    ])
    
    # Category URL mappings
    category_urls: Dict[str, str] = field(default_factory=lambda: {
        "groceries": "/groceries/",
        "food": "/groceries/",
        "supermarket": "/groceries/",
        "phones": "/phones-tablets/",
        "mobiles": "/phones-tablets/",
        "computing": "/computing/",
        "laptops": "/computing/",
        "electronics": "/electronics/",
        "health-beauty": "/health-beauty/",
        "health": "/health-beauty/",
        "home-office": "/home-office/",
        "furniture": "/home-office-furniture/",
        "appliances": "/home-office-appliances/",
        "fashion": "/fashion/",
        "clothing": "/womens-fashion/",
    })
    
    requests_per_minute: float = 10


# =============================================================================
# Jumia Scraper
# =============================================================================

class JumiaScraper(EcommerceScraper):
    """
    Jumia Kenya (jumia.co.ke) scraper.
    
    Jumia is Kenya's largest e-commerce platform, providing:
    - Standard retail prices
    - Discount tracking
    - Daily essentials pricing
    - Consumer electronics trends
    """
    
    def __init__(
        self,
        config: Optional[JumiaScraperConfig] = None,
    ):
        super().__init__(config or JumiaScraperConfig())
        self.config: JumiaScraperConfig = self.config
        self._session = None
    
    @property
    def source(self) -> str:
        return "jumia"
    
    async def _initialize(self) -> None:
        """Initialize HTTP session."""
        try:
            import httpx
            
            self._session = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                follow_redirects=True,
                timeout=30.0,
            )
            
            logger.info("Jumia scraper initialized")
            
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise
    
    async def _scrape_category(
        self,
        category: str,
        limit: int,
    ) -> List[PriceData]:
        """Scrape products from a Jumia category."""
        from bs4 import BeautifulSoup
        
        category_path = self.config.category_urls.get(
            category.lower(), f"/{category}/"
        )
        url = urljoin(self.config.base_url, category_path)
        
        results = []
        page = 1
        
        while len(results) < limit:
            page_url = f"{url}?page={page}" if page > 1 else url
            
            try:
                response = await self._session.get(page_url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product cards
                products = soup.select('article.prd')
                if not products:
                    products = soup.select('.sku')
                
                if not products:
                    logger.debug(f"No products found on {page_url}")
                    break
                
                for product in products:
                    if len(results) >= limit:
                        break
                    
                    try:
                        price_data = self._parse_product_card(product, category)
                        if price_data:
                            results.append(price_data)
                    except Exception as e:
                        logger.debug(f"Failed to parse Jumia product: {e}")
                        continue
                
                page += 1
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Failed to scrape Jumia page {page_url}: {e}")
                break
        
        return results
    
    def _parse_product_card(
        self,
        product: Any,
        category: str,
    ) -> Optional[PriceData]:
        """Parse a Jumia product card to PriceData."""
        try:
            # Extract product link and ID
            link_elem = product.select_one('a.core')
            if not link_elem:
                return None
            
            href = link_elem.get('href', '')
            product_url = urljoin(self.config.base_url, href)
            
            # Extract product ID from URL
            product_id = href.split('-')[-1].replace('.html', '') if href else None
            if not product_id:
                return None
            
            # Extract title
            title_elem = product.select_one('.name')
            title = title_elem.get_text(strip=True) if title_elem else None
            
            if not title:
                return None
            
            # Extract current price
            price_elem = product.select_one('.prc')
            price_text = price_elem.get_text(strip=True) if price_elem else None
            
            price = self._parse_price(price_text)
            if not price or price <= 0:
                return None
            
            # Extract original price (if discounted)
            old_price_elem = product.select_one('.old')
            old_price = None
            discount = None
            if old_price_elem:
                old_price = self._parse_price(old_price_elem.get_text(strip=True))
                if old_price and old_price > price:
                    discount = ((old_price - price) / old_price) * 100
            
            # Extract rating
            rating_elem = product.select_one('.stars')
            rating = None
            if rating_elem:
                rating_text = rating_elem.get('data-rating', '')
                try:
                    rating = float(rating_text)
                except:
                    pass
            
            # Check stock status
            in_stock = True
            if product.select_one('.out'):
                in_stock = False
            
            return PriceData(
                source=self.source,
                product_url=product_url,
                product_id=product_id,
                product_name=title,
                category=category,
                price_kes=price,
                original_price_kes=old_price,
                discount_percent=discount,
                economic_domain=map_category_to_domain(category),
                in_stock=in_stock,
                seller_rating=rating,
                scraped_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Jumia product: {e}")
            return None
    
    def _parse_price(self, price_text: str) -> Optional[float]:
        """Parse Jumia price string to float."""
        if not price_text:
            return None
        
        # Clean the price string
        price_text = price_text.upper().replace('KSH', '').replace('KES', '')
        price_text = price_text.replace(',', '').replace(' ', '')
        
        match = re.search(r'[\d.]+', price_text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None
    
    async def _scrape_product(
        self,
        url: str,
    ) -> Optional[PriceData]:
        """Scrape a single Jumia product page."""
        from bs4 import BeautifulSoup
        
        try:
            response = await self._session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.select_one('h1.-fs20')
            title = title_elem.get_text(strip=True) if title_elem else None
            
            # Extract price
            price_elem = soup.select_one('.-fs24')
            price_text = price_elem.get_text(strip=True) if price_elem else None
            price = self._parse_price(price_text)
            
            if not title or not price:
                return None
            
            # Extract product ID from URL
            product_id = url.split('-')[-1].replace('.html', '')
            
            # Get category from breadcrumbs
            breadcrumb = soup.select_one('.brcr')
            category = "general"
            if breadcrumb:
                crumbs = breadcrumb.select('a')
                if len(crumbs) >= 2:
                    category = crumbs[1].get_text(strip=True).lower()
            
            return PriceData(
                source=self.source,
                product_url=url,
                product_id=product_id,
                product_name=title,
                category=category,
                price_kes=price,
                economic_domain=map_category_to_domain(category),
                scraped_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.warning(f"Failed to scrape Jumia product {url}: {e}")
            return None
    
    async def scrape_flash_sales(
        self,
        limit: int = 50,
    ) -> List[PriceData]:
        """
        Scrape Flash Sales for discount tracking.
        
        Flash sale activity can indicate economic sentiment.
        """
        from bs4 import BeautifulSoup
        
        url = urljoin(self.config.base_url, "/flash-sales/")
        results = []
        
        try:
            response = await self._session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            products = soup.select('article.prd')
            
            for product in products[:limit]:
                price_data = self._parse_product_card(product, "flash-sale")
                if price_data:
                    results.append(price_data)
            
            logger.info(f"Scraped {len(results)} flash sale products")
            
        except Exception as e:
            logger.warning(f"Failed to scrape Jumia flash sales: {e}")
        
        return results
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.aclose()
            self._session = None
        
        await super().close()


# =============================================================================
# Factory Function
# =============================================================================

def create_jumia_scraper(
    categories: Optional[List[str]] = None,
) -> JumiaScraper:
    """Create Jumia scraper."""
    config = JumiaScraperConfig(
        categories=categories or JumiaScraperConfig().categories,
    )
    return JumiaScraper(config)

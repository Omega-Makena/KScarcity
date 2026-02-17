"""
Kilimall Kenya Scraper for KShield Pulse

Scrapes Kilimall (kilimall.co.ke) for:
- Electronics prices
- Fashion and clothing
- Home appliances
- Consumer goods

Usage:
    async with KilimallScraper() as scraper:
        prices = await scraper.scrape_categories(["electronics", "fashion"])

Note:
    Kilimall is a popular Kenyan e-commerce platform with
    competitive pricing on electronics and consumer goods.
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

logger = logging.getLogger("kshield.pulse.scrapers.ecommerce.kilimall")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class KilimallScraperConfig(EcommerceScraperConfig):
    """Configuration for Kilimall scraper."""
    
    base_url: str = "https://www.kilimall.co.ke"
    
    # Default categories
    categories: List[str] = field(default_factory=lambda: [
        "electronics",
        "phones",
        "fashion",
        "home-living",
        "health-beauty",
    ])
    
    # Category URL mappings
    category_urls: Dict[str, str] = field(default_factory=lambda: {
        "electronics": "/category/electronics",
        "phones": "/category/phones-tablets",
        "computing": "/category/computing",
        "fashion": "/category/fashion",
        "clothing": "/category/clothing",
        "home-living": "/category/home-living",
        "furniture": "/category/furniture",
        "appliances": "/category/home-appliances",
        "health-beauty": "/category/health-beauty",
        "groceries": "/category/groceries",
    })
    
    requests_per_minute: float = 10


# =============================================================================
# Kilimall Scraper
# =============================================================================

class KilimallScraper(EcommerceScraper):
    """
    Kilimall Kenya scraper.
    
    Kilimall provides competitive pricing on:
    - Electronics and gadgets
    - Fashion items
    - Home appliances
    - Health and beauty products
    """
    
    def __init__(
        self,
        config: Optional[KilimallScraperConfig] = None,
    ):
        super().__init__(config or KilimallScraperConfig())
        self.config: KilimallScraperConfig = self.config
        self._session = None
    
    @property
    def source(self) -> str:
        return "kilimall"
    
    async def _initialize(self) -> None:
        """Initialize HTTP session."""
        try:
            import httpx
            
            self._session = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                follow_redirects=True,
                timeout=30.0,
            )
            
            logger.info("Kilimall scraper initialized")
            
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise
    
    async def _scrape_category(
        self,
        category: str,
        limit: int,
    ) -> List[PriceData]:
        """Scrape products from a Kilimall category."""
        from bs4 import BeautifulSoup
        
        category_path = self.config.category_urls.get(
            category.lower(), f"/category/{category}"
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
                
                # Find product elements
                products = soup.select('.product-item')
                if not products:
                    products = soup.select('.product-card')
                if not products:
                    products = soup.select('[class*="product"]')
                
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
                        logger.debug(f"Failed to parse Kilimall product: {e}")
                        continue
                
                page += 1
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Failed to scrape Kilimall page {page_url}: {e}")
                break
        
        return results
    
    def _parse_product_card(
        self,
        product: Any,
        category: str,
    ) -> Optional[PriceData]:
        """Parse a Kilimall product card to PriceData."""
        try:
            # Extract link
            link_elem = product.select_one('a[href*="/product"]')
            if not link_elem:
                link_elem = product.select_one('a')
            
            if not link_elem:
                return None
            
            href = link_elem.get('href', '')
            product_url = urljoin(self.config.base_url, href)
            
            # Extract product ID
            product_id = None
            if '/product/' in href:
                product_id = href.split('/product/')[-1].split('/')[0]
            if not product_id:
                product_id = str(hash(href))
            
            # Extract title
            title_elem = product.select_one('.product-name') or product.select_one('.title')
            if not title_elem:
                title_elem = product.select_one('a')
            
            title = title_elem.get_text(strip=True) if title_elem else None
            if not title:
                return None
            
            # Extract price
            price_elem = product.select_one('.product-price') or product.select_one('.price')
            price_text = price_elem.get_text(strip=True) if price_elem else None
            
            price = self._parse_price(price_text)
            if not price or price <= 0:
                return None
            
            # Extract original price
            old_price_elem = product.select_one('.original-price') or product.select_one('.old-price')
            old_price = None
            discount = None
            if old_price_elem:
                old_price = self._parse_price(old_price_elem.get_text(strip=True))
                if old_price and old_price > price:
                    discount = ((old_price - price) / old_price) * 100
            
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
                scraped_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Kilimall product: {e}")
            return None
    
    def _parse_price(self, price_text: str) -> Optional[float]:
        """Parse Kilimall price string to float."""
        if not price_text:
            return None
        
        # Clean the price string
        price_text = price_text.upper().replace('KSH', '').replace('KES', '')
        price_text = price_text.replace('SH', '').replace(',', '').replace(' ', '')
        
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
        """Scrape a single Kilimall product page."""
        from bs4 import BeautifulSoup
        
        try:
            response = await self._session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.select_one('h1')
            title = title_elem.get_text(strip=True) if title_elem else None
            
            # Extract price
            price_elem = soup.select_one('.product-price') or soup.select_one('.price')
            price_text = price_elem.get_text(strip=True) if price_elem else None
            price = self._parse_price(price_text)
            
            if not title or not price:
                return None
            
            # Extract product ID
            product_id = url.split('/product/')[-1].split('/')[0] if '/product/' in url else str(hash(url))
            
            return PriceData(
                source=self.source,
                product_url=url,
                product_id=product_id,
                product_name=title,
                category="general",
                price_kes=price,
                economic_domain=ResourceDomain.GENERAL,
                scraped_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.warning(f"Failed to scrape Kilimall product {url}: {e}")
            return None
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.aclose()
            self._session = None
        
        await super().close()


# =============================================================================
# Factory Function
# =============================================================================

def create_kilimall_scraper(
    categories: Optional[List[str]] = None,
) -> KilimallScraper:
    """Create Kilimall scraper."""
    config = KilimallScraperConfig(
        categories=categories or KilimallScraperConfig().categories,
    )
    return KilimallScraper(config)

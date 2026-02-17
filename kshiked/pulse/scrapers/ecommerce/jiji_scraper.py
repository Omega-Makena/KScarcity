"""
Jiji Kenya Scraper for KShield Pulse

Scrapes Jiji Kenya (jiji.co.ke) for:
- Product prices and listings
- Property rentals (housing costs)
- Vehicle prices
- Used goods (economic stress indicator)

Usage:
    async with JijiScraper() as scraper:
        prices = await scraper.scrape_categories(["vehicles", "property"])

Note:
    Jiji is a C2C marketplace - prices reflect what regular
    Kenyans are asking for goods, making it a good indicator
    of economic conditions.
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

logger = logging.getLogger("kshield.pulse.scrapers.ecommerce.jiji")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JijiScraperConfig(EcommerceScraperConfig):
    """Configuration for Jiji scraper."""
    
    base_url: str = "https://jiji.co.ke"
    
    # Default categories to monitor
    categories: List[str] = field(default_factory=lambda: [
        "vehicles",           # Cars, motorbikes
        "property",           # Houses, apartments, land
        "electronics",        # Phones, computers
        "home-garden",        # Furniture, appliances
        "fashion",            # Clothes, shoes
    ])
    
    # Category URL mappings
    category_urls: Dict[str, str] = field(default_factory=lambda: {
        "vehicles": "/vehicles",
        "cars": "/cars",
        "motorcycles": "/motorcycles",
        "property": "/real-estate",
        "houses": "/houses-apartments-for-rent",
        "land": "/land-plots",
        "electronics": "/electronics",
        "phones": "/mobile-phones",
        "computers": "/computers-laptops",
        "home-garden": "/home-garden",
        "furniture": "/furniture",
        "appliances": "/home-appliances",
        "fashion": "/fashion",
        "clothes": "/clothing",
    })
    
    # Rate limiting
    requests_per_minute: float = 10


# =============================================================================
# Jiji Scraper
# =============================================================================

class JijiScraper(EcommerceScraper):
    """
    Jiji Kenya (jiji.co.ke) scraper.
    
    Jiji is Kenya's largest C2C marketplace, providing insight into:
    - Real market prices for used goods
    - Housing rental costs
    - Vehicle prices
    - Consumer electronics prices
    """
    
    def __init__(
        self,
        config: Optional[JijiScraperConfig] = None,
    ):
        super().__init__(config or JijiScraperConfig())
        self.config: JijiScraperConfig = self.config
        self._session = None
    
    @property
    def source(self) -> str:
        return "jiji"
    
    async def _initialize(self) -> None:
        """Initialize HTTP session and Playwright if needed."""
        try:
            import httpx
            from bs4 import BeautifulSoup
            
            self._session = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                follow_redirects=True,
                timeout=30.0,
            )
            
            logger.info("Jiji scraper initialized")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}. Run: pip install httpx beautifulsoup4")
            raise
    
    async def _scrape_category(
        self,
        category: str,
        limit: int,
    ) -> List[PriceData]:
        """Scrape products from a Jiji category."""
        from bs4 import BeautifulSoup
        
        # Get category URL
        category_path = self.config.category_urls.get(
            category.lower(), f"/{category}"
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
                
                # Find product listings
                listings = soup.select('div[data-testid="listing"]')
                if not listings:
                    # Try alternative selectors
                    listings = soup.select('.b-list-advert__item') or soup.select('article')
                
                if not listings:
                    logger.debug(f"No listings found on {page_url}")
                    break
                
                for listing in listings:
                    if len(results) >= limit:
                        break
                    
                    try:
                        price_data = self._parse_listing(listing, category, url)
                        if price_data:
                            results.append(price_data)
                    except Exception as e:
                        logger.debug(f"Failed to parse listing: {e}")
                        continue
                
                page += 1
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to scrape Jiji page {page_url}: {e}")
                break
        
        return results
    
    def _parse_listing(
        self,
        listing: Any,
        category: str,
        category_url: str,
    ) -> Optional[PriceData]:
        """Parse a Jiji listing element to PriceData."""
        try:
            # Extract title
            title_elem = (
                listing.select_one('[data-testid="title"]') or
                listing.select_one('.b-advert-title-inner') or
                listing.select_one('h2') or
                listing.select_one('a')
            )
            title = title_elem.get_text(strip=True) if title_elem else None
            
            if not title:
                return None
            
            # Extract price
            price_elem = (
                listing.select_one('[data-testid="price"]') or
                listing.select_one('.b-list-advert__price-text') or
                listing.select_one('.qa-advert-price')
            )
            price_text = price_elem.get_text(strip=True) if price_elem else None
            
            if not price_text:
                return None
            
            price = self._parse_price(price_text)
            if price is None or price <= 0:
                return None
            
            # Extract URL
            link_elem = listing.select_one('a[href]')
            product_url = ""
            product_id = ""
            if link_elem and link_elem.get('href'):
                href = link_elem['href']
                product_url = urljoin(self.config.base_url, href)
                # Extract ID from URL
                product_id = href.split('-')[-1].replace('.html', '') if href else str(hash(title))
            
            # Extract location
            location_elem = listing.select_one('.b-list-advert__region')
            seller_location = location_elem.get_text(strip=True) if location_elem else None
            
            # Map to economic domain
            domain = map_category_to_domain(category)
            
            return PriceData(
                source=self.source,
                product_url=product_url,
                product_id=product_id or str(hash(title)),
                product_name=title,
                category=category,
                price_kes=price,
                economic_domain=domain,
                seller_location=seller_location,
                scraped_at=datetime.utcnow(),
                raw_data={"listing_html": str(listing)[:500]},
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Jiji listing: {e}")
            return None
    
    def _parse_price(self, price_text: str) -> Optional[float]:
        """Parse Jiji price string to float."""
        if not price_text:
            return None
        
        # Clean the price string
        price_text = price_text.upper().replace('KSH', '').replace('KES', '')
        price_text = price_text.replace('SH', '').replace(',', '').replace(' ', '')
        
        # Handle K/M suffixes
        multiplier = 1
        if 'K' in price_text:
            multiplier = 1000
            price_text = price_text.replace('K', '')
        elif 'M' in price_text:
            multiplier = 1000000
            price_text = price_text.replace('M', '')
        
        # Extract numeric value
        match = re.search(r'[\d.]+', price_text)
        if match:
            try:
                return float(match.group()) * multiplier
            except ValueError:
                return None
        
        return None
    
    async def _scrape_product(
        self,
        url: str,
    ) -> Optional[PriceData]:
        """Scrape a single Jiji product page."""
        from bs4 import BeautifulSoup
        
        try:
            response = await self._session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.select_one('h1')
            title = title_elem.get_text(strip=True) if title_elem else None
            
            # Extract price
            price_elem = soup.select_one('.qa-advert-price')
            price_text = price_elem.get_text(strip=True) if price_elem else None
            price = self._parse_price(price_text) if price_text else None
            
            if not title or not price:
                return None
            
            # Extract category from breadcrumbs
            breadcrumb = soup.select_one('.b-breadcrumbs')
            category = "general"
            if breadcrumb:
                crumbs = breadcrumb.select('a')
                if len(crumbs) >= 2:
                    category = crumbs[1].get_text(strip=True).lower()
            
            # Extract seller info
            seller_elem = soup.select_one('.b-seller-info__name')
            seller_name = seller_elem.get_text(strip=True) if seller_elem else None
            
            return PriceData(
                source=self.source,
                product_url=url,
                product_id=url.split('-')[-1].replace('.html', ''),
                product_name=title,
                category=category,
                price_kes=price,
                economic_domain=map_category_to_domain(category),
                seller_name=seller_name,
                scraped_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.warning(f"Failed to scrape Jiji product {url}: {e}")
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

def create_jiji_scraper(
    categories: Optional[List[str]] = None,
) -> JijiScraper:
    """
    Create Jiji scraper with specified configuration.
    
    Args:
        categories: List of categories to monitor.
        
    Returns:
        Configured JijiScraper instance.
    """
    config = JijiScraperConfig(
        categories=categories or JijiScraperConfig().categories,
    )
    
    return JijiScraper(config)

"""
Telegram Scraper for KShield Pulse

Monitors Kenya-related public Telegram channels and groups using Telethon.

Usage:
    config = TelegramScraperConfig(
        api_id=12345,
        api_hash="your_api_hash",
        channels=["@kikimugo_ke", "@KenyaNews"],
    )
    
    async with TelegramScraper(config) as scraper:
        posts = await scraper.scrape("maandamano", limit=100)

Note:
    Requires Telegram API credentials from https://my.telegram.org
    Phone verification required on first use.
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
    AuthenticationError, Platform, RateLimiter, RetryPolicy,
)

logger = logging.getLogger("kshield.pulse.scrapers.telegram")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TelegramScraperConfig:
    """Configuration for Telegram scraper."""
    
    # Telegram API credentials (from https://my.telegram.org)
    api_id: int = 0
    api_hash: str = ""
    
    # Optional: session name for persistence
    session_name: str = "kshield_pulse"
    
    # Kenya-focused public channels
    channels: List[str] = field(default_factory=lambda: [
        # Add Kenya public channels here
        # Example: "@KenyaNewsUpdates", "@NairobiGossip"
    ])
    
    # Kenya-focused keywords for search
    kenya_keywords: List[str] = field(default_factory=lambda: [
        "Kenya", "Nairobi", "Mombasa", "Ruto", "Raila",
        "maandamano", "protest", "haki yetu", "strike",
        "unga", "fuel", "cost of living",
    ])
    
    # Rate limiting
    requests_per_minute: float = 20  # Telegram is sensitive


# =============================================================================
# Telegram Scraper Implementation
# =============================================================================

class TelegramScraper(BaseScraper):
    """
    Telegram scraper using Telethon.
    
    Features:
    - Monitors public channels
    - Searches messages by keyword
    - Handles message history retrieval
    - Respects Telegram's rate limits
    """
    
    def __init__(
        self,
        config: Optional[TelegramScraperConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        super().__init__(
            rate_limiter=rate_limiter or RateLimiter(
                requests_per_minute=config.requests_per_minute if config else 20
            ),
            retry_policy=retry_policy,
        )
        self.config = config or TelegramScraperConfig()
        self._client = None
        self._connected = False
    
    @property
    def platform(self) -> Platform:
        return Platform.TELEGRAM
    
    async def _initialize(self) -> None:
        """Initialize Telethon client."""
        if not self.config.api_id or not self.config.api_hash:
            logger.warning(
                "Telegram API credentials not configured. "
                "Get them from https://my.telegram.org"
            )
            return
        
        try:
            from telethon import TelegramClient
            from telethon.sessions import StringSession
            
            # Create client
            self._client = TelegramClient(
                self.config.session_name,
                self.config.api_id,
                self.config.api_hash,
            )
            
            # Connect (will prompt for phone on first use)
            await self._client.connect()
            
            if not await self._client.is_user_authorized():
                logger.warning(
                    "Telegram not authorized. Run interactively first "
                    "to complete phone verification."
                )
            else:
                self._connected = True
                logger.info("Telegram scraper initialized and connected")
            
        except ImportError:
            logger.warning("Telethon not installed. Run: pip install telethon")
        except Exception as e:
            logger.warning(f"Failed to initialize Telethon: {e}")
    
    async def has_api_credentials(self) -> bool:
        """Check if Telegram API credentials are configured."""
        return bool(self.config.api_id and self.config.api_hash)
    
    async def _scrape_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape Telegram channels for matching messages.
        
        Searches configured channels for messages matching query.
        """
        if not self._client or not self._connected:
            raise ScraperError(
                "Telegram not connected. Configure api_id and api_hash, "
                "then run interactively for phone verification."
            )
        
        results = []
        
        try:
            from telethon.tl.functions.messages import SearchRequest
            from telethon.tl.types import InputMessagesFilterEmpty
            
            # Search each configured channel
            for channel_name in self.config.channels:
                try:
                    channel = await self._client.get_entity(channel_name)
                    
                    # Get messages matching query
                    messages = await self._client.get_messages(
                        channel,
                        limit=limit,
                        search=query,
                        offset_date=since,
                    )
                    
                    for msg in messages:
                        result = self._parse_message(msg, channel_name)
                        if result:
                            results.append(result)
                            
                            if len(results) >= limit:
                                break
                    
                    logger.debug(f"Got {len(messages)} messages from {channel_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape {channel_name}: {e}")
                    continue
                
                if len(results) >= limit:
                    break
            
            logger.info(f"Telegram returned {len(results)} messages for '{query}'")
            return results
            
        except Exception as e:
            if "flood" in str(e).lower():
                raise RateLimitError(f"Telegram flood wait: {e}")
            raise ScraperError(f"Telegram scraping error: {e}")
    
    async def _scrape_via_api_impl(
        self,
        query: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Scrape using Telegram API (same as scrape_impl with Telethon).
        
        Telethon IS the API, so implementation is the same.
        """
        return await self._scrape_impl(query, limit, since)
    
    async def scrape_channels(
        self,
        limit_per_channel: int = 50,
        since: Optional[datetime] = None,
    ) -> List[ScraperResult]:
        """
        Get recent messages from all configured channels.
        
        Returns:
            List of recent messages from all channels.
        """
        if not self._client or not self._connected:
            await self.initialize()
            if not self._connected:
                return []
        
        results = []
        since = since or (datetime.utcnow() - timedelta(hours=24))
        
        for channel_name in self.config.channels:
            try:
                channel = await self._client.get_entity(channel_name)
                
                messages = await self._client.get_messages(
                    channel,
                    limit=limit_per_channel,
                    offset_date=since,
                )
                
                for msg in messages:
                    result = self._parse_message(msg, channel_name)
                    if result:
                        results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to get messages from {channel_name}: {e}")
                continue
        
        return results
    
    async def monitor_channel(
        self,
        channel_name: str,
    ):
        """
        Generator that yields new messages from a channel in real-time.
        
        Usage:
            async for message in scraper.monitor_channel("@KenyaNews"):
                process(message)
        """
        if not self._client or not self._connected:
            raise ScraperError("Telegram not connected")
        
        try:
            from telethon import events
            
            channel = await self._client.get_entity(channel_name)
            
            @self._client.on(events.NewMessage(chats=[channel]))
            async def handler(event):
                result = self._parse_message(event.message, channel_name)
                if result:
                    yield result
            
            # Keep running
            await self._client.run_until_disconnected()
            
        except Exception as e:
            raise ScraperError(f"Failed to monitor {channel_name}: {e}")
    
    def _parse_message(
        self, 
        message: Any, 
        channel_name: str,
    ) -> Optional[ScraperResult]:
        """Parse Telethon message to ScraperResult."""
        try:
            # Skip empty messages
            if not message.text:
                return None
            
            text = message.text
            
            # Extract entities
            urls = []
            hashtags = []
            mentions = []
            
            if hasattr(message, 'entities') and message.entities:
                for entity in message.entities:
                    entity_type = type(entity).__name__
                    
                    if 'Url' in entity_type:
                        url = text[entity.offset:entity.offset + entity.length]
                        urls.append(url)
                    elif 'Hashtag' in entity_type:
                        tag = text[entity.offset:entity.offset + entity.length]
                        hashtags.append(tag.lstrip('#'))
                    elif 'Mention' in entity_type:
                        mention = text[entity.offset:entity.offset + entity.length]
                        mentions.append(mention.lstrip('@'))
            
            # Get sender info
            sender_id = None
            sender_username = None
            if hasattr(message, 'sender') and message.sender:
                sender_id = str(message.sender.id)
                sender_username = getattr(message.sender, 'username', None)
            
            return ScraperResult(
                platform=Platform.TELEGRAM,
                platform_id=str(message.id),
                text=text,
                author_id=sender_id,
                author_username=sender_username,
                likes=getattr(message, 'forwards', 0) or 0,
                shares=0,
                replies=getattr(message, 'replies', {}).get('replies', 0) if hasattr(message, 'replies') and message.replies else 0,
                views=getattr(message, 'views', None),
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                reply_to_id=str(message.reply_to_msg_id) if hasattr(message, 'reply_to_msg_id') and message.reply_to_msg_id else None,
                posted_at=message.date.replace(tzinfo=None) if message.date else datetime.utcnow(),
                scraped_at=datetime.utcnow(),
                raw_data={
                    "source": "telethon",
                    "channel": channel_name,
                    "message_id": message.id,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse Telegram message: {e}")
            return None
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.disconnect()
            self._client = None
            self._connected = False


# =============================================================================
# Factory Function
# =============================================================================

def create_telegram_scraper(
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    channels: Optional[List[str]] = None,
) -> TelegramScraper:
    """
    Create Telegram scraper with specified configuration.
    
    Args:
        api_id: Telegram API ID.
        api_hash: Telegram API hash.
        channels: List of public channels to monitor.
        
    Returns:
        Configured TelegramScraper instance.
    """
    config = TelegramScraperConfig(
        api_id=api_id or 0,
        api_hash=api_hash or "",
        channels=channels or [],
    )
    
    return TelegramScraper(config)

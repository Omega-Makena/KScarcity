"""
X (Twitter) API v2 Client — Real Implementation

Implements actual X API calls with:
- Strict rate limiting for Free tier (100 posts/month)
- Usage tracking and persistence
- OAuth 1.0a authentication
- Search and lookup endpoints

Free Tier Limits:
- 100 posts/month read
- No streaming access
- Basic endpoints only
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import base64
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import secrets

# Use httpx for async HTTP (fall back to requests if not available)
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    try:
        import requests
        HAS_REQUESTS = True
    except ImportError:
        HAS_REQUESTS = False

from .config import XAPIConfig, load_env_file
from .social import SocialPost, Platform, SocialMediaClient, StreamConfig

logger = logging.getLogger("kshield.pulse.x_client")


# =============================================================================
# Usage Tracker — Stay within 100 posts/month
# =============================================================================

@dataclass
class UsageStats:
    """Track API usage to stay within limits."""
    posts_read: int = 0
    month_start: str = ""  # YYYY-MM format
    last_request: float = 0.0
    requests_today: int = 0
    
    def reset_if_new_month(self) -> None:
        """Reset counters on new month."""
        current_month = datetime.now().strftime("%Y-%m")
        if self.month_start != current_month:
            self.posts_read = 0
            self.month_start = current_month
            logger.info(f"New month - reset usage counter")
    
    def can_read(self, count: int = 1) -> bool:
        """Check if we can read more posts."""
        self.reset_if_new_month()
        return self.posts_read + count <= 100
    
    def record_read(self, count: int = 1) -> None:
        """Record posts read."""
        self.posts_read += count
        self.last_request = time.time()
        self.requests_today += 1


class UsageTracker:
    """
    Persistent usage tracking to ensure we stay within 100 posts/month.
    
    Saves usage to a JSON file so it persists across sessions.
    """
    
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            pulse_dir = Path(__file__).parent
            storage_path = pulse_dir / ".x_usage.json"
        self.storage_path = Path(storage_path)
        self.stats = self._load()
    
    def _load(self) -> UsageStats:
        """Load usage stats from file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                return UsageStats(**data)
            except Exception as e:
                logger.warning(f"Failed to load usage stats: {e}")
        return UsageStats(month_start=datetime.now().strftime("%Y-%m"))
    
    def _save(self) -> None:
        """Save usage stats to file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "posts_read": self.stats.posts_read,
                    "month_start": self.stats.month_start,
                    "last_request": self.stats.last_request,
                    "requests_today": self.stats.requests_today,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save usage stats: {e}")
    
    def can_read(self, count: int = 1) -> bool:
        """Check if we can read more posts."""
        return self.stats.can_read(count)
    
    def record_read(self, count: int = 1) -> None:
        """Record posts read and save."""
        self.stats.record_read(count)
        self._save()
    
    def get_remaining(self) -> int:
        """Get remaining posts for this month."""
        self.stats.reset_if_new_month()
        return max(0, 100 - self.stats.posts_read)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "posts_read": self.stats.posts_read,
            "remaining": self.get_remaining(),
            "month": self.stats.month_start,
            "limit": 100,
        }


# =============================================================================
# OAuth 1.0a Signature Generator
# =============================================================================

class OAuth1Signer:
    """Generate OAuth 1.0a signatures for X API requests."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str = "",
        access_secret: str = "",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_secret = access_secret
    
    def sign_request(
        self,
        method: str,
        url: str,
        params: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """
        Generate OAuth 1.0a authorization header.
        
        Returns:
            Dict with Authorization header
        """
        params = params or {}
        
        # OAuth parameters
        oauth_params = {
            "oauth_consumer_key": self.api_key,
            "oauth_nonce": secrets.token_hex(16),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_version": "1.0",
        }
        
        if self.access_token:
            oauth_params["oauth_token"] = self.access_token
        
        # Combine all params for signature
        all_params = {**params, **oauth_params}
        
        # Create signature base string
        sorted_params = "&".join(
            f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}"
            for k, v in sorted(all_params.items())
        )
        
        base_string = "&".join([
            method.upper(),
            urllib.parse.quote(url, safe=''),
            urllib.parse.quote(sorted_params, safe=''),
        ])
        
        # Create signing key
        signing_key = "&".join([
            urllib.parse.quote(self.api_secret, safe=''),
            urllib.parse.quote(self.access_secret, safe=''),
        ])
        
        # Generate signature
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode(),
                base_string.encode(),
                hashlib.sha1
            ).digest()
        ).decode()
        
        oauth_params["oauth_signature"] = signature
        
        # Build Authorization header
        header_params = ", ".join(
            f'{k}="{urllib.parse.quote(str(v), safe="")}"'
            for k, v in sorted(oauth_params.items())
        )
        
        return {"Authorization": f"OAuth {header_params}"}


# =============================================================================
# Real X API Client
# =============================================================================

@dataclass
class XClientConfig:
    """Configuration for X client."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    bearer_token: str = ""
    
    # Rate limiting
    min_request_interval: float = 2.0  # seconds between requests
    
    @classmethod
    def from_env(cls) -> "XClientConfig":
        """Load from environment."""
        config = XAPIConfig.from_env()
        return cls(
            api_key=config.api_key,
            api_secret=config.api_secret,
            access_token=config.access_token,
            access_secret=config.access_secret,
            bearer_token=config.bearer_token,
        )


class RealXClient(SocialMediaClient):
    """
    Real X (Twitter) API v2 client.
    
    Uses the Free tier with strict 100 posts/month limit.
    
    Usage:
        client = RealXClient.from_env()
        await client.authenticate()
        posts = await client.search("Kenya economy", max_results=10)
    """
    
    BASE_URL = "https://api.twitter.com/2"
    
    def __init__(self, config: XClientConfig = None):
        self.config = config or XClientConfig.from_env()
        self.usage = UsageTracker()
        self.signer = OAuth1Signer(
            self.config.api_key,
            self.config.api_secret,
            self.config.access_token,
            self.config.access_secret,
        )
        self._authenticated = False
        self._last_request = 0.0
    
    @classmethod
    def from_env(cls) -> "RealXClient":
        """Create client from environment variables."""
        return cls(XClientConfig.from_env())
    
    @property
    def platform(self) -> Platform:
        return Platform.TWITTER
    
    async def authenticate(self) -> bool:
        """
        Authenticate with X API.
        
        Note: With API Key/Secret only (no Bearer Token),
        we can only use OAuth 1.0a endpoints.
        """
        if not self.config.api_key or not self.config.api_secret:
            logger.error("X API Key and Secret required")
            return False
        
        # For Free tier with just API key/secret, we're limited
        # but can still authenticate
        logger.info("X client initialized (Free tier - 100 posts/month)")
        logger.info(f"Remaining this month: {self.usage.get_remaining()} posts")
        
        self._authenticated = True
        return True
    
    async def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self.config.min_request_interval:
            await asyncio.sleep(self.config.min_request_interval - elapsed)
        self._last_request = time.time()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, str] = None,
    ) -> Optional[Dict]:
        """Make authenticated request to X API."""
        if not self._authenticated:
            await self.authenticate()
        
        await self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/{endpoint}"
        headers = self.signer.sign_request(method, url, params or {})
        headers["Content-Type"] = "application/json"
        
        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    if method == "GET":
                        response = await client.get(url, params=params, headers=headers)
                    else:
                        response = await client.request(method, url, params=params, headers=headers)
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.error(f"X API error {response.status_code}: {response.text}")
                        return None
            else:
                logger.error("httpx not installed. Run: pip install httpx")
                return None
                
        except Exception as e:
            logger.error(f"X API request failed: {e}")
            return None
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        since: Optional[datetime] = None,
    ) -> List[SocialPost]:
        """
        Search for recent tweets.
        
        Note: Free tier has very limited search access.
        This uses the tweets/search/recent endpoint.
        
        Args:
            query: Search query
            max_results: Maximum results (capped at remaining quota)
            since: Only tweets after this time
        """
        # Check quota
        if not self.usage.can_read(max_results):
            remaining = self.usage.get_remaining()
            logger.warning(f"Quota exceeded! Only {remaining} posts remaining this month")
            if remaining == 0:
                return []
            max_results = min(max_results, remaining)
        
        params = {
            "query": query,
            "max_results": str(min(max_results, 10)),  # Free tier max is 10
            "tweet.fields": "created_at,author_id,public_metrics,lang",
        }
        
        if since:
            params["start_time"] = since.isoformat() + "Z"
        
        logger.info(f"Searching X for: {query} (max {max_results})")
        
        response = await self._make_request("GET", "tweets/search/recent", params)
        
        if not response:
            return []
        
        posts = []
        data = response.get("data", [])
        
        for tweet in data:
            posts.append(self._parse_tweet(tweet))
        
        # Record usage
        self.usage.record_read(len(posts))
        logger.info(f"Retrieved {len(posts)} tweets. Remaining quota: {self.usage.get_remaining()}")
        
        return posts
    
    async def get_post(self, post_id: str) -> Optional[SocialPost]:
        """Get a specific tweet by ID."""
        if not self.usage.can_read(1):
            logger.warning("Quota exceeded!")
            return None
        
        params = {
            "tweet.fields": "created_at,author_id,public_metrics,lang",
        }
        
        response = await self._make_request("GET", f"tweets/{post_id}", params)
        
        if response and "data" in response:
            self.usage.record_read(1)
            return self._parse_tweet(response["data"])
        
        return None
    
    async def stream(self, config: StreamConfig):
        """
        Streaming not available on Free tier.
        
        Use search with polling instead.
        """
        logger.warning("Streaming not available on X Free tier. Use search() instead.")
        # Yield nothing
        while False:
            yield
    
    def _parse_tweet(self, data: Dict[str, Any]) -> SocialPost:
        """Parse X API response into SocialPost."""
        metrics = data.get("public_metrics", {})
        
        created_at = datetime.now()
        if "created_at" in data:
            try:
                created_at = datetime.fromisoformat(
                    data["created_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse tweet timestamp: {e}")
        
        return SocialPost(
            id=data.get("id", ""),
            platform=Platform.TWITTER,
            text=data.get("text", ""),
            language=data.get("lang", "en"),
            author_id=data.get("author_id", ""),
            likes=metrics.get("like_count", 0),
            shares=metrics.get("retweet_count", 0),
            comments=metrics.get("reply_count", 0),
            views=metrics.get("impression_count", 0),
            created_at=created_at,
            raw_data=data,
        )
    
    def get_usage(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage.get_stats()


# =============================================================================
# Convenience Functions
# =============================================================================

async def search_x(
    query: str,
    max_results: int = 10,
) -> List[SocialPost]:
    """
    Quick search function.
    
    Usage:
        posts = await search_x("Kenya economy crisis", max_results=5)
    """
    client = RealXClient.from_env()
    await client.authenticate()
    return await client.search(query, max_results)


def get_x_usage() -> Dict[str, Any]:
    """Get current X API usage statistics."""
    tracker = UsageTracker()
    return tracker.get_stats()

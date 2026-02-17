"""
Pulse Engine Configuration

Loads API credentials and configuration from environment variables
or .env files. NEVER hardcode credentials in source code.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kshield.pulse.config")


def load_env_file(env_path: str = None) -> None:
    """
    Load environment variables from .env file.
    
    Simple implementation without external dependencies.
    For production, consider using python-dotenv.
    """
    if env_path is None:
        # Look for .env in pulse directory
        pulse_dir = Path(__file__).parent
        env_path = pulse_dir / ".env"
    else:
        env_path = Path(env_path)
    
    if not env_path.exists():
        logger.debug(f"No .env file found at {env_path}")
        return
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Don't override existing env vars
                    if key not in os.environ:
                        os.environ[key] = value
        logger.info(f"Loaded environment from {env_path}")
    except Exception as e:
        logger.warning(f"Failed to load .env file: {e}")


@dataclass
class XAPIConfig:
    """X (Twitter) API configuration."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    bearer_token: str = ""
    
    @classmethod
    def from_env(cls) -> "XAPIConfig":
        """Load from environment variables."""
        load_env_file()
        return cls(
            api_key=os.getenv("X_API_KEY", ""),
            api_secret=os.getenv("X_API_SECRET", ""),
            access_token=os.getenv("X_ACCESS_TOKEN", ""),
            access_secret=os.getenv("X_ACCESS_SECRET", ""),
            bearer_token=os.getenv("X_BEARER_TOKEN", ""),
        )
    
    def is_configured(self) -> bool:
        """Check if basic credentials are set."""
        return bool(self.api_key and self.api_secret)


@dataclass
class TikTokAPIConfig:
    """TikTok API configuration."""
    client_key: str = ""
    client_secret: str = ""
    access_token: str = ""
    
    @classmethod
    def from_env(cls) -> "TikTokAPIConfig":
        load_env_file()
        return cls(
            client_key=os.getenv("TIKTOK_CLIENT_KEY", ""),
            client_secret=os.getenv("TIKTOK_CLIENT_SECRET", ""),
            access_token=os.getenv("TIKTOK_ACCESS_TOKEN", ""),
        )


@dataclass
class InstagramAPIConfig:
    """Instagram API configuration."""
    access_token: str = ""
    app_id: str = ""
    app_secret: str = ""
    
    @classmethod
    def from_env(cls) -> "InstagramAPIConfig":
        load_env_file()
        return cls(
            access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN", ""),
            app_id=os.getenv("INSTAGRAM_APP_ID", ""),
            app_secret=os.getenv("INSTAGRAM_APP_SECRET", ""),
        )


@dataclass  
class PulseConfig:
    """
    Master configuration for Pulse Engine.
    
    Usage:
        config = PulseConfig.from_env()
        if config.x_api.is_configured():
            client = TwitterClient(config.x_api)
    """
    x_api: XAPIConfig = field(default_factory=XAPIConfig)
    tiktok_api: TikTokAPIConfig = field(default_factory=TikTokAPIConfig)
    instagram_api: InstagramAPIConfig = field(default_factory=InstagramAPIConfig)
    
    # Sensor settings
    use_nlp: bool = True
    min_intensity_threshold: float = 0.1
    min_confidence_threshold: float = 0.3
    
    # Bridge settings
    shock_interval_seconds: float = 300
    enable_probabilistic_shocks: bool = True
    
    @classmethod
    def from_env(cls) -> "PulseConfig":
        """Load all configuration from environment."""
        return cls(
            x_api=XAPIConfig.from_env(),
            tiktok_api=TikTokAPIConfig.from_env(),
            instagram_api=InstagramAPIConfig.from_env(),
            use_nlp=os.getenv("PULSE_USE_NLP", "true").lower() == "true",
            min_intensity_threshold=float(os.getenv("PULSE_MIN_INTENSITY", "0.1")),
            min_confidence_threshold=float(os.getenv("PULSE_MIN_CONFIDENCE", "0.3")),
            shock_interval_seconds=float(os.getenv("PULSE_SHOCK_INTERVAL", "300")),
            enable_probabilistic_shocks=os.getenv("PULSE_PROBABILISTIC", "true").lower() == "true",
        )


# =============================================================================
# New Scraper Configs (for ingestion pipeline)
# =============================================================================

@dataclass
class RedditAPIConfig:
    """Reddit API configuration."""
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "KShieldPulse/1.0"
    subreddits: list = field(default_factory=lambda: ["Kenya", "NairobiCity", "africa"])
    
    @classmethod
    def from_env(cls) -> "RedditAPIConfig":
        load_env_file()
        return cls(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.getenv("REDDIT_USER_AGENT", "KShieldPulse/1.0"),
        )
    
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


@dataclass
class TelegramAPIConfig:
    """Telegram API configuration."""
    api_id: int = 0
    api_hash: str = ""
    phone: str = ""
    channels: list = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> "TelegramAPIConfig":
        load_env_file()
        return cls(
            api_id=int(os.getenv("TELEGRAM_API_ID", "0")),
            api_hash=os.getenv("TELEGRAM_API_HASH", ""),
            phone=os.getenv("TELEGRAM_PHONE", ""),
        )
    
    def is_configured(self) -> bool:
        return bool(self.api_id and self.api_hash)


@dataclass
class FacebookAPIConfig:
    """Facebook/Meta API configuration."""
    email: str = ""
    password: str = ""
    pages: list = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> "FacebookAPIConfig":
        load_env_file()
        return cls(
            email=os.getenv("FACEBOOK_EMAIL", ""),
            password=os.getenv("FACEBOOK_PASSWORD", ""),
        )
    
    def is_configured(self) -> bool:
        return bool(self.email and self.password)


@dataclass
class GeminiAPIConfig:
    """Google Gemini API configuration."""
    api_key: str = ""
    model: str = "gemini-1.5-flash"
    temperature: float = 0.3
    
    @classmethod
    def from_env(cls) -> "GeminiAPIConfig":
        load_env_file()
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        )
    
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = ""
    pool_size: int = 5
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        load_env_file()
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///pulse.db"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        )


@dataclass
class NewsAPIConfig:
    """NewsAPI.org configuration."""
    api_key: str = "9b404b4922ed4ff7a71c9f2247b5c722"  # Default confirmed key
    
    @classmethod
    def from_env(cls) -> "NewsAPIConfig":
        load_env_file()
        return cls(
            api_key=os.getenv("NEWS_API_KEY", "9b404b4922ed4ff7a71c9f2247b5c722"),
        )
    
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class OllamaConfig:
    """Ollama API configuration."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout: int = 120


@dataclass
class ScraperConfig:
    """
    Unified configuration for all scrapers.
    
    Usage:
        config = ScraperConfig.from_env()
        if config.reddit.is_configured():
            scraper = RedditScraper(config.reddit)
    """
    # Social media
    x_api: XAPIConfig = field(default_factory=XAPIConfig)
    reddit: RedditAPIConfig = field(default_factory=RedditAPIConfig)
    telegram: TelegramAPIConfig = field(default_factory=TelegramAPIConfig)
    instagram: InstagramAPIConfig = field(default_factory=InstagramAPIConfig)
    facebook: FacebookAPIConfig = field(default_factory=FacebookAPIConfig)
    news_api: NewsAPIConfig = field(default_factory=NewsAPIConfig)
    
    # LLM
    gemini: GeminiAPIConfig = field(default_factory=GeminiAPIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    
    # Database
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Scraping settings
    social_scrape_interval_minutes: int = 30
    ecommerce_scrape_interval_hours: int = 6
    max_posts_per_scrape: int = 100
    max_products_per_category: int = 50
    
    @classmethod
    def from_env(cls) -> "ScraperConfig":
        """Load all scraper configuration from environment."""
        return cls(
            x_api=XAPIConfig.from_env(),
            reddit=RedditAPIConfig.from_env(),
            telegram=TelegramAPIConfig.from_env(),
            instagram=InstagramAPIConfig.from_env(),
            facebook=FacebookAPIConfig.from_env(),
            news_api=NewsAPIConfig.from_env(),
            gemini=GeminiAPIConfig.from_env(),
            database=DatabaseConfig.from_env(),
            social_scrape_interval_minutes=int(os.getenv("SCRAPE_INTERVAL_MINUTES", "30")),
            ecommerce_scrape_interval_hours=int(os.getenv("ECOMMERCE_INTERVAL_HOURS", "6")),
            max_posts_per_scrape=int(os.getenv("MAX_POSTS_PER_SCRAPE", "100")),
            max_products_per_category=int(os.getenv("MAX_PRODUCTS_PER_CATEGORY", "50")),
        )


# Convenience functions
def get_config() -> PulseConfig:
    """Get Pulse configuration from environment."""
    return PulseConfig.from_env()


def get_scraper_config() -> ScraperConfig:
    """Get unified scraper configuration from environment."""
    return ScraperConfig.from_env()


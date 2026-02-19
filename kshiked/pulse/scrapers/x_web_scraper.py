"""
X (Twitter) Web Scraper for KShield Pulse — Kenya Focus

Production-grade Twitter scraper using `twikit` (Twitter's internal API)
with built-in Kenya location extraction and account data capture.

Features:
- No official API key required (uses guest/session auth)
- Kenya geo-filter: hardcoded county list + NLP location extraction
- Account data: handle, display name, followers, verified status, bio, location
- Outputs to CSV matching the dashboard schema
- Rate-limit aware with exponential backoff
- Multi-session cookie + proxy rotation
- Checkpoint/resume support (tweet IDs + query cursor/page progress)
- Deduplication by tweet ID

Architecture:
    twikit authenticates via Twitter's internal login flow and keeps
    session cookies.  First run requires username+password; subsequent
    runs reuse saved cookies from `data/pulse/.x_cookies.json`.

Usage (standalone):
    python -m kshiked.pulse.scrapers.x_web_scraper --user <handle> --pw <pass>

Usage (library):
    from kshiked.pulse.scrapers.x_web_scraper import KenyaXScraper

    async with KenyaXScraper("handle", "password") as scraper:
        tweets, accounts = await scraper.scrape_kenya(limit=200)
        scraper.save_tweets_csv(tweets)
        scraper.save_accounts_csv(accounts)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("kshield.pulse.scrapers.x_web")

# ─── Project paths ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"
_PULSE_DIR = _DATA_DIR / "pulse"
_COOKIE_PATH = _PULSE_DIR / ".x_cookies.json"
_CHECKPOINT_PATH = _PULSE_DIR / "x_scraper_checkpoint.json"
_TWEETS_OUT = _DATA_DIR / "pulse" / "x_kenya_tweets.csv"
_ACCOUNTS_OUT = _DATA_DIR / "pulse" / "x_kenya_accounts.csv"


def _split_csv_env(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_path(raw_path: str, default_root: Path = _PROJECT_ROOT) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = default_root / path
    return path


def _load_fallback_credentials() -> Dict[str, str]:
    """Load X credentials from backend helper when available, else env vars."""
    username = os.getenv("X_USERNAME", "").strip()
    password = os.getenv("X_PASSWORD", "").strip()
    email = os.getenv("X_EMAIL", "").strip()

    try:
        from backend.secrets.x_credentials import get_x_credentials  # type: ignore
    except Exception:
        return {"username": username, "password": password, "email": email}

    try:
        creds = get_x_credentials() or {}
    except Exception:
        creds = {}

    return {
        "username": username or str(creds.get("username", "")).strip(),
        "password": password or str(creds.get("password", "")).strip(),
        "email": email or str(creds.get("email", "")).strip(),
    }


def _derive_cookie_path(base_path: Path, session_number: int) -> Path:
    if session_number <= 1:
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    return base_path.with_name(f"{stem}_{session_number}{suffix}")


# ═══════════════════════════════════════════════════════════════════════════
# Kenya Location Extraction
# ═══════════════════════════════════════════════════════════════════════════

KENYA_COUNTIES: Dict[str, Tuple[float, float]] = {
    "Baringo": (0.4919, 35.9722),
    "Bomet": (-0.7813, 35.3416),
    "Bungoma": (0.5695, 34.5584),
    "Busia": (0.4608, 34.1115),
    "Elgeyo-Marakwet": (0.8047, 35.5088),
    "Embu": (-0.5389, 37.4596),
    "Garissa": (-0.4533, 39.6461),
    "Homa Bay": (-0.5273, 34.4571),
    "Isiolo": (0.3546, 37.5822),
    "Kajiado": (-1.8524, 36.7768),
    "Kakamega": (0.2827, 34.7519),
    "Kericho": (-0.3692, 35.2863),
    "Kiambu": (-1.1714, 36.8356),
    "Kilifi": (-3.6305, 39.8499),
    "Kirinyaga": (-0.6591, 37.3827),
    "Kisii": (-0.6816, 34.7668),
    "Kisumu": (-0.1022, 34.7617),
    "Kitui": (-1.3668, 38.0106),
    "Kwale": (-4.1816, 39.4521),
    "Laikipia": (0.3606, 36.7819),
    "Lamu": (-2.2717, 40.9020),
    "Machakos": (-1.5177, 37.2634),
    "Makueni": (-1.8039, 37.6200),
    "Mandera": (3.9373, 41.8569),
    "Marsabit": (2.3284, 37.9911),
    "Meru": (0.0515, 37.6493),
    "Migori": (-1.0634, 34.4731),
    "Mombasa": (-4.0435, 39.6682),
    "Murang'a": (-0.7839, 37.0400),
    "Nairobi": (-1.2921, 36.8219),
    "Nakuru": (-0.3031, 36.0666),
    "Nandi": (0.1836, 35.1269),
    "Narok": (-1.0875, 35.8714),
    "Nyamira": (-0.5633, 34.9348),
    "Nyandarua": (-0.1804, 36.5231),
    "Nyeri": (-0.4246, 36.9514),
    "Samburu": (1.2128, 36.9540),
    "Siaya": (-0.0617, 34.2422),
    "Taita-Taveta": (-3.3161, 38.4850),
    "Tana River": (-1.6471, 40.0030),
    "Tharaka-Nithi": (-0.3074, 37.8019),
    "Trans-Nzoia": (1.0567, 34.9507),
    "Turkana": (3.1067, 35.5918),
    "Uasin Gishu": (0.5528, 35.3027),
    "Vihiga": (0.0834, 34.7072),
    "Wajir": (1.7471, 40.0573),
    "West Pokot": (1.6210, 35.3905),
}

# Major towns / landmarks mapped to their county
_TOWN_TO_COUNTY: Dict[str, str] = {
    "nairobi": "Nairobi",
    "nrb": "Nairobi",
    "nbi": "Nairobi",
    "westlands": "Nairobi",
    "kibera": "Nairobi",
    "langata": "Nairobi",
    "kasarani": "Nairobi",
    "eastleigh": "Nairobi",
    "cbd": "Nairobi",
    "uhuru gardens": "Nairobi",
    "mombasa": "Mombasa",
    "mvita": "Mombasa",
    "nyali": "Mombasa",
    "likoni": "Mombasa",
    "kisumu": "Kisumu",
    "nakuru": "Nakuru",
    "eldoret": "Uasin Gishu",
    "thika": "Kiambu",
    "juja": "Kiambu",
    "ruiru": "Kiambu",
    "kikuyu": "Kiambu",
    "limuru": "Kiambu",
    "malindi": "Kilifi",
    "diani": "Kwale",
    "nanyuki": "Laikipia",
    "nyahururu": "Nyandarua",
    "naivasha": "Nakuru",
    "kitale": "Trans-Nzoia",
    "machakos": "Machakos",
    "athi river": "Machakos",
    "garissa": "Garissa",
    "lodwar": "Turkana",
    "lamu": "Lamu",
    "meru": "Meru",
    "embu": "Embu",
    "nyeri": "Nyeri",
    "karatina": "Nyeri",
    "murang'a": "Murang'a",
    "kericho": "Kericho",
    "bomet": "Bomet",
    "narok": "Narok",
    "maasai mara": "Narok",
    "migori": "Migori",
    "homa bay": "Homa Bay",
    "siaya": "Siaya",
    "bungoma": "Bungoma",
    "busia": "Busia",
    "kakamega": "Kakamega",
    "vihiga": "Vihiga",
    "kisii": "Kisii",
    "isiolo": "Isiolo",
    "marsabit": "Marsabit",
    "mandera": "Mandera",
    "wajir": "Wajir",
    "kitui": "Kitui",
    "makueni": "Makueni",
    "kajiado": "Kajiado",
    "ngong": "Kajiado",
    "rongai": "Kajiado",
    "kilifi": "Kilifi",
    "watamu": "Kilifi",
    "kwale": "Kwale",
    "taveta": "Taita-Taveta",
    "voi": "Taita-Taveta",
    "samburu": "Samburu",
    "baringo": "Baringo",
    "nandi": "Nandi",
    "nyamira": "Nyamira",
    "kirinyaga": "Kirinyaga",
    "kerugoya": "Kirinyaga",
}

# Pre-compile location regexes (county names + towns, case-insensitive)
_LOCATION_PATTERNS: List[Tuple[re.Pattern, str]] = []
# Add county patterns (longer first to match "Taita-Taveta" before "Taita")
for _county in sorted(KENYA_COUNTIES, key=len, reverse=True):
    _LOCATION_PATTERNS.append(
        (re.compile(r"\b" + re.escape(_county) + r"\b", re.IGNORECASE), _county)
    )
# Add town patterns
for _town, _county in sorted(_TOWN_TO_COUNTY.items(), key=lambda x: len(x[0]), reverse=True):
    _LOCATION_PATTERNS.append(
        (re.compile(r"\b" + re.escape(_town) + r"\b", re.IGNORECASE), _county)
    )


def extract_kenya_locations(text: str) -> List[str]:
    """Extract Kenya county names from text using pattern matching.

    Returns a deduplicated list of county names found in the text.
    """
    found: Dict[str, None] = {}  # ordered set
    for pattern, county in _LOCATION_PATTERNS:
        if pattern.search(text):
            found[county] = None
    return list(found.keys())


def extract_primary_county(text: str, user_location: str = "") -> Tuple[str, float, float]:
    """Return (county, lat, lon) from text + user bio location.

    Falls back to '' and 0.0, 0.0 if no location detected.
    """
    combined = f"{text} {user_location}"
    counties = extract_kenya_locations(combined)
    if counties:
        county = counties[0]
        lat, lon = KENYA_COUNTIES[county]
        return county, lat, lon
    return "", 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScrapedTweet:
    """A single scraped tweet with all extracted metadata."""
    tweet_id: str
    text: str
    created_at: datetime
    language: str = "en"

    # Engagement
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    view_count: int = 0
    bookmark_count: int = 0

    # Author snapshot (denormalised for CSV convenience)
    author_id: str = ""
    author_username: str = ""
    author_display_name: str = ""
    author_followers: int = 0
    author_following: int = 0
    author_verified: bool = False
    author_bio: str = ""
    author_location: str = ""
    author_created_at: str = ""
    author_tweet_count: int = 0
    author_profile_image: str = ""

    # Thread context
    reply_to_tweet_id: str = ""
    reply_to_user: str = ""
    conversation_id: str = ""
    is_retweet: bool = False
    is_quote: bool = False

    # Extracted location
    location_county: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    mentioned_counties: str = ""  # comma-separated

    # Content features
    hashtags: str = ""  # comma-separated
    mentions: str = ""  # comma-separated
    urls: str = ""  # comma-separated
    media_urls: str = ""  # comma-separated

    # Metadata
    source: str = "twikit"
    scraped_at: str = ""


@dataclass
class ScrapedAccount:
    """A Twitter/X account profile captured during scraping."""
    user_id: str
    username: str
    display_name: str
    bio: str = ""
    location: str = ""
    website: str = ""
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0
    verified: bool = False
    created_at: str = ""
    profile_image_url: str = ""
    profile_banner_url: str = ""

    # Kenya-specific
    is_kenyan: bool = False
    detected_county: str = ""

    # Scrape metadata
    scraped_at: str = ""
    scrape_source: str = "twikit"


@dataclass
class XSessionConfig:
    """One Twikit session, bound to an optional proxy and cookie file."""

    name: str
    username: str
    password: str
    email: str
    cookie_path: Path
    proxy: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Kenya-focused search queries
# ═══════════════════════════════════════════════════════════════════════════

KENYA_SEARCH_QUERIES: List[str] = [
    # General Kenya
    "Kenya",
    "Nairobi",
    "Kenya news",
    # Politics
    "Ruto",
    "Raila",
    "Kenya politics",
    "parliament Kenya",
    "Kenya Kwanza",
    "Azimio",
    # Economy
    "cost of living Kenya",
    "unga prices Kenya",
    "fuel prices Kenya",
    "Kenya economy",
    "KRA tax",
    # Social issues
    "maandamano",
    "hustler Kenya",
    "NHIF SHIF",
    "housing levy Kenya",
    "gen z kenya",
    # Security
    "Kenya security",
    "police Kenya",
    # Counties
    "Mombasa",
    "Kisumu",
    "Nakuru",
    "Eldoret",
]


# ═══════════════════════════════════════════════════════════════════════════
# Core Scraper
# ═══════════════════════════════════════════════════════════════════════════

class KenyaXScraper:
    """
    Production-grade X scraper using twikit, focused on Kenya.

    Handles:
    - Authentication via username/password with cookie persistence
    - Search with Kenya-focused queries
    - User timeline scraping
    - Location extraction from tweet text + user profiles
    - Account data capture
    - CSV output in dashboard-compatible schema
    - Rate-limit backoff
    """

    def __init__(
        self,
        username: str = "",
        password: str = "",
        email: str = "",
        cookie_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        session_configs: Optional[List[XSessionConfig]] = None,
        proxies: Optional[List[str]] = None,
        session_cookie_paths: Optional[List[Path]] = None,
        checkpoint_path: Optional[Path] = None,
        enable_checkpoint: bool = True,
        resume_from_checkpoint: bool = False,
        checkpoint_every_pages: int = 1,
        rotate_on_rate_limit: bool = True,
        rotate_on_detection: bool = True,
        request_delay_s: float = 1.2,
        query_delay_s: float = 3.0,
        request_jitter_s: float = 0.0,
        query_jitter_s: float = 0.0,
        detection_cooldown_hours: float = 0.0,
        wait_if_cooldown_active: bool = False,
        client_factory: Optional[Callable[[str], Any]] = None,
    ):
        if not (username and password and email):
            creds = _load_fallback_credentials()
            username = username or creds["username"]
            password = password or creds["password"]
            email = email or creds["email"]
        self.username = username
        self.password = password
        self.email = email
        self.cookie_path = cookie_path or _COOKIE_PATH
        self.output_dir = output_dir or _PULSE_DIR

        self.session_configs = session_configs or self._build_default_session_configs(
            base_cookie_path=self.cookie_path,
            proxies=proxies,
            session_cookie_paths=session_cookie_paths,
        )
        self._client_factory = client_factory
        self._clients: List[Any] = []
        self._active_session_configs: List[XSessionConfig] = []
        self._active_session_idx = 0
        self._client = None

        self.enable_checkpoint = enable_checkpoint
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_every_pages = max(1, int(checkpoint_every_pages))
        self.checkpoint_path = checkpoint_path or _CHECKPOINT_PATH
        self._query_progress: Dict[str, Dict[str, Any]] = {}
        self._current_query_index = 0

        self.rotate_on_rate_limit = rotate_on_rate_limit
        self.rotate_on_detection = rotate_on_detection
        self.request_delay_s = max(0.0, float(request_delay_s))
        self.query_delay_s = max(0.0, float(query_delay_s))
        self.request_jitter_s = max(0.0, float(request_jitter_s))
        self.query_jitter_s = max(0.0, float(query_jitter_s))
        self.detection_cooldown_hours = max(0.0, float(detection_cooldown_hours))
        self.wait_if_cooldown_active = bool(wait_if_cooldown_active)
        self._cooldown_until = ""
        self._abort_scrape = False

        self._seen_ids: Set[str] = set()
        self._accounts: Dict[str, ScrapedAccount] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *exc):
        pass  # twikit doesn't need explicit close

    async def initialize(self):
        """Authenticate all configured sessions and prepare rotation pool."""
        self._clients = []
        self._active_session_configs = []

        if self.resume_from_checkpoint and self.enable_checkpoint:
            self._load_checkpoint()
            await self._enforce_cooldown_if_active()

        for cfg in self.session_configs:
            try:
                client = self._make_client(cfg.proxy)
                await self._authenticate_session(client, cfg)
                self._clients.append(client)
                self._active_session_configs.append(cfg)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize {cfg.name} (proxy='{cfg.proxy}'): {e}"
                )

        if not self._clients:
            raise RuntimeError(
                "No X sessions could be initialized. Check credentials, cookies, and proxies."
            )

        self._active_session_idx = min(
            max(self._active_session_idx, 0), len(self._clients) - 1
        )
        self._client = self._clients[self._active_session_idx]

        if self.enable_checkpoint:
            self._save_checkpoint(force=True)

    def _build_default_session_configs(
        self,
        base_cookie_path: Path,
        proxies: Optional[List[str]],
        session_cookie_paths: Optional[List[Path]],
    ) -> List[XSessionConfig]:
        env_proxies = _split_csv_env(os.getenv("X_PROXIES", ""))
        raw_proxies = proxies if proxies is not None else env_proxies
        clean_proxies = [p.strip() for p in raw_proxies if p and p.strip()]

        env_cookie_paths = _split_csv_env(os.getenv("X_SESSION_COOKIES", ""))
        raw_cookie_paths: List[Path] = []
        if session_cookie_paths is not None:
            raw_cookie_paths = list(session_cookie_paths)
        elif env_cookie_paths:
            raw_cookie_paths = [_resolve_path(p) for p in env_cookie_paths]
        elif base_cookie_path:
            raw_cookie_paths = [base_cookie_path]

        session_count = max(1, len(clean_proxies), len(raw_cookie_paths))
        configs: List[XSessionConfig] = []
        for idx in range(session_count):
            session_number = idx + 1
            cookie_path = (
                raw_cookie_paths[idx]
                if idx < len(raw_cookie_paths)
                else _derive_cookie_path(base_cookie_path, session_number)
            )
            proxy = clean_proxies[idx % len(clean_proxies)] if clean_proxies else ""
            configs.append(
                XSessionConfig(
                    name=f"session-{session_number}",
                    username=self.username,
                    password=self.password,
                    email=self.email,
                    cookie_path=cookie_path,
                    proxy=proxy,
                )
            )
        return configs

    def _make_client(self, proxy: str):
        if self._client_factory:
            return self._client_factory(proxy)

        from twikit import Client

        if proxy:
            try:
                return Client("en-US", proxy=proxy)
            except TypeError:
                logger.warning(
                    "twikit Client proxy kwarg unavailable; continuing without explicit proxy."
                )
        return Client("en-US")

    async def _authenticate_session(self, client: Any, session_cfg: XSessionConfig) -> None:
        # Try loading saved cookies first
        if session_cfg.cookie_path.exists():
            try:
                client.load_cookies(str(session_cfg.cookie_path))
                logger.info(
                    f"Loaded cookies for {session_cfg.name} from {session_cfg.cookie_path}"
                )
                return
            except Exception as e:
                logger.warning(
                    f"Invalid cookies for {session_cfg.name}, re-authenticating: {e}"
                )

        if not session_cfg.username or not session_cfg.password:
            raise RuntimeError(
                "X credentials required. Set X_USERNAME/X_PASSWORD/X_EMAIL or pass --user/--pw/--email."
            )

        logger.info(
            f"Authenticating {session_cfg.name} as @{session_cfg.username} "
            f"(proxy='{session_cfg.proxy or 'none'}')"
        )
        await client.login(
            auth_info_1=session_cfg.username,
            auth_info_2=session_cfg.email,
            password=session_cfg.password,
        )

        session_cfg.cookie_path.parent.mkdir(parents=True, exist_ok=True)
        client.save_cookies(str(session_cfg.cookie_path))
        logger.info(
            f"Authenticated {session_cfg.name} and saved cookies to {session_cfg.cookie_path}"
        )

    def _is_rate_limited_error(self, error: Exception) -> bool:
        err = str(error).lower()
        return (
            "rate limit" in err
            or "429" in err
            or "too many" in err
            or "temporarily locked" in err
        )

    def _is_detection_error(self, error: Exception) -> bool:
        err = str(error).lower()
        return any(
            token in err
            for token in ("captcha", "challenge", "suspicious", "forbidden", "detected")
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_iso_datetime(raw: str) -> Optional[datetime]:
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    async def _sleep_with_jitter(self, base_seconds: float, jitter_seconds: float) -> None:
        delay = max(0.0, float(base_seconds))
        jitter = max(0.0, float(jitter_seconds))
        if jitter > 0:
            delay += random.uniform(0.0, jitter)
        if delay > 0:
            await asyncio.sleep(delay)

    async def _enforce_cooldown_if_active(self) -> None:
        cooldown_dt = self._parse_iso_datetime(self._cooldown_until)
        if not cooldown_dt:
            return
        now = self._utc_now()
        if now >= cooldown_dt:
            self._cooldown_until = ""
            self._abort_scrape = False
            return

        seconds_left = int((cooldown_dt - now).total_seconds())
        if self.wait_if_cooldown_active:
            logger.warning(
                f"Detection cooldown active until {cooldown_dt.isoformat()}. "
                f"Waiting {seconds_left}s before scraping resumes."
            )
            await asyncio.sleep(max(0, seconds_left))
            self._cooldown_until = ""
            self._abort_scrape = False
            return

        raise RuntimeError(
            "Detection cooldown active. Earliest resume: "
            f"{cooldown_dt.isoformat()} (in {seconds_left}s). "
            "Use --wait-cooldown to block until resume time."
        )

    def _trigger_detection_cooldown(self, context: str) -> None:
        if self.detection_cooldown_hours <= 0:
            return
        cooldown_dt = self._utc_now().replace(microsecond=0)
        cooldown_dt = cooldown_dt + timedelta(hours=self.detection_cooldown_hours)
        self._cooldown_until = cooldown_dt.isoformat()
        self._abort_scrape = True
        logger.warning(
            f"Detection cooldown set after {context}. Next allowed scrape at {self._cooldown_until}"
        )
        if self.enable_checkpoint:
            self._save_checkpoint(force=True)

    def _rotate_session(self, reason: str) -> bool:
        if len(self._clients) <= 1:
            return False
        prev_idx = self._active_session_idx
        self._active_session_idx = (self._active_session_idx + 1) % len(self._clients)
        self._client = self._clients[self._active_session_idx]

        prev_name = self._active_session_configs[prev_idx].name
        next_name = self._active_session_configs[self._active_session_idx].name
        logger.warning(f"Rotated session {prev_name} -> {next_name} ({reason})")

        if self.enable_checkpoint:
            self._save_checkpoint(force=True)
        return True

    def _extract_cursor(self, response: Any) -> str:
        for attr in ("next_cursor", "cursor", "bottom_cursor", "next_cursor_str"):
            value = getattr(response, attr, None)
            if value:
                return str(value)
        if isinstance(response, dict):
            for key in ("next_cursor", "cursor", "bottom_cursor"):
                value = response.get(key)
                if value:
                    return str(value)
        return ""

    async def _search_initial_page(self, query: str, cursor: str = "") -> Tuple[Any, bool]:
        kwargs = {"product": "Latest", "count": 20}
        if cursor:
            kwargs["cursor"] = cursor
            try:
                return await self._client.search_tweet(query, **kwargs), True
            except TypeError:
                kwargs.pop("cursor", None)

        return await self._client.search_tweet(query, **kwargs), False

    async def _skip_pages(self, response: Any, pages_to_skip: int) -> Any:
        skipped = 0
        while response and skipped < pages_to_skip:
            response = await response.next()
            skipped += 1
        return response

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return

        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {self.checkpoint_path}: {e}")
            return

        self._seen_ids = set(payload.get("seen_tweet_ids", []))
        self._query_progress = dict(payload.get("query_state", {}))
        self._current_query_index = int(payload.get("current_query_index", 0) or 0)
        self._active_session_idx = int(payload.get("session_index", 0) or 0)
        self._cooldown_until = str(payload.get("cooldown_until", "") or "")
        logger.info(
            f"Resumed checkpoint from {self.checkpoint_path}: "
            f"{len(self._seen_ids)} seen tweets"
        )

    def _save_checkpoint(self, force: bool = False) -> None:
        if not self.enable_checkpoint:
            return

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "session_index": self._active_session_idx,
            "current_query_index": self._current_query_index,
            "query_state": self._query_progress,
            "seen_tweet_ids": list(self._seen_ids),
            "cooldown_until": self._cooldown_until,
        }
        self.checkpoint_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _update_query_checkpoint(
        self,
        query: str,
        page: int,
        cursor: str,
        tweets_collected: int,
        completed: bool = False,
    ) -> None:
        state = self._query_progress.get(query, {})
        state["page"] = page
        state["cursor"] = cursor
        state["tweets_collected"] = tweets_collected
        state["completed"] = completed
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._query_progress[query] = state

    # ── High-level scrape methods ────────────────────────────────────────

    async def scrape_kenya(
        self,
        queries: Optional[List[str]] = None,
        limit_per_query: int = 40,
        max_total: int = 500,
    ) -> Tuple[List[ScrapedTweet], List[ScrapedAccount]]:
        """
        Run a full Kenya-focused scrape across multiple search queries.

        Returns (tweets, accounts):
            tweets   – deduplicated list of ScrapedTweet
            accounts – deduplicated list of ScrapedAccount
        """
        queries = queries or KENYA_SEARCH_QUERIES
        all_tweets: List[ScrapedTweet] = []
        start_query_index = self._current_query_index if self.resume_from_checkpoint else 0
        start_query_index = max(0, min(start_query_index, len(queries)))

        for idx in range(start_query_index, len(queries)):
            if self._abort_scrape:
                logger.warning("Stopping scrape early due to active detection cooldown.")
                break
            query = queries[idx]
            if len(all_tweets) >= max_total:
                logger.info(f"Reached {max_total} tweets, stopping early.")
                break

            self._current_query_index = idx
            remaining = max_total - len(all_tweets)
            per_query_target = min(limit_per_query, remaining)

            try:
                tweets = await self.search_tweets(
                    query,
                    limit=per_query_target,
                )
                if len(tweets) > remaining:
                    tweets = tweets[:remaining]
                all_tweets.extend(tweets)
                self._update_query_checkpoint(
                    query=query,
                    page=int(self._query_progress.get(query, {}).get("page", 0)),
                    cursor=str(self._query_progress.get(query, {}).get("cursor", "")),
                    tweets_collected=int(
                        self._query_progress.get(query, {}).get("tweets_collected", 0)
                    ),
                    completed=True,
                )
                logger.info(
                    f"  [{len(all_tweets):>4}/{max_total}] "
                    f"'{query}' → {len(tweets)} tweets"
                )
            except Exception as e:
                logger.warning(f"  Query '{query}' failed: {e}")
            finally:
                if self.enable_checkpoint:
                    self._save_checkpoint(force=True)

            # Polite delay between queries
            await self._sleep_with_jitter(self.query_delay_s, self.query_jitter_s)

        accounts = list(self._accounts.values())
        logger.info(
            f"Scrape complete: {len(all_tweets)} tweets, "
            f"{len(accounts)} unique accounts"
        )
        return all_tweets, accounts

    async def search_tweets(
        self,
        query: str,
        limit: int = 40,
        max_pages: int = 100,
        max_stale_pages: int = 3,
    ) -> List[ScrapedTweet]:
        """Search tweets with deep cursor pagination and rate-limit backoff."""
        results: List[ScrapedTweet] = []
        stale_pages = 0
        retries = 0
        response = None
        query_state = self._query_progress.get(query, {})
        pages_processed = (
            int(query_state.get("page", 0)) if self.resume_from_checkpoint else 0
        )
        cursor = str(query_state.get("cursor", "")) if self.resume_from_checkpoint else ""

        while len(results) < limit and pages_processed < max_pages and retries < 5:
            try:
                if response is None:
                    response, used_cursor = await self._search_initial_page(
                        query=query,
                        cursor=cursor,
                    )
                    if pages_processed > 0 and not used_cursor:
                        response = await self._skip_pages(response, pages_processed)
                else:
                    response = await response.next()

                if not response:
                    break

                new_count = 0
                for tweet in response:
                    parsed = self._parse_tweet(tweet)
                    if parsed and parsed.tweet_id not in self._seen_ids:
                        self._seen_ids.add(parsed.tweet_id)
                        results.append(parsed)
                        new_count += 1
                        if len(results) >= limit:
                            break

                pages_processed += 1
                cursor = self._extract_cursor(response)
                if new_count == 0:
                    stale_pages += 1
                    if stale_pages >= max_stale_pages:
                        break
                else:
                    stale_pages = 0

                self._update_query_checkpoint(
                    query=query,
                    page=pages_processed,
                    cursor=cursor,
                    tweets_collected=int(
                        self._query_progress.get(query, {}).get("tweets_collected", 0)
                    )
                    + new_count,
                    completed=False,
                )
                if self.enable_checkpoint and pages_processed % self.checkpoint_every_pages == 0:
                    self._save_checkpoint(force=True)

                retries = 0
                await self._sleep_with_jitter(self.request_delay_s, self.request_jitter_s)

            except Exception as e:
                retries += 1
                if self._is_rate_limited_error(e):
                    wait = min((2 ** (retries - 1)) * 15, 120)
                    logger.warning(
                        f"Rate-limited on '{query}', waiting {wait}s "
                        f"(retry {retries}/5)"
                    )
                    if self.rotate_on_rate_limit:
                        self._rotate_session("rate limit")
                    await self._sleep_with_jitter(wait, self.request_jitter_s)
                    response = None
                elif self._is_detection_error(e):
                    if self.detection_cooldown_hours > 0:
                        self._trigger_detection_cooldown(f"query='{query}'")
                        break
                    wait = min((2 ** (retries - 1)) * 10, 60)
                    logger.warning(
                        f"Detection/challenge on '{query}', waiting {wait}s "
                        f"(retry {retries}/5)"
                    )
                    if self.rotate_on_detection:
                        self._rotate_session("challenge/detection")
                    await self._sleep_with_jitter(wait, self.request_jitter_s)
                    response = None
                else:
                    logger.warning(f"Search error for '{query}': {e}")
                    break

        self._update_query_checkpoint(
            query=query,
            page=pages_processed,
            cursor=cursor,
            tweets_collected=int(self._query_progress.get(query, {}).get("tweets_collected", 0)),
            completed=False,
        )
        if self.enable_checkpoint:
            self._save_checkpoint(force=True)

        return results

    async def scrape_user_timeline(
        self,
        username: str,
        limit: int = 50,
    ) -> List[ScrapedTweet]:
        """Scrape a specific user's recent tweets."""
        results: List[ScrapedTweet] = []
        if self._abort_scrape:
            return results
        retries = 0
        while retries < 5:
            try:
                user = await self._client.get_user_by_screen_name(username)
                if not user:
                    return results

                tweets = await self._client.get_user_tweets(
                    user.id, tweet_type="Tweets", count=limit
                )
                for tweet in tweets or []:
                    parsed = self._parse_tweet(tweet)
                    if parsed and parsed.tweet_id not in self._seen_ids:
                        self._seen_ids.add(parsed.tweet_id)
                        results.append(parsed)
                        if len(results) >= limit:
                            break
                return results

            except Exception as e:
                retries += 1
                if self._is_rate_limited_error(e):
                    if self.rotate_on_rate_limit:
                        self._rotate_session("timeline-rate-limit")
                    await self._sleep_with_jitter(
                        min((2 ** (retries - 1)) * 15, 120),
                        self.request_jitter_s,
                    )
                    continue
                if self._is_detection_error(e):
                    if self.detection_cooldown_hours > 0:
                        self._trigger_detection_cooldown(f"timeline='@{username}'")
                        break
                    if self.rotate_on_detection:
                        self._rotate_session("timeline-detection")
                    await self._sleep_with_jitter(
                        min((2 ** (retries - 1)) * 10, 60),
                        self.request_jitter_s,
                    )
                    continue
                logger.warning(f"Failed to scrape @{username}: {e}")
                break

        return results

    # ── Parsing ──────────────────────────────────────────────────────────

    def _parse_tweet(self, tweet: Any) -> Optional[ScrapedTweet]:
        """Parse a twikit Tweet object into our dataclass."""
        try:
            text = getattr(tweet, "full_text", "") or getattr(tweet, "text", "") or ""

            # Author info
            user = getattr(tweet, "user", None)
            author_id = str(getattr(user, "id", "")) if user else ""
            author_username = getattr(user, "screen_name", "") or getattr(user, "username", "") or ""
            author_display = getattr(user, "name", "") or ""
            author_followers = int(getattr(user, "followers_count", 0) or 0)
            author_following = int(getattr(user, "friends_count", 0) or getattr(user, "following_count", 0) or 0)
            author_verified = bool(getattr(user, "verified", False) or getattr(user, "is_blue_verified", False))
            author_bio = getattr(user, "description", "") or ""
            author_loc = getattr(user, "location", "") or ""
            author_created = str(getattr(user, "created_at", "")) or ""
            author_tweet_count = int(getattr(user, "statuses_count", 0) or 0)
            author_img = getattr(user, "profile_image_url_https", "") or getattr(user, "profile_image_url", "") or ""

            # Capture account
            if author_id:
                self._capture_account(
                    user_id=author_id,
                    username=author_username,
                    display_name=author_display,
                    bio=author_bio,
                    location=author_loc,
                    followers=author_followers,
                    following=author_following,
                    tweet_count=author_tweet_count,
                    verified=author_verified,
                    created_at=author_created,
                    profile_image=author_img,
                    user_obj=user,
                )

            # Location extraction
            county, lat, lon = extract_primary_county(text, author_loc)
            all_counties = extract_kenya_locations(f"{text} {author_loc}")

            # Content features
            hashtags = re.findall(r"#(\w+)", text)
            mentions = re.findall(r"@(\w+)", text)
            urls = re.findall(r"https?://\S+", text)

            # Media
            media_urls = []
            media = getattr(tweet, "media", None)
            if media:
                for m in media:
                    url = getattr(m, "media_url_https", None) or getattr(m, "url", None)
                    if url:
                        media_urls.append(url)

            # Thread context
            reply_to = getattr(tweet, "in_reply_to_tweet_id", "") or ""
            reply_to_user = getattr(tweet, "in_reply_to_screen_name", "") or ""
            conv_id = getattr(tweet, "conversation_id", "") or ""
            is_retweet = bool(getattr(tweet, "retweeted_tweet", None))
            is_quote = bool(getattr(tweet, "quoted_tweet", None))

            # Timestamps
            created = getattr(tweet, "created_at", "")
            if isinstance(created, str) and created:
                try:
                    created_dt = datetime.strptime(
                        created, "%a %b %d %H:%M:%S %z %Y"
                    )
                except (ValueError, TypeError):
                    created_dt = datetime.now(timezone.utc)
            elif isinstance(created, datetime):
                created_dt = created
            else:
                created_dt = datetime.now(timezone.utc)

            return ScrapedTweet(
                tweet_id=str(getattr(tweet, "id", "") or hash(text)),
                text=text,
                created_at=created_dt,
                language=getattr(tweet, "lang", "en") or "en",
                like_count=int(getattr(tweet, "favorite_count", 0) or 0),
                retweet_count=int(getattr(tweet, "retweet_count", 0) or 0),
                reply_count=int(getattr(tweet, "reply_count", 0) or 0),
                quote_count=int(getattr(tweet, "quote_count", 0) or 0),
                view_count=int(getattr(tweet, "view_count", 0) or 0),
                bookmark_count=int(getattr(tweet, "bookmark_count", 0) or 0),
                author_id=author_id,
                author_username=author_username,
                author_display_name=author_display,
                author_followers=author_followers,
                author_following=author_following,
                author_verified=author_verified,
                author_bio=author_bio,
                author_location=author_loc,
                author_created_at=author_created,
                author_tweet_count=author_tweet_count,
                author_profile_image=author_img,
                reply_to_tweet_id=str(reply_to),
                reply_to_user=str(reply_to_user),
                conversation_id=str(conv_id),
                is_retweet=is_retweet,
                is_quote=is_quote,
                location_county=county,
                latitude=lat,
                longitude=lon,
                mentioned_counties=",".join(all_counties),
                hashtags=",".join(hashtags),
                mentions=",".join(mentions),
                urls=",".join(urls),
                media_urls=",".join(media_urls),
                source="twikit",
                scraped_at=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            logger.debug(f"Failed to parse tweet: {e}")
            return None

    def _capture_account(
        self,
        user_id: str,
        username: str,
        display_name: str,
        bio: str,
        location: str,
        followers: int,
        following: int,
        tweet_count: int,
        verified: bool,
        created_at: str,
        profile_image: str,
        user_obj: Any = None,
    ):
        """Deduplicate and store account data."""
        if user_id in self._accounts:
            return  # already captured

        # Kenya detection: check bio + location for county references
        detected_county, _, _ = extract_primary_county(bio, location)
        is_kenyan = bool(detected_county) or any(
            kw in (bio + " " + location).lower()
            for kw in ["kenya", "nairobi", "mombasa", "ke", "254"]
        )

        # Additional fields from user object
        website = ""
        banner = ""
        listed_count = 0
        if user_obj:
            website = (
                getattr(user_obj, "url", "")
                or getattr(user_obj, "website", "")
                or ""
            )
            banner = getattr(user_obj, "profile_banner_url", "") or ""
            listed_count = int(getattr(user_obj, "listed_count", 0) or 0)

        self._accounts[user_id] = ScrapedAccount(
            user_id=user_id,
            username=username,
            display_name=display_name,
            bio=bio,
            location=location,
            website=website,
            followers_count=followers,
            following_count=following,
            tweet_count=tweet_count,
            listed_count=listed_count,
            verified=verified,
            created_at=created_at,
            profile_image_url=profile_image,
            profile_banner_url=banner,
            is_kenyan=is_kenyan,
            detected_county=detected_county,
            scraped_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_accounts(self) -> List[ScrapedAccount]:
        """Return deduplicated accounts captured so far."""
        return list(self._accounts.values())

    # ── CSV Output ───────────────────────────────────────────────────────

    def save_tweets_csv(
        self,
        tweets: List[ScrapedTweet],
        path: Optional[Path] = None,
    ) -> Path:
        """Write tweets to CSV.  Appends if file already exists."""
        path = path or _TWEETS_OUT
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine if we need to write header
        write_header = not path.exists() or path.stat().st_size == 0

        # Load existing IDs for dedup on append
        existing_ids: Set[str] = set()
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_ids.add(row.get("tweet_id", ""))

        new_count = 0
        with open(path, "a", encoding="utf-8", newline="") as f:
            fieldnames = [fd.name for fd in ScrapedTweet.__dataclass_fields__.values()]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            for tw in tweets:
                if tw.tweet_id not in existing_ids:
                    row = asdict(tw)
                    row["created_at"] = str(row["created_at"])
                    writer.writerow(row)
                    existing_ids.add(tw.tweet_id)
                    new_count += 1

        logger.info(f"Saved {new_count} new tweets to {path} (total {len(existing_ids)})")
        return path

    def save_accounts_csv(
        self,
        accounts: List[ScrapedAccount],
        path: Optional[Path] = None,
    ) -> Path:
        """Write account profiles to CSV.  Appends / deduplicates."""
        path = path or _ACCOUNTS_OUT
        path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not path.exists() or path.stat().st_size == 0

        existing_ids: Set[str] = set()
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_ids.add(row.get("user_id", ""))

        new_count = 0
        with open(path, "a", encoding="utf-8", newline="") as f:
            fieldnames = [fd.name for fd in ScrapedAccount.__dataclass_fields__.values()]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            for acc in accounts:
                if acc.user_id not in existing_ids:
                    writer.writerow(asdict(acc))
                    existing_ids.add(acc.user_id)
                    new_count += 1

        logger.info(f"Saved {new_count} new accounts to {path} (total {len(existing_ids)})")
        return path

    def export_dashboard_csv(
        self,
        tweets: List[ScrapedTweet],
        path: Optional[Path] = None,
    ) -> Path:
        """Export tweets in the dashboard-compatible schema
        (matches data/synthetic_kenya/tweets.csv).

        This allows the PulseConnector to ingest real scraped tweets
        alongside synthetic data.
        """
        path = path or (_DATA_DIR / "pulse" / "x_kenya_dashboard.csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "post_id", "account_id", "timestamp", "text", "intent",
            "interaction_type", "reply_to_post_id", "source_label",
            "like_count", "retweet_count", "location_county",
            "latitude", "longitude", "imperative_rate", "urgency_rate",
            "coordination_score", "escalation_score", "threat_score",
        ]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for tw in tweets:
                # Classify interaction type
                if tw.is_retweet:
                    itype = "retweet"
                elif tw.reply_to_tweet_id:
                    itype = "reply"
                elif tw.is_quote:
                    itype = "quote"
                else:
                    itype = "original"

                writer.writerow({
                    "post_id": tw.tweet_id,
                    "account_id": tw.author_id,
                    "timestamp": str(tw.created_at),
                    "text": tw.text,
                    "intent": "scraped",
                    "interaction_type": itype,
                    "reply_to_post_id": tw.reply_to_tweet_id,
                    "source_label": f"x/@{tw.author_username}",
                    "like_count": tw.like_count,
                    "retweet_count": tw.retweet_count,
                    "location_county": tw.location_county,
                    "latitude": tw.latitude,
                    "longitude": tw.longitude,
                    "imperative_rate": 0.0,  # needs NLP scoring
                    "urgency_rate": 0.0,
                    "coordination_score": 0.0,
                    "escalation_score": 0.0,
                    "threat_score": 0.0,
                })

        logger.info(f"Dashboard CSV exported: {path} ({len(tweets)} rows)")
        return path


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def _load_proxies_from_file(path: Path) -> List[str]:
    proxies: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Proxy file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        proxies.append(cleaned)
    return proxies


def _load_session_configs_from_file(
    path: Path,
    default_username: str,
    default_password: str,
    default_email: str,
    base_cookie_path: Path,
) -> List[XSessionConfig]:
    if not path.exists():
        raise FileNotFoundError(f"Session config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sessions = payload.get("sessions", payload if isinstance(payload, list) else [])
    if not isinstance(raw_sessions, list):
        raise ValueError("Session config must be a list or have a top-level 'sessions' list.")

    configs: List[XSessionConfig] = []
    for idx, session in enumerate(raw_sessions):
        if not isinstance(session, dict):
            continue
        session_number = idx + 1
        raw_cookie_path = session.get("cookie_path", "")
        cookie_path = (
            _resolve_path(str(raw_cookie_path))
            if raw_cookie_path
            else _derive_cookie_path(base_cookie_path, session_number)
        )
        configs.append(
            XSessionConfig(
                name=str(session.get("name", f"session-{session_number}")),
                username=str(session.get("username", default_username)),
                password=str(session.get("password", default_password)),
                email=str(session.get("email", default_email)),
                cookie_path=cookie_path,
                proxy=str(session.get("proxy", "")),
            )
        )

    if not configs:
        raise ValueError(f"No valid session entries found in {path}")
    return configs


async def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="K-SHIELD X/Twitter Kenya Scraper"
    )
    parser.add_argument("--user", "-u", default="", help="X username")
    parser.add_argument("--pw", "-p", default="", help="X password")
    parser.add_argument("--email", "-e", default="", help="X email")
    parser.add_argument(
        "--limit", "-l", type=int, default=200,
        help="Max tweets to scrape (default: 200)"
    )
    parser.add_argument(
        "--queries", "-q", nargs="*", default=None,
        help="Custom search queries (default: Kenya-focused set)"
    )
    parser.add_argument(
        "--per-query", type=int, default=40,
        help="Tweets per query (default: 40)"
    )
    parser.add_argument(
        "--timeline", "-t", nargs="*", default=None,
        help="Specific usernames to scrape timelines"
    )
    parser.add_argument(
        "--timeline-limit", type=int, default=50,
        help="Tweets per timeline account (default: 50)",
    )
    parser.add_argument(
        "--proxy",
        action="append",
        default=None,
        help="Proxy URL (repeatable). Example: --proxy http://user:pass@host:port",
    )
    parser.add_argument(
        "--proxy-file",
        default="",
        help="Path to a text file with one proxy URL per line.",
    )
    parser.add_argument(
        "--session-cookie",
        action="append",
        default=None,
        help="Cookie path per session (repeatable).",
    )
    parser.add_argument(
        "--session-config",
        default=os.getenv("X_SESSION_CONFIG", ""),
        help=(
            "JSON file describing sessions. Supports top-level list or "
            "{'sessions': [...]}; each session may define "
            "name, username, password, email, cookie_path, proxy."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("X_CHECKPOINT_PATH", str(_CHECKPOINT_PATH)),
        help=f"Checkpoint file path (default: {str(_CHECKPOINT_PATH)})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (also supports X_RESUME_CHECKPOINT=1).",
    )
    parser.add_argument(
        "--checkpoint-every-pages",
        type=int,
        default=1,
        help="Persist checkpoint every N pages (default: 1).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint writing.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.2,
        help="Delay between paginated requests in seconds (default: 1.2).",
    )
    parser.add_argument(
        "--query-delay",
        type=float,
        default=3.0,
        help="Delay between queries in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--request-jitter",
        type=float,
        default=0.0,
        help="Random jitter added to request delay in seconds (default: 0.0).",
    )
    parser.add_argument(
        "--query-jitter",
        type=float,
        default=0.0,
        help="Random jitter added to query delay in seconds (default: 0.0).",
    )
    parser.add_argument(
        "--detection-cooldown-hours",
        type=float,
        default=24.0,
        help=(
            "Hours to pause scraping after detection/challenge signal "
            "(default: 24). Set 0 to disable cooldown."
        ),
    )
    parser.add_argument(
        "--wait-cooldown",
        action="store_true",
        help="If cooldown is active, wait until it expires instead of exiting.",
    )
    parser.add_argument(
        "--conservative-mode",
        action="store_true",
        help=(
            "Apply slower pacing defaults to reduce request pressure "
            "(request_delay>=4s, query_delay>=15s, with jitter)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )

    fallback_creds = _load_fallback_credentials()
    username = args.user or fallback_creds["username"]
    password = args.pw or fallback_creds["password"]
    email = args.email or fallback_creds["email"]

    session_configs: Optional[List[XSessionConfig]] = None
    proxy_list: Optional[List[str]] = args.proxy[:] if args.proxy else None
    session_cookie_paths: Optional[List[Path]] = (
        [_resolve_path(p) for p in args.session_cookie] if args.session_cookie else None
    )

    if args.proxy_file:
        file_proxies = _load_proxies_from_file(_resolve_path(args.proxy_file))
        proxy_list = (proxy_list or []) + file_proxies

    if args.session_config:
        session_configs = _load_session_configs_from_file(
            path=_resolve_path(args.session_config),
            default_username=username,
            default_password=password,
            default_email=email,
            base_cookie_path=_COOKIE_PATH,
        )
        if proxy_list or session_cookie_paths:
            logger.warning(
                "Session config file provided; ignoring --proxy/--proxy-file/--session-cookie overrides."
            )

    env_resume = os.getenv("X_RESUME_CHECKPOINT", "").strip().lower()
    resume_enabled = args.resume or env_resume in {"1", "true", "yes", "y"}
    checkpoint_path = _resolve_path(args.checkpoint)

    request_delay = args.request_delay
    query_delay = args.query_delay
    request_jitter = args.request_jitter
    query_jitter = args.query_jitter
    per_query_limit = args.per_query
    if args.conservative_mode:
        request_delay = max(request_delay, 4.0)
        query_delay = max(query_delay, 15.0)
        request_jitter = max(request_jitter, 1.0)
        query_jitter = max(query_jitter, 3.0)
        per_query_limit = min(per_query_limit, 20)

    try:
        async with KenyaXScraper(
            username=username,
            password=password,
            email=email,
            session_configs=session_configs,
            proxies=proxy_list,
            session_cookie_paths=session_cookie_paths,
            checkpoint_path=checkpoint_path,
            enable_checkpoint=not args.no_checkpoint,
            resume_from_checkpoint=resume_enabled,
            checkpoint_every_pages=args.checkpoint_every_pages,
            request_delay_s=request_delay,
            query_delay_s=query_delay,
            request_jitter_s=request_jitter,
            query_jitter_s=query_jitter,
            detection_cooldown_hours=args.detection_cooldown_hours,
            wait_if_cooldown_active=args.wait_cooldown,
        ) as scraper:
            tweets, accounts = await scraper.scrape_kenya(
                queries=args.queries,
                limit_per_query=per_query_limit,
                max_total=args.limit,
            )

            # Optionally scrape specific timelines
            if args.timeline:
                remaining = max(args.limit - len(tweets), 0)
                for username in args.timeline:
                    if remaining <= 0:
                        break
                    logger.info(f"Scraping @{username} timeline …")
                    tl_limit = min(args.timeline_limit, remaining)
                    tl = await scraper.scrape_user_timeline(username, limit=tl_limit)
                    if len(tl) > remaining:
                        tl = tl[:remaining]
                    tweets.extend(tl)
                    remaining = max(args.limit - len(tweets), 0)
                    await scraper._sleep_with_jitter(
                        scraper.query_delay_s, scraper.query_jitter_s
                    )

            # Save results
            accounts = scraper.get_accounts()
            scraper.save_tweets_csv(tweets)
            scraper.save_accounts_csv(accounts)
            scraper.export_dashboard_csv(tweets)

            # Summary
            kenya_tweets = [t for t in tweets if t.location_county]
            print(f"\n{'=' * 60}")
            print(f"  SCRAPE COMPLETE")
            print(f"  Total tweets:      {len(tweets)}")
            print(f"  Unique accounts:   {len(accounts)}")
            print(f"  Kenya-located:     {len(kenya_tweets)} ({len(kenya_tweets)*100//max(len(tweets),1)}%)")
            print(f"  Kenyan accounts:   {sum(1 for a in accounts if a.is_kenyan)}")
            print(f"{'=' * 60}")

            # County breakdown
            county_counts: Dict[str, int] = {}
            for t in tweets:
                if t.location_county:
                    county_counts[t.location_county] = county_counts.get(t.location_county, 0) + 1
            if county_counts:
                print("\n  COUNTY DISTRIBUTION:")
                for county, count in sorted(county_counts.items(), key=lambda x: -x[1])[:15]:
                    bar = "█" * min(count, 40)
                    print(f"    {county:<20} {count:>4}  {bar}")
            print()
    except RuntimeError as e:
        logger.error(str(e))
        return


if __name__ == "__main__":
    asyncio.run(_main())

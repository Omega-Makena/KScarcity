"""URL fetching and full-content extraction for NewsAPI articles."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:  # pragma: no cover
    HAS_BS4 = False

logger = logging.getLogger("kshield.pulse.news_content")

BLOCKED_HTTP_STATUSES = {401, 403, 451}


@dataclass
class ExtractionResult:
    """Extraction payload for persistence."""

    url: str
    canonical_url: str
    status: str
    extracted_text: str = ""
    extraction_method: str = ""
    content_hash: str = ""
    raw_html: str = ""
    http_status: int = 0
    error_reason: str = ""


class NewsContentExtractor:
    """Fetches URL content with robots safety and extracts article text."""

    def __init__(
        self,
        timeout_seconds: int = 10,
        retries: int = 2,
        user_agent: str = "ScarcityNewsBot/1.0 (+https://localhost)",
    ):
        self.timeout_seconds = timeout_seconds
        self.retries = max(0, int(retries))
        self.user_agent = user_agent
        self._robots_cache: Dict[str, RobotFileParser] = {}

    def extract(self, url: str) -> ExtractionResult:
        canonical_input = self._canonicalize_url(url)
        if not requests:
            return ExtractionResult(
                url=url,
                canonical_url=canonical_input,
                status="error",
                error_reason="requests_not_available",
            )

        allowed = self._is_allowed(url)
        if not allowed:
            return ExtractionResult(
                url=url,
                canonical_url=canonical_input,
                status="blocked",
                extraction_method="robots",
                error_reason="blocked_by_robots",
            )

        headers = {"User-Agent": self.user_agent, "Accept-Language": "en-US,en;q=0.8"}
        last_error = ""
        response = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    allow_redirects=True,
                )
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = str(exc)
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))

        if response is None:
            return ExtractionResult(
                url=url,
                canonical_url=canonical_input,
                status="error",
                error_reason=f"request_failed:{last_error or 'unknown'}",
            )

        final_url = response.url or url
        canonical_url = self._canonicalize_url(final_url)
        http_status = int(response.status_code)

        if http_status in BLOCKED_HTTP_STATUSES:
            return ExtractionResult(
                url=url,
                canonical_url=canonical_url,
                status="blocked",
                extraction_method="http_status",
                http_status=http_status,
                error_reason=f"http_{http_status}",
            )

        if http_status >= 400:
            return ExtractionResult(
                url=url,
                canonical_url=canonical_url,
                status="error",
                extraction_method="http_status",
                http_status=http_status,
                error_reason=f"http_{http_status}",
            )

        html = response.text or ""
        extracted_text, method, canonical_hint = self._extract_main_text(html, canonical_url)
        if canonical_hint:
            canonical_url = self._canonicalize_url(canonical_hint)

        if not extracted_text.strip():
            return ExtractionResult(
                url=url,
                canonical_url=canonical_url,
                status="error",
                extraction_method=method or "empty",
                http_status=http_status,
                raw_html=html,
                error_reason="no_extractable_text",
            )

        normalized_text = re.sub(r"\s+", " ", extracted_text).strip()
        content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

        return ExtractionResult(
            url=url,
            canonical_url=canonical_url,
            status="ok",
            extracted_text=normalized_text,
            extraction_method=method,
            content_hash=content_hash,
            raw_html=html,
            http_status=http_status,
        )

    def _is_allowed(self, url: str) -> bool:
        parsed = urlsplit(url)
        domain = parsed.netloc.lower()
        if not domain:
            return True

        if domain not in self._robots_cache:
            self._robots_cache[domain] = self._fetch_robots(parsed.scheme or "https", domain)

        parser = self._robots_cache[domain]
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:
            return True

    def _fetch_robots(self, scheme: str, domain: str) -> RobotFileParser:
        parser = RobotFileParser()
        if not requests:
            parser.parse("")
            return parser

        robots_url = f"{scheme}://{domain}/robots.txt"
        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=max(5, self.timeout_seconds),
            )
            if response.status_code == 200 and response.text:
                parser.parse(response.text.splitlines())
            else:
                parser.parse(["User-agent: *", "Disallow:"])
        except Exception:
            parser.parse(["User-agent: *", "Disallow:"])
        return parser

    def _extract_main_text(self, html: str, base_url: str) -> tuple[str, str, Optional[str]]:
        if HAS_BS4:
            return self._extract_with_bs4(html, base_url)
        return self._extract_with_regex(html), "regex", None

    def _extract_with_bs4(self, html: str, base_url: str) -> tuple[str, str, Optional[str]]:
        soup = BeautifulSoup(html, "html.parser")
        canonical_hint = None
        canonical_tag = soup.find("link", rel=lambda x: x and "canonical" in str(x).lower())
        if canonical_tag and canonical_tag.get("href"):
            canonical_hint = urljoin(base_url, str(canonical_tag.get("href")))

        for tag_name in ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        candidates = []

        def _add_candidate(node, method: str) -> None:
            text = node.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            if len(text) < 120:
                return
            punct = sum(text.count(p) for p in ".,;:!?")
            score = len(text) + punct * 12
            candidates.append((score, text, method))

        for selector, method in [("article", "article"), ("main", "main")]:
            for node in soup.select(selector):
                _add_candidate(node, method)

        for node in soup.find_all("div"):
            if node.find_all("p"):
                _add_candidate(node, "div_paragraph_cluster")

        if not candidates:
            paragraphs = []
            for p in soup.find_all("p"):
                text = p.get_text(" ", strip=True)
                text = re.sub(r"\s+", " ", text)
                if len(text) >= 40:
                    paragraphs.append(text)
            if paragraphs:
                joined = "\n".join(paragraphs)
                return joined, "paragraphs", canonical_hint

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1], candidates[0][2], canonical_hint

        fallback = soup.get_text(" ", strip=True)
        fallback = re.sub(r"\s+", " ", fallback)
        return fallback, "full_text", canonical_hint

    def _extract_with_regex(self, html: str) -> str:
        text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _canonicalize_url(self, url: str) -> str:
        parsed = urlsplit(url)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        clean_params = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            key_l = key.lower()
            if key_l.startswith("utm_") or key_l in {"fbclid", "gclid", "mc_cid", "mc_eid"}:
                continue
            clean_params.append((key, value))
        query = "&".join(
            f"{k}={v}" if v != "" else k
            for k, v in sorted(clean_params)
        )
        return urlunsplit((scheme, netloc, path, query, ""))


def build_excerpt(text: str, max_chars: int = 260) -> str:
    """Compact evidence excerpt from extracted article text."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."

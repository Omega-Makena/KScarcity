"""
Policy Bill Extractor — Parse Bills into Structured Provisions

Accepts any Kenyan policy document (bill, gazette notice, executive order)
and extracts structured provisions via LLM analysis.

Supports:
- Pasted text
- PDF upload (PyMuPDF / pdfplumber)
- URL scraping (Kenya Gazette, Parliament)

Not limited to Finance Bills — handles ALL policy sectors:
TAXATION, HEALTH, HOUSING, FUEL_ENERGY, EDUCATION, DIGITAL,
SECURITY, AGRICULTURE, DEVOLUTION, CONSTITUTIONAL, TRANSPORT, EMPLOYMENT

Usage:
    extractor = PolicyExtractor(provider)
    bill = await extractor.extract_from_text(text, title="SHIF Amendment 2026")
    bill = await extractor.extract_from_pdf(pdf_bytes)
    bill = await extractor.extract_from_url("https://parliament.go.ke/bills/...")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import OllamaConfig, AnalysisTask

logger = logging.getLogger(__name__)

# Try importing PDF libraries (optional)
try:
    import fitz as pymupdf  # type: ignore[import-untyped]  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

VALID_SECTORS = [
    "taxation", "health", "housing", "fuel_energy", "education", "digital",
    "security", "agriculture", "devolution", "constitutional", "transport",
    "employment", "general",
]

VALID_GRIEVANCE_TYPES = [
    "economic", "social", "governance", "security", "health",
    "education", "environmental", "none",
]


@dataclass
class BillProvision:
    """Single provision/clause extracted from a policy document."""
    clause_id: str = ""
    description: str = ""
    sector: str = "general"
    affected_groups: List[str] = field(default_factory=list)
    affected_counties: List[str] = field(default_factory=list)
    monetary_impact: str = ""
    severity: float = 0.5
    keywords_en: List[str] = field(default_factory=list)
    keywords_sw: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BillAnalysis:
    """Complete bill/policy analysis output."""
    title: str = ""
    source_type: str = "paste"  # paste, pdf, url
    raw_text: str = ""
    summary: str = ""
    sectors: List[str] = field(default_factory=list)
    provisions: List[BillProvision] = field(default_factory=list)
    total_severity: float = 0.0
    hashtags: List[str] = field(default_factory=list)
    keywords_en: List[str] = field(default_factory=list)
    keywords_sw: List[str] = field(default_factory=list)
    matched_historical_event: str = ""
    match_similarity: float = 0.0
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provisions"] = [p.to_dict() for p in self.provisions]
        return d

    @property
    def top_provisions(self) -> List[BillProvision]:
        """Top 5 provisions by severity."""
        return sorted(self.provisions, key=lambda p: p.severity, reverse=True)[:5]

    @property
    def provision_count(self) -> int:
        return len(self.provisions)


# ═══════════════════════════════════════════════════════════════════════════
# Extraction Prompts
# ═══════════════════════════════════════════════════════════════════════════

BILL_EXTRACTION_SYSTEM = """You are a Kenyan legislative analyst. Extract structured provisions from policy documents.

You handle ALL types of Kenyan policy:
- Parliamentary Bills (Finance, Health, Education, etc.)
- Gazette Notices (tax changes, regulatory updates)
- Executive Orders
- County Government Policies
- Regulatory Changes (EPRA fuel pricing, NHIF/SHIF directives, etc.)

For each provision/clause/change you find, identify:
1. clause_id: Reference number (Section X, Clause Y, or invented short ID)
2. description: Plain-language summary a citizen would understand
3. sector: One of: taxation, health, housing, fuel_energy, education, digital, security, agriculture, devolution, constitutional, transport, employment, general
4. affected_groups: Who is impacted (e.g., ["low_income", "youth", "traders", "bodaboda", "jua_kali", "students", "civil_servants"])
5. affected_counties: Which counties or "nationwide"
6. monetary_impact: Financial effect (e.g., "+16% VAT on bread", "KES 3,000/month deduction")
7. severity: 0.0-1.0 how much social disruption this will cause
8. keywords_en: English search terms
9. keywords_sw: Swahili/Sheng search terms

Also provide:
- summary: One-paragraph bill overview
- hashtags: Likely organic hashtags (#RejectFinanceBill, etc.)
- total_severity: Overall bill severity (0-1)

Return JSON:
{
  "title": "bill title",
  "summary": "one paragraph overview",
  "sectors": ["sector1", "sector2"],
  "total_severity": 0.0-1.0,
  "hashtags": ["#tag1", "#tag2"],
  "keywords_en": ["keyword1"],
  "keywords_sw": ["neno1"],
  "provisions": [
    {
      "clause_id": "Section 12",
      "description": "plain language summary",
      "sector": "taxation",
      "affected_groups": ["low_income", "traders"],
      "affected_counties": ["nationwide"],
      "monetary_impact": "+16% VAT on bread",
      "severity": 0.85,
      "keywords_en": ["vat", "bread"],
      "keywords_sw": ["ushuru", "mkate"]
    }
  ]
}"""


BILL_SUMMARY_SYSTEM = """You are a Kenyan policy analyst. Given a policy document title and optional context,
provide a brief summary of what this policy does and who it affects.
Focus on the Kenyan context. Be concise (2-3 sentences).
If you don't recognize the specific bill, describe what a bill in that sector typically covers in Kenya."""


# ═══════════════════════════════════════════════════════════════════════════
# Extractor Engine
# ═══════════════════════════════════════════════════════════════════════════

class PolicyExtractor:
    """
    Extract structured provisions from policy documents via LLM.
    
    Works with any OllamaProvider instance.
    """

    def __init__(self, provider):
        """
        Args:
            provider: OllamaProvider instance (must have _generate_json / _generate_text)
        """
        self.provider = provider

    # ─── Core Extraction ─────────────────────────────────────────────────

    async def extract_from_text(
        self,
        text: str,
        title: str = "",
        source_type: str = "paste",
    ) -> BillAnalysis:
        """
        Extract provisions from raw policy text.
        
        Args:
            text: Full bill/policy text
            title: Optional bill title
            source_type: "paste", "pdf", or "url"
            
        Returns:
            BillAnalysis with extracted provisions
        """
        # Truncate very long texts to fit context window
        max_chars = 24000
        truncated = text[:max_chars]
        if len(text) > max_chars:
            truncated += f"\n\n[... truncated, {len(text) - max_chars} chars omitted ...]"

        prompt = f"""Extract all provisions from this Kenyan policy document.

TITLE: {title or 'Unknown — infer from content'}

DOCUMENT TEXT:
\"\"\"
{truncated}
\"\"\"

Return a JSON object with title, summary, sectors, total_severity, hashtags, keywords_en, keywords_sw, and provisions array."""

        data = await self.provider._generate_json(
            prompt, BILL_EXTRACTION_SYSTEM,
            task=AnalysisTask.POLICY_IMPACT,
        )

        if not data:
            logger.warning("LLM returned empty extraction — building minimal analysis")
            return BillAnalysis(
                title=title or "Unknown Policy",
                source_type=source_type,
                raw_text=text,
                summary="Extraction failed — please try again or provide a clearer document.",
            )

        return self._parse_extraction(data, text, title, source_type)

    async def extract_from_title(self, title: str) -> BillAnalysis:
        """
        Extract a basic analysis from just a bill title/name.
        Useful when the user mentions a bill by name without full text.
        """
        prompt = f"""The user mentioned this Kenyan policy: "{title}"

Based on your knowledge of Kenyan policy and the current 2025-2026 landscape,
provide a structured analysis as if you were reading the bill.
Include likely provisions, affected groups, severity, and hashtags.
If this is a known bill (Finance Bill, Housing Levy, SHIF, etc.), be specific.
If unknown, provide reasonable estimates for a bill in that sector.

Return the full JSON structure with provisions."""

        data = await self.provider._generate_json(
            prompt, BILL_EXTRACTION_SYSTEM,
            task=AnalysisTask.POLICY_IMPACT,
        )

        if not data:
            return BillAnalysis(title=title, source_type="title_only")

        return self._parse_extraction(data, "", title, "title_only")

    async def extract_from_pdf(self, pdf_bytes: bytes, title: str = "") -> BillAnalysis:
        """
        Extract provisions from PDF bytes.
        
        Tries PyMuPDF first, then pdfplumber, then raw text fallback.
        """
        text = self._extract_pdf_text(pdf_bytes)
        if not text or len(text.strip()) < 50:
            logger.warning("PDF extraction yielded minimal text")
            return BillAnalysis(
                title=title or "PDF Document",
                source_type="pdf",
                summary="Could not extract readable text from this PDF. "
                        "Try pasting the text directly.",
            )
        return await self.extract_from_text(text, title=title, source_type="pdf")

    async def extract_from_url(self, url: str, title: str = "") -> BillAnalysis:
        """
        Scrape and extract provisions from a URL.
        
        Supports Kenya Gazette, Parliament, and general news URLs.
        """
        text = await self._scrape_url(url)
        if not text or len(text.strip()) < 50:
            logger.warning(f"URL scrape yielded minimal text from {url}")
            # Fall back to title-based analysis if we have a title
            if title:
                logger.info(f"Falling back to title-based analysis for: {title}")
                bill = await self.extract_from_title(title)
                bill.summary = (
                    f"Note: Could not extract text from {url} "
                    f"(page may require JavaScript or login). "
                    f"Analysis below is based on the bill title.\n\n"
                    + (bill.summary or "")
                )
                bill.source_type = "url"
                return bill
            return BillAnalysis(
                title=title or url,
                source_type="url",
                summary=f"Could not extract readable text from {url}. "
                        "The page may require JavaScript or login. "
                        "Try pasting the bill text directly instead.",
            )
        logger.info(f"URL extraction successful: {len(text)} chars from {url}")
        return await self.extract_from_text(text, title=title, source_type="url")

    # ─── PDF Handling ────────────────────────────────────────────────────

    def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using available library."""
        # Try PyMuPDF first (faster, more reliable)
        if HAS_PYMUPDF:
            try:
                doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                return "\n".join(text_parts)
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")

        # Try pdfplumber
        if HAS_PDFPLUMBER:
            try:
                import io
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    return "\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")

        logger.error("No PDF library available (install PyMuPDF or pdfplumber)")
        return ""

    # ─── URL Scraping ────────────────────────────────────────────────────

    async def _scrape_url(self, url: str) -> str:
        """Scrape text content from a URL using a real browser UA."""
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as resp:
                    if resp.status != 200:
                        logger.warning(f"URL returned HTTP {resp.status} for {url}")
                        return ""
                    html = await resp.text()

            if not html or len(html) < 100:
                return ""

            if HAS_BS4:
                soup = BeautifulSoup(html, "html.parser")

                # Remove non-content elements
                for tag in soup(["script", "style", "nav", "header", "footer",
                                 "aside", "form", "iframe", "noscript", "svg"]):
                    tag.decompose()

                # Try to find the main article content first
                article = (
                    soup.find("article")
                    or soup.find("main")
                    or soup.find("div", class_=re.compile(r"content|article|body|post|entry", re.I))
                    or soup.find("div", id=re.compile(r"content|article|body|main", re.I))
                )

                target = article if article else soup.body or soup

                # Extract text from the target
                text = target.get_text(separator="\n", strip=True)
                # Clean excessive whitespace
                text = re.sub(r"\n{3,}", "\n\n", text)
                # Remove very short lines (likely UI elements)
                lines = [l for l in text.split("\n") if len(l.strip()) > 20 or l.strip() == ""]
                text = "\n".join(lines)

                logger.info(f"URL scrape extracted {len(text)} chars from {url}")
                return text
            else:
                # Basic HTML tag stripping
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text)
                return text.strip()

        except Exception as e:
            logger.error(f"URL scrape failed for {url}: {e}")
            return ""

    # ─── Parse LLM Output ───────────────────────────────────────────────

    def _parse_extraction(
        self,
        data: Dict[str, Any],
        raw_text: str,
        title: str,
        source_type: str,
    ) -> BillAnalysis:
        """Parse LLM JSON output into BillAnalysis."""
        provisions = []
        for p in data.get("provisions", []):
            sector = str(p.get("sector", "general")).lower()
            if sector not in VALID_SECTORS:
                sector = "general"

            provisions.append(BillProvision(
                clause_id=str(p.get("clause_id", "")),
                description=str(p.get("description", "")),
                sector=sector,
                affected_groups=_ensure_list(p.get("affected_groups", [])),
                affected_counties=_ensure_list(p.get("affected_counties", ["nationwide"])),
                monetary_impact=str(p.get("monetary_impact", "")),
                severity=_clamp(float(p.get("severity", 0.5))),
                keywords_en=_ensure_list(p.get("keywords_en", [])),
                keywords_sw=_ensure_list(p.get("keywords_sw", [])),
            ))

        # Deduplicate sectors
        sectors_raw = _ensure_list(data.get("sectors", []))
        sectors = list({s.lower() for s in sectors_raw if s.lower() in VALID_SECTORS})
        if not sectors and provisions:
            sectors = list({p.sector for p in provisions})

        return BillAnalysis(
            title=data.get("title", title) or title or "Unknown Policy",
            source_type=source_type,
            raw_text=raw_text,
            summary=str(data.get("summary", "")),
            sectors=sectors,
            provisions=provisions,
            total_severity=_clamp(float(data.get("total_severity", 0.0))),
            hashtags=_ensure_list(data.get("hashtags", [])),
            keywords_en=_ensure_list(data.get("keywords_en", [])),
            keywords_sw=_ensure_list(data.get("keywords_sw", [])),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _ensure_list(val) -> list:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return []

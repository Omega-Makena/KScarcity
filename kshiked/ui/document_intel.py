"""
Document Intelligence + News Integration.

Manages:
1. PDF Analysis (local documents)
2. News Feed (via NewsIngestor / NewsAPI)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..pulse.news import get_news_ingestor

logger = logging.getLogger("sentinel.document_intel")

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "been", "has", "have", "had", "will", "would", "could", "should", "than",
    "then", "into", "onto", "over", "under", "about", "between", "within",
    "while", "where", "when", "what", "which", "who", "whom", "whose", "why",
    "how", "a", "an", "of", "to", "in", "on", "at", "by", "as", "is", "it",
    "its", "or", "not", "no", "yes", "if", "but", "so", "we", "our", "us",
    "they", "their", "them", "he", "she", "his", "her", "you", "your", "i",
    "me", "my", "mine", "these", "those", "there", "here", "also", "more",
    "most", "some", "any", "all", "each", "few", "many", "such", "new",
    "may", "might", "can", "shall", "than", "per", "via", "re", "vs",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIR = PROJECT_ROOT / "random"
DEFAULT_CACHE_DIR = DEFAULT_PDF_DIR

DEFAULT_UPDATE_SECONDS = 24 * 60 * 60
MAX_TEXT_CHARS = 30000
MAX_KEYWORDS = 20
MAX_BIGRAMS = 10
THEME_EXCERPT_CHARS = 1000

@dataclass
class DocumentInfo:
    name: str
    path: str
    mtime: float
    word_count: int
    keywords: List[str]
    text: str
    theme: str
    theme_terms: List[str]


def _load_json(path: Path) -> Dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to read cache {path}: {exc}")
    return {}


def _write_json(path: Path, payload: Dict) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Failed to write cache {path}: {exc}")


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9']{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _extract_keywords(text: str, stem_tokens: Optional[List[str]] = None) -> List[str]:
    tokens = _tokenize(text)
    if not tokens:
        return stem_tokens or []

    counts = Counter(tokens)
    top_words = [w for w, _ in counts.most_common(MAX_KEYWORDS)]

    bigrams = Counter(
        f"{a} {b}" for a, b in zip(tokens, tokens[1:])
        if a not in STOPWORDS and b not in STOPWORDS
    )
    top_bigrams = [b for b, _ in bigrams.most_common(MAX_BIGRAMS)]

    combined = []
    for item in (stem_tokens or []) + top_words + top_bigrams:
        if item and item not in combined:
            combined.append(item)
    return combined


def _extract_pdf_text(path: Path) -> str:
    text_parts: List[str] = []
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
        except Exception as exc:
            logger.warning(f"PDF extraction failed for {path.name}: {exc}")

    text = _normalize_text(" ".join(text_parts))
    if len(text) > MAX_TEXT_CHARS:
        return text[:MAX_TEXT_CHARS]
    return text


def _stem_from_filename(path: Path) -> List[str]:
    stem_tokens = re.split(r"[_\W]+", path.stem.lower())
    return [t for t in stem_tokens if len(t) > 2 and t not in STOPWORDS]


def build_document_index(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    cache_path: Optional[Path] = None,
) -> List[DocumentInfo]:
    cache_path = cache_path or (DEFAULT_CACHE_DIR / "_pdf_index.json")
    cached = _load_json(cache_path)
    cached_docs = {doc["path"]: doc for doc in cached.get("documents", [])}

    documents: List[DocumentInfo] = []
    if not pdf_dir.exists():
        return documents

    pdfs = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    for pdf in pdfs:
        rel_path = str(pdf.relative_to(pdf_dir))
        mtime = pdf.stat().st_mtime
        cached_doc = cached_docs.get(rel_path)
        if cached_doc and cached_doc.get("mtime") == mtime:
            documents.append(DocumentInfo(
                name=cached_doc.get("name", pdf.name),
                path=rel_path,
                mtime=mtime,
                word_count=cached_doc.get("word_count", 0),
                keywords=cached_doc.get("keywords", []),
                text=cached_doc.get("text", ""),
                theme="uncategorized",
                theme_terms=[],
            ))
            continue

        text = _extract_pdf_text(pdf)
        stem_tokens = _stem_from_filename(pdf)
        keywords = _extract_keywords(text, stem_tokens=stem_tokens)
        word_count = len(_tokenize(text))
        documents.append(DocumentInfo(
            name=pdf.name,
            path=rel_path,
            mtime=mtime,
            word_count=word_count,
            keywords=keywords,
            text=text,
            theme="uncategorized",
            theme_terms=[],
        ))

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "documents": [
            {
                "name": doc.name,
                "path": doc.path,
                "mtime": doc.mtime,
                "word_count": doc.word_count,
                "keywords": doc.keywords,
                "text": doc.text,
            }
            for doc in documents
        ],
    }
    _write_json(cache_path, payload)
    return documents


class DocumentIntel:
    def __init__(
        self,
        pdf_dir: Path = DEFAULT_PDF_DIR,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        update_seconds: int = DEFAULT_UPDATE_SECONDS,
    ) -> None:
        self.pdf_dir = pdf_dir
        self.cache_dir = cache_dir
        self.update_seconds = update_seconds
        self.cache_path = cache_dir / "_document_intel_index.json"
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._news_ingestor = get_news_ingestor()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._runner, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _runner(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception as exc:
                logger.warning(f"DocumentIntel refresh failed: {exc}")
            self._stop.wait(self.update_seconds)

    def refresh(self) -> Dict:
        with self._lock:
            # 1. Build local document index
            documents = build_document_index(self.pdf_dir, self.cache_dir / "_pdf_index.json")
            
            # 2. Fetch news from pipeline
            news_data = self._news_ingestor.fetch_all()
            
            payload = {
                "last_update": datetime.utcnow().isoformat(),
                "documents": [
                    {
                        "name": doc.name,
                        "path": doc.path,
                        "mtime": doc.mtime,
                        "word_count": doc.word_count,
                        "keywords": doc.keywords,
                    }
                    for doc in documents
                ],
                "news": news_data,  # Dict[category, List[Article]]
            }
            _write_json(self.cache_path, payload)
            return payload

    def get_snapshot(self) -> Dict:
        """Return cached data or refresh if missing."""
        cached = _load_json(self.cache_path)
        if cached and cached.get("news"):
            return cached
        return self.refresh()


_DOC_INTEL: Optional[DocumentIntel] = None


def get_document_intel() -> DocumentIntel:
    global _DOC_INTEL
    if _DOC_INTEL is None:
        _DOC_INTEL = DocumentIntel()
    return _DOC_INTEL

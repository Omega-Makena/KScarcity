"""
Policy Search Engine — Multi-Source Evidence Retrieval

Searches across all available data sources to find evidence relevant
to a policy bill's provisions:

1. Tweet corpus (synthetic_kenya_policy/tweets.csv) — historical policy reactions
2. General tweets (synthetic_kenya/tweets.csv) — broader social context
3. News cache (data/news_cache/*.json) — media context
4. Incident data (kshield_kenya_unified_incidents_*.csv) — protest/violence history
5. Policy events (scarcity/synthetic/policy_events.py) — historical bill patterns

Uses keyword pre-filtering + semantic embeddings for efficient search.

Usage:
    searcher = PolicySearchEngine(embeddings)
    results = await searcher.search_all(bill, top_k=30)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..db.news_db import NewsDatabase
from ..news_content import build_excerpt

logger = logging.getLogger(__name__)

# Project data root — resolve relative to this file
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = _PROJECT_ROOT / "data"


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """Single search result from any source."""
    source: str          # "tweet", "news", "incident", "policy_event"
    text: str            # The content
    similarity: float    # Cosine similarity to query (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResults:
    """Aggregated results from all sources."""
    query: str
    tweets: List[SearchResult] = field(default_factory=list)
    news: List[SearchResult] = field(default_factory=list)
    incidents: List[SearchResult] = field(default_factory=list)
    policy_events: List[SearchResult] = field(default_factory=list)
    total_found: int = 0

    @property
    def all_results(self) -> List[SearchResult]:
        combined = self.tweets + self.news + self.incidents + self.policy_events
        return sorted(combined, key=lambda r: r.similarity, reverse=True)

    @property
    def top_evidence(self) -> List[SearchResult]:
        return self.all_results[:20]

    def summary_text(self, max_items: int = 10) -> str:
        """Format top results as context for LLM."""
        lines = []
        for r in self.all_results[:max_items]:
            lines.append(f"[{r.source}|sim={r.similarity:.2f}] {r.text[:200]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Search Engine
# ═══════════════════════════════════════════════════════════════════════════

class PolicySearchEngine:
    """
    Multi-source search engine for policy impact evidence.
    
    Uses OllamaEmbeddings for semantic search + keyword pre-filtering.
    """

    def __init__(self, embeddings=None, data_dir: Optional[Path] = None):
        """
        Args:
            embeddings: OllamaEmbeddings instance (for semantic search)
            data_dir: Override data directory path
        """
        self.embeddings = embeddings
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._tweet_cache: Optional[List[Dict]] = None
        self._policy_tweet_cache: Optional[List[Dict]] = None
        self._news_cache: Optional[Dict[str, List[Dict]]] = None
        self._news_db_cache: Optional[List[Dict]] = None
        self._news_db = NewsDatabase()

    # ─── Main Search ────────────────────────────────────────────────────

    async def search_all(
        self,
        bill,
        top_k: int = 30,
    ) -> SearchResults:
        """
        Search all sources for evidence related to a BillAnalysis.
        
        Args:
            bill: BillAnalysis instance
            top_k: Max results per source
            
        Returns:
            SearchResults with evidence from all sources
        """
        # Build search query from bill keywords
        query_parts = []
        if bill.title:
            query_parts.append(bill.title)
        query_parts.extend(bill.keywords_en[:5])
        query_parts.extend(bill.keywords_sw[:3])
        for p in bill.top_provisions[:3]:
            query_parts.append(p.description[:100])
        query = " ".join(query_parts)

        # Build keyword set for pre-filtering
        keywords = set()
        for kw in bill.keywords_en + bill.keywords_sw + bill.hashtags:
            keywords.add(kw.lower().strip("#"))
        for p in bill.provisions:
            keywords.update(k.lower() for k in p.keywords_en)
            keywords.update(k.lower() for k in p.keywords_sw)

        results = SearchResults(query=query)

        # Search each source (sequentially to avoid overloading Ollama)
        tweets = await self._search_tweets(query, keywords, top_k)
        results.tweets = tweets

        news = await self._search_news(query, keywords, top_k)
        results.news = news

        incidents = self._search_incidents(keywords, top_k)
        results.incidents = incidents

        policy_matches = self._search_policy_events(bill, top_k)
        results.policy_events = policy_matches

        results.total_found = len(results.all_results)
        return results

    async def search_query(
        self,
        query: str,
        top_k: int = 20,
    ) -> SearchResults:
        """
        Free-text search across all sources.
        
        Args:
            query: Natural language search query
            top_k: Max results per source
        """
        keywords = set(re.findall(r'\w+', query.lower()))

        results = SearchResults(query=query)
        results.tweets = await self._search_tweets(query, keywords, top_k)
        results.news = await self._search_news(query, keywords, top_k)
        results.incidents = self._search_incidents(keywords, top_k)
        results.total_found = len(results.all_results)
        return results

    # ─── Tweet Search ───────────────────────────────────────────────────

    async def _search_tweets(
        self,
        query: str,
        keywords: set,
        top_k: int,
    ) -> List[SearchResult]:
        """Search tweet corpus with keyword pre-filter + semantic ranking."""
        # Load both tweet corpora
        policy_tweets = self._load_policy_tweets()
        general_tweets = self._load_general_tweets()
        all_tweets = policy_tweets + general_tweets

        if not all_tweets:
            logger.info("No tweet data available for search")
            return []

        # Keyword pre-filter (reduce corpus before embedding)
        candidates = []
        for tweet in all_tweets:
            text = tweet.get("text", "").lower()
            if any(kw in text for kw in keywords if len(kw) > 2):
                candidates.append(tweet)

        # If too few keyword matches, take random sample
        if len(candidates) < 10:
            import random
            sample_size = min(200, len(all_tweets))
            candidates = random.sample(all_tweets, sample_size)

        # Cap candidates for embedding (memory/speed)
        candidates = candidates[:500]

        # Semantic ranking if embeddings available
        if self.embeddings and candidates:
            texts = [c.get("text", "") for c in candidates]
            try:
                similar = await self.embeddings.find_similar(query, texts, top_k=top_k)
                results = []
                for idx, sim, text in similar:
                    meta = {k: v for k, v in candidates[idx].items() if k != "text"}
                    results.append(SearchResult(
                        source="tweet",
                        text=text,
                        similarity=sim,
                        metadata=meta,
                    ))
                return results
            except Exception as e:
                logger.warning(f"Semantic tweet search failed: {e}")

        # Fallback: keyword-only ranking
        results = []
        for tweet in candidates[:top_k]:
            text = tweet.get("text", "")
            # Score by keyword overlap
            text_lower = text.lower()
            score = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
            meta = {k: v for k, v in tweet.items() if k != "text"}
            results.append(SearchResult(
                source="tweet", text=text, similarity=min(score, 1.0), metadata=meta
            ))
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    # ─── News Search ────────────────────────────────────────────────────

    async def _search_news(
        self,
        query: str,
        keywords: set,
        top_k: int,
    ) -> List[SearchResult]:
        """Search news history with full-content traceability."""
        all_articles = []

        # 1) Prefer DB-backed full-content records (URL + extracted text + IDs).
        for row in self._load_news_from_db():
            all_articles.append(
                {
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "source": row.get("source", ""),
                    "published_at": row.get("published_at", ""),
                    "description": row.get("description", ""),
                    "content": row.get("content", ""),
                    "extracted_text": row.get("extracted_text", ""),
                    "evidence_excerpt": build_excerpt(str(row.get("extracted_text", "") or "")),
                    "_topic": row.get("category", ""),
                    "article_id": row.get("article_id"),
                    "content_record_id": row.get("content_record_id"),
                    "extraction_status": row.get("extraction_status", ""),
                    "content_storage_path": row.get("storage_path", ""),
                }
            )

        # 2) Fall back to cache files (with lower traceability detail).
        news = self._load_news_cache()
        for topic, articles in news.items():
            for article in articles:
                if isinstance(article, dict):
                    all_articles.append({**article, "_topic": topic})
                elif isinstance(article, str):
                    all_articles.append({
                        "title": article[:200],
                        "content": article,
                        "_topic": topic,
                    })

        if not all_articles:
            return []

        # Keyword pre-filter
        candidates = []
        for article in all_articles:
            searchable = (
                article.get("title", "") + " " +
                article.get("extracted_text", "") + " " +
                article.get("content", "") + " " +
                article.get("description", "")
            ).lower()
            if any(kw in searchable for kw in keywords if len(kw) > 2):
                candidates.append(article)

        if not candidates:
            candidates = all_articles[:100]

        candidates = candidates[:200]

        # Semantic ranking if available
        if self.embeddings and candidates:
            texts = [
                f"{a.get('title', '')}. "
                f"{(a.get('extracted_text') or a.get('content') or a.get('description') or '')[:1200]}"
                for a in candidates
            ]
            try:
                similar = await self.embeddings.find_similar(query, texts, top_k=top_k)
                results = []
                for idx, sim, text in similar:
                    a = candidates[idx]
                    results.append(SearchResult(
                        source="news",
                        text=text,
                        similarity=sim,
                        metadata={
                            "title": a.get("title", ""),
                            "url": a.get("url", ""),
                            "source": a.get("source", ""),
                            "topic": a.get("_topic", ""),
                            "published_at": a.get("published_at", ""),
                            "article_id": a.get("article_id"),
                            "content_record_id": a.get("content_record_id"),
                            "content_storage_path": a.get("content_storage_path", ""),
                            "extraction_status": a.get("extraction_status", ""),
                            "evidence_excerpt": a.get("evidence_excerpt") or build_excerpt(
                                str(a.get("extracted_text", "") or a.get("content", "") or "")
                            ),
                        },
                    ))
                return results
            except Exception as e:
                logger.warning(f"Semantic news search failed: {e}")

        # Keyword fallback
        results = []
        for a in candidates[:top_k]:
            text = (
                f"{a.get('title', '')}. "
                f"{(a.get('extracted_text') or a.get('content') or a.get('description') or '')[:800]}"
            )
            text_lower = text.lower()
            score = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
            results.append(SearchResult(
                source="news", text=text, similarity=min(score, 1.0),
                metadata={
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "source": a.get("source", ""),
                    "topic": a.get("_topic", ""),
                    "published_at": a.get("published_at", ""),
                    "article_id": a.get("article_id"),
                    "content_record_id": a.get("content_record_id"),
                    "content_storage_path": a.get("content_storage_path", ""),
                    "extraction_status": a.get("extraction_status", ""),
                    "evidence_excerpt": a.get("evidence_excerpt") or build_excerpt(
                        str(a.get("extracted_text", "") or a.get("content", "") or "")
                    ),
                },
            ))
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    # ─── Incident Search ────────────────────────────────────────────────

    def _search_incidents(
        self,
        keywords: set,
        top_k: int,
    ) -> List[SearchResult]:
        """Search historical incident CSV data."""
        incident_files = list(self.data_dir.glob("kshield_kenya_unified_incidents*.csv"))
        if not incident_files:
            return []

        results = []
        for fpath in incident_files:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        searchable = " ".join(str(v) for v in row.values()).lower()
                        score = sum(1 for kw in keywords if kw in searchable) / max(len(keywords), 1)
                        if score > 0:
                            desc = row.get("description", row.get("notes", str(row)))
                            results.append(SearchResult(
                                source="incident",
                                text=str(desc)[:500],
                                similarity=min(score, 1.0),
                                metadata={
                                    "date": row.get("date", row.get("event_date", "")),
                                    "county": row.get("county", row.get("location", "")),
                                    "type": row.get("event_type", row.get("type", "")),
                                    "fatalities": row.get("fatalities", ""),
                                },
                            ))
            except Exception as e:
                logger.warning(f"Incident file read error: {e}")

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    # ─── Policy Event Matching ──────────────────────────────────────────

    def _search_policy_events(
        self,
        bill,
        top_k: int,
    ) -> List[SearchResult]:
        """Match bill against known policy events from policy_events.py."""
        try:
            from scarcity.synthetic.policy_events import build_kenya_2026_events, PolicyEvent
            from datetime import datetime
            events = build_kenya_2026_events(datetime(2026, 1, 1))
        except ImportError:
            logger.info("policy_events.py not available")
            return []

        bill_keywords = set(
            kw.lower() for kw in
            (bill.keywords_en + bill.keywords_sw + bill.hashtags +
             [bill.title.lower()] + bill.sectors)
        )

        results = []
        for evt in events:
            evt_keywords = set(
                kw.lower() for kw in
                (evt.keywords_en + evt.keywords_sw + evt.hashtags +
                 [evt.name.lower(), evt.sector.value, evt.description.lower()])
            )
            overlap = bill_keywords & evt_keywords
            if not overlap:
                # Check if bill title words appear in event
                title_words = set(bill.title.lower().split())
                name_words = set(evt.name.lower().split())
                overlap = title_words & name_words

            score = len(overlap) / max(len(bill_keywords), 1)
            if score > 0:
                results.append(SearchResult(
                    source="policy_event",
                    text=f"{evt.name}: {evt.description}",
                    similarity=min(score * 2, 1.0),  # Boost policy event scores
                    metadata={
                        "event_id": evt.event_id,
                        "sector": evt.sector.value,
                        "severity": evt.severity,
                        "hashtags": evt.hashtags[:5],
                        "affected_counties": evt.affected_counties[:10],
                    },
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    # ─── Data Loaders ───────────────────────────────────────────────────

    def _load_policy_tweets(self) -> List[Dict]:
        """Load policy-specific tweets from CSV."""
        if self._policy_tweet_cache is not None:
            return self._policy_tweet_cache

        path = self.data_dir / "synthetic_kenya_policy" / "tweets.csv"
        self._policy_tweet_cache = self._load_csv(path, max_rows=10000)
        return self._policy_tweet_cache

    def _load_general_tweets(self) -> List[Dict]:
        """Load general tweets from CSV."""
        if self._tweet_cache is not None:
            return self._tweet_cache

        path = self.data_dir / "synthetic_kenya" / "tweets.csv"
        self._tweet_cache = self._load_csv(path, max_rows=5000)
        return self._tweet_cache

    def _load_news_cache(self) -> Dict[str, List[Dict]]:
        """Load all news cache JSON files."""
        if self._news_cache is not None:
            return self._news_cache

        self._news_cache = {}
        cache_dir = self.data_dir / "news_cache"
        if not cache_dir.exists():
            return self._news_cache

        for fpath in cache_dir.glob("*.json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    topic = fpath.stem
                    if isinstance(data, list):
                        self._news_cache[topic] = data
                    elif isinstance(data, dict) and "articles" in data:
                        articles = data["articles"]
                        if isinstance(articles, list):
                            self._news_cache[topic] = articles
                        else:
                            self._news_cache[topic] = []
                    elif isinstance(data, dict):
                        self._news_cache[topic] = [data]
                    else:
                        self._news_cache[topic] = []
            except Exception as e:
                logger.warning(f"Failed to load news cache {fpath.name}: {e}")

        return self._news_cache

    def _load_news_from_db(self, limit: int = 1000) -> List[Dict]:
        """Load traceable news records from SQLite (article+content join)."""
        if self._news_db_cache is not None:
            return self._news_db_cache
        try:
            self._news_db_cache = self._news_db.get_recent_articles_with_content(limit=limit)
        except Exception as exc:
            logger.warning(f"Failed loading news from db: {exc}")
            self._news_db_cache = []
        return self._news_db_cache

    @staticmethod
    def _load_csv(path: Path, max_rows: int = 10000) -> List[Dict]:
        """Load CSV file into list of dicts."""
        if not path.exists():
            return []
        rows = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(dict(row))
        except Exception as e:
            logger.warning(f"CSV load error {path}: {e}")
        return rows

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kshiked.pulse.db import Database, DatabaseConfig
from kshiked.pulse.db.models import LLMAnalysis, ProcessedSignal, Platform as DBPlatform
from kshiked.pulse.ingestion.orchestrator import IngestionConfig, IngestionOrchestrator
from kshiked.pulse.ingestion.pipeline import PipelineIntegration, run_full_pipeline
from kshiked.pulse.llm.base import ThreatClassification, ThreatTier
from kshiked.pulse.mapper import SignalDetection, SignalID
from kshiked.pulse.scrapers.base import Platform, ScraperResult
from kshiked.pulse.scrapers.x_scraper import XScraper, XScraperConfig


def _sample_scraper_result(platform_id: str = "x-1") -> ScraperResult:
    return ScraperResult(
        platform=Platform.TWITTER,
        platform_id=platform_id,
        text="Kenya protest updates from Nairobi",
        author_id="author-1",
        author_username="handle1",
        posted_at=datetime.now(timezone.utc),
        scraped_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_xscraper_web_primary_maps_web_tweet(monkeypatch):
    tweet = SimpleNamespace(
        tweet_id="123",
        text="Nairobi update",
        created_at=datetime.now(timezone.utc),
        language="en",
        like_count=7,
        retweet_count=3,
        reply_count=2,
        view_count=99,
        author_id="a1",
        author_username="user1",
        author_display_name="User One",
        author_followers=250,
        author_verified=True,
        hashtags="Kenya,Nairobi",
        mentions="gov_ke",
        urls="https://example.com",
        media_urls="https://img.example/1.jpg",
        location_county="Nairobi",
        mentioned_counties="Nairobi,Kiambu",
        reply_to_tweet_id="",
        conversation_id="conv-1",
        source="twikit",
        reply_to_user="",
        is_retweet=False,
        is_quote=False,
        latitude=-1.29,
        longitude=36.82,
        author_location="Kenya",
        author_bio="bio",
        scraped_at=datetime.now(timezone.utc).isoformat(),
    )

    class _FakeWeb:
        async def search_tweets(self, query: str, limit: int = 40):
            return [tweet]

        def save_tweets_csv(self, _tweets):
            return None

        def save_accounts_csv(self, _accounts):
            return None

        def get_accounts(self):
            return []

        def export_dashboard_csv(self, _tweets):
            return None

    scraper = XScraper(
        XScraperConfig(
            backend_mode="web_primary",
            web_export_csv=False,
        )
    )
    scraper._web_scraper = _FakeWeb()

    results = await scraper._scrape_impl("Kenya", limit=1)

    assert len(results) == 1
    result = results[0]
    assert result.platform == Platform.TWITTER
    assert result.platform_id == "123"
    assert result.likes == 7
    assert result.shares == 3
    assert result.geo_location == "Nairobi"
    assert result.hashtags == ["Kenya", "Nairobi"]
    assert result.mentioned_locations == ["Nairobi", "Kiambu"]


@pytest.mark.asyncio
async def test_xscraper_web_primary_falls_back_to_legacy(monkeypatch):
    class _FailingWeb:
        async def search_tweets(self, query: str, limit: int = 40):
            raise RuntimeError("web backend unavailable")

    expected = _sample_scraper_result("legacy-1")
    scraper = XScraper(XScraperConfig(backend_mode="web_primary", web_export_csv=False))
    scraper._web_scraper = _FailingWeb()
    scraper._twscrape_api = object()

    async def _fake_twscrape(query: str, limit: int, since=None):
        return [expected]

    monkeypatch.setattr(scraper, "_scrape_twscrape", _fake_twscrape)

    results = await scraper._scrape_impl("Kenya", limit=1)

    assert len(results) == 1
    assert results[0].platform_id == "legacy-1"


@pytest.mark.asyncio
async def test_orchestrator_scrape_social_media_dedups_x_posts(tmp_path: Path, monkeypatch):
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orchestrator_test.db'}"
    config = IngestionConfig(database_url=db_url)

    duplicate = _sample_scraper_result("dup-1")

    class _FakeXScraper:
        async def scrape(self, query: str, limit: int):
            return [duplicate, duplicate]

        async def close(self):
            return None

    async def _fake_init_scrapers(self):
        self._scrapers = {"x": _FakeXScraper()}

    async def _fake_init_ecommerce_scrapers(self):
        self._ecommerce_scrapers = {}

    monkeypatch.setattr(IngestionOrchestrator, "_init_scrapers", _fake_init_scrapers)
    monkeypatch.setattr(
        IngestionOrchestrator,
        "_init_ecommerce_scrapers",
        _fake_init_ecommerce_scrapers,
    )

    async with IngestionOrchestrator(config) as orchestrator:
        posts = await orchestrator.scrape_social_media(
            search_terms=["Kenya"],
            limit=5,
        )
        assert len(posts) == 1
        stats = await orchestrator._db.get_stats()
        assert stats["posts"] == 1


@pytest.mark.asyncio
async def test_pipeline_store_analysis_persists_required_fields(tmp_path: Path):
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'pipeline_store.db'}"
    db = Database(DatabaseConfig(url=db_url))
    await db.connect()
    try:
        post_result = _sample_scraper_result("persist-1")
        await db.add(post_result.to_social_post())

        pipeline = PipelineIntegration(database=db)
        classification = ThreatClassification(
            tier=ThreatTier.TIER_3,
            confidence=0.85,
            reasoning="Mobilization language detected",
            base_risk=0.64,
            intent_score=0.7,
            capability_score=0.5,
            specificity_score=0.8,
            reach_score=0.6,
            model_name="test-llm",
            prompt_tokens=111,
            completion_tokens=22,
        )
        detections = [
            SignalDetection(
                signal_id=SignalID.MOBILIZATION_LANGUAGE,
                intensity=0.9,
                confidence=0.8,
                raw_score=1.25,
                context={"matched_keywords": ["rise up"]},
            )
        ]

        await pipeline._store_analysis(post_result, classification, detections)

        async with db.session() as session:
            analysis_rows = (await session.execute(select(LLMAnalysis))).scalars().all()
            signal_rows = (await session.execute(select(ProcessedSignal))).scalars().all()

        assert len(analysis_rows) == 1
        stored_tier = getattr(analysis_rows[0].threat_tier, "value", analysis_rows[0].threat_tier)
        assert stored_tier == ThreatTier.TIER_3.value
        assert analysis_rows[0].threat_confidence == pytest.approx(0.85)
        assert len(signal_rows) == 1
        assert signal_rows[0].signal_id == "MOBILIZATION_LANGUAGE"
        assert signal_rows[0].signal_category
        assert signal_rows[0].raw_score == pytest.approx(1.25)
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_run_full_pipeline_smoke_with_mocked_orchestrator(tmp_path: Path, monkeypatch):
    import kshiked.pulse.ingestion as ingestion_pkg

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'full_pipeline_smoke.db'}"
    post = _sample_scraper_result("smoke-1").to_social_post()
    post.platform = DBPlatform.TWITTER

    class _FakeConfig:
        @classmethod
        def from_env(cls):
            return SimpleNamespace(
                database_url=db_url,
                gemini_api_key="",
            )

    class _FakeOrchestrator:
        def __init__(self, _config):
            self.config = _config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def scrape_social_media(self, search_terms=None):
            return [post]

    monkeypatch.setattr(ingestion_pkg, "IngestionConfig", _FakeConfig)
    monkeypatch.setattr(ingestion_pkg, "IngestionOrchestrator", _FakeOrchestrator)

    result = await run_full_pipeline(search_terms=["Kenya"])

    assert result["posts_processed"] == 1
    assert "threat_summary" in result
    assert "results" in result

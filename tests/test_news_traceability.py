from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kshiked.pulse.db.news_db import NewsDatabase


def test_news_content_upsert_and_trace(tmp_path: Path):
    db = NewsDatabase(db_path=tmp_path / "news.sqlite")

    db.add_articles(
        "technology",
        [
            {
                "url": "https://news.example.com/a1",
                "title": "A1",
                "source": "Example",
                "published_at": "2026-02-18T00:00:00Z",
                "description": "desc",
                "content": "content",
            }
        ],
    )

    record = db.upsert_content_extraction(
        "https://news.example.com/a1",
        {
            "canonical_url": "https://news.example.com/a1",
            "extracted_text": "Full extracted article body for traceability.",
            "extraction_method": "unit_test",
            "content_hash": "abc123",
            "status": "ok",
            "http_status": 200,
        },
    )

    assert record.get("article_id") is not None
    assert record.get("status") == "ok"
    assert "traceability" in record.get("extracted_text", "")

    enriched = db.get_recent_articles_with_content(limit=5)
    assert enriched
    assert enriched[0].get("content_record_id") is not None

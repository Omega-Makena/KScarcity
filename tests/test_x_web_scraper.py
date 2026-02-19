from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kshiked.pulse.scrapers.x_web_scraper import KenyaXScraper, ScrapedTweet, XSessionConfig


class _FakeTweet:
    def __init__(self, tweet_id: int):
        self.id = tweet_id
        self.text = f"tweet-{tweet_id}"


class _FakeResponse:
    def __init__(self, tweet_ids: list[int], next_cursor: str = ""):
        self._tweets = [_FakeTweet(tid) for tid in tweet_ids]
        self._next: _FakeResponse | None = None
        self.next_cursor = next_cursor

    def __iter__(self):
        return iter(self._tweets)

    async def next(self):
        return self._next


class _FakeClient:
    def __init__(
        self,
        first_response: _FakeResponse,
        cursor_map: dict[str, _FakeResponse] | None = None,
        raise_rate_limit_once: bool = False,
        raise_detection_once: bool = False,
    ):
        self.first_response = first_response
        self.cursor_map = cursor_map or {}
        self.raise_rate_limit_once = raise_rate_limit_once
        self.raise_detection_once = raise_detection_once
        self.search_calls = 0

    async def search_tweet(self, query: str, **kwargs):
        self.search_calls += 1
        if self.raise_rate_limit_once:
            self.raise_rate_limit_once = False
            raise RuntimeError("429 Too many requests")
        if self.raise_detection_once:
            self.raise_detection_once = False
            raise RuntimeError("challenge required")

        cursor = str(kwargs.get("cursor", "") or "")
        if cursor and cursor in self.cursor_map:
            return self.cursor_map[cursor]
        return self.first_response


def _build_response_chain(pages: list[list[int]]) -> tuple[_FakeResponse, dict[str, _FakeResponse]]:
    responses: list[_FakeResponse] = []
    for idx, page_ids in enumerate(pages):
        next_cursor = f"c{idx + 1}" if idx + 1 < len(pages) else ""
        responses.append(_FakeResponse(page_ids, next_cursor=next_cursor))

    for idx in range(len(responses) - 1):
        responses[idx]._next = responses[idx + 1]

    cursor_map: dict[str, _FakeResponse] = {}
    for idx in range(len(responses) - 1):
        cursor_map[f"c{idx + 1}"] = responses[idx + 1]
    return responses[0], cursor_map


def _scraped(tweet: _FakeTweet) -> ScrapedTweet:
    return ScrapedTweet(
        tweet_id=str(tweet.id),
        text=tweet.text,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def no_sleep(monkeypatch):
    async def _noop_sleep(_seconds: float):
        return None

    monkeypatch.setattr("kshiked.pulse.scrapers.x_web_scraper.asyncio.sleep", _noop_sleep)


@pytest.mark.asyncio
async def test_search_tweets_deep_pagination_collects_to_limit(no_sleep):
    first, cursor_map = _build_response_chain([[1, 2], [3, 4], [5, 6], [7, 8]])
    client = _FakeClient(first_response=first, cursor_map=cursor_map)

    scraper = KenyaXScraper(
        username="u",
        password="p",
        email="e@example.com",
        enable_checkpoint=False,
        request_delay_s=0,
        query_delay_s=0,
    )
    scraper._client = client
    scraper._clients = [client]
    scraper._active_session_configs = [
        XSessionConfig("session-1", "u", "p", "e@example.com", Path("cookies.json"), "")
    ]
    scraper._parse_tweet = _scraped  # type: ignore[method-assign]

    tweets = await scraper.search_tweets("Kenya", limit=7, max_pages=20)

    assert len(tweets) == 7
    assert [t.tweet_id for t in tweets] == ["1", "2", "3", "4", "5", "6", "7"]


@pytest.mark.asyncio
async def test_search_tweets_rotates_session_on_rate_limit(no_sleep):
    first, cursor_map = _build_response_chain([[11, 12]])
    rate_limited_client = _FakeClient(
        first_response=first,
        cursor_map=cursor_map,
        raise_rate_limit_once=True,
    )
    healthy_client = _FakeClient(first_response=first, cursor_map=cursor_map)

    scraper = KenyaXScraper(
        username="u",
        password="p",
        email="e@example.com",
        enable_checkpoint=False,
        request_delay_s=0,
        query_delay_s=0,
    )
    scraper._clients = [rate_limited_client, healthy_client]
    scraper._active_session_configs = [
        XSessionConfig("session-1", "u", "p", "e@example.com", Path("c1.json"), "http://p1"),
        XSessionConfig("session-2", "u", "p", "e@example.com", Path("c2.json"), "http://p2"),
    ]
    scraper._active_session_idx = 0
    scraper._client = rate_limited_client
    scraper._parse_tweet = _scraped  # type: ignore[method-assign]

    tweets = await scraper.search_tweets("Kenya", limit=2, max_pages=5)

    assert len(tweets) == 2
    assert scraper._active_session_idx == 1
    assert healthy_client.search_calls >= 1


@pytest.mark.asyncio
async def test_checkpoint_resume_continues_without_duplicates(no_sleep, tmp_path: Path):
    checkpoint_path = tmp_path / "x_scraper_checkpoint.json"
    pages = [[1, 2], [3, 4], [5, 6]]
    first_resp, first_cursor_map = _build_response_chain(pages)

    scraper_first = KenyaXScraper(
        username="u",
        password="p",
        email="e@example.com",
        checkpoint_path=checkpoint_path,
        enable_checkpoint=True,
        request_delay_s=0,
        query_delay_s=0,
    )
    client_first = _FakeClient(first_response=first_resp, cursor_map=first_cursor_map)
    scraper_first._client = client_first
    scraper_first._clients = [client_first]
    scraper_first._active_session_configs = [
        XSessionConfig("session-1", "u", "p", "e@example.com", Path("cookies.json"), "")
    ]
    scraper_first._parse_tweet = _scraped  # type: ignore[method-assign]

    first_batch = await scraper_first.search_tweets("Kenya", limit=2, max_pages=10)
    assert [t.tweet_id for t in first_batch] == ["1", "2"]
    assert checkpoint_path.exists()

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["query_state"]["Kenya"]["page"] >= 1

    second_resp, second_cursor_map = _build_response_chain(pages)
    scraper_second = KenyaXScraper(
        username="u",
        password="p",
        email="e@example.com",
        checkpoint_path=checkpoint_path,
        enable_checkpoint=True,
        resume_from_checkpoint=True,
        request_delay_s=0,
        query_delay_s=0,
    )
    scraper_second._load_checkpoint()
    client_second = _FakeClient(first_response=second_resp, cursor_map=second_cursor_map)
    scraper_second._client = client_second
    scraper_second._clients = [client_second]
    scraper_second._active_session_configs = [
        XSessionConfig("session-1", "u", "p", "e@example.com", Path("cookies.json"), "")
    ]
    scraper_second._parse_tweet = _scraped  # type: ignore[method-assign]

    resumed_batch = await scraper_second.search_tweets("Kenya", limit=4, max_pages=10)

    assert [t.tweet_id for t in resumed_batch] == ["3", "4", "5", "6"]


@pytest.mark.asyncio
async def test_detection_sets_24h_cooldown_checkpoint(no_sleep, tmp_path: Path):
    checkpoint_path = tmp_path / "x_scraper_checkpoint.json"
    first, cursor_map = _build_response_chain([[21, 22]])
    detecting_client = _FakeClient(
        first_response=first,
        cursor_map=cursor_map,
        raise_detection_once=True,
    )

    scraper = KenyaXScraper(
        username="u",
        password="p",
        email="e@example.com",
        checkpoint_path=checkpoint_path,
        enable_checkpoint=True,
        detection_cooldown_hours=24,
        request_delay_s=0,
        query_delay_s=0,
    )
    scraper._client = detecting_client
    scraper._clients = [detecting_client]
    scraper._active_session_configs = [
        XSessionConfig("session-1", "u", "p", "e@example.com", Path("cookies.json"), "")
    ]
    scraper._parse_tweet = _scraped  # type: ignore[method-assign]

    tweets = await scraper.search_tweets("Kenya", limit=2, max_pages=5)
    assert tweets == []
    assert scraper._abort_scrape is True
    assert scraper._cooldown_until
    assert checkpoint_path.exists()

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload.get("cooldown_until")

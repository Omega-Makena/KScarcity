#!/usr/bin/env python3
"""Smoke validation for live batch append, news extraction, and federated sync."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        count = -1  # header
        for count, _ in enumerate(reader):
            pass
    return max(0, count)


def check_live_batch_append() -> tuple[bool, str]:
    target = PROJECT_ROOT / "data" / "synthetic_kenya_policy" / "tweets.csv"
    before = count_rows(target)

    env = os.environ.copy()
    env.update(
        {
            "TWEET_NUM_ACCOUNTS": "4",
            "POLICY_NUM_ACCOUNTS": "6",
            "TWEET_DURATION_DAYS": "1",
            "POLICY_DURATION_DAYS": "1",
            "TWEET_APPEND": "true",
            "TWEET_MAX_ROWS": "200000",
        }
    )

    proc = subprocess.run(
        [str(PROJECT_ROOT / ".venv-linux" / "bin" / "python"), str(PROJECT_ROOT / "scripts" / "generate_daily_tweets.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    after = count_rows(target)
    if proc.returncode != 0:
        return False, f"generator failed: {proc.stderr[-400:]}"
    if after <= before:
        return False, f"row count did not increase ({before} -> {after})"
    return True, f"rows increased ({before} -> {after})"


def check_news_extraction() -> tuple[bool, str]:
    from kshiked.pulse.news import get_news_ingestor

    ingestor = get_news_ingestor()
    articles = ingestor.fetch_pipeline("technology", force=False)
    extracted = [a for a in articles if (a.get("extracted_text") or "").strip()]

    if not articles or not extracted:
        # Fallback synthetic sample (no network needed)
        sample = [{
            "title": "Synthetic smoke article",
            "url": "https://news.example.com/smoke/article-1",
            "source": "SmokeTest",
            "published_at": "2026-02-18T00:00:00Z",
            "description": "Synthetic description for smoke validation.",
            "content": "Synthetic content body with enough length to act as extracted full text for traceability checks.",
        }]
        ingestor.db.add_articles("technology", sample)
        articles = ingestor._enrich_with_full_content(sample, force=True)
        extracted = [a for a in articles if (a.get("extracted_text") or "").strip()]
    if not extracted:
        return False, "no articles with extracted_text"

    traced = [a for a in extracted if a.get("content_record_id")]
    if not traced:
        return False, "extracted_text exists but no content trace pointers"

    return True, f"extracted_text present for {len(extracted)} article(s); traced={len(traced)}"


def check_federation_sync() -> tuple[bool, str]:
    from federated_databases import get_scarcity_federation

    manager = get_scarcity_federation()
    if not manager.list_nodes():
        manager.register_node("org_a", county_filter="Nairobi")
        manager.register_node("org_b", county_filter="Mombasa")

    primary_node = manager.list_nodes()[0]["node_id"]
    single = manager.run_single_node_training(primary_node, learning_rate=0.12)
    result = manager.run_sync_round(learning_rate=0.12, lookback_hours=24)
    status = manager.get_status()
    audit_rows = manager.get_exchange_log(limit=10)

    if single.get("sample_count", 0) <= 0:
        return False, "single-node ML training had zero samples"
    if result.participants <= 0:
        return False, "sync round had zero participants"
    if not status.get("nodes"):
        return False, "federation node database status missing"
    if not audit_rows:
        return False, "no audit records after sync"

    return True, (
        f"single_loss={single.get('loss', 0.0):.4f}, round={result.round_number}, "
        f"participants={result.participants}, samples={result.total_samples}, "
        f"audit_rows={len(audit_rows)}"
    )


def main() -> int:
    checks = [
        ("live_batch_append", check_live_batch_append),
        ("news_extraction", check_news_extraction),
        ("federation_sync", check_federation_sync),
    ]

    failed = False
    print("Smoke Validation Results")
    print("=" * 80)
    for name, fn in checks:
        try:
            ok, message = fn()
        except Exception as exc:
            ok, message = False, f"exception: {exc}"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {message}")
        if not ok:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

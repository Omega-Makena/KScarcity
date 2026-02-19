#!/usr/bin/env python3
"""
Daily Synthetic Tweet Generator — Replaces Twitter Bearer Token Dependency

Generates realistic Kenyan social media data every 24 hours using the existing
Scarcity synthetic pipeline. Designed to run as a scheduled task (cron/Task Scheduler).

Outputs:
  - data/synthetic_kenya/tweets.csv        (general tweets, refreshed daily)
  - data/synthetic_kenya_policy/tweets.csv  (policy-reaction tweets, refreshed daily)
  - data/synthetic_kenya/accounts.csv       (account profiles)
  - data/synthetic_kenya_policy/accounts.csv

Schedule:
  Windows Task Scheduler:
    schtasks /create /tn "KShield Daily Tweets" /tr "python scripts/generate_daily_tweets.py" /sc daily /st 02:00
  Linux cron:
    0 2 * * * cd /path/to/scace4 && python scripts/generate_daily_tweets.py

Settings can be tuned via environment variables or the CONFIG section below.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scarcity.synthetic.accounts import AccountGenerator
from scarcity.synthetic.content import ContentGenerator
from scarcity.synthetic.behavior import BehaviorSimulator
from scarcity.synthetic.vocabulary import COUNTY_COORDINATES, INTERACTION_WEIGHTS
from scarcity.synthetic.scenarios import ScenarioManager
from scarcity.synthetic.policy_events import (
    PolicyEventInjector,
    PolicyPhase,
    PolicySector,
    build_kenya_2026_events,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_tweets")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — tune these or override via env vars
# ═══════════════════════════════════════════════════════════════════════════

# General tweet generation
NUM_ACCOUNTS          = int(os.getenv("TWEET_NUM_ACCOUNTS", "150"))
DURATION_DAYS         = int(os.getenv("TWEET_DURATION_DAYS", "7"))     # Rolling 7-day window
APPEND_MODE           = os.getenv("TWEET_APPEND", "true").lower() == "true"  # Append or overwrite
MAX_ROWS              = int(os.getenv("TWEET_MAX_ROWS", "120000"))     # Cap per CSV

# Policy-specific overrides
POLICY_NUM_ACCOUNTS   = int(os.getenv("POLICY_NUM_ACCOUNTS", "200"))
POLICY_DURATION_DAYS  = int(os.getenv("POLICY_DURATION_DAYS", "7"))

# Output directories
GENERAL_DIR  = PROJECT_ROOT / "data" / "synthetic_kenya"
POLICY_DIR   = PROJECT_ROOT / "data" / "synthetic_kenya_policy"

# Lockfile to prevent overlapping runs
LOCK_FILE = PROJECT_ROOT / "data" / ".daily_tweets.lock"

# Metadata log
META_FILE = PROJECT_ROOT / "data" / ".daily_tweets_meta.json"


# ═══════════════════════════════════════════════════════════════════════════
# GENERAL-PURPOSE COLUMNS (18) — matches synthetic_kenya/tweets.csv
# ═══════════════════════════════════════════════════════════════════════════

GENERAL_COLUMNS = [
    "post_id", "account_id", "timestamp", "text", "intent",
    "interaction_type", "reply_to_post_id", "source_label",
    "like_count", "retweet_count", "location_county", "latitude",
    "longitude", "imperative_rate", "urgency_rate",
    "coordination_score", "escalation_score", "threat_score",
]

# ═══════════════════════════════════════════════════════════════════════════
# POLICY COLUMNS (24) — matches synthetic_kenya_policy/tweets.csv
# ═══════════════════════════════════════════════════════════════════════════

POLICY_COLUMNS = GENERAL_COLUMNS + [
    "sentiment_score", "stance_score", "policy_event_id",
    "policy_phase", "topic_cluster", "policy_severity",
]


def _acquire_lock() -> bool:
    """Simple file-based lock to prevent concurrent runs."""
    if LOCK_FILE.exists():
        # Check if stale (> 2 hours old)
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < 7200:
            logger.warning("Another generation run is active (lockfile exists). Exiting.")
            return False
        logger.info("Removing stale lockfile.")
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock():
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


def _read_existing_csv(path: Path, columns: list) -> list:
    """Read existing CSV rows (as list of dicts). Returns [] if file missing."""
    if not path.exists():
        return []
    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return []


def _write_csv(path: Path, columns: list, rows: list):
    """Write rows to CSV, ensuring directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_meta(general_count: int, policy_count: int, duration_sec: float):
    """Write generation metadata for monitoring."""
    meta = {
        "last_run": datetime.now().isoformat(),
        "general_tweets": general_count,
        "policy_tweets": policy_count,
        "duration_seconds": round(duration_sec, 1),
        "config": {
            "num_accounts": NUM_ACCOUNTS,
            "policy_num_accounts": POLICY_NUM_ACCOUNTS,
            "duration_days": DURATION_DAYS,
            "policy_duration_days": POLICY_DURATION_DAYS,
            "append_mode": APPEND_MODE,
            "max_rows": MAX_ROWS,
        },
    }
    META_FILE.write_text(json.dumps(meta, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# Core Generation — wraps SyntheticPipeline but writes to our target dirs
# ═══════════════════════════════════════════════════════════════════════════

def generate_batch(
    output_dir: Path,
    columns: list,
    num_accounts: int,
    duration_days: int,
    append: bool = True,
    max_rows: int = MAX_ROWS,
    seed: int | None = None,
) -> int:
    """
    Generate a batch of synthetic tweets.

    Uses the full Scarcity pipeline (accounts → behavior → content → scores)
    to produce realistic Kenyan social media data with policy-reaction modeling.

    Returns the total number of rows written.
    """
    # Use current timestamp as seed for variety each run
    if seed is None:
        seed = int(datetime.now().timestamp()) % (2**31)
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Generating {num_accounts} accounts × {duration_days} days (seed={seed})")

    account_gen = AccountGenerator()
    account_gen.__init__(seed=seed)  # re-seed
    content_gen = ContentGenerator(seed=seed)
    behavior_sim = BehaviorSimulator()

    start_date = datetime.now() - timedelta(days=duration_days)
    accounts = account_gen.generate_accounts(num_accounts)
    scenario_manager = ScenarioManager(start_date, duration_days)

    # Collect all activities
    all_activities = []
    for account in accounts:
        activity_log = behavior_sim.generate_activity_schedule(
            account, start_date, duration_days, scenario_manager
        )
        for act in activity_log:
            act["account"] = account
            all_activities.append(act)

    all_activities.sort(key=lambda x: x["timestamp"])
    logger.info(f"Generated {len(all_activities)} activities to render as tweets")

    all_tweets = []
    recent_tweets = []

    for activity in all_activities:
        account = activity["account"]
        intent = activity["intent"]
        policy_event_id = activity.get("policy_event_id")
        policy_phase = activity.get("policy_phase")
        stance = activity.get("stance")
        acc_type = account.get("account_type", "Individual")

        # Interaction type
        weights = INTERACTION_WEIGHTS.get(acc_type, INTERACTION_WEIGHTS["Individual"])
        interaction_type = random.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]

        # Reference tweet for interactions
        ref_tweet = None
        if interaction_type in ["Retweet", "Reply", "Quote"] and len(recent_tweets) > 10:
            if policy_event_id and acc_type == "Bot":
                candidates = [
                    t for t in recent_tweets[-200:]
                    if t.get("policy_event_id") == policy_event_id
                ]
                if not candidates:
                    candidates = [
                        t for t in recent_tweets[-200:]
                        if t.get("escalation_score", 0) > 0.5
                    ]
                ref_tweet = random.choice(candidates) if candidates else random.choice(recent_tweets[-100:])
            elif acc_type == "Bot":
                candidates = [t for t in recent_tweets[-200:]
                              if t.get("escalation_score", 0) > 0.5]
                ref_tweet = random.choice(candidates) if candidates else random.choice(recent_tweets[-100:])
            else:
                ref_tweet = random.choice(recent_tweets[-100:])

        if not ref_tweet:
            interaction_type = "Tweet"

        # Generate content
        is_policy_tweet = policy_event_id is not None
        pe = None
        phase_enum = None
        if is_policy_tweet:
            pe = scenario_manager.policy_injector.get_event_by_id(policy_event_id)
            phase_enum = PolicyPhase(policy_phase)

        if interaction_type == "Retweet":
            tweet_text = f"RT @{ref_tweet['account_id'][:8]}: {ref_tweet['text']}"
            scores = {k: v for k, v in ref_tweet.items()
                      if "score" in k or "_rate" in k
                      or k in ("policy_event_id", "policy_phase",
                               "topic_cluster", "stance_score",
                               "sentiment_score", "policy_severity")}
        elif interaction_type == "Reply":
            if is_policy_tweet and pe:
                reply_text = content_gen.generate_policy_tweet(account, pe, phase_enum, stance or "neutral")
                scores = content_gen.calculate_policy_scores(reply_text, pe, phase_enum, stance or "neutral")
            else:
                reply_text = content_gen.generate_tweet(account, intent=intent)
                scores = content_gen.calculate_scores(reply_text, intent)
            tweet_text = f"@{ref_tweet['account_id'][:8]} {reply_text}"
        elif interaction_type == "Quote":
            if is_policy_tweet and pe:
                quote_text = content_gen.generate_policy_tweet(account, pe, phase_enum, stance or "neutral")
                scores = content_gen.calculate_policy_scores(quote_text, pe, phase_enum, stance or "neutral")
            else:
                quote_text = content_gen.generate_tweet(account, intent=intent)
                scores = content_gen.calculate_scores(quote_text, intent)
            tweet_text = f"{quote_text} QT @{ref_tweet['account_id'][:8]}: {ref_tweet['text'][:50]}..."
        else:
            if is_policy_tweet and pe:
                tweet_text = content_gen.generate_policy_tweet(account, pe, phase_enum, stance or "neutral")
                scores = content_gen.calculate_policy_scores(tweet_text, pe, phase_enum, stance or "neutral")
            else:
                tweet_text = content_gen.generate_tweet(account, intent=intent)
                scores = content_gen.calculate_scores(tweet_text, intent)

        # Ensure policy fields
        scores.setdefault("sentiment_score", 0.0)
        scores.setdefault("stance_score", 0.0)
        scores.setdefault("policy_event_id", None)
        scores.setdefault("policy_phase", None)
        scores.setdefault("topic_cluster", None)
        scores.setdefault("policy_severity", 0.0)

        # Geo + engagement
        home_county = account.get("home_county", "Nairobi")
        base_lat, base_lon = COUNTY_COORDINATES.get(home_county, (-1.2921, 36.8219))
        lat = base_lat + random.uniform(-0.04, 0.04)
        lon = base_lon + random.uniform(-0.04, 0.04)

        followers = account.get("followers_count", 100)
        base_engagement = np.random.lognormal(mean=np.log(max(1, followers / 100)), sigma=1.0)
        likes = int(base_engagement)
        retweets = int(base_engagement * 0.3)
        if acc_type == "Bot":
            likes = int(likes * 0.1)
            retweets = int(retweets * 2.0)

        tweet_record = {
            "post_id": f"tweet_{uuid.uuid4().hex[:12]}",
            "account_id": account["account_id"],
            "timestamp": activity["timestamp"],
            "text": tweet_text,
            "intent": intent,
            "interaction_type": interaction_type,
            "reply_to_post_id": ref_tweet["post_id"] if ref_tweet else "",
            "source_label": account.get("primary_device", "Twitter for Android"),
            "like_count": likes,
            "retweet_count": retweets,
            "location_county": home_county,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "imperative_rate": scores.get("imperative_rate", 0.0),
            "urgency_rate": scores.get("urgency_rate", 0.0),
            "coordination_score": scores.get("coordination_score", 0.0),
            "escalation_score": scores.get("escalation_score", 0.0),
            "threat_score": scores.get("threat_score", 0.0),
            "sentiment_score": scores.get("sentiment_score", 0.0),
            "stance_score": scores.get("stance_score", 0.0),
            "policy_event_id": scores.get("policy_event_id", ""),
            "policy_phase": scores.get("policy_phase", ""),
            "topic_cluster": scores.get("topic_cluster", ""),
            "policy_severity": scores.get("policy_severity", 0.0),
        }

        all_tweets.append(tweet_record)
        if interaction_type == "Tweet":
            recent_tweets.append(tweet_record)

    logger.info(f"Generated {len(all_tweets)} new tweets")

    # Merge with existing if append mode
    tweets_path = output_dir / "tweets.csv"
    if append:
        existing = _read_existing_csv(tweets_path, columns)
        logger.info(f"Existing tweets: {len(existing)}")

        # Trim old rows if combined exceeds max
        combined = existing + all_tweets
        if len(combined) > max_rows:
            # Keep the newest
            combined = combined[-max_rows:]
            logger.info(f"Trimmed to {max_rows} rows (removed oldest)")
        all_tweets = combined

    _write_csv(tweets_path, columns, all_tweets)
    logger.info(f"Wrote {len(all_tweets)} tweets to {tweets_path}")

    # Also write accounts
    import pandas as pd
    acc_records = []
    for account in accounts:
        acc_record = {k: v for k, v in account.items() if k != "account"}
        acc_records.append(acc_record)
    acc_df = pd.DataFrame(acc_records)
    acc_path = output_dir / "accounts.csv"
    acc_df.to_csv(acc_path, index=False)
    logger.info(f"Wrote {len(acc_records)} accounts to {acc_path}")

    return len(all_tweets)


# ═══════════════════════════════════════════════════════════════════════════
# News Cache Refresh — also generate fresh fake news articles
# ═══════════════════════════════════════════════════════════════════════════

def refresh_news_cache():
    """Generate fresh synthetic news articles in the news_cache/*.json format."""
    news_dir = PROJECT_ROOT / "data" / "news_cache"
    if not news_dir.exists():
        news_dir.mkdir(parents=True, exist_ok=True)

    topics = [
        ("kenya_finance_bill", "Finance Bill 2026", "taxation"),
        ("kenya_shif_health", "SHIF Phase 2 Implementation", "health"),
        ("kenya_housing_levy", "Housing Levy Increase", "housing"),
        ("kenya_fuel_prices", "Fuel Levy and EPRA Pricing", "fuel_energy"),
        ("kenya_education", "University Funding Model", "education"),
        ("kenya_digital_tax", "Digital Services Tax", "digital"),
        ("kenya_security", "Security Operations Expansion", "security"),
        ("kenya_agriculture", "Agricultural Import Policy", "agriculture"),
        ("kenya_devolution", "County Revenue Allocation", "devolution"),
    ]

    headlines = {
        "taxation": [
            "Finance Bill 2026: New tax proposals spark nationwide debate",
            "KRA targets Sh3.6 trillion revenue with expanded tax base",
            "Youth groups mobilize against proposed tax hikes in Finance Bill",
            "Business community warns Finance Bill will hurt economy",
            "MPs propose amendments to controversial Finance Bill clauses",
        ],
        "health": [
            "SHIF Phase 2 rollout faces implementation challenges",
            "Kenyans struggle to register for new SHIF health coverage",
            "Hospitals turn away patients amid SHA transition confusion",
            "Government defends SHIF as superior to old NHIF system",
            "Health workers union demands clarity on SHIF payment structure",
        ],
        "housing": [
            "Housing Levy increase to 3% draws worker pushback",
            "Affordable housing program delivers first 1,500 units in Nairobi",
            "Court challenge filed against Housing Levy deductions",
            "Workers union calls for Housing Levy suspension",
            "Government targets 200,000 housing units by 2027",
        ],
        "fuel_energy": [
            "EPRA announces new fuel prices amid public outcry",
            "Fuel levy hike pushes transport costs up 15%",
            "Matatu operators threaten strike over diesel prices",
            "Government maintains fuel stabilization fund is depleted",
            "Opposition calls for fuel subsidy reinstatement",
        ],
        "education": [
            "University funding model leaves students scrambling",
            "Higher education loans board overwhelmed with applications",
            "Parents protest new university fee structure",
            "Government promises no student will miss education due to fees",
            "Vice-chancellors endorse new differentiated funding approach",
        ],
        "digital": [
            "Digital Services Tax hits tech startups and content creators",
            "Kenya Revenue Authority expands digital tax net",
            "E-commerce platforms face new compliance requirements",
            "Tech community warns digital tax will stifle innovation",
            "Government projects Sh15B from digital services taxation",
        ],
        "security": [
            "Government expands security operations in northern counties",
            "Police reform commission tables new accountability measures",
            "Security budget increase draws mixed reactions",
            "Human rights groups monitor security operation conduct",
            "Community policing initiative launched in 15 counties",
        ],
        "agriculture": [
            "Agricultural import policy changes affect local farmers",
            "Sugar import permits spark protests in Western Kenya",
            "Government subsidizes fertilizer for smallholder farmers",
            "Tea and coffee sector reforms face stakeholder resistance",
            "Food security strategy targets self-sufficiency by 2030",
        ],
        "devolution": [
            "County revenue allocation formula faces Supreme Court challenge",
            "Governors demand increased share of national revenue",
            "Devolution review commission proposes structural changes",
            "County governments struggle with development budget execution",
            "New equalization fund disbursement criteria announced",
        ],
    }

    now = datetime.now()
    for slug, topic_name, sector in topics:
        articles = []
        sector_headlines = headlines.get(sector, headlines["taxation"])
        for i, headline in enumerate(sector_headlines):
            pub_date = (now - timedelta(hours=random.randint(1, 72))).isoformat()
            articles.append({
                "title": headline,
                "description": f"{headline}. {topic_name} continues to dominate Kenyan policy discourse as stakeholders weigh in on the implications for ordinary citizens.",
                "url": f"https://news.example.com/kenya/{slug}/{i+1}",
                "source": random.choice([
                    "The Standard", "Daily Nation", "The Star",
                    "Business Daily", "People Daily", "Citizen Digital",
                ]),
                "publishedAt": pub_date,
                "content": f"{headline}. The {topic_name} has been a subject of intense debate across Kenya. "
                           f"Analysts say the policy will have significant implications for the {sector} sector. "
                           f"Various stakeholders including civil society groups, business associations, and county governments "
                           f"have expressed diverse opinions on the matter."
            })

        cache_data = {
            "articles": articles,
            "last_fetched": now.isoformat(),
            "status": "ok",
            "query": topic_name,
        }

        cache_file = news_dir / f"{slug}.json"
        cache_file.write_text(json.dumps(cache_data, indent=2))

    logger.info(f"Refreshed {len(topics)} news cache files in {news_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run the full daily generation cycle."""
    logger.info("=" * 60)
    logger.info("DAILY SYNTHETIC TWEET GENERATION — START")
    logger.info("=" * 60)

    if not _acquire_lock():
        return

    start = time.time()

    try:
        # 1. General tweets
        logger.info("\n--- Generating GENERAL tweets ---")
        general_count = generate_batch(
            output_dir=GENERAL_DIR,
            columns=GENERAL_COLUMNS,
            num_accounts=NUM_ACCOUNTS,
            duration_days=DURATION_DAYS,
            append=APPEND_MODE,
            max_rows=MAX_ROWS,
        )

        # 2. Policy tweets (more accounts, policy-aware)
        logger.info("\n--- Generating POLICY tweets ---")
        policy_count = generate_batch(
            output_dir=POLICY_DIR,
            columns=POLICY_COLUMNS,
            num_accounts=POLICY_NUM_ACCOUNTS,
            duration_days=POLICY_DURATION_DAYS,
            append=APPEND_MODE,
            max_rows=MAX_ROWS,
        )

        # 3. Refresh news cache
        logger.info("\n--- Refreshing NEWS CACHE ---")
        refresh_news_cache()

        duration = time.time() - start
        _save_meta(general_count, policy_count, duration)

        logger.info("=" * 60)
        logger.info(f"COMPLETE — {general_count} general + {policy_count} policy tweets in {duration:.1f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()

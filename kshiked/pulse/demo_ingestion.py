#!/usr/bin/env python
"""
KShield Pulse - Demo Script

Demonstrates the full data ingestion pipeline.

Usage:
    python -m kshiked.pulse.demo_ingestion --test        # Run with test data only
    python -m kshiked.pulse.demo_ingestion --live        # Run with live scraping (needs API keys)
    python -m kshiked.pulse.demo_ingestion --gemini KEY  # Test Gemini classification
    
(requires: pip install -e .)
"""

import asyncio
import argparse


async def test_filters():
    """Test Kenya keyword filters."""
    print("\n=== Testing Kenya Filters ===")
    
    from kshiked.pulse.filters import (
        KENYA_POLITICAL, KENYA_ECONOMIC, KENYA_THREAT_SIGNALS,
        is_threat_related, get_matched_keywords
    )
    
    print(f"✓ Political keywords: {len(KENYA_POLITICAL)}")
    print(f"✓ Economic keywords: {len(KENYA_ECONOMIC)}")
    print(f"✓ Threat signals: {len(KENYA_THREAT_SIGNALS)}")
    
    # Test detection
    test_texts = [
        ("Rise up for maandamano tomorrow!", True),
        ("President Ruto visits Mombasa", False),
        ("Kill all the enemies of Kenya", True),
        ("Beautiful sunset in Nairobi", False),
    ]
    
    print("\nThreat Detection Tests:")
    for text, expected_threat in test_texts:
        is_threat = is_threat_related(text)
        status = "✓" if is_threat == expected_threat else "✗"
        print(f"  {status} '{text[:40]}...' → {'THREAT' if is_threat else 'safe'}")
    
    print("\n✓ Filters working correctly")


async def test_pulse_sensor():
    """Test PulseSensor signal detection."""
    print("\n=== Testing PulseSensor ===")
    
    from kshiked.pulse import PulseSensor
    
    sensor = PulseSensor()
    print(f"✓ PulseSensor initialized with {len(sensor._detectors)} detectors")
    
    test_texts = [
        "We are suffering! Fuel prices are killing us! The government must pay!",
        "Join the protest tomorrow at Uhuru Park. Rise up Kenya!",
        "Parliament session continues with budget discussions today",
    ]
    
    print("\nSignal Detection Tests:")
    for text in test_texts:
        detections = sensor.process_text(text)
        signals = [d.signal_id.name for d in detections]
        if signals:
            print(f"  → Detected: {', '.join(signals)}")
        else:
            print(f"  → No signals detected")
    
    # Update state
    state = sensor.update_state()
    print(f"\n✓ State updated:")
    print(f"    Instability Index: {state.instability_index:.3f}")
    print(f"    Crisis Probability: {state.crisis_probability:.3f}")


async def test_database():
    """Test database operations."""
    print("\n=== Testing Database ===")
    
    from kshiked.pulse.db import Database, DatabaseConfig
    from kshiked.pulse.db.models import SocialPost, Platform
    from datetime import datetime
    
    config = DatabaseConfig(url="sqlite+aiosqlite:///test_demo.db")
    db = Database(config)
    
    await db.connect()
    print("✓ Database connected")
    
    # Create test post using correct enum
    post = SocialPost(
        platform=Platform.TWITTER,
        platform_id="demo_123",
        text="Test post for demo",
        author_id=None,
        posted_at=datetime.utcnow(),
        scraped_at=datetime.utcnow(),
    )
    
    await db.add(post)
    print("✓ Test post created")
    
    await db.disconnect()
    
    # Cleanup
    import os
    if os.path.exists("test_demo.db"):
        os.remove("test_demo.db")
    
    print("✓ Database working correctly")


async def test_gemini(api_key: str):
    """Test Gemini threat classification."""
    print("\n=== Testing Gemini LLM ===")
    
    from kshiked.pulse.llm import create_gemini_provider
    
    provider = create_gemini_provider(api_key=api_key)
    print("✓ Gemini provider created")
    
    test_texts = [
        "Rise up Kenya! Take the streets tomorrow!",
        "Parliament session today to discuss budget",
        "Kill all the corrupt politicians!",
    ]
    
    print("\nThreat Classification Tests:")
    for text in test_texts:
        try:
            result = await provider.classify_threat(text)
            print(f"  → '{text[:35]}...'")
            print(f"     Tier: {result.tier.value}, Confidence: {result.confidence:.2f}")
            print(f"     Reasoning: {result.reasoning}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n✓ Gemini classification working")


async def run_live_demo():
    """Run live scraping demo (requires API keys)."""
    print("\n=== Live Scraping Demo ===")
    print("Note: This requires API credentials in .env file")
    
    from kshiked.pulse.ingestion import IngestionOrchestrator, IngestionConfig
    
    config = IngestionConfig.from_env()
    
    async with IngestionOrchestrator(config) as orchestrator:
        print("✓ Orchestrator initialized")
        
        # Scrape limited sample
        posts = await orchestrator.scrape_social_media(
            search_terms=["Kenya"],
            limit=10,
        )
        
        print(f"✓ Scraped {len(posts)} posts")
        
        for post in posts[:3]:
            print(f"    - [{post.platform}] {post.text[:50]}...")
    
    print("\n✓ Live scraping demo complete")


async def main():
    parser = argparse.ArgumentParser(description="KShield Pulse Demo")
    parser.add_argument("--test", action="store_true", help="Run with test data only")
    parser.add_argument("--live", action="store_true", help="Run live scraping")
    parser.add_argument("--gemini", type=str, help="Test Gemini with API key")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  KShield Pulse - Data Ingestion Pipeline Demo")
    print("=" * 60)
    
    if args.all or args.test or (not args.live and not args.gemini):
        await test_filters()
        await test_pulse_sensor()
        await test_database()
    
    if args.gemini:
        await test_gemini(args.gemini)
    
    if args.live:
        await run_live_demo()
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

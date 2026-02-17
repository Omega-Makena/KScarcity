"""
Pulse Engine Demo — Showcases the full pipeline with sample data

This demo uses realistic sample social media posts to demonstrate:
1. Signal detection from text
2. NLP analysis (sentiment, emotion, entities)
3. Primitive updates
4. Risk scoring
5. Shock generation

No API calls are made - uses embedded sample data.

Usage: python -m kshiked.pulse.demo
(requires: pip install -e .)
"""

import time
from datetime import datetime, timedelta
from kshiked.pulse import (
    PulseSensor, PulseState, SignalID, RiskScorer,
    create_kshield_bridge, SocialPost, Platform,
)
from kshiked.pulse.nlp import NLPPipeline


# =============================================================================
# Sample Social Media Data (Simulating Kenya economic crisis posts)
# =============================================================================

SAMPLE_POSTS = [
    # Distress signals
    SocialPost(
        id="1", platform=Platform.TWITTER,
        text="The cost of living is unbearable! Food prices have doubled, fuel is unaffordable. How are ordinary Kenyans supposed to survive? #CostOfLivingCrisis",
        author_name="KenyanCitizen", author_followers=1200, likes=458, shares=89,
        hashtags=["CostOfLivingCrisis"], created_at=datetime.now() - timedelta(hours=2)
    ),
    SocialPost(
        id="2", platform=Platform.TWITTER,
        text="My family is starving. We can't afford ugali anymore. This government has failed us completely. Children going to sleep hungry every night.",
        author_name="MamaKenya", author_followers=345, likes=234, shares=67,
        created_at=datetime.now() - timedelta(hours=3)
    ),
    SocialPost(
        id="3", platform=Platform.TWITTER,
        text="I'm exhausted. Mentally and physically drained. No hope left. Worked 10 years and still can't afford rent. What's the point anymore?",
        author_name="TiredWorker", author_followers=567, likes=189, shares=45,
        created_at=datetime.now() - timedelta(hours=4)
    ),
    
    # Anger signals
    SocialPost(
        id="4", platform=Platform.TWITTER,
        text="These corrupt politicians are stealing from us while we die of hunger! They must face justice! #CorruptionKE #StepDown",
        author_name="AngryCitizen", author_followers=2300, likes=1234, shares=456,
        hashtags=["CorruptionKE", "StepDown"], created_at=datetime.now() - timedelta(hours=1)
    ),
    SocialPost(
        id="5", platform=Platform.TWITTER,
        text="The dictator and his cronies have looted this country dry. They are enemies of the people. Revolution is coming!",
        author_name="ResistKE", author_followers=8900, likes=3456, shares=890,
        created_at=datetime.now() - timedelta(hours=2)
    ),
    
    # Mobilization signals
    SocialPost(
        id="6", platform=Platform.TWITTER,
        text="RISE UP KENYANS! 🔥 Take the streets! General strike Monday! Uhuru Park 9AM! Bring water and first aid. Spread the word! #MondayProtests #Maandamano",
        author_name="ProtestOrganizer", author_followers=15000, likes=5678, shares=2345,
        hashtags=["MondayProtests", "Maandamano"], created_at=datetime.now() - timedelta(minutes=30)
    ),
    SocialPost(
        id="7", platform=Platform.TWITTER,
        text="Join our Telegram group for protest coordination: Safety tips, meeting points, legal aid contacts. We stand together! Link in bio.",
        author_name="YouthMovement", author_followers=23000, likes=4567, shares=1890,
        created_at=datetime.now() - timedelta(hours=1)
    ),
    
    # Institutional friction
    SocialPost(
        id="8", platform=Platform.TWITTER,
        text="BREAKING: Police fire tear gas at peaceful protesters in CBD. Several injured. Military vehicles spotted heading to Nairobi. Stay safe everyone!",
        author_name="KenyaNewsNow", author_followers=150000, likes=12345, shares=8901,
        created_at=datetime.now() - timedelta(minutes=10)
    ),
    SocialPost(
        id="9", platform=Platform.TWITTER,
        text="Banks reporting unusual withdrawal activity. People queuing for hours. Currency losing value rapidly. Economic collapse imminent?",
        author_name="FinanceWatch", author_followers=45000, likes=6789, shares=3456,
        created_at=datetime.now() - timedelta(hours=1)
    ),
    
    # Information warfare
    SocialPost(
        id="10", platform=Platform.TWITTER,
        text="DON'T BELIEVE THE GOVERNMENT PROPAGANDA! They're lying about the true inflation numbers. The real story is being hidden from you. Wake up!",
        author_name="TruthSeeker", author_followers=12000, likes=3456, shares=1234,
        created_at=datetime.now() - timedelta(hours=2)
    ),
]


# =============================================================================
# Demo Functions
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_nlp_analysis():
    """Demonstrate NLP analysis on sample posts."""
    print_header("NLP ANALYSIS DEMO")
    
    pipeline = NLPPipeline()
    
    for i, post in enumerate(SAMPLE_POSTS[:3]):
        print(f"\n📱 Post {i+1}: {post.text[:60]}...")
        result = pipeline.analyze(post.text)
        
        print(f"   Sentiment: {result.sentiment.compound:+.2f} ", end="")
        if result.sentiment.compound < -0.3:
            print("(Negative 😠)")
        elif result.sentiment.compound > 0.3:
            print("(Positive 😊)")
        else:
            print("(Neutral 😐)")
        
        print(f"   Emotion: {result.emotions.dominant.upper()} (arousal: {result.emotions.arousal:.1%})")
        if result.entities:
            print(f"   Entities: {', '.join(e.text for e in result.entities[:3])}")
        if result.hashtags:
            print(f"   Hashtags: #{', #'.join(result.hashtags)}")


def demo_signal_detection():
    """Demonstrate signal detection on sample posts."""
    print_header("SIGNAL DETECTION DEMO")
    
    sensor = PulseSensor(use_nlp=True)
    sensor.config.update_interval = 0
    
    all_detections = []
    
    for post in SAMPLE_POSTS:
        detections = sensor.process_text(post.text, post.to_pulse_metadata())
        all_detections.extend(detections)
    
    print(f"\n📊 Processed {len(SAMPLE_POSTS)} posts")
    print(f"🎯 Detected {len(all_detections)} signals")
    
    # Count by signal
    signal_counts = {}
    for d in all_detections:
        name = d.signal_id.name
        if name not in signal_counts:
            signal_counts[name] = {"count": 0, "intensity": 0}
        signal_counts[name]["count"] += 1
        signal_counts[name]["intensity"] = max(signal_counts[name]["intensity"], d.intensity)
    
    print("\n📈 Signal Summary:")
    for name, data in sorted(signal_counts.items(), key=lambda x: x[1]["intensity"], reverse=True):
        bar = "█" * int(data["intensity"] * 20)
        print(f"   {name[:25]:<25} {bar} {data['intensity']:.0%} (x{data['count']})")
    
    return sensor


def demo_primitive_state(sensor: PulseSensor):
    """Demonstrate primitive state updates."""
    print_header("PRIMITIVE STATE DEMO")
    
    sensor.update_state()
    state = sensor.state
    
    print("\n🔴 SCARCITY VECTOR:")
    from kshiked.pulse import ResourceDomain
    for domain in ResourceDomain:
        val = state.scarcity.get(domain)
        bar = "█" * int(val * 20)
        print(f"   {domain.value:<12} {bar} {val:.0%}")
    
    print(f"\n   Aggregate: {state.scarcity.aggregate_score():.0%}")
    
    print("\n😰 ACTOR STRESS:")
    from kshiked.pulse import ActorType
    for actor in ActorType:
        val = state.stress.get_stress(actor)
        if val < 0:
            bar = "▓" * int(abs(val) * 20)
            print(f"   {actor.value:<12} -{bar} {val:+.2f}")
        else:
            bar = "░" * int(val * 20)
            print(f"   {actor.value:<12} +{bar} {val:+.2f}")
    
    print(f"\n   Total System Stress: {state.stress.total_system_stress():.2f}")
    
    print("\n🤝 SOCIAL BONDS:")
    print(f"   National Cohesion:  {'█' * int(state.bonds.national_cohesion * 20)} {state.bonds.national_cohesion:.0%}")
    print(f"   Class Solidarity:   {'█' * int(state.bonds.class_solidarity * 20)} {state.bonds.class_solidarity:.0%}")
    print(f"   Regional Unity:     {'█' * int(state.bonds.regional_unity * 20)} {state.bonds.regional_unity:.0%}")
    print(f"\n   Fragility Score: {state.bonds.fragility_score():.0%}")
    
    return state


def demo_risk_scoring(sensor: PulseSensor):
    """Demonstrate risk scoring."""
    print_header("RISK SCORING DEMO")
    
    scorer = RiskScorer()
    
    # Add detections to scorer
    for post in SAMPLE_POSTS:
        for d in sensor.process_text(post.text):
            scorer.add_detection(d)
    
    score = scorer.compute()
    
    print(f"\n⚠️  OVERALL RISK: {score.overall:.1%}")
    print(f"   Trend: {score.trend.upper()}")
    print(f"   Anomaly Score: {score.anomaly_score:.1%}")
    
    print("\n📊 BY CATEGORY:")
    for cat, val in sorted(score.by_category.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(val * 30)
        print(f"   {cat:<15} {bar} {val:.0%}")
    
    return scorer


def demo_shock_generation():
    """Demonstrate shock generation for simulation."""
    print_header("SHOCK GENERATION DEMO")
    
    bridge, sensor = create_kshield_bridge(use_nlp=True)
    sensor.config.update_interval = 0
    
    # Track shocks
    shocks = []
    bridge.register_handler(lambda s: shocks.append(s))
    
    # Process all sample posts
    for post in SAMPLE_POSTS * 2:  # Double to increase intensity
        for d in sensor.process_text(post.text):
            bridge.scorer.add_detection(d)
    
    # Run bridge cycle
    generated = bridge.process_cycle()
    
    print(f"\n⚡ Generated {len(generated)} economic shocks:")
    for shock in generated[:5]:
        direction = "📉" if shock.magnitude < 0 else "📈"
        print(f"   {direction} {shock.target_variable[:40]:<40} {shock.magnitude:+.2%}")
    
    if len(generated) > 5:
        print(f"   ... and {len(generated) - 5} more")
    
    print(f"\n📋 Bridge Stats: {bridge.get_stats()}")


def demo_full_pipeline():
    """Run the complete demo."""
    print("\n" + "🔴" * 30)
    print("   PULSE ENGINE DEMONSTRATION")
    print("   Social Signal Intelligence for Economic Simulation")
    print("🔴" * 30)
    
    print(f"\n📱 Sample Data: {len(SAMPLE_POSTS)} simulated social media posts")
    print("   (Realistic crisis scenario based on cost-of-living protests)")
    
    # Run demos
    demo_nlp_analysis()
    sensor = demo_signal_detection()
    state = demo_primitive_state(sensor)
    scorer = demo_risk_scoring(sensor)
    demo_shock_generation()
    
    print_header("DEMO COMPLETE")
    print("""
✅ This demonstration shows the full Pulse Engine pipeline:

   Social Media Text → NLP Analysis → Signal Detection → 
   Primitive Updates → Risk Scoring → Economic Shocks

🔑 When connected to real X API:
   - Replace sample posts with real search results
   - Same pipeline processes live data
   - Shocks feed into economic simulation

📊 Your X API Status:
   - 100 posts/month available
   - Usage tracked automatically
   - Ready for live data when needed
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_full_pipeline()

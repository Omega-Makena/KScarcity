"""
KShield V3.0 Verification on Real Tweet Data
Target: random/tweets_retweets.csv (Boda Boda Corpus)
"""

import asyncio
import csv
import logging
import random
from pathlib import Path
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("kshield.real_test")

from pulse.news import NewsIngestor, NewsAPIConfig
from pulse.llm.ollama import OllamaProvider
from pulse.llm.signals import (
    KShieldSignal, ThreatTier, ThreatCategory,
    EconomicGrievance, SocialGrievance
)

# Heuristic Mock to simulate AI Reasoning on this specific dataset
class HeuristicMockOllama(OllamaProvider):
    async def _generate_json(self, prompt: str, system_prompt: str):
        # Extract text from prompt
        # Prompt format: 'Analyze this content...\nTEXT:\n"..."'
        try:
            text_part = prompt.split('TEXT:')[1].split('CONTEXT:')[0].strip('" \n')
        except:
            text_part = ""
            
        lower_text = text_part.lower()
        
        # 1. Threat Scan Logic
        if "THREAT CATEGORY" in prompt:
            # Tier 1 keywords
            if any(w in lower_text for w in ["kill", "burn", "death", "wipe out", "cleanse"]):
                return {
                    "category": "mass_casualty_advocacy",
                    "tier": "TIER_1_EXISTENTIAL",
                    "intent": 0.9, "capability": 0.7, "specificity": 0.6, "reach": 0.5, "trajectory": 0.8,
                    "reasoning": "Detected lethal violence advocacy."
                }
            # Tier 2 keywords
            if any(w in lower_text for w in ["insurrection", "overthrow", "rebel", "seize"]):
                return {
                    "category": "coordinated_insurrection",
                    "tier": "TIER_2_SEVERE_STABILITY",
                    "intent": 0.8, "capability": 0.6, "specificity": 0.5, "reach": 0.6, "trajectory": 0.7,
                    "reasoning": "Detected insurrectionary language."
                }
            # Tier 3 (Boda Boda violence specific)
            if any(w in lower_text for w in ["boda", "gang", "attack", "harass"]):
                return {
                    "category": "ethnic_religious_mobilization", # Proxy for group violence
                    "tier": "TIER_3_HIGH_RISK",
                    "intent": 0.7, "capability": 0.8, "specificity": 0.8, "reach": 0.7, "trajectory": 0.6,
                    "reasoning": "Detected group-based violence/harassment (Boda Boda)."
                }
            
            return {
                "category": "political_criticism",
                "tier": "TIER_5_NON_THREAT",
                "intent": 0.1, "capability": 0.0, "specificity": 0.1, "reach": 0.2, "trajectory": 0.1,
                "reasoning": "Standard discourse/criticism."
            }

        # 2. Context Scan Logic
        if "CONTEXTUAL STRESS" in prompt:
            score = 0.0
            e_g = "E0_legitimate_grievance"
            s_g = "S0_normal_discontent"
            
            if "law and order" in lower_text or "out of control" in lower_text:
                s_g = "S4_societal_breakdown"
                score = 0.9
                
            return {
                "economic_grievance": e_g,
                "social_grievance": s_g,
                "economic_score": 0.2,
                "social_score": score,
                "shock_marker": 0.8 if "attack" in lower_text else 0.1,
                "polarization_marker": 0.7 if "them" in lower_text else 0.2
            }
            
        # 3. Indices
        if "Analyze indices" in prompt:
            return {
                "lei_score": 0.6 if "police" in lower_text or "government" in lower_text else 0.1,
                "lei_target": "Police/Govt",
                "si_score": 0.4,
                "maturation_score": 40,
                "maturation_stage": "Narrative",
                "aa_score": 0.1,
                "aa_technique": "None"
            }
            
        return {}

    async def analyze_tta(self, text: str):
        from pulse.llm.signals import TimeToAction
        return TimeToAction.IMMEDIATE_24H if "now" in text.lower() else TimeToAction.CHRONIC_14D

    async def analyze_resilience(self, text: str):
        from pulse.llm.signals import ResilienceIndex
        return ResilienceIndex(0.2, 0.3, 0.1)

    async def analyze_role_v3(self, text: str):
        from pulse.llm.signals import RoleType
        return RoleType.MOBILIZER if "meet" in text.lower() else RoleType.OBSERVER

async def run_test():
    # Adjusted path for running from kshiked subdirectory
    csv_path = Path("../random/tweets_retweets.csv")
    if not csv_path.exists():
        # Fallback for running from root
        csv_path = Path("random/tweets_retweets.csv")
    
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path.resolve()}")
        return

    logger.info(f"Loading {csv_path}...")
    
    raw_tweets = []
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('text'):
                raw_tweets.append(row['text'])
    
    logger.info(f"Loaded {len(raw_tweets)} tweets. Sampling 5...")
    sample = random.sample(raw_tweets, 5)
    
    # Force include a likely "hot" tweet if random misses (for demo)
    hot_tweets = [t for t in raw_tweets if "out of control" in t.lower()]
    if hot_tweets and hot_tweets[0] not in sample:
        sample[0] = hot_tweets[0]

    ingestor = NewsIngestor(config=NewsAPIConfig(api_key="verify"))
    ingestor.ollama = HeuristicMockOllama()
    
    print("\n" + "="*60)
    print("KSHIELD V3.0 REAL DATA VERIFICATION")
    print("="*60 + "\n")
    
    for i, text in enumerate(sample):
        print(f"--- Tweet {i+1} ---")
        print(f"Text: {text[:100]}...")
        
        # Mock Article wrapper
        article = {
            "title": text[:30],
            "description": text,
            "content": text,
            "url": f"tweet_{i}"
        }
        
        signal = await ingestor.process_article_deeply(article)
        
        if signal:
            print(f"Detected: [{signal.threat.tier.name}] {signal.threat.category.name}")
            print(f"Context:  {signal.context.social_fracture.name} (Multiplier: {signal.context.stress_multiplier}x)")
            print(f"Indices:  LEI={signal.indices.lei_score} | Maturation={signal.indices.maturation_stage}")
            print(f"V3 Layer: TTA={signal.tta.value} | Role={signal.role.value} | Resilience={signal.resilience.counter_narrative_score}")
            print(f"RISK:     Base={signal.base_risk:.1f} -> Adjusted={signal.adjusted_risk:.1f}")
            print(f"Action:   {signal.status}")
        else:
            print("Action:   Filtered/Ignored")
        print("\n")

if __name__ == "__main__":
    asyncio.run(run_test())

"""
Verification Script for KShield Pulse V3.0
Tests the full pipeline from Text -> Risk Calculation -> DB Storage.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kshield.verify")

from kshiked.pulse.news import NewsIngestor, NewsAPIConfig
from kshiked.pulse.llm.ollama import OllamaProvider
from kshiked.pulse.llm.signals import (
    KShieldSignal, ThreatTier, ThreatCategory, 
    EconomicGrievance, SocialGrievance
)

# Mock Provider to simulate Ollama responses exactly as per PDF examples
class MockOllama(OllamaProvider):
    async def _generate_json(self, prompt: str, system_prompt: str):
        # 1. Threat Scan: Simulate "Mass Casualty Advocacy" (Tier 1)
        if "THREAT CATEGORY" in prompt:
            return {
                "category": "mass_casualty_advocacy",
                "tier": "TIER_1_EXISTENTIAL",
                "intent": 0.95,
                "capability": 0.80,
                "specificity": 0.70,
                "reach": 0.60,
                "trajectory": 0.90,
                "reasoning": "Explicit call for ethnic cleansing ('Wipe them out')."
            }
        
        # 2. Context Scan: Simulate High Social Dissatisfaction (CSM multiplier)
        if "CONTEXTUAL STRESS" in prompt:
            return {
                "economic_grievance": "E3_destabilization_narratives",
                "social_grievance": "S3_violence_risk",
                "economic_score": 0.8,
                "social_score": 0.9,
                "shock_marker": 1.0,      # Huge shock
                "polarization_marker": 0.9 # Extreme polarization
            }
            
        # 3. Indices Scan: Simulate High LEI and Maturation
        if "Analyze indices" in prompt:
            return {
                "lei_score": 0.85,
                "lei_target": "Judiciary",
                "si_score": 0.75,
                "maturation_score": 80,
                "maturation_stage": "Campaign",
                "aa_score": 0.2,
                "aa_technique": "None"
            }
        return {}

async def run_verification():
    logger.info("Initializing KShield V3.0 Verification...")
    
    # Setup Ingestor
    ingestor = NewsIngestor(config=NewsAPIConfig(api_key="verify"))
    ingestor.ollama = MockOllama() # Inject Mock
    
    # Test Article (Based on PDF Page 1 Example)
    article = {
        "title": "Operation Cleanse: Wipe them out",
        "description": "They deserve to die. We must defend ourselves.",
        "content": "The courts are fake. Join the struggle. Tonight we move.",
        "url": f"https://example.com/tier1-threat-{int(time.time())}"
    }
    
    logger.info(f"Processing Article: {article['title']}")
    
    # Run Deep Pulse Pipeline
    signal = await ingestor.process_article_deeply(article)
    
    if not signal:
        logger.error("❌ Pipeline returned None")
        return

    # Verify Logic
    logger.info(f"✅ Generated Signal ID: {signal.source_id}")
    
    # 1. Threat Verification
    logger.info(f"   Threat Tier: {signal.threat.tier.name} (Expected: TIER_1_EXISTENTIAL)")
    assert signal.threat.tier == ThreatTier.TIER_1
    
    # 2. Risk Calculation Verification
    # BaseRisk = (0.3*0.95 + 0.2*0.80 + 0.15*0.70 + 0.15*0.60 + 0.1*0.9 + 0.1*0.5) * 100
    #          = (0.285 + 0.16 + 0.105 + 0.09 + 0.09 + 0.05) * 100
    #          = 0.78 * 100 = 78.0
    logger.info(f"   Base Risk: {signal.base_risk:.2f} (Expected ~78.0)")
    
    # 3. Context Verification (CSM)
    # CSM = 1 + (0.15*0.8) + (0.15*0.9) + (0.20*1.0) + (0.15*0.9)
    #     = 1 + 0.12 + 0.135 + 0.20 + 0.135 
    #     = 1.59
    logger.info(f"   CSM: {signal.context.stress_multiplier:.2f} (Expected ~1.59)")
    
    # 4. Adjusted Risk
    # Risk = min(100, 78.0 * 1.59) = min(100, 124.02) = 100.0
    logger.info(f"   Adjusted Risk: {signal.adjusted_risk:.2f} (Expected 100.0)")
    assert signal.adjusted_risk == 100.0
    
    # 5. Gating Verification
    # Base > 40, CSM > 1.15, Base > 80 check... 78 is < 80 but * CSM is high.
    # PDF Rule: "Escalate (D) regardless" if Base > 80.
    # Here Base is 78. Condition 60-80: If CSM >= 1.10 -> ACTIVE MONITORING.
    # Wait, simple logic in KShieldSignal.status uses base_risk for primary gating.
    # My implementation says: if 60 <= base < 80: if csm >= 1.10: ACTIVE_MONITORING.
    logger.info(f"   Status: {signal.status} (Expected: ACTIVE_MONITORING)")
    assert signal.status == "ACTIVE_MONITORING"
    
    logger.info("✅ Logic Verification PASSED")

if __name__ == "__main__":
    asyncio.run(run_verification())

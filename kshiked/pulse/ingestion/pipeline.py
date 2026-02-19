"""
Pipeline Integration Bridge for KShield Pulse

Connects the data ingestion pipeline to the existing PulseSensor.

This bridge:
1. Takes scraped social media posts
2. Runs them through LLM for threat classification
3. Feeds them into PulseSensor for signal detection
4. Stores results in database

Usage:
    bridge = PipelineIntegration(
        sensor=PulseSensor(),
        llm_provider=GeminiProvider(...),
        database=Database(...),
    )
    
    # Process scraped posts
    await bridge.process_posts(scraped_posts)
    
    # Get current threat state
    state = bridge.get_state()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..sensor import PulseSensor, AsyncPulseSensor
from ..primitives import PulseState
from ..mapper import SIGNAL_CATEGORIES
from ..db import Database
from ..db.models import SocialPost as DBSocialPost, LLMAnalysis, ProcessedSignal
from ..llm.base import LLMProvider, ThreatClassification, ThreatTier
from ..scrapers.base import ScraperResult

logger = logging.getLogger("kshield.pulse.ingestion.pipeline")


# =============================================================================
# Pipeline Integration
# =============================================================================

class PipelineIntegration:
    """
    Bridges the data ingestion pipeline with the PulseSensor.
    
    Flow:
    1. Scrapers → ScraperResult
    2. LLM → ThreatClassification  
    3. PulseSensor → SignalDetection
    4. Database → Storage
    """
    
    def __init__(
        self,
        sensor: Optional[PulseSensor] = None,
        llm_provider: Optional[LLMProvider] = None,
        database: Optional[Database] = None,
        use_nlp_detectors: bool = True,
    ):
        """
        Initialize pipeline integration.
        
        Args:
            sensor: PulseSensor instance for signal detection.
            llm_provider: LLM provider for threat classification.
            database: Database for storage.
            use_nlp_detectors: Use NLP-enhanced detectors in sensor.
        """
        self.sensor = sensor or PulseSensor(use_nlp=use_nlp_detectors)
        self.llm = llm_provider
        self.db = database
        
        self._stats = {
            "posts_processed": 0,
            "threats_detected": 0,
            "signals_triggered": 0,
            "errors": 0,
        }
    
    async def process_posts(
        self,
        posts: List[ScraperResult],
        batch_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process scraped posts through the full pipeline.
        
        Args:
            posts: Scraped posts from any platform.
            batch_llm: Use batch LLM classification.
            
        Returns:
            List of processing results.
        """
        results = []
        
        # Batch LLM classification
        classifications = []
        if self.llm and posts:
            try:
                if batch_llm and len(posts) > 1:
                    texts = [p.text for p in posts]
                    classifications = await self.llm.batch_classify(texts)
                else:
                    for post in posts:
                        cls = await self.llm.classify_threat(
                            post.text,
                            context={
                                "platform": post.platform.value,
                                "followers": post.author_followers or 0,
                            }
                        )
                        classifications.append(cls)
            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
                classifications = [None] * len(posts)
                self._stats["errors"] += 1
        else:
            classifications = [None] * len(posts)
        
        # Process each post
        for i, post in enumerate(posts):
            try:
                result = await self._process_single(
                    post,
                    classifications[i] if i < len(classifications) else None,
                )
                results.append(result)
                self._stats["posts_processed"] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process post: {e}")
                self._stats["errors"] += 1
                continue
        
        # Update sensor state
        self.sensor.update_state()
        
        return results
    
    async def _process_single(
        self,
        post: ScraperResult,
        classification: Optional[ThreatClassification],
    ) -> Dict[str, Any]:
        """Process a single post."""
        result = {
            "post_id": post.platform_id,
            "platform": post.platform.value,
            "tier": None,
            "signals": [],
        }
        
        # 1. LLM classification result
        if classification:
            result["tier"] = classification.tier.value
            result["confidence"] = classification.confidence
            result["reasoning"] = classification.reasoning
            
            if classification.is_threat:
                self._stats["threats_detected"] += 1
        
        # 2. Run through PulseSensor for signal detection
        metadata = {
            "platform": post.platform.value,
            "author": post.author_username,
            "timestamp": post.posted_at.isoformat() if post.posted_at else None,
            "threat_tier": classification.tier.value if classification else None,
        }
        
        detections = self.sensor.process_text(post.text, metadata)
        
        result["signals"] = [
            {
                "signal_id": d.signal_id.name,
                "intensity": d.intensity,
                "confidence": d.confidence,
            }
            for d in detections
        ]
        
        self._stats["signals_triggered"] += len(detections)
        
        # 3. Store LLM analysis in database
        if self.db and classification:
            await self._store_analysis(post, classification, detections)
        
        return result
    
    async def _store_analysis(
        self,
        post: ScraperResult,
        classification: ThreatClassification,
        detections: list,
    ) -> None:
        """Store analysis results in database."""
        try:
            # Get or create the post in DB
            db_post = await self.db.get_post(post.platform.value, post.platform_id)
            
            if db_post:
                # Store LLM analysis
                analysis = LLMAnalysis(
                    post_id=db_post.id,
                    threat_tier=classification.tier.value,
                    threat_confidence=classification.confidence,
                    threat_reasoning=classification.reasoning,
                    base_risk=classification.base_risk,
                    intent_score=classification.intent_score,
                    capability_score=classification.capability_score,
                    specificity_score=classification.specificity_score,
                    reach_score=classification.reach_score,
                    model_name=classification.model_name or "unknown",
                    prompt_tokens=classification.prompt_tokens,
                    completion_tokens=classification.completion_tokens,
                    analyzed_at=datetime.utcnow(),
                )
                await self.db.add(analysis)
                
                # Store signal detections
                for detection in detections:
                    signal_category = SIGNAL_CATEGORIES.get(detection.signal_id)
                    signal = ProcessedSignal(
                        post_id=db_post.id,
                        signal_id=detection.signal_id.name,
                        signal_category=(
                            signal_category.name.lower()
                            if signal_category is not None
                            else "distress"
                        ),
                        intensity=detection.intensity,
                        confidence=detection.confidence,
                        raw_score=float(getattr(detection, "raw_score", detection.intensity)),
                        matched_keywords=(
                            detection.context.get("matched_keywords")
                            if isinstance(detection.context, dict)
                            else None
                        ),
                        sentiment_score=(
                            float(detection.context["sentiment"])
                            if isinstance(detection.context, dict)
                            and isinstance(detection.context.get("sentiment"), (int, float))
                            else None
                        ),
                        emotion_scores=(
                            detection.context.get("emotions")
                            if isinstance(detection.context, dict)
                            and isinstance(detection.context.get("emotions"), dict)
                            else None
                        ),
                        detected_at=datetime.utcnow(),
                    )
                    await self.db.add(signal)
                
        except Exception as e:
            logger.warning(f"Failed to store analysis: {e}")
    
    def get_state(self) -> PulseState:
        """Get current PulseSensor state."""
        return self.sensor.state
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get current threat summary."""
        state = self.sensor.state
        metrics = self.sensor.get_metrics()
        
        # Determine overall threat level
        crisis_prob = state.crisis_probability
        if crisis_prob > 0.7:
            threat_level = "CRITICAL"
        elif crisis_prob > 0.5:
            threat_level = "HIGH"
        elif crisis_prob > 0.3:
            threat_level = "ELEVATED"
        elif crisis_prob > 0.1:
            threat_level = "GUARDED"
        else:
            threat_level = "LOW"
        
        return {
            "threat_level": threat_level,
            "crisis_probability": crisis_prob,
            "instability_index": state.instability_index,
            "metrics": metrics,
            "processing_stats": self._stats,
        }
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.sensor.reset()
        self._stats = {
            "posts_processed": 0,
            "threats_detected": 0,
            "signals_triggered": 0,
            "errors": 0,
        }


# =============================================================================
# Full Pipeline Runner
# =============================================================================

async def run_full_pipeline(
    search_terms: Optional[List[str]] = None,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full ingestion + analysis pipeline.
    
    This is the main entry point for running KShield Pulse.
    
    Args:
        search_terms: Kenya-focused search terms.
        gemini_api_key: Gemini API key for classification.
        
    Returns:
        Pipeline results and threat summary.
    """
    from ..ingestion import IngestionOrchestrator, IngestionConfig
    from ..llm import create_gemini_provider
    from ..db import Database, DatabaseConfig
    
    # Load configuration
    config = IngestionConfig.from_env()
    
    # Override with provided values
    if gemini_api_key:
        config.gemini_api_key = gemini_api_key
    
    # Initialize components
    db_config = DatabaseConfig(url=config.database_url)
    db = Database(db_config)
    await db.connect()
    
    llm = None
    if config.gemini_api_key:
        llm = create_gemini_provider(config.gemini_api_key)
    
    # Create pipeline integration
    pipeline = PipelineIntegration(
        sensor=PulseSensor(use_nlp=True),
        llm_provider=llm,
        database=db,
    )
    
    # Run orchestrator to scrape
    async with IngestionOrchestrator(config) as orchestrator:
        posts = await orchestrator.scrape_social_media(search_terms)
        
        # Convert DB posts to ScraperResult format
        scraper_results = []
        for post in posts:
            # Re-create ScraperResult from DB model
            from ..scrapers.base import Platform as ScraperPlatform
            db_platform = getattr(post.platform, "value", post.platform)
            try:
                platform = ScraperPlatform(str(db_platform))
            except Exception:
                platform = ScraperPlatform.TWITTER

            author = getattr(post, "author", None)
            mentioned_locations = []
            raw_locations = getattr(post, "mentioned_locations", None)
            if isinstance(raw_locations, dict):
                locs = raw_locations.get("locations")
                if isinstance(locs, list):
                    mentioned_locations = [str(loc) for loc in locs if loc]

            result = ScraperResult(
                platform=platform,
                platform_id=post.platform_id,
                text=post.text,
                language=getattr(post, "language", "en"),
                author_id=(
                    getattr(author, "platform_id", None)
                    if author is not None
                    else None
                ) or None,
                author_username=(
                    getattr(author, "username", None)
                    if author is not None
                    else None
                ),
                author_display_name=(
                    getattr(author, "display_name", None)
                    if author is not None
                    else None
                ),
                author_followers=(
                    getattr(author, "follower_count", None)
                    if author is not None
                    else None
                ),
                author_verified=bool(
                    getattr(author, "verified", False) if author is not None else False
                ),
                likes=getattr(post, "likes", 0) or 0,
                shares=getattr(post, "shares", 0) or 0,
                replies=getattr(post, "replies", 0) or 0,
                views=getattr(post, "views", None),
                hashtags=list(getattr(post, "hashtags", []) or []),
                mentions=list(getattr(post, "mentions", []) or []),
                urls=list(getattr(post, "urls", []) or []),
                media_urls=list(getattr(post, "media_urls", []) or []),
                geo_location=getattr(post, "geo_location", None),
                mentioned_locations=mentioned_locations,
                reply_to_id=getattr(post, "reply_to_id", None),
                conversation_id=getattr(post, "conversation_id", None),
                posted_at=post.posted_at,
                scraped_at=post.scraped_at,
                raw_data=getattr(post, "raw_data", None),
            )
            scraper_results.append(result)
        
        # Process through pipeline
        results = await pipeline.process_posts(scraper_results)
    
    # Get summary
    summary = pipeline.get_threat_summary()
    
    await db.disconnect()
    
    return {
        "posts_processed": len(results),
        "threat_summary": summary,
        "results": results[:10],  # First 10 for preview
    }


# =============================================================================
# Factory Functions
# =============================================================================

def create_pipeline(
    gemini_api_key: Optional[str] = None,
    database: Optional[Database] = None,
) -> PipelineIntegration:
    """Create a pipeline integration instance."""
    from ..llm import create_gemini_provider
    
    llm = None
    if gemini_api_key:
        llm = create_gemini_provider(gemini_api_key)
    
    return PipelineIntegration(
        sensor=PulseSensor(use_nlp=True),
        llm_provider=llm,
        database=database,
    )

"""
NLP Signal Detectors — Enhanced Detectors using NLP Pipeline

Replaces simple keyword matching with:
- Sentiment-weighted detection
- Entity-aware context
- Emotion analysis
- Semantic scoring

Each detector implements the SignalDetector protocol and uses the NLP pipeline.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import numpy as np

from .sensor import SignalDetector, SignalDetection
from .mapper import SignalID
from .nlp import (
    NLPPipeline,
    NLPResult,
    SentimentAnalyzer,
    EmotionDetector,
    EntityRecognizer,
    TextPreprocessor,
)

logger = logging.getLogger("kshield.pulse.detectors")


# =============================================================================
# Base NLP Detector
# =============================================================================

class NLPSignalDetector(SignalDetector):
    """
    Base class for NLP-enhanced signal detectors.
    
    Provides common functionality:
    - NLP pipeline integration
    - Keyword boosting
    - Entity context
    - Confidence calculation
    """
    
    def __init__(
        self, 
        signal_id: SignalID,
        keywords: List[str] = None,
        negative_keywords: List[str] = None,
        target_emotions: List[str] = None,
        sentiment_weight: float = 0.3,
        keyword_weight: float = 0.4,
        emotion_weight: float = 0.3,
    ):
        self._signal_id = signal_id
        self.keywords = set(k.lower() for k in (keywords or []))
        self.negative_keywords = set(k.lower() for k in (negative_keywords or []))
        self.target_emotions = target_emotions or []
        
        self.sentiment_weight = sentiment_weight
        self.keyword_weight = keyword_weight
        self.emotion_weight = emotion_weight
        
        self.nlp = NLPPipeline()
        self.preprocessor = TextPreprocessor()
    
    @property
    def signal_id(self) -> SignalID:
        return self._signal_id
    
    def detect(self, text: str, metadata: Dict[str, Any] = None) -> Optional[SignalDetection]:
        """Detect signal using NLP analysis."""
        metadata = metadata or {}
        
        # Run NLP analysis
        nlp_result = self.nlp.analyze(text)
        
        # Compute component scores
        keyword_score = self._compute_keyword_score(text)
        sentiment_score = self._compute_sentiment_score(nlp_result)
        emotion_score = self._compute_emotion_score(nlp_result)
        
        # Weighted combination
        raw_score = (
            self.keyword_weight * keyword_score +
            self.sentiment_weight * sentiment_score +
            self.emotion_weight * emotion_score
        )
        
        # Apply entity bonus
        entity_bonus = self._compute_entity_bonus(nlp_result)
        raw_score *= (1.0 + entity_bonus)
        
        # Check threshold
        if raw_score < 0.2:
            return None
        
        # Normalize intensity
        intensity = min(1.0, raw_score)
        
        # Compute confidence based on text length and match quality
        confidence = self._compute_confidence(nlp_result, keyword_score)
        
        return SignalDetection(
            signal_id=self._signal_id,
            intensity=intensity,
            confidence=confidence,
            raw_score=raw_score,
            context={
                "sentiment": nlp_result.sentiment.compound,
                "dominant_emotion": nlp_result.emotions.dominant,
                "entities": [e.text for e in nlp_result.entities[:5]],
                "keyword_score": keyword_score,
            },
            timestamp=time.time()
        )
    
    def _compute_keyword_score(self, text: str) -> float:
        """Score based on keyword matches."""
        tokens = set(self.preprocessor.tokenize(text))
        
        matches = len(tokens & self.keywords)
        negatives = len(tokens & self.negative_keywords)
        
        if not self.keywords:
            return 0.0
        
        score = matches / len(self.keywords)
        score -= 0.5 * (negatives / max(1, len(self.negative_keywords))) if self.negative_keywords else 0
        
        return max(0.0, min(1.0, score * 2.0))  # Scale up
    
    def _compute_sentiment_score(self, nlp_result: NLPResult) -> float:
        """Score based on sentiment (default: negative sentiment is signal)."""
        # Most signals are negative, override in subclass if needed
        return max(0.0, -nlp_result.sentiment.compound)
    
    def _compute_emotion_score(self, nlp_result: NLPResult) -> float:
        """Score based on target emotions."""
        if not self.target_emotions:
            return 0.0
        
        scores = [
            nlp_result.emotions.emotions.get(e, 0.0) 
            for e in self.target_emotions
        ]
        return np.mean(scores) if scores else 0.0
    
    def _compute_entity_bonus(self, nlp_result: NLPResult) -> float:
        """Bonus for relevant entities."""
        # Base implementation: bonus for any entity
        entity_count = len(nlp_result.entities)
        return min(0.3, entity_count * 0.1)
    
    def _compute_confidence(self, nlp_result: NLPResult, keyword_score: float) -> float:
        """Compute detection confidence."""
        # Longer text = more confidence
        length_factor = min(1.0, nlp_result.token_count / 20.0)
        
        # Keyword matches = more confidence
        keyword_factor = min(1.0, keyword_score * 1.5)
        
        # Emotion intensity = more confidence
        emotion_factor = nlp_result.emotions.arousal
        
        return min(0.95, 0.4 + 0.2 * length_factor + 0.2 * keyword_factor + 0.2 * emotion_factor)


# =============================================================================
# Signal-Specific Detectors
# =============================================================================

class SurvivalCostStressDetector(NLPSignalDetector):
    """Signal 1: Survival Cost Stress - complaints about basic living costs."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.SURVIVAL_COST_STRESS,
            keywords=[
                "expensive", "afford", "costly", "price", "inflation",
                "rent", "food", "fuel", "electricity", "water",
                "salary", "wages", "unemployed", "jobless", "broke",
                "hungry", "starving", "bills", "debt", "loan",
            ],
            target_emotions=["sadness", "fear", "anger"],
            sentiment_weight=0.3,
            keyword_weight=0.5,
            emotion_weight=0.2,
        )


class DistressFramingDetector(NLPSignalDetector):
    """Signal 2: Distress Framing - crisis language and suffering narratives."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.DISTRESS_FRAMING,
            keywords=[
                "suffering", "dying", "crisis", "emergency", "disaster",
                "starvation", "collapse", "catastrophe", "desperate",
                "help", "save", "mercy", "please", "urgent",
            ],
            target_emotions=["sadness", "fear"],
            sentiment_weight=0.4,
            keyword_weight=0.4,
            emotion_weight=0.2,
        )


class EmotionalExhaustionDetector(NLPSignalDetector):
    """Signal 3: Emotional Exhaustion - hopelessness and fatigue."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.EMOTIONAL_EXHAUSTION,
            keywords=[
                "tired", "exhausted", "hopeless", "given", "up",
                "enough", "fed", "burnt", "depressed", "done",
                "no", "hope", "point", "why", "bother",
            ],
            target_emotions=["sadness"],
            sentiment_weight=0.4,
            keyword_weight=0.3,
            emotion_weight=0.3,
        )


class DirectedRageDetector(NLPSignalDetector):
    """Signal 4: Directed Rage - specific anger at leaders/groups."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.DIRECTED_RAGE,
            keywords=[
                "hate", "destroy", "kill", "death", "punish",
                "revenge", "traitor", "enemy", "corrupt", "thief",
                "criminal", "evil", "monster", "demon",
            ],
            target_emotions=["anger", "disgust"],
            sentiment_weight=0.3,
            keyword_weight=0.4,
            emotion_weight=0.3,
        )
    
    def _compute_entity_bonus(self, nlp_result: NLPResult) -> float:
        """Higher bonus for naming specific targets."""
        # Look for PERSON or INSTITUTION entities
        target_entities = [
            e for e in nlp_result.entities 
            if e.label in ("PERSON", "INSTITUTION")
        ]
        return min(0.5, len(target_entities) * 0.2)


class RotatingRegimeSlangDetector(NLPSignalDetector):
    """Signal 5: Rotating Regime Slang - coded mockery of regime."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.ROTATING_REGIME_SLANG,
            keywords=[
                "dictator", "tyrant", "puppet", "regime", "oppressor",
                "clown", "fool", "incompetent", "useless", "failure",
            ],
            target_emotions=["anger", "disgust"],
            sentiment_weight=0.3,
            keyword_weight=0.5,
            emotion_weight=0.2,
        )
    
    def detect(self, text: str, metadata: Dict[str, Any] = None) -> Optional[SignalDetection]:
        # Check for hashtags (often used for coded slang)
        hashtags = self.preprocessor.extract_hashtags(text)
        
        detection = super().detect(text, metadata)
        
        if detection and hashtags:
            # Boost for hashtag usage
            detection.intensity = min(1.0, detection.intensity * 1.2)
            detection.context["hashtags"] = hashtags
        
        return detection


class DehumanizationLanguageDetector(NLPSignalDetector):
    """Signal 6: Dehumanization Language - degrading group labels."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.DEHUMANIZATION_LANGUAGE,
            keywords=[
                "cockroaches", "snakes", "rats", "vermin", "animals",
                "savages", "plague", "infestation", "cleanse", "eliminate",
                "exterminate", "wipe", "parasites", "subhuman",
            ],
            target_emotions=["disgust", "anger"],
            sentiment_weight=0.2,
            keyword_weight=0.6,
            emotion_weight=0.2,
        )
    
    def _compute_keyword_score(self, text: str) -> float:
        """Dehumanization keywords are high-signal, boost score."""
        base_score = super()._compute_keyword_score(text)
        return min(1.0, base_score * 1.5)  # Extra boost


class LegitimacyRejectionDetector(NLPSignalDetector):
    """Signal 7: Legitimacy Rejection - rejecting authority of state."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.LEGITIMACY_REJECTION,
            keywords=[
                "fake", "election", "stolen", "illegitimate", "rigged",
                "fraud", "sham", "puppet", "not", "my", "president",
                "reject", "refuse", "invalid", "unconstitutional",
            ],
            target_emotions=["anger", "disgust"],
            sentiment_weight=0.3,
            keyword_weight=0.5,
            emotion_weight=0.2,
        )


class SecurityForceFrictionDetector(NLPSignalDetector):
    """Signal 8: Security Force Friction - police/military tension."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.SECURITY_FORCE_FRICTION,
            keywords=[
                "police", "brutality", "military", "soldier", "violence",
                "abused", "beaten", "shot", "killed", "defect",
                "refuse", "orders", "mutiny", "protest",
            ],
            target_emotions=["anger", "fear"],
            sentiment_weight=0.3,
            keyword_weight=0.5,
            emotion_weight=0.2,
        )
    
    def _compute_entity_bonus(self, nlp_result: NLPResult) -> float:
        """Bonus for security-related entities."""
        security_entities = [
            e for e in nlp_result.entities 
            if "police" in e.text.lower() or "military" in e.text.lower()
               or "army" in e.text.lower() or "soldier" in e.text.lower()
        ]
        return min(0.4, len(security_entities) * 0.2)


class EconomicCascadeFailureDetector(NLPSignalDetector):
    """Signal 9: Economic Cascade Failure - systemic economic collapse signals."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.ECONOMIC_CASCADE_FAILURE,
            keywords=[
                "bank", "run", "collapse", "crash", "bankruptcy",
                "currency", "hyperinflation", "default", "crisis",
                "closing", "shutdown", "layoffs", "recession",
            ],
            target_emotions=["fear", "sadness"],
            sentiment_weight=0.4,
            keyword_weight=0.4,
            emotion_weight=0.2,
        )


class EliteFractureDetector(NLPSignalDetector):
    """Signal 10: Elite Fracture - divisions among elites."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.ELITE_FRACTURE,
            keywords=[
                "resign", "defect", "split", "division", "disagree",
                "conflict", "fight", "against", "betray", "leak",
                "expose", "reveal", "insider", "source",
            ],
            target_emotions=["surprise", "anger"],
            sentiment_weight=0.3,
            keyword_weight=0.5,
            emotion_weight=0.2,
        )


class EthnoRegionalFramingDetector(NLPSignalDetector):
    """Signal 11: Ethno-Regional Framing - ethnic/regional identity language."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.ETHNO_REGIONAL_FRAMING,
            keywords=[
                "tribe", "ethnic", "our", "people", "them", "they",
                "region", "ancestral", "homeland", "taking", "ours",
                "belong", "outsiders", "foreigners",
            ],
            negative_keywords=["unity", "together", "peace", "reconciliation"],
            target_emotions=["anger", "fear"],
            sentiment_weight=0.2,
            keyword_weight=0.5,
            emotion_weight=0.3,
        )


class MobilizationLanguageDetector(NLPSignalDetector):
    """Signal 12: Mobilization Language - calls to action."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.MOBILIZATION_LANGUAGE,
            keywords=[
                "rise", "up", "streets", "protest", "march",
                "strike", "shut", "down", "join", "come",
                "fight", "resist", "stand", "together", "now",
            ],
            target_emotions=["anger", "anticipation"],
            sentiment_weight=0.2,
            keyword_weight=0.5,
            emotion_weight=0.3,
        )
    
    def _compute_sentiment_score(self, nlp_result: NLPResult) -> float:
        # Mobilization can be positive (empowering) or negative
        return nlp_result.emotions.arousal  # Use arousal instead


class CoordinationInfrastructureDetector(NLPSignalDetector):
    """Signal 13: Coordination Infrastructure - organizing activity."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.COORDINATION_INFRASTRUCTURE,
            keywords=[
                "telegram", "group", "channel", "join", "link",
                "location", "meeting", "point", "coordinate", "organize",
                "supplies", "bring", "share", "forward", "spread",
            ],
            target_emotions=["anticipation"],
            sentiment_weight=0.1,
            keyword_weight=0.7,
            emotion_weight=0.2,
        )


class RumorVelocityPanicDetector(NLPSignalDetector):
    """Signal 14: Rumor Velocity & Panic - rapid spread of unverified claims."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.RUMOR_VELOCITY_PANIC,
            keywords=[
                "heard", "say", "unconfirmed", "spreading", "viral",
                "panic", "stockpile", "run", "emergency", "rumor",
                "apparently", "supposedly", "breaking",
            ],
            target_emotions=["fear", "surprise"],
            sentiment_weight=0.3,
            keyword_weight=0.4,
            emotion_weight=0.3,
        )


class CounterNarrativeActivationDetector(NLPSignalDetector):
    """Signal 15: Counter-Narrative Activation - info warfare."""
    
    def __init__(self):
        super().__init__(
            signal_id=SignalID.COUNTER_NARRATIVE_ACTIVATION,
            keywords=[
                "propaganda", "fake", "news", "lie", "truth",
                "cover", "up", "hidden", "real", "story",
                "believe", "sheep", "wake", "expose",
            ],
            target_emotions=["anger", "disgust"],
            sentiment_weight=0.3,
            keyword_weight=0.4,
            emotion_weight=0.3,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_nlp_detectors() -> Dict[SignalID, NLPSignalDetector]:
    """Create all NLP-enhanced signal detectors."""
    return {
        SignalID.SURVIVAL_COST_STRESS: SurvivalCostStressDetector(),
        SignalID.DISTRESS_FRAMING: DistressFramingDetector(),
        SignalID.EMOTIONAL_EXHAUSTION: EmotionalExhaustionDetector(),
        SignalID.DIRECTED_RAGE: DirectedRageDetector(),
        SignalID.ROTATING_REGIME_SLANG: RotatingRegimeSlangDetector(),
        SignalID.DEHUMANIZATION_LANGUAGE: DehumanizationLanguageDetector(),
        SignalID.LEGITIMACY_REJECTION: LegitimacyRejectionDetector(),
        SignalID.SECURITY_FORCE_FRICTION: SecurityForceFrictionDetector(),
        SignalID.ECONOMIC_CASCADE_FAILURE: EconomicCascadeFailureDetector(),
        SignalID.ELITE_FRACTURE: EliteFractureDetector(),
        SignalID.ETHNO_REGIONAL_FRAMING: EthnoRegionalFramingDetector(),
        SignalID.MOBILIZATION_LANGUAGE: MobilizationLanguageDetector(),
        SignalID.COORDINATION_INFRASTRUCTURE: CoordinationInfrastructureDetector(),
        SignalID.RUMOR_VELOCITY_PANIC: RumorVelocityPanicDetector(),
        SignalID.COUNTER_NARRATIVE_ACTIVATION: CounterNarrativeActivationDetector(),
    }

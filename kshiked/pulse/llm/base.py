"""
Base LLM Provider Interface for KShield Pulse

Provides:
- Abstract LLMProvider interface (swappable)
- Standard classification result types
- Common utilities

Design allows swapping between providers:
- GeminiProvider (default, using your API key)
- OpenAIProvider (future)
- ClaudeProvider (future)
- LocalProvider (future, for fine-tuned models)

Usage:
    provider = GeminiProvider(api_key="...")
    
    result = await provider.classify_threat(
        text="Rise up! Take the streets!",
        context={"platform": "twitter", "author_followers": 10000}
    )
    
    print(result.tier)  # ThreatTier.TIER_3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class ThreatTier(str, Enum):
    """
    Threat severity tiers from KShield taxonomy.
    
    Tier 0: Protected speech (not a threat)
    Tier 1: Existential threats (critical danger)
    Tier 2: Severe stability threats
    Tier 3: High-risk destabilization
    Tier 4: Emerging threats
    Tier 5: Non-threat (background noise)
    """
    TIER_0 = "tier_0"  # Protected speech
    TIER_1 = "tier_1"  # Existential threats
    TIER_2 = "tier_2"  # Severe stability threats
    TIER_3 = "tier_3"  # High-risk destabilization
    TIER_4 = "tier_4"  # Emerging threats
    TIER_5 = "tier_5"  # Non-threat
    
    @property
    def severity(self) -> int:
        """Get numeric severity (1 = highest, 5 = lowest)."""
        if self == ThreatTier.TIER_0:
            return 0  # Not a threat
        return int(self.value.replace("tier_", ""))


class RoleType(str, Enum):
    """
    Actor roles in threat networks (from KShield taxonomy).
    """
    IDEOLOGUE = "ideologue"       # Produces justification narratives
    MOBILIZER = "mobilizer"       # Calls for action, coordinates
    AMPLIFIER = "amplifier"       # High-volume resharing
    BROKER = "broker"             # Connects communities
    LEGITIMIZER = "legitimizer"   # Adds authority cues
    GATEKEEPER = "gatekeeper"     # Controls channels
    UNKNOWN = "unknown"


class NarrativeMaturity(str, Enum):
    """Narrative maturation stages."""
    RUMOR = "rumor"           # Initial, unverified
    NARRATIVE = "narrative"   # Established, repeated
    CAMPAIGN = "campaign"     # Coordinated, organized


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class ThreatClassification:
    """
    Result of threat tier classification.
    
    Contains the tier, confidence, and supporting details.
    """
    # Primary classification
    tier: ThreatTier
    confidence: float  # 0-1
    
    # Risk scores (from KShield formula)
    base_risk: float = 0.0
    intent_score: float = 0.0      # 0-1
    capability_score: float = 0.0   # 0-1
    specificity_score: float = 0.0  # 0-1
    reach_score: float = 0.0        # 0-1
    
    # Explanation
    reasoning: str = ""
    matched_signals: List[str] = field(default_factory=list)
    
    # Metadata
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    
    @property
    def is_threat(self) -> bool:
        """Check if this is classified as a threat."""
        return self.tier not in [ThreatTier.TIER_0, ThreatTier.TIER_5]
    
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical threat (Tier 1-2)."""
        return self.tier in [ThreatTier.TIER_1, ThreatTier.TIER_2]


@dataclass
class RoleClassification:
    """
    Result of actor role classification.
    """
    role: RoleType
    confidence: float
    
    # Supporting evidence
    reasoning: str = ""
    behavioral_signals: List[str] = field(default_factory=list)
    
    # Network position indicators
    is_hub: bool = False  # High connectivity
    is_bridge: bool = False  # Connects communities
    
    # Metadata
    model_name: str = ""
    latency_ms: float = 0.0


@dataclass
class NarrativeAnalysis:
    """
    Result of narrative pattern analysis.
    """
    # Narrative identification
    narrative_type: str  # e.g., "government_corruption", "economic_hardship"
    maturity: NarrativeMaturity
    
    # Key themes
    themes: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    
    # Coordination signals
    is_coordinated: bool = False
    coordination_confidence: float = 0.0
    
    # Sentiment and emotion
    dominant_emotion: str = ""
    call_to_action: bool = False
    
    # Metadata
    sample_size: int = 0
    model_name: str = ""


# =============================================================================
# Abstract LLM Provider
# =============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implement this interface for:
    - GeminiProvider (default)
    - OpenAIProvider
    - ClaudeProvider
    - LocalProvider (fine-tuned models)
    """
    
    @abstractmethod
    async def classify_threat(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ThreatClassification:
        """
        Classify a post's threat tier.
        
        Args:
            text: The post text to classify.
            context: Optional context (platform, author info, etc.)
            
        Returns:
            ThreatClassification with tier and confidence.
        """
        pass
    
    @abstractmethod
    async def identify_role(
        self,
        author_posts: List[str],
        author_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoleClassification:
        """
        Identify an author's role in threat networks.
        
        Args:
            author_posts: Recent posts from the author.
            author_metadata: Optional author info (followers, etc.)
            
        Returns:
            RoleClassification with role and confidence.
        """
        pass
    
    @abstractmethod
    async def analyze_narrative(
        self,
        posts: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> NarrativeAnalysis:
        """
        Analyze narrative patterns across posts.
        
        Args:
            posts: Collection of related posts.
            context: Optional context (time range, hashtags, etc.)
            
        Returns:
            NarrativeAnalysis with patterns and themes.
        """
        pass
    
    @abstractmethod
    async def batch_classify(
        self,
        texts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ThreatClassification]:
        """
        Classify multiple texts efficiently.
        
        Args:
            texts: List of post texts.
            contexts: Optional list of contexts (same length as texts).
            
        Returns:
            List of ThreatClassification results.
        """
        pass
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

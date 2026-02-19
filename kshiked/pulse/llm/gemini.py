"""
Gemini LLM Provider for KShield Pulse

Google Gemini implementation of the LLMProvider interface.

Usage:
    provider = GeminiProvider(api_key="your-api-key")
    
    result = await provider.classify_threat(
        "Rise up and take the streets!",
        context={"platform": "twitter"}
    )
    
    print(result.tier)  # ThreatTier.TIER_3

Models:
    - gemini-1.5-flash: Fast, cost-effective (default)
    - gemini-1.5-pro: Higher accuracy for complex analysis
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .base import (
    LLMProvider, ThreatClassification, RoleClassification,
    NarrativeAnalysis, ThreatTier, RoleType, NarrativeMaturity,
)
from .prompts import (
    THREAT_CLASSIFIER_SYSTEM, ROLE_CLASSIFIER_SYSTEM,
    NARRATIVE_ANALYZER_SYSTEM, format_threat_prompt,
    format_role_prompt, format_narrative_prompt,
    BATCH_CLASSIFICATION_PROMPT,
)

logger = logging.getLogger("kshield.pulse.llm.gemini")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GeminiConfig:
    """Configuration for Gemini provider."""
    
    api_key: str
    model: str = "gemini-1.5-flash"  # or "gemini-1.5-pro"
    
    # Generation settings
    temperature: float = 0.3  # Lower for more consistent classification
    max_output_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 40
    
    # Rate limiting
    requests_per_minute: int = 60
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


# =============================================================================
# Gemini Provider
# =============================================================================

class GeminiProvider(LLMProvider):
    """
    Google Gemini implementation of LLMProvider.
    
    Uses google-generativeai SDK for API calls.
    """
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self._model = None
        self._initialized = False
        self._request_times: List[float] = []
    
    async def _initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._initialized:
            return
        
        try:
            import google.generativeai as genai
            
            # Configure API
            genai.configure(api_key=self.config.api_key)
            
            # Create model instance
            self._model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                },
            )
            
            self._initialized = True
            logger.info(f"Gemini provider initialized with model {self.config.model}")
            
        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old request times
        self._request_times = [t for t in self._request_times if t > minute_ago]
        
        # Wait if at limit
        if len(self._request_times) >= self.config.requests_per_minute:
            wait_time = self._request_times[0] - minute_ago
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        self._request_times.append(now)
    
    async def _generate(
        self,
        prompt: str,
        system_prompt: str,
    ) -> tuple[str, int, int, float]:
        """
        Generate response from Gemini.
        
        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens, latency_ms)
        """
        await self._initialize()
        await self._rate_limit()
        
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                # Combine system prompt and user prompt
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
                # Run in executor since SDK is synchronous
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._model.generate_content(full_prompt)
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract token counts if available
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                
                return response.text, prompt_tokens, completion_tokens, latency_ms
                
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def _parse_json(self, text: str) -> Dict:
        """Extract and parse JSON from response text.
        
        Raises:
            ValueError: If JSON cannot be parsed from response.
        """
        # Try to find JSON in the response
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object or array
            match = re.search(r'(\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"Failed to parse JSON from response: {text[:200]}...")
            raise ValueError(f"Failed to parse JSON from LLM response: {text[:200]}...")
    
    async def classify_threat(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ThreatClassification:
        """Classify a post's threat tier using Gemini."""
        context = context or {}
        
        prompt = format_threat_prompt(
            text=text,
            platform=context.get("platform", "unknown"),
            followers=context.get("followers", 0),
            timestamp=context.get("timestamp", "unknown"),
            location=context.get("location", "unknown"),
        )
        
        response_text, prompt_tokens, completion_tokens, latency_ms = await self._generate(
            prompt=prompt,
            system_prompt=THREAT_CLASSIFIER_SYSTEM,
        )
        
        # Parse response
        try:
            data = self._parse_json(response_text)
        except ValueError:
            data = {}
        
        # Map tier string to enum
        tier_str = data.get("tier", "TIER_5").upper().replace("-", "_")
        try:
            tier = ThreatTier(tier_str.lower())
        except ValueError:
            tier = ThreatTier.TIER_5
        
        return ThreatClassification(
            tier=tier,
            confidence=float(data.get("confidence", 0.5)),
            base_risk=self._calculate_base_risk(data),
            intent_score=float(data.get("intent_score", 0.0)),
            capability_score=float(data.get("capability_score", 0.0)),
            specificity_score=float(data.get("specificity_score", 0.0)),
            reach_score=float(data.get("reach_score", 0.0)),
            reasoning=data.get("reasoning", ""),
            matched_signals=data.get("matched_signals", []),
            model_name=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
    
    def _calculate_base_risk(self, data: Dict) -> float:
        """Calculate base risk score from component scores."""
        intent = float(data.get("intent_score", 0.0))
        capability = float(data.get("capability_score", 0.0))
        specificity = float(data.get("specificity_score", 0.0))
        reach = float(data.get("reach_score", 0.0))
        
        # Weighted average (from KShield formula)
        # Intent and specificity weighted higher
        base_risk = (
            intent * 0.35 +
            capability * 0.20 +
            specificity * 0.30 +
            reach * 0.15
        )
        
        return min(1.0, max(0.0, base_risk))
    
    async def identify_role(
        self,
        author_posts: List[str],
        author_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoleClassification:
        """Identify an author's role using Gemini."""
        metadata = author_metadata or {}
        
        prompt = format_role_prompt(
            posts=author_posts,
            platform=metadata.get("platform", "unknown"),
            followers=metadata.get("followers", 0),
            account_age=metadata.get("account_age", "unknown"),
        )
        
        response_text, _, _, latency_ms = await self._generate(
            prompt=prompt,
            system_prompt=ROLE_CLASSIFIER_SYSTEM,
        )
        
        try:
            data = self._parse_json(response_text)
        except ValueError:
            data = {}
        
        # Map role string to enum
        role_str = data.get("role", "UNKNOWN").upper()
        try:
            role = RoleType(role_str.lower())
        except ValueError:
            role = RoleType.UNKNOWN
        
        return RoleClassification(
            role=role,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            behavioral_signals=data.get("behavioral_signals", []),
            is_hub=data.get("is_hub", False),
            is_bridge=data.get("is_bridge", False),
            model_name=self.config.model,
            latency_ms=latency_ms,
        )
    
    async def analyze_narrative(
        self,
        posts: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> NarrativeAnalysis:
        """Analyze narrative patterns using Gemini."""
        context = context or {}
        
        prompt = format_narrative_prompt(
            posts=posts,
            time_range=context.get("time_range", "unknown"),
            hashtags=context.get("hashtags", []),
            platform=context.get("platform", "unknown"),
        )
        
        response_text, _, _, _ = await self._generate(
            prompt=prompt,
            system_prompt=NARRATIVE_ANALYZER_SYSTEM,
        )
        
        try:
            data = self._parse_json(response_text)
        except ValueError:
            data = {}
        
        # Map maturity string to enum
        maturity_str = data.get("maturity", "RUMOR").upper()
        try:
            maturity = NarrativeMaturity(maturity_str.lower())
        except ValueError:
            maturity = NarrativeMaturity.RUMOR
        
        return NarrativeAnalysis(
            narrative_type=data.get("narrative_type", "unknown"),
            maturity=maturity,
            themes=data.get("themes", []),
            target_entities=data.get("target_entities", []),
            is_coordinated=data.get("is_coordinated", False),
            coordination_confidence=float(data.get("coordination_confidence", 0.0)),
            dominant_emotion=data.get("dominant_emotion", "unknown"),
            call_to_action=data.get("call_to_action", False),
            sample_size=len(posts),
            model_name=self.config.model,
        )
    
    async def batch_classify(
        self,
        texts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ThreatClassification]:
        """Classify multiple texts in a single API call."""
        if not texts:
            return []
        
        # For small batches, use individual calls
        if len(texts) <= 3:
            results = []
            for i, text in enumerate(texts):
                ctx = contexts[i] if contexts and i < len(contexts) else None
                result = await self.classify_threat(text, ctx)
                results.append(result)
            return results
        
        # For larger batches, use batch prompt
        posts_data = [
            {"index": i, "text": text[:500]}  # Truncate long texts
            for i, text in enumerate(texts)
        ]
        
        prompt = BATCH_CLASSIFICATION_PROMPT.format(
            posts_json=json.dumps(posts_data, indent=2)
        )
        
        response_text, prompt_tokens, completion_tokens, latency_ms = await self._generate(
            prompt=prompt,
            system_prompt=THREAT_CLASSIFIER_SYSTEM,
        )
        
        # Parse batch response
        try:
            data = self._parse_json(response_text)
        except ValueError:
            data = []
        
        results = []
        if isinstance(data, list):
            for item in data:
                tier_str = item.get("tier", "TIER_5").upper().replace("-", "_")
                try:
                    tier = ThreatTier(tier_str.lower())
                except ValueError:
                    tier = ThreatTier.TIER_5
                
                results.append(ThreatClassification(
                    tier=tier,
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=item.get("reasoning", ""),
                    model_name=self.config.model,
                    prompt_tokens=prompt_tokens // len(texts),
                    completion_tokens=completion_tokens // len(texts),
                    latency_ms=latency_ms / len(texts),
                ))
        
        # Fill in missing results with defaults
        while len(results) < len(texts):
            results.append(ThreatClassification(
                tier=ThreatTier.TIER_5,
                confidence=0.0,
                reasoning="Failed to classify",
                model_name=self.config.model,
            ))
        
        return results


# =============================================================================
# Factory Function
# =============================================================================

def create_gemini_provider(
    api_key: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.3,
) -> GeminiProvider:
    """
    Create a Gemini provider instance.
    
    Args:
        api_key: Google AI API key.
        model: Model name (gemini-1.5-flash or gemini-1.5-pro).
        temperature: Generation temperature (lower = more consistent).
        
    Returns:
        Configured GeminiProvider instance.
    """
    config = GeminiConfig(
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
    
    return GeminiProvider(config)

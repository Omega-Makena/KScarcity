"""
Ollama LLM Provider for KShield Pulse (KShield Engine v3.0)

Implements the Dual-Layer Risk Architecture:
1. Threat Analysis (BaseRisk)
2. Context Analysis (CSM)
3. Advanced Indices (LEI, SI, MS)
"""

import logging
import json
import aiohttp
from typing import Dict, List, Optional, Any, Tuple

from .base import LLMProvider
from .signals import (
    KShieldSignal,
    ThreatSignal, 
    ContextAnalysis, 
    AdvancedIndices,
    ThreatCategory, 
    ThreatTier, 
    EconomicGrievance, 
    SocialGrievance,
    TimeToAction, ResilienceIndex, RoleType as RoleTypeV3
)

from .prompts import (
    THREAT_TAXONOMY_SYSTEM,
    CONTEXT_ANALYST_SYSTEM,
    INDICES_SYSTEM,
    format_threat_v3_prompt,
    format_context_v3_prompt,
    format_indices_v3_prompt,
    format_tta_prompt, format_resilience_prompt, format_role_v3_prompt,
    TTA_SYSTEM, RESILIENCE_SYSTEM, ROLE_V3_SYSTEM
)

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """
    Provider for local Ollama LLM (v3.0 Architecture).
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model
        self.session = None

    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def _generate_json(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Generate JSON response from Ollama (Robust)."""
        await self._ensure_session()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1, 
                "seed": 42,
                "num_ctx": 4096 
            }
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Ollama Error {resp.status}: {await resp.text()}")
                    return {}
                
                result = await resp.json()
                response_text = result.get("response", "{}")
                return json.loads(response_text)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return {}

    # =========================================================================
    # KShield V3.0 Core Methods
    # =========================================================================

    async def analyze_threat_landscape(self, text: str, context:  Optional[Dict] = None) -> Tuple[Optional[ThreatSignal], Optional[ContextAnalysis]]:
        """
        Dual-Layer Analysis:
        1. Threat (BaseRisk)
        2. Context (CSM)
        Returns: (ThreatSignal, ContextAnalysis)
        """
        # Parallelize these in production, sequential for safety now
        
        # 1. Threat Scan
        t_prompt = format_threat_v3_prompt(text, context)
        t_data = await self._generate_json(t_prompt, THREAT_TAXONOMY_SYSTEM)
        
        if not t_data.get("category"):
            logger.warning("Failed to extract threat data")
            return None, None

        threat_signal = ThreatSignal(
            category=ThreatCategory(t_data.get("category", "unknown")),
            tier=ThreatTier(t_data.get("tier", "TIER_5_NON_THREAT")),
            intent=t_data.get("intent", 0.0),
            capability=t_data.get("capability", 0.0),
            specificity=t_data.get("specificity", 0.0),
            reach=t_data.get("reach", 0.0),
            trajectory=t_data.get("trajectory", 0.0),
            classification_reason=t_data.get("reasoning", "")
        )

        # 2. Context Scan
        c_prompt = format_context_v3_prompt(text)
        c_data = await self._generate_json(c_prompt, CONTEXT_ANALYST_SYSTEM)
        
        context_analysis = ContextAnalysis(
            economic_strain=EconomicGrievance(c_data.get("economic_grievance", "E0_legitimate_grievance")),
            social_fracture=SocialGrievance(c_data.get("social_grievance", "S0_normal_discontent")),
            economic_dissatisfaction_score=c_data.get("economic_score", 0.0),
            social_dissatisfaction_score=c_data.get("social_score", 0.0),
            shock_marker=c_data.get("shock_marker", 0.0),
            polarization_marker=c_data.get("polarization_marker", 0.0)
        )
        
        return threat_signal, context_analysis

    async def analyze_indices(self, text: str) -> Optional[AdvancedIndices]:
        """
        Deep Scan: LEI, SI, MS, AA
        """
        prompt = format_indices_v3_prompt(text)
        data = await self._generate_json(prompt, INDICES_SYSTEM)
        
        if not data: return None
        
        return AdvancedIndices(
            lei_score=data.get("lei_score", 0.0),
            institution_target=data.get("lei_target", ""),
            si_score=data.get("si_score", 0.0),
            cognitive_rigidity=0.0, # detailed breakdown could be added
            identity_fusion=0.0,
            maturation_score=data.get("maturation_score", 0.0),
            maturation_stage=data.get("maturation_stage", "Rumor"),
            aa_score=data.get("aa_score", 0.0),
            evasion_technique=data.get("aa_technique", "None")
        )

    async def analyze_tta(self, text: str) -> TimeToAction:
        """Analyze Time-To-Action (V3)."""
        prompt = format_tta_prompt(text)
        data = await self._generate_json(prompt, TTA_SYSTEM)
        try:
            return TimeToAction(data.get("tta_category", "chronic_14d").lower())
        except ValueError:
            return TimeToAction.CHRONIC_14D

    async def analyze_resilience(self, text: str) -> ResilienceIndex:
        """Analyze Resilience Factors (V3)."""
        prompt = format_resilience_prompt(text)
        data = await self._generate_json(prompt, RESILIENCE_SYSTEM)
        return ResilienceIndex(
            counter_narrative_score=float(data.get("counter_narrative_score", 0.0)),
            community_resilience=float(data.get("community_resilience", 0.0)),
            confusion_factor=float(data.get("confusion_factor", 0.0))
        )

    async def analyze_role_v3(self, text: str) -> RoleTypeV3:
        """Analyze Author Role (V3)."""
        prompt = format_role_v3_prompt(text)
        data = await self._generate_json(prompt, ROLE_V3_SYSTEM)
        
        # Normalize
        role_str = data.get("role", "OBSERVER").lower()
        if role_str == "amplifier": role_str = "unwitting_amplifier"
        
        try:
            return RoleTypeV3(role_str)
        except ValueError:
            return RoleTypeV3.OBSERVER

    # =========================================================================
    # Fallback / Legacy Support (Abstract Base Class)
    # =========================================================================
    
    async def classify_threat(self, text: str, context: Optional[Dict[str, Any]] = None):
        # Map V3 back to legacy if needed, or raise NotImplemented
        pass

    async def identify_role(self, author_posts: List[str], author_metadata: Optional[Dict[str, Any]] = None):
        pass

    async def analyze_narrative(self, posts: List[str], context: Optional[Dict[str, Any]] = None):
        pass

    async def batch_classify(self, texts: List[str], contexts: Optional[List[Dict[str, Any]]] = None):
        pass

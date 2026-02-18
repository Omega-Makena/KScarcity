"""
Ollama LLM Provider for KShield Pulse (v3.0 Architecture)

Production-ready Ollama integration implementing:
- Dual-Layer Risk Architecture (BaseRisk + CSM)
- Multi-model task routing
- Retry logic with exponential backoff
- Health checks and model management
- Full legacy API compatibility
- Kenyan Sheng/Swahili awareness
- Latency tracking

Usage:
    from kshiked.pulse.llm.ollama import OllamaProvider
    from kshiked.pulse.llm.config import OllamaConfig
    
    config = OllamaConfig.single_model("llama3.1:8b")
    async with OllamaProvider(config=config) as provider:
        threat, context = await provider.analyze_threat_landscape(text)
        signal = await provider.full_analysis(text)
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import aiohttp

from .base import (
    LLMProvider,
    ThreatClassification,
    RoleClassification,
    NarrativeAnalysis,
    ThreatTier as LegacyThreatTier,
    RoleType as LegacyRoleType,
    NarrativeMaturity,
)
from .signals import (
    KShieldSignal,
    ThreatSignal,
    ContextAnalysis,
    AdvancedIndices,
    ThreatCategory,
    ThreatTier,
    EconomicGrievance,
    SocialGrievance,
    TimeToAction,
    ResilienceIndex,
    RoleType as RoleTypeV3,
)
from .prompts import (
    THREAT_TAXONOMY_SYSTEM,
    CONTEXT_ANALYST_SYSTEM,
    INDICES_SYSTEM,
    TTA_SYSTEM,
    RESILIENCE_SYSTEM,
    ROLE_V3_SYSTEM,
    format_threat_v3_prompt,
    format_context_v3_prompt,
    format_indices_v3_prompt,
    format_tta_prompt,
    format_resilience_prompt,
    format_role_v3_prompt,
)
from .config import (
    OllamaConfig,
    AnalysisTask,
    InferenceMetrics,
    SessionMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# V3 Tier → Legacy Tier Mapping
# =============================================================================

_V3_TO_LEGACY_TIER = {
    ThreatTier.TIER_1: LegacyThreatTier.TIER_1,
    ThreatTier.TIER_2: LegacyThreatTier.TIER_2,
    ThreatTier.TIER_3: LegacyThreatTier.TIER_3,
    ThreatTier.TIER_4: LegacyThreatTier.TIER_4,
    ThreatTier.TIER_5: LegacyThreatTier.TIER_5,
}

_V3_TO_LEGACY_ROLE = {
    RoleTypeV3.IDEOLOGUE: LegacyRoleType.IDEOLOGUE,
    RoleTypeV3.MOBILIZER: LegacyRoleType.MOBILIZER,
    RoleTypeV3.BROKER: LegacyRoleType.BROKER,
    RoleTypeV3.OPERATIONAL_SIGNALER: LegacyRoleType.GATEKEEPER,
    RoleTypeV3.UNWITTING_AMPLIFIER: LegacyRoleType.AMPLIFIER,
    RoleTypeV3.OBSERVER: LegacyRoleType.UNKNOWN,
}


# =============================================================================
# Robust Enum Parsing Helpers
# =============================================================================

def _parse_threat_category(raw: str) -> ThreatCategory:
    """Fuzzy-match LLM output to valid ThreatCategory enum."""
    if not raw:
        return ThreatCategory.UNKNOWN
    raw_lower = raw.lower().strip()
    # Direct match
    for member in ThreatCategory:
        if raw_lower == member.value.lower():
            return member
    # Partial match: check if the raw output contains an enum value or vice versa
    for member in ThreatCategory:
        if member.value.lower() in raw_lower or raw_lower in member.value.lower():
            return member
    # Keyword fallback mapping
    _KEYWORD_MAP = {
        "mass_casualty": ThreatCategory.CAT_1_MASS_VIOLENCE,
        "mass_violence": ThreatCategory.CAT_1_MASS_VIOLENCE,
        "genocide": ThreatCategory.CAT_1_MASS_VIOLENCE,
        "terrorism": ThreatCategory.CAT_2_TERRORISM_SUPPORT,
        "infrastructure": ThreatCategory.CAT_3_INFRA_SABOTAGE,
        "sabotage": ThreatCategory.CAT_3_INFRA_SABOTAGE,
        "insurrection": ThreatCategory.CAT_4_INSURRECTION,
        "rebellion": ThreatCategory.CAT_4_INSURRECTION,
        "election": ThreatCategory.CAT_5_ELECTION_SUBVERSION,
        "official": ThreatCategory.CAT_6_OFFICIAL_THREATS,
        "ethnic": ThreatCategory.CAT_7_ETHNIC_MOBILIZATION,
        "religious": ThreatCategory.CAT_7_ETHNIC_MOBILIZATION,
        "disinfo": ThreatCategory.CAT_8_DISINFO_CAMPAIGNS,
        "disinformation": ThreatCategory.CAT_8_DISINFO_CAMPAIGNS,
        "financial": ThreatCategory.CAT_9_FINANCIAL_WARFARE,
        "economic": ThreatCategory.CAT_9_FINANCIAL_WARFARE,
        "radicalization": ThreatCategory.CAT_10_RADICALIZATION,
        "hate": ThreatCategory.CAT_11_HATE_NETWORKS,
        "foreign": ThreatCategory.CAT_12_FOREIGN_INFLUENCE,
        "criticism": ThreatCategory.CAT_13_POLITICAL_CRITICISM,
        "political": ThreatCategory.CAT_13_POLITICAL_CRITICISM,
        "satire": ThreatCategory.CAT_14_SATIRE_PROTEST,
        "protest": ThreatCategory.CAT_14_SATIRE_PROTEST,
        "protected": ThreatCategory.CAT_13_POLITICAL_CRITICISM,
        "non_threat": ThreatCategory.CAT_13_POLITICAL_CRITICISM,
        "non-threat": ThreatCategory.CAT_13_POLITICAL_CRITICISM,
    }
    for keyword, cat in _KEYWORD_MAP.items():
        if keyword in raw_lower:
            return cat
    logger.warning(f"Unknown threat category from LLM: {raw!r}, defaulting to UNKNOWN")
    return ThreatCategory.UNKNOWN


def _parse_threat_tier(raw: str) -> ThreatTier:
    """Fuzzy-match LLM output to valid ThreatTier enum."""
    if not raw:
        return ThreatTier.TIER_5
    raw_lower = raw.lower().strip()
    # Direct match
    for member in ThreatTier:
        if raw_lower == member.value.lower():
            return member
    # Check for tier number
    for tier_num, tier in [
        ("1", ThreatTier.TIER_1), ("2", ThreatTier.TIER_2),
        ("3", ThreatTier.TIER_3), ("4", ThreatTier.TIER_4),
        ("5", ThreatTier.TIER_5),
    ]:
        if f"tier_{tier_num}" in raw_lower or f"tier {tier_num}" in raw_lower:
            return tier
    # Keyword fallback
    if "existential" in raw_lower or "catastrophic" in raw_lower:
        return ThreatTier.TIER_1
    if "severe" in raw_lower or "stability" in raw_lower:
        return ThreatTier.TIER_2
    if "high" in raw_lower:
        return ThreatTier.TIER_3
    if "emerging" in raw_lower or "latent" in raw_lower:
        return ThreatTier.TIER_4
    if "non" in raw_lower or "protected" in raw_lower or "low" in raw_lower:
        return ThreatTier.TIER_5
    logger.warning(f"Unknown threat tier from LLM: {raw!r}, defaulting to TIER_5")
    return ThreatTier.TIER_5


def _parse_economic_grievance(raw: str) -> EconomicGrievance:
    """Fuzzy-match LLM output to valid EconomicGrievance enum."""
    if not raw:
        return EconomicGrievance.E0_LEGITIMATE
    raw_lower = raw.lower().strip()
    # Direct match
    for member in EconomicGrievance:
        if raw_lower == member.value.lower():
            return member
    # Shorthand match (E0, E1, E2, E3, E4)
    _SHORT = {
        "e0": EconomicGrievance.E0_LEGITIMATE,
        "e1": EconomicGrievance.E1_DELEGITIMIZATION,
        "e2": EconomicGrievance.E2_MOBILIZATION,
        "e3": EconomicGrievance.E3_DESTABILIZATION,
        "e4": EconomicGrievance.E4_SABOTAGE,
    }
    for key, val in _SHORT.items():
        if raw_lower.startswith(key):
            return val
    # Keyword fallback
    if "sabotage" in raw_lower:
        return EconomicGrievance.E4_SABOTAGE
    if "destabiliz" in raw_lower:
        return EconomicGrievance.E3_DESTABILIZATION
    if "mobiliz" in raw_lower:
        return EconomicGrievance.E2_MOBILIZATION
    if "delegitim" in raw_lower or "anger" in raw_lower:
        return EconomicGrievance.E1_DELEGITIMIZATION
    return EconomicGrievance.E0_LEGITIMATE


def _parse_social_grievance(raw: str) -> SocialGrievance:
    """Fuzzy-match LLM output to valid SocialGrievance enum."""
    if not raw:
        return SocialGrievance.S0_DISCONTENT
    raw_lower = raw.lower().strip()
    # Direct match
    for member in SocialGrievance:
        if raw_lower == member.value.lower():
            return member
    # Shorthand match (S0, S1, S2, S3, S4)
    _SHORT = {
        "s0": SocialGrievance.S0_DISCONTENT,
        "s1": SocialGrievance.S1_POLARIZATION,
        "s2": SocialGrievance.S2_MOBILIZATION,
        "s3": SocialGrievance.S3_FRACTURE,
        "s4": SocialGrievance.S4_BREAKDOWN,
    }
    for key, val in _SHORT.items():
        if raw_lower.startswith(key):
            return val
    # Keyword fallback
    if "conflict" in raw_lower or "civil war" in raw_lower or "breakdown" in raw_lower:
        return SocialGrievance.S4_BREAKDOWN
    if "fracture" in raw_lower or "punishment" in raw_lower:
        return SocialGrievance.S3_FRACTURE
    if "mobiliz" in raw_lower:
        return SocialGrievance.S2_MOBILIZATION
    if "polariz" in raw_lower:
        return SocialGrievance.S1_POLARIZATION
    return SocialGrievance.S0_DISCONTENT


class OllamaProvider(LLMProvider):
    """
    Production Ollama LLM Provider (KShield v3.0).
    
    Features:
    - Multi-model task routing (different models for different tasks)
    - Automatic retry with exponential backoff
    - Connection health monitoring
    - Full V3 signal extraction pipeline
    - Legacy API compatibility (classify_threat, identify_role, etc.)
    - Inference latency tracking
    - Sheng/Swahili-aware prompting
    
    Architecture:
        Text → [Sheng Detect] → [Threat Scan] → ThreatSignal (BaseRisk)
                                → [Context Scan] → ContextAnalysis (CSM)
                                → [Indices Scan] → AdvancedIndices (LEI/SI/MS/AA)
                                → [TTA Scan]     → TimeToAction  
                                → [Resilience]   → ResilienceIndex
                                → [Role Scan]    → RoleType V3
                                ──────────────────→ KShieldSignal (aggregate)
    """

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        # Legacy compat: allow positional base_url/model 
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
    ):
        self.config = config or OllamaConfig(base_url=base_url, default_model=model)
        self.base_url = self.config.base_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._healthy: bool = False
        self._available_models: List[str] = []
        self.metrics = SessionMetrics()

    # =========================================================================
    # Session Management
    # =========================================================================

    async def _ensure_session(self):
        """Lazily create aiohttp session with configured timeouts."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.config.connect_timeout,
                total=self.config.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Clean up HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # Health & Model Management
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        await self._ensure_session()
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._available_models = [
                        m.get("name", "") for m in data.get("models", [])
                    ]
                    self._healthy = True
                    logger.info(
                        f"Ollama healthy: {len(self._available_models)} models available"
                    )
                    return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
        self._healthy = False
        return False

    async def list_models(self) -> List[str]:
        """List available Ollama models."""
        await self.health_check()
        return self._available_models

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        await self._ensure_session()
        try:
            payload = {"name": model_name, "stream": False}
            async with self._session.post(
                f"{self.base_url}/api/pull", json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Successfully pulled model: {model_name}")
                    return True
                logger.error(f"Failed to pull {model_name}: {resp.status}")
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
        return False

    async def ensure_model(self, model_name: str) -> bool:
        """Ensure a model is available, pulling if necessary."""
        models = await self.list_models()
        # Check if model is already available (partial match for tags)
        for m in models:
            if model_name in m or m in model_name:
                return True
        logger.info(f"Model {model_name} not found locally, pulling...")
        return await self.pull_model(model_name)

    # =========================================================================
    # Core Generation (with retries)
    # =========================================================================

    async def _generate_json(
        self,
        prompt: str,
        system_prompt: str,
        task: AnalysisTask = AnalysisTask.THREAT_CLASSIFICATION,
    ) -> Dict[str, Any]:
        """
        Generate JSON response from Ollama with retry logic.
        
        Args:
            prompt: The user prompt
            system_prompt: System instructions
            task: Which analysis task (for model routing + metrics)
            
        Returns:
            Parsed JSON dict, empty dict on failure
        """
        await self._ensure_session()

        model = self.config.get_model_for_task(task)
        options = self.config.get_options_for_task(task)

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": options,
        }

        if self.config.log_prompts:
            logger.debug(f"[{task.value}] Prompt: {prompt[:200]}...")

        for attempt in range(self.config.max_retries):
            start_time = time.monotonic()
            metric = InferenceMetrics(task=task, model=model)

            try:
                async with self._session.post(
                    f"{self.base_url}/api/generate", json=payload
                ) as resp:
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    metric.latency_ms = elapsed_ms

                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            f"Ollama HTTP {resp.status} (attempt {attempt+1}): "
                            f"{error_text[:200]}"
                        )
                        metric.success = False
                        metric.error = f"HTTP {resp.status}"
                        self.metrics.record(metric)

                        if resp.status == 404:
                            # Model not found — try pulling it
                            logger.info(f"Model {model} not found, attempting pull...")
                            await self.pull_model(model)
                        
                        if attempt < self.config.max_retries - 1:
                            delay = self.config.retry_delay * (
                                self.config.retry_backoff ** attempt
                            )
                            await asyncio.sleep(delay)
                            continue
                        return {}

                    result = await resp.json()
                    response_text = result.get("response", "{}")
                    
                    # Track token usage
                    metric.prompt_tokens = result.get("prompt_eval_count", 0)
                    metric.completion_tokens = result.get("eval_count", 0)
                    metric.total_tokens = metric.prompt_tokens + metric.completion_tokens

                    if self.config.log_responses:
                        logger.debug(
                            f"[{task.value}] Response ({elapsed_ms:.0f}ms): "
                            f"{response_text[:300]}"
                        )

                    # Parse JSON
                    parsed = json.loads(response_text)
                    metric.success = True
                    self.metrics.record(metric)
                    return parsed

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse error (attempt {attempt+1}): {e}"
                )
                metric.success = False
                metric.error = f"JSON parse: {e}"
                self.metrics.record(metric)

            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                logger.warning(
                    f"Ollama timeout after {elapsed_ms:.0f}ms (attempt {attempt+1})"
                )
                metric.latency_ms = elapsed_ms
                metric.success = False
                metric.error = "timeout"
                self.metrics.record(metric)

            except aiohttp.ClientError as e:
                logger.warning(f"Connection error (attempt {attempt+1}): {e}")
                metric.success = False
                metric.error = str(e)
                self.metrics.record(metric)

            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt+1}): {e}")
                metric.success = False
                metric.error = str(e)
                self.metrics.record(metric)

            # Retry delay
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (
                    self.config.retry_backoff ** attempt
                )
                await asyncio.sleep(delay)

        logger.error(f"All {self.config.max_retries} attempts failed for {task.value}")
        return {}

    async def _generate_text(
        self,
        prompt: str,
        system_prompt: str,
        task: AnalysisTask = AnalysisTask.SUMMARY,
    ) -> str:
        """Generate plain text response (non-JSON)."""
        await self._ensure_session()
        model = self.config.get_model_for_task(task)
        options = self.config.get_options_for_task(task)

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": options,
        }

        try:
            async with self._session.post(
                f"{self.base_url}/api/generate", json=payload
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("response", "")
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
        return ""

    # =========================================================================
    # V3 Core Analysis Methods
    # =========================================================================

    async def analyze_threat_landscape(
        self,
        text: str,
        context: Optional[Dict] = None,
    ) -> Tuple[Optional[ThreatSignal], Optional[ContextAnalysis]]:
        """
        Dual-Layer Analysis:
        1. Threat Scan → ThreatSignal (BaseRisk)
        2. Context Scan → ContextAnalysis (CSM)
        
        Returns (ThreatSignal, ContextAnalysis) or (None, None) on failure.
        """
        # Run threat and context analysis in parallel
        t_prompt = format_threat_v3_prompt(text, context)
        c_prompt = format_context_v3_prompt(text)

        t_data, c_data = await asyncio.gather(
            self._generate_json(
                t_prompt, THREAT_TAXONOMY_SYSTEM,
                task=AnalysisTask.THREAT_CLASSIFICATION,
            ),
            self._generate_json(
                c_prompt, CONTEXT_ANALYST_SYSTEM,
                task=AnalysisTask.CONTEXT_ANALYSIS,
            ),
        )

        # Parse threat signal
        if not t_data.get("category"):
            logger.warning("Failed to extract threat data")
            return None, None

        try:
            threat_signal = ThreatSignal(
                category=_parse_threat_category(t_data.get("category", "unknown")),
                tier=_parse_threat_tier(t_data.get("tier", "TIER_5_NON_THREAT")),
                intent=float(t_data.get("intent", 0.0)),
                capability=float(t_data.get("capability", 0.0)),
                specificity=float(t_data.get("specificity", 0.0)),
                reach=float(t_data.get("reach", 0.0)),
                trajectory=float(t_data.get("trajectory", 0.0)),
                classification_reason=t_data.get("reasoning", ""),
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Threat signal parse error: {e}")
            return None, None

        # Parse context analysis
        try:
            context_analysis = ContextAnalysis(
                economic_strain=_parse_economic_grievance(
                    c_data.get("economic_grievance", "E0_legitimate_grievance")
                ),
                social_fracture=_parse_social_grievance(
                    c_data.get("social_grievance", "S0_normal_discontent")
                ),
                economic_dissatisfaction_score=float(
                    c_data.get("economic_score", 0.0)
                ),
                social_dissatisfaction_score=float(
                    c_data.get("social_score", 0.0)
                ),
                shock_marker=float(c_data.get("shock_marker", 0.0)),
                polarization_marker=float(c_data.get("polarization_marker", 0.0)),
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Context analysis parse error: {e}")
            context_analysis = ContextAnalysis(
                economic_strain=EconomicGrievance.E0_LEGITIMATE,
                social_fracture=SocialGrievance.S0_DISCONTENT,
                economic_dissatisfaction_score=0.0,
                social_dissatisfaction_score=0.0,
                shock_marker=0.0,
                polarization_marker=0.0,
            )

        return threat_signal, context_analysis

    async def analyze_indices(self, text: str) -> Optional[AdvancedIndices]:
        """Extract Advanced Indices: LEI, SI, MS, AA."""
        prompt = format_indices_v3_prompt(text)
        data = await self._generate_json(
            prompt, INDICES_SYSTEM, task=AnalysisTask.INDICES_EXTRACTION
        )

        if not data:
            return None

        try:
            return AdvancedIndices(
                lei_score=float(data.get("lei_score", 0.0)),
                institution_target=str(data.get("lei_target", "")),
                si_score=float(data.get("si_score", 0.0)),
                cognitive_rigidity=float(data.get("cognitive_rigidity", 0.0)),
                identity_fusion=float(data.get("identity_fusion", 0.0)),
                maturation_score=float(data.get("maturation_score", 0.0)),
                maturation_stage=str(data.get("maturation_stage", "Rumor")),
                aa_score=float(data.get("aa_score", 0.0)),
                evasion_technique=str(data.get("aa_technique", "None")),
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Indices parse error: {e}")
            return None

    async def analyze_tta(self, text: str) -> TimeToAction:
        """Analyze Time-To-Action urgency."""
        prompt = format_tta_prompt(text)
        data = await self._generate_json(
            prompt, TTA_SYSTEM, task=AnalysisTask.TIME_TO_ACTION
        )
        try:
            return TimeToAction(data.get("tta_category", "chronic_14d").lower())
        except ValueError:
            return TimeToAction.CHRONIC_14D

    async def analyze_resilience(self, text: str) -> ResilienceIndex:
        """Analyze counter-narrative resilience factors."""
        prompt = format_resilience_prompt(text)
        data = await self._generate_json(
            prompt, RESILIENCE_SYSTEM, task=AnalysisTask.RESILIENCE_ANALYSIS
        )
        return ResilienceIndex(
            counter_narrative_score=float(data.get("counter_narrative_score", 0.0)),
            community_resilience=float(data.get("community_resilience", 0.0)),
            confusion_factor=float(data.get("confusion_factor", 0.0)),
        )

    async def analyze_role_v3(self, text: str) -> RoleTypeV3:
        """Classify author's network role (V3 taxonomy)."""
        prompt = format_role_v3_prompt(text)
        data = await self._generate_json(
            prompt, ROLE_V3_SYSTEM, task=AnalysisTask.ROLE_CLASSIFICATION
        )

        role_str = data.get("role", "OBSERVER").lower()
        # Normalize common aliases
        alias_map = {
            "amplifier": "unwitting_amplifier",
            "op_signaler": "op_signaler",
            "operational_signaler": "op_signaler",
            "signaler": "op_signaler",
        }
        role_str = alias_map.get(role_str, role_str)

        try:
            return RoleTypeV3(role_str)
        except ValueError:
            return RoleTypeV3.OBSERVER

    # =========================================================================
    # Full V3 Pipeline (All signals in one call)
    # =========================================================================

    async def full_analysis(
        self,
        text: str,
        source_id: str = "",
        context: Optional[Dict] = None,
        skip_indices: bool = False,
        skip_resilience: bool = False,
    ) -> Optional[KShieldSignal]:
        """
        Run the COMPLETE V3 analysis pipeline on a single text.
        
        Returns a fully populated KShieldSignal with:
        - ThreatSignal (BaseRisk)
        - ContextAnalysis (CSM)
        - AdvancedIndices (LEI/SI/MS/AA)
        - TimeToAction
        - ResilienceIndex
        - RoleType V3
        - Calculated adjusted_risk
        
        Args:
            text: Content to analyze
            source_id: Unique identifier for the content
            context: Optional context dict (platform, author info, etc.)
            skip_indices: Skip advanced indices (faster)
            skip_resilience: Skip resilience analysis (faster)
        """
        # Phase 1: Dual-layer (parallel)
        threat, ctx = await self.analyze_threat_landscape(text, context)
        if threat is None or ctx is None:
            logger.warning(f"Full analysis failed for source={source_id}")
            return None

        # Phase 2: Supplementary analyses (parallel where possible)
        tasks = [self.analyze_tta(text), self.analyze_role_v3(text)]
        
        if not skip_indices:
            tasks.append(self.analyze_indices(text))
        if not skip_resilience:
            tasks.append(self.analyze_resilience(text))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        tta = results[0] if not isinstance(results[0], Exception) else TimeToAction.CHRONIC_14D
        role = results[1] if not isinstance(results[1], Exception) else RoleTypeV3.OBSERVER
        
        idx = 2
        indices = None
        if not skip_indices:
            indices = results[idx] if not isinstance(results[idx], Exception) else None
            idx += 1
        
        resilience = None
        if not skip_resilience:
            resilience = results[idx] if not isinstance(results[idx], Exception) else None

        # Default indices if skipped or failed
        if indices is None:
            indices = AdvancedIndices(
                lei_score=0.0, institution_target="",
                si_score=0.0, cognitive_rigidity=0.0, identity_fusion=0.0,
                maturation_score=0.0, maturation_stage="Rumor",
                aa_score=0.0, evasion_technique="None",
            )

        # Build aggregate signal
        signal = KShieldSignal(
            source_id=source_id,
            timestamp=datetime.now(),
            content_text=text,
            threat=threat,
            context=ctx,
            indices=indices,
            tta=tta,
            resilience=resilience,
            role=role,
        )
        signal.calculate_risk()

        return signal

    # =========================================================================
    # Legacy API (Abstract Base Class Implementation)
    # =========================================================================

    async def classify_threat(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ThreatClassification:
        """
        Legacy threat classification via V3 pipeline.
        Maps V3 ThreatSignal → Legacy ThreatClassification.
        """
        threat, ctx = await self.analyze_threat_landscape(text, context)

        if threat is None:
            return ThreatClassification(
                tier=LegacyThreatTier.TIER_5,
                confidence=0.0,
                reasoning="Analysis failed",
            )

        # Map V3 tier → legacy tier
        legacy_tier = _V3_TO_LEGACY_TIER.get(
            threat.tier, LegacyThreatTier.TIER_5
        )

        # Derive confidence from score spread
        scores = [threat.intent, threat.capability, threat.specificity, threat.reach]
        confidence = 1.0 - (max(scores) - min(scores)) if scores else 0.5

        return ThreatClassification(
            tier=legacy_tier,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            base_risk=threat.base_risk_score,
            intent_score=threat.intent,
            capability_score=threat.capability,
            specificity_score=threat.specificity,
            reach_score=threat.reach,
            reasoning=threat.classification_reason,
            model_name=self.config.get_model_for_task(
                AnalysisTask.THREAT_CLASSIFICATION
            ),
        )

    async def identify_role(
        self,
        author_posts: List[str],
        author_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoleClassification:
        """
        Legacy role identification via V3 pipeline.
        Analyzes multiple posts and returns dominant role.
        """
        if not author_posts:
            return RoleClassification(
                role=LegacyRoleType.UNKNOWN,
                confidence=0.0,
                reasoning="No posts provided",
            )

        # Analyze each post's role
        roles = []
        for post in author_posts[:10]:  # Cap at 10 posts
            v3_role = await self.analyze_role_v3(post)
            roles.append(v3_role)

        # Find dominant role
        from collections import Counter
        role_counts = Counter(roles)
        dominant_v3, count = role_counts.most_common(1)[0]
        confidence = count / len(roles)

        legacy_role = _V3_TO_LEGACY_ROLE.get(dominant_v3, LegacyRoleType.UNKNOWN)

        return RoleClassification(
            role=legacy_role,
            confidence=round(confidence, 3),
            reasoning=f"Dominant role across {len(roles)} posts: {dominant_v3.value}",
            behavioral_signals=[r.value for r in roles],
            model_name=self.config.get_model_for_task(
                AnalysisTask.ROLE_CLASSIFICATION
            ),
        )

    async def analyze_narrative(
        self,
        posts: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> NarrativeAnalysis:
        """
        Legacy narrative analysis via V3 pipeline.
        Analyzes multiple posts for narrative patterns.
        """
        if not posts:
            return NarrativeAnalysis(
                narrative_type="unknown",
                maturity=NarrativeMaturity.RUMOR,
                sample_size=0,
            )

        # Combine posts for narrative context
        combined = "\n---\n".join(posts[:20])  # Cap at 20 posts
        
        system_prompt = """You are the KShield Narrative Analyst.
Analyze these posts for narrative patterns. Identify:
1. The dominant narrative type (e.g. "government_corruption", "economic_hardship", "ethnic_tension")
2. Key themes and target entities
3. Whether the narrative appears coordinated
4. The maturation stage (Rumor/Narrative/Campaign)
5. Dominant emotion and whether there's a call to action

Return JSON:
{
  "narrative_type": "type_name",
  "maturity": "Rumor" | "Narrative" | "Campaign",
  "themes": ["theme1", "theme2"],
  "target_entities": ["entity1"],
  "is_coordinated": true/false,
  "coordination_confidence": 0.0-1.0,
  "dominant_emotion": "emotion",
  "call_to_action": true/false
}"""
        
        prompt = f"""Analyze these {len(posts)} posts for narrative patterns:

{combined}

Context: {json.dumps(context or {})}"""

        data = await self._generate_json(
            prompt, system_prompt, task=AnalysisTask.NARRATIVE_ANALYSIS
        )

        maturity_str = data.get("maturity", "Rumor").lower()
        maturity_map = {
            "rumor": NarrativeMaturity.RUMOR,
            "narrative": NarrativeMaturity.NARRATIVE,
            "campaign": NarrativeMaturity.CAMPAIGN,
        }

        return NarrativeAnalysis(
            narrative_type=data.get("narrative_type", "unknown"),
            maturity=maturity_map.get(maturity_str, NarrativeMaturity.RUMOR),
            themes=data.get("themes", []),
            target_entities=data.get("target_entities", []),
            is_coordinated=bool(data.get("is_coordinated", False)),
            coordination_confidence=float(
                data.get("coordination_confidence", 0.0)
            ),
            dominant_emotion=data.get("dominant_emotion", ""),
            call_to_action=bool(data.get("call_to_action", False)),
            sample_size=len(posts),
            model_name=self.config.get_model_for_task(
                AnalysisTask.NARRATIVE_ANALYSIS
            ),
        )

    async def batch_classify(
        self,
        texts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ThreatClassification]:
        """
        Batch threat classification.
        Processes texts with controlled concurrency.
        """
        if not texts:
            return []

        contexts = contexts or [None] * len(texts)
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def _classify_one(text: str, ctx: Optional[Dict]) -> ThreatClassification:
            async with semaphore:
                result = await self.classify_threat(text, ctx)
                await asyncio.sleep(self.config.batch_delay)
                return result

        tasks = [
            _classify_one(text, ctx)
            for text, ctx in zip(texts, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Replace exceptions with default
        final = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Batch classify error: {r}")
                final.append(ThreatClassification(
                    tier=LegacyThreatTier.TIER_5,
                    confidence=0.0,
                    reasoning=f"Error: {r}",
                ))
            else:
                final.append(r)
        
        return final

    # =========================================================================
    # Sheng/Swahili Detection
    # =========================================================================

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language mix in text (English/Swahili/Sheng).
        
        Returns:
            {
                "primary_language": "sheng"|"swahili"|"english"|"mixed",
                "english_pct": 0.0-1.0,
                "swahili_pct": 0.0-1.0,
                "sheng_pct": 0.0-1.0,
                "code_switched": true/false,
                "translation": "English translation if not English"
            }
        """
        system = """You are a Kenyan linguistics expert specializing in:
- Standard Swahili (Kiswahili sanifu)
- Sheng (Nairobi street slang mixing English/Swahili/ethnic languages)
- Code-switching patterns common in Kenyan social media

Identify the language composition and provide translation to English if needed.

Return JSON:
{
  "primary_language": "sheng" | "swahili" | "english" | "mixed",
  "english_pct": 0.0-1.0,
  "swahili_pct": 0.0-1.0,
  "sheng_pct": 0.0-1.0,
  "code_switched": true/false,
  "translation": "English translation"
}"""
        prompt = f'Analyze language: "{text}"'
        
        return await self._generate_json(
            prompt, system, task=AnalysisTask.SHENG_TRANSLATION
        )

    async def translate_to_english(self, text: str) -> str:
        """Translate Sheng/Swahili text to English for downstream analysis."""
        lang = await self.detect_language(text)
        if lang.get("primary_language") == "english":
            return text
        return lang.get("translation", text)

    # =========================================================================
    # Policy Event Analysis
    # =========================================================================

    async def analyze_policy_impact(
        self,
        text: str,
        policy_event: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Analyze text for policy impact signals (Kenya-specific).
        
        Args:
            text: Social media post
            policy_event: Optional policy event context dict
            
        Returns:
            {
                "policy_relevance": 0.0-1.0,
                "stance": "anti" | "pro" | "neutral",
                "stance_intensity": 0.0-1.0,
                "mobilization_potential": 0.0-1.0,
                "economic_impact_signal": 0.0-1.0,
                "grievance_type": "economic" | "social" | "governance" | "none",
                "call_to_action": true/false,
                "target_institution": "institution name or none"
            }
        """
        system = """You are a Kenyan Policy Impact Analyst. 
Analyze social media posts for their relationship to government policy events.

CONTEXT: Kenya 2025-2026 — active policy landscape including Finance Bills, 
housing levies, digital economy taxes, healthcare mandates, and security operations.

Understand Sheng slang: "serikali" (government), "mwananchi" (citizen), 
"kupiga kelele" (protest), "taxes ni mob" (taxes are too much), etc.

Evaluate: policy relevance, stance, mobilization potential, and institutional targeting.

Return JSON:
{
  "policy_relevance": 0.0-1.0,
  "stance": "anti" | "pro" | "neutral",
  "stance_intensity": 0.0-1.0,
  "mobilization_potential": 0.0-1.0,
  "economic_impact_signal": 0.0-1.0,
  "grievance_type": "economic" | "social" | "governance" | "none",
  "call_to_action": true/false,
  "target_institution": "institution name or none"
}"""

        ctx = ""
        if policy_event:
            ctx = f"\nActive Policy Event: {json.dumps(policy_event)}"

        prompt = f'Analyze policy impact:\n\nText: "{text}"{ctx}'

        return await self._generate_json(
            prompt, system, task=AnalysisTask.POLICY_IMPACT
        )

    # =========================================================================
    # Utility
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get session inference metrics summary."""
        return self.metrics.summary()

    def reset_metrics(self):
        """Reset accumulated metrics."""
        self.metrics = SessionMetrics()

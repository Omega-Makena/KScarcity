"""
KShield Pulse End-to-End Analyzer

Orchestrates the complete intelligence pipeline:
    Raw Text → Language Detection → NLP Preprocessing → LLM Analysis → 
    Signal Assembly → Risk Scoring → Gating Decision → Bridge Output

This is the main entry point for analyzing content through the full KShield stack.

Usage:
    from kshiked.pulse.llm.analyzer import KShieldAnalyzer
    
    analyzer = KShieldAnalyzer()
    async with analyzer:
        # Single text
        report = await analyzer.analyze("Serikali wezi! Twende streets!")
        
        # Full pipeline with signal output
        signal = await analyzer.analyze_to_signal("text here")
        
        # Batch with embeddings
        reports = await analyzer.analyze_batch(tweets_df)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .config import OllamaConfig, AnalysisTask, SessionMetrics
from .ollama import OllamaProvider
from .embeddings import OllamaEmbeddings
from .signals import (
    KShieldSignal,
    ThreatSignal,
    ContextAnalysis,
    AdvancedIndices,
    ThreatCategory,
    ThreatTier,
    TimeToAction,
    ResilienceIndex,
    RoleType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Analysis Report
# =============================================================================

@dataclass
class AnalysisReport:
    """
    Human-readable analysis report for a single text.
    
    This is the primary output format — combines all V3 signals
    into a structured, actionable intelligence report.
    """
    # Source
    source_id: str = ""
    text: str = ""
    timestamp: str = ""
    
    # Language
    language: str = "english"
    translation: str = ""
    sheng_detected: bool = False
    
    # Threat Assessment
    threat_category: str = "unknown"
    threat_tier: str = "TIER_5_NON_THREAT"
    threat_tier_label: str = "Non-Threat"
    base_risk: float = 0.0
    adjusted_risk: float = 0.0
    
    # Risk Decomposition
    intent: float = 0.0
    capability: float = 0.0
    specificity: float = 0.0
    reach: float = 0.0
    trajectory: float = 0.0
    
    # Context
    economic_grievance: str = "E0"
    social_grievance: str = "S0"
    csm: float = 1.0
    shock_detected: bool = False
    polarization_level: str = "low"
    
    # Advanced Indices
    lei_score: float = 0.0
    si_score: float = 0.0
    maturation_stage: str = "Rumor"
    aa_score: float = 0.0
    
    # Temporal & Role
    time_to_action: str = "chronic_14d"
    role: str = "observer"
    
    # Resilience 
    counter_narrative_strength: float = 0.0
    community_resilience: float = 0.0
    
    # Policy
    policy_relevance: float = 0.0
    policy_stance: str = "neutral"
    mobilization_potential: float = 0.0
    
    # Decision
    gating_status: str = "MONITOR_CONTEXT"
    recommended_action: str = "No action required"
    reasoning: str = ""
    
    # Performance
    total_latency_ms: float = 0.0
    llm_calls: int = 0
    
    @property
    def is_actionable(self) -> bool:
        """Does this require human attention?"""
        return self.gating_status in [
            "IMMEDIATE_ESCALATION",
            "ACTIVE_MONITORING",
            "ANALYST_REVIEW",
        ]
    
    @property
    def risk_level(self) -> str:
        """Human-readable risk level."""
        if self.adjusted_risk >= 80:
            return "CRITICAL"
        elif self.adjusted_risk >= 60:
            return "HIGH"
        elif self.adjusted_risk >= 40:
            return "MEDIUM"
        elif self.adjusted_risk >= 20:
            return "LOW"
        return "MINIMAL"
    
    def summary(self) -> str:
        """One-line summary for dashboards."""
        return (
            f"[{self.risk_level}] {self.threat_tier_label} | "
            f"Risk: {self.adjusted_risk:.0f} | "
            f"Status: {self.gating_status} | "
            f"Role: {self.role} | "
            f"Lang: {self.language}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# Tier Labels
# =============================================================================

TIER_LABELS = {
    "TIER_1_EXISTENTIAL": ("Tier 1: Existential Threat", "IMMEDIATE_ESCALATION"),
    "TIER_2_SEVERE_STABILITY": ("Tier 2: Severe Stability Threat", "IMMEDIATE_ESCALATION"),
    "TIER_3_HIGH_RISK": ("Tier 3: High-Risk Destabilization", "ACTIVE_MONITORING"),
    "TIER_4_EMERGING": ("Tier 4: Emerging Threat", "ANALYST_REVIEW"),
    "TIER_5_NON_THREAT": ("Tier 5: Non-Threat / Protected Speech", "MONITOR_CONTEXT"),
}

GATING_ACTIONS = {
    "IMMEDIATE_ESCALATION": "Escalate to security team immediately. Requires human review within 1 hour.",
    "ACTIVE_MONITORING": "Add to active monitoring queue. Requires analyst review within 4 hours.",
    "ANALYST_REVIEW": "Flag for periodic analyst review. Review within 24 hours.",
    "ROUTINE_MONITORING": "Add to routine monitoring. Weekly batch review.",
    "LOG_ONLY": "Log for trend analysis. No immediate action required.",
    "MONITOR_CONTEXT": "No threat detected. Monitor contextual stress levels only.",
}


# =============================================================================
# KShield Analyzer
# =============================================================================

class KShieldAnalyzer:
    """
    End-to-end KShield intelligence analyzer.
    
    Orchestrates:
    1. Language detection (Sheng/Swahili/English)
    2. LLM-based threat analysis (14-category taxonomy)
    3. Context stress analysis (E0-E4/S0-S4)
    4. Advanced indices (LEI/SI/MS/AA)
    5. Time-to-Action urgency
    6. Resilience factors
    7. Role classification
    8. Policy impact assessment
    9. Risk scoring (BaseRisk × CSM)
    10. Gating decision + recommended action
    
    All powered by local Ollama models.
    """

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        enable_embeddings: bool = False,
    ):
        self.config = config or OllamaConfig()
        self.provider: Optional[OllamaProvider] = None
        self.embedder: Optional[OllamaEmbeddings] = None
        self._enable_embeddings = enable_embeddings
        self.metrics = SessionMetrics()

    async def _ensure_ready(self):
        """Initialize providers."""
        if self.provider is None:
            self.provider = OllamaProvider(config=self.config)
            await self.provider._ensure_session()
        
        if self._enable_embeddings and self.embedder is None:
            self.embedder = OllamaEmbeddings(config=self.config)
            await self.embedder._ensure_session()

    async def close(self):
        """Clean up all resources."""
        if self.provider:
            await self.provider.close()
        if self.embedder:
            await self.embedder.close()
        self.provider = None
        self.embedder = None

    async def __aenter__(self):
        await self._ensure_ready()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # Primary Analysis Method
    # =========================================================================

    async def analyze(
        self,
        text: str,
        source_id: str = "",
        context: Optional[Dict] = None,
        deep: bool = False,
    ) -> AnalysisReport:
        """
        Analyze a single text through the full KShield pipeline.
        
        Args:
            text: Content to analyze
            source_id: Unique identifier
            context: Optional context (platform, author, policy_event, etc.)
            deep: Run all analyses including indices and resilience
            
        Returns:
            AnalysisReport with complete intelligence assessment
        """
        await self._ensure_ready()
        report = AnalysisReport(
            source_id=source_id,
            text=text,
            timestamp=datetime.now().isoformat(),
        )
        start = time.monotonic()

        try:
            # Step 1: Language Detection
            if self.config.enable_sheng_detection:
                lang_data = await self.provider.detect_language(text)
                report.language = lang_data.get("primary_language", "english")
                report.translation = lang_data.get("translation", "")
                report.sheng_detected = lang_data.get("sheng_pct", 0) > 0.2
                report.llm_calls += 1

            # Step 2: Dual-Layer Analysis (Threat + Context — parallel)
            threat, ctx = await self.provider.analyze_threat_landscape(text, context)
            report.llm_calls += 2  # Two parallel calls

            if threat:
                report.threat_category = threat.category.value
                report.threat_tier = threat.tier.value
                report.base_risk = threat.base_risk_score
                report.intent = threat.intent
                report.capability = threat.capability
                report.specificity = threat.specificity
                report.reach = threat.reach
                report.trajectory = threat.trajectory
                report.reasoning = threat.classification_reason
                
                # Get tier label
                tier_info = TIER_LABELS.get(
                    threat.tier.value,
                    ("Unknown", "MONITOR_CONTEXT"),
                )
                report.threat_tier_label = tier_info[0]

            if ctx:
                report.economic_grievance = ctx.economic_strain.value
                report.social_grievance = ctx.social_fracture.value
                report.csm = ctx.stress_multiplier
                report.shock_detected = ctx.shock_marker > 0.5
                
                # Polarization level
                if ctx.polarization_marker > 0.7:
                    report.polarization_level = "critical"
                elif ctx.polarization_marker > 0.4:
                    report.polarization_level = "high"
                elif ctx.polarization_marker > 0.2:
                    report.polarization_level = "moderate"
                else:
                    report.polarization_level = "low"
                
                report.adjusted_risk = min(100.0, report.base_risk * ctx.stress_multiplier)

            # Step 3: Supplementary analyses (sequential for VRAM safety)
            supp_tasks = ["tta", "role"]
            if deep:
                supp_tasks.extend(["indices", "resilience"])
            if self.config.enable_policy_context:
                supp_tasks.append("policy")

            for name in supp_tasks:
                try:
                    if name == "tta":
                        res = await self.provider.analyze_tta(text)
                        report.time_to_action = res.value if isinstance(res, TimeToAction) else "chronic_14d"
                    elif name == "role":
                        res = await self.provider.analyze_role_v3(text)
                        report.role = res.value if isinstance(res, RoleType) else "observer"
                    elif name == "indices":
                        res = await self.provider.analyze_indices(text)
                        if res is not None:
                            report.lei_score = res.lei_score
                            report.si_score = res.si_score
                            report.maturation_stage = res.maturation_stage
                            report.aa_score = res.aa_score
                    elif name == "resilience":
                        res = await self.provider.analyze_resilience(text)
                        if res is not None:
                            report.counter_narrative_strength = res.counter_narrative_score
                            report.community_resilience = res.community_resilience
                    elif name == "policy":
                        policy_ctx = context.get("policy_event") if context else None
                        res = await self.provider.analyze_policy_impact(text, policy_ctx)
                        if isinstance(res, dict):
                            report.policy_relevance = float(res.get("policy_relevance", 0))
                            report.policy_stance = res.get("stance", "neutral")
                            report.mobilization_potential = float(
                                res.get("mobilization_potential", 0)
                            )
                    report.llm_calls += 1
                except Exception as e:
                    logger.warning(f"Supplementary analysis '{name}' failed: {e}")

            # Step 4: Gating Decision
            report.gating_status = self._compute_gating(report)
            report.recommended_action = GATING_ACTIONS.get(
                report.gating_status, "Monitor"
            )

        except Exception as e:
            logger.error(f"Analysis pipeline error: {e}")
            report.reasoning = f"Pipeline error: {e}"

        report.total_latency_ms = (time.monotonic() - start) * 1000
        return report

    # =========================================================================
    # Signal Output (for bridge.py integration)
    # =========================================================================

    async def analyze_to_signal(
        self,
        text: str,
        source_id: str = "",
        context: Optional[Dict] = None,
    ) -> Optional[KShieldSignal]:
        """
        Analyze text and return a KShieldSignal for bridge integration.
        
        Use this when you need to feed results into the Pulse → Simulation bridge.
        """
        await self._ensure_ready()
        return await self.provider.full_analysis(text, source_id, context)

    # =========================================================================
    # Batch Analysis
    # =========================================================================

    async def analyze_batch(
        self,
        texts: List[str],
        source_ids: Optional[List[str]] = None,
        deep: bool = False,
    ) -> List[AnalysisReport]:
        """
        Analyze multiple texts with controlled concurrency.
        
        Args:
            texts: List of texts
            source_ids: Optional identifiers
            deep: Run full analysis (slower)
            
        Returns:
            List of AnalysisReport objects
        """
        await self._ensure_ready()
        
        source_ids = source_ids or [str(i) for i in range(len(texts))]
        reports = []
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def _analyze_one(text: str, sid: str) -> AnalysisReport:
            async with semaphore:
                report = await self.analyze(text, sid, deep=deep)
                await asyncio.sleep(self.config.batch_delay)
                return report

        for i in range(0, len(texts), self.config.batch_size):
            batch = [
                _analyze_one(texts[j], source_ids[j])
                for j in range(i, min(i + self.config.batch_size, len(texts)))
            ]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Batch analysis error: {r}")
                    reports.append(AnalysisReport(reasoning=f"Error: {r}"))
                else:
                    reports.append(r)
            
            processed = min(i + self.config.batch_size, len(texts))
            logger.info(f"Batch progress: {processed}/{len(texts)}")

        return reports

    # =========================================================================
    # Semantic Search (requires embeddings)
    # =========================================================================

    async def semantic_search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Find semantically similar texts using embeddings.
        
        Args:
            query: Search query
            corpus: Texts to search through
            top_k: Number of results
            
        Returns:
            List of (index, similarity, text) tuples
        """
        if self.embedder is None:
            self._enable_embeddings = True
            await self._ensure_ready()
        
        return await self.embedder.find_similar(query, corpus, top_k)

    # =========================================================================
    # Narrative Clustering
    # =========================================================================

    async def cluster_narratives(
        self,
        texts: List[str],
        n_clusters: int = 8,
    ) -> Dict[str, Any]:
        """
        Cluster texts by semantic similarity + narrative analysis.
        
        Returns cluster info with representative texts and narrative labels.
        """
        if self.embedder is None:
            self._enable_embeddings = True
            await self._ensure_ready()

        # Step 1: Semantic clustering via embeddings
        clusters = await self.embedder.cluster_texts(texts, n_clusters)
        
        # Step 2: Label each cluster using narrative analysis
        for cluster_id, rep_text in clusters.get("representatives", {}).items():
            narrative = await self.provider.analyze_narrative([rep_text])
            clusters.setdefault("narrative_labels", {})[cluster_id] = {
                "type": narrative.narrative_type,
                "maturity": narrative.maturity.value,
                "emotion": narrative.dominant_emotion,
            }
        
        return clusters

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _compute_gating(self, report: AnalysisReport) -> str:
        """
        Apply KShield gating rules to determine action status.
        
        Gating Rules (from Architecture PDF):
        - BaseRisk >= 80 → IMMEDIATE_ESCALATION
        - 60-80, CSM >= 1.10 → ACTIVE_MONITORING
        - 60-80, CSM < 1.10 → ROUTINE_MONITORING
        - 40-60, CSM >= 1.15 → ANALYST_REVIEW
        - 40-60, CSM < 1.15 → LOG_ONLY
        - < 40 → MONITOR_CONTEXT
        
        Additional modifiers:
        - TTA = IMMEDIATE_24H → escalate one level
        - Mobilizer/Ideologue role → escalate one level
        """
        base = report.base_risk
        csm = report.csm
        
        # Base gating
        if base >= 80:
            status = "IMMEDIATE_ESCALATION"
        elif 60 <= base < 80:
            status = "ACTIVE_MONITORING" if csm >= 1.10 else "ROUTINE_MONITORING"
        elif 40 <= base < 60:
            status = "ANALYST_REVIEW" if csm >= 1.15 else "LOG_ONLY"
        else:
            status = "MONITOR_CONTEXT"
        
        # TTA escalation
        if report.time_to_action == "immediate_24h":
            status = self._escalate(status)
        
        # Role escalation (mobilizers and ideologues are higher priority)
        if report.role in ("mobilizer", "ideologue", "op_signaler"):
            status = self._escalate(status)
        
        # High mobilization potential escalation
        if report.mobilization_potential > 0.7:
            status = self._escalate(status)
        
        return status

    @staticmethod
    def _escalate(current_status: str) -> str:
        """Escalate one gating level."""
        escalation_chain = [
            "MONITOR_CONTEXT",
            "LOG_ONLY",
            "ROUTINE_MONITORING",
            "ANALYST_REVIEW",
            "ACTIVE_MONITORING",
            "IMMEDIATE_ESCALATION",
        ]
        try:
            idx = escalation_chain.index(current_status)
            return escalation_chain[min(idx + 1, len(escalation_chain) - 1)]
        except ValueError:
            return current_status

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics from provider and embedder."""
        metrics = {}
        if self.provider:
            metrics["provider"] = self.provider.get_metrics()
        if self.embedder:
            metrics["embeddings"] = self.embedder.cache_stats()
        return metrics

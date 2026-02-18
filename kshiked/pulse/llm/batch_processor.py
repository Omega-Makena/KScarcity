"""
KShield Pulse Batch Processor

Processes large datasets (100K+ tweets) through the Ollama LLM pipeline
with controlled concurrency, progress tracking, checkpointing, and CSV output.

Designed for the synthetic Kenya tweet dataset (98K tweets with policy events).

Usage:
    from kshiked.pulse.llm.batch_processor import BatchProcessor
    from kshiked.pulse.llm.config import OllamaConfig
    
    config = OllamaConfig.single_model("llama3.1:8b")
    processor = BatchProcessor(config=config)
    
    # Process from CSV
    await processor.process_csv(
        input_path="data/synthetic_kenya_policy/synthetic_tweets.csv",
        output_path="data/synthetic_kenya_policy/analyzed_tweets.csv",
        text_column="text",
    )
"""

import asyncio
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .config import OllamaConfig, AnalysisTask
from .ollama import OllamaProvider
from .signals import (
    KShieldSignal,
    ThreatSignal,
    ContextAnalysis,
    ThreatCategory,
    ThreatTier,
    EconomicGrievance,
    SocialGrievance,
    TimeToAction,
    RoleType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Processing Modes
# =============================================================================

@dataclass
class ProcessingMode:
    """Defines what analyses to run per text."""
    
    # Core (always runs)
    threat_analysis: bool = True
    context_analysis: bool = True
    
    # Extended
    indices_extraction: bool = False
    time_to_action: bool = True
    resilience_analysis: bool = False
    role_classification: bool = True
    
    # Kenya-specific
    language_detection: bool = True
    policy_impact: bool = True
    
    # Embedding
    compute_embedding: bool = False

    @classmethod
    def fast(cls) -> "ProcessingMode":
        """Fast mode: threat + context only (~2 LLM calls per text)."""
        return cls(
            threat_analysis=True,
            context_analysis=True,
            indices_extraction=False,
            time_to_action=False,
            resilience_analysis=False,
            role_classification=False,
            language_detection=False,
            policy_impact=False,
            compute_embedding=False,
        )

    @classmethod
    def standard(cls) -> "ProcessingMode":
        """Standard mode: threat + context + role + TTA (~4-5 calls)."""
        return cls(
            threat_analysis=True,
            context_analysis=True,
            indices_extraction=False,
            time_to_action=True,
            resilience_analysis=False,
            role_classification=True,
            language_detection=True,
            policy_impact=True,
            compute_embedding=False,
        )

    @classmethod
    def full(cls) -> "ProcessingMode":
        """Full mode: all analyses (~7-8 LLM calls per text)."""
        return cls(
            threat_analysis=True,
            context_analysis=True,
            indices_extraction=True,
            time_to_action=True,
            resilience_analysis=True,
            role_classification=True,
            language_detection=True,
            policy_impact=True,
            compute_embedding=True,
        )


# =============================================================================
# Batch Result
# =============================================================================

@dataclass
class AnalysisResult:
    """Result of analyzing a single text."""
    source_id: str = ""
    text: str = ""
    
    # Core
    threat_category: str = "unknown"
    threat_tier: str = "TIER_5_NON_THREAT"
    base_risk: float = 0.0
    adjusted_risk: float = 0.0
    intent: float = 0.0
    capability: float = 0.0
    specificity: float = 0.0
    reach: float = 0.0
    trajectory: float = 0.0
    
    # Context
    economic_grievance: str = "E0_legitimate_grievance"
    social_grievance: str = "S0_normal_discontent"
    economic_score: float = 0.0
    social_score: float = 0.0
    shock_marker: float = 0.0
    polarization_marker: float = 0.0
    csm: float = 1.0
    
    # Indices
    lei_score: float = 0.0
    si_score: float = 0.0
    maturation_score: float = 0.0
    maturation_stage: str = "Rumor"
    aa_score: float = 0.0
    
    # TTA & Resilience
    tta: str = "chronic_14d"
    resilience_counter: float = 0.0
    resilience_community: float = 0.0
    
    # Role
    role: str = "observer"
    
    # Language
    language_primary: str = "english"
    language_sheng_pct: float = 0.0
    language_swahili_pct: float = 0.0
    
    # Policy
    policy_relevance: float = 0.0
    policy_stance: str = "neutral"
    policy_mobilization: float = 0.0
    
    # Status
    gating_status: str = "UNKNOWN"
    reasoning: str = ""
    processing_ms: float = 0.0
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dict for CSV output."""
        return {
            "source_id": self.source_id,
            "text": self.text[:500],  # Truncate for CSV
            "threat_category": self.threat_category,
            "threat_tier": self.threat_tier,
            "base_risk": round(self.base_risk, 2),
            "adjusted_risk": round(self.adjusted_risk, 2),
            "intent": round(self.intent, 3),
            "capability": round(self.capability, 3),
            "specificity": round(self.specificity, 3),
            "reach": round(self.reach, 3),
            "trajectory": round(self.trajectory, 3),
            "economic_grievance": self.economic_grievance,
            "social_grievance": self.social_grievance,
            "economic_score": round(self.economic_score, 3),
            "social_score": round(self.social_score, 3),
            "shock_marker": round(self.shock_marker, 3),
            "polarization_marker": round(self.polarization_marker, 3),
            "csm": round(self.csm, 3),
            "lei_score": round(self.lei_score, 3),
            "si_score": round(self.si_score, 3),
            "maturation_score": round(self.maturation_score, 1),
            "maturation_stage": self.maturation_stage,
            "aa_score": round(self.aa_score, 3),
            "tta": self.tta,
            "resilience_counter": round(self.resilience_counter, 3),
            "resilience_community": round(self.resilience_community, 3),
            "role": self.role,
            "language_primary": self.language_primary,
            "language_sheng_pct": round(self.language_sheng_pct, 2),
            "language_swahili_pct": round(self.language_swahili_pct, 2),
            "policy_relevance": round(self.policy_relevance, 3),
            "policy_stance": self.policy_stance,
            "policy_mobilization": round(self.policy_mobilization, 3),
            "gating_status": self.gating_status,
            "reasoning": self.reasoning[:200],
            "processing_ms": round(self.processing_ms, 1),
            "success": self.success,
        }


# =============================================================================
# Progress Tracking
# =============================================================================

@dataclass
class BatchProgress:
    """Track batch processing progress."""
    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    start_time: float = 0.0
    
    # Tier distribution
    tier_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time if self.start_time else 0
    
    @property  
    def rate(self) -> float:
        """Texts per second."""
        elapsed = self.elapsed_seconds
        return self.processed / elapsed if elapsed > 0 else 0
    
    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining."""
        if self.rate <= 0:
            return float("inf")
        remaining = self.total - self.processed
        return remaining / self.rate
    
    @property
    def pct(self) -> float:
        return (self.processed / self.total * 100) if self.total > 0 else 0
    
    def log_status(self):
        """Log current progress."""
        logger.info(
            f"Progress: {self.processed}/{self.total} ({self.pct:.1f}%) | "
            f"Rate: {self.rate:.1f}/s | ETA: {self.eta_seconds:.0f}s | "
            f"Success: {self.succeeded} | Failed: {self.failed}"
        )


# =============================================================================
# Batch Processor
# =============================================================================

class BatchProcessor:
    """
    Process large datasets through the Ollama LLM pipeline.
    
    Features:
    - Controlled concurrency (VRAM-aware)
    - Checkpointing (resume from interruption)
    - Progress logging
    - CSV input/output
    - Configurable analysis depth (fast/standard/full)
    - Per-text error handling (doesn't crash on individual failures)
    """

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        mode: Optional[ProcessingMode] = None,
    ):
        self.config = config or OllamaConfig()
        self.mode = mode or ProcessingMode.standard()
        self.provider: Optional[OllamaProvider] = None
        self.progress = BatchProgress()

    async def _ensure_provider(self):
        """Create provider if needed."""
        if self.provider is None:
            self.provider = OllamaProvider(config=self.config)
            await self.provider._ensure_session()

    async def close(self):
        """Clean up resources."""
        if self.provider:
            await self.provider.close()
            self.provider = None

    # =========================================================================
    # Single Text Analysis
    # =========================================================================

    async def analyze_one(
        self,
        text: str,
        source_id: str = "",
        context: Optional[Dict] = None,
    ) -> AnalysisResult:
        """
        Analyze a single text through the configured pipeline.
        
        Returns AnalysisResult (always succeeds — errors captured in result).
        """
        await self._ensure_provider()
        result = AnalysisResult(source_id=source_id, text=text)
        start = time.monotonic()

        try:
            # Phase 1: Threat + Context (parallel in provider)
            if self.mode.threat_analysis and self.mode.context_analysis:
                threat, ctx = await self.provider.analyze_threat_landscape(
                    text, context
                )
                if threat:
                    result.threat_category = threat.category.value
                    result.threat_tier = threat.tier.value
                    result.base_risk = threat.base_risk_score
                    result.intent = threat.intent
                    result.capability = threat.capability
                    result.specificity = threat.specificity
                    result.reach = threat.reach
                    result.trajectory = threat.trajectory
                    result.reasoning = threat.classification_reason

                if ctx:
                    result.economic_grievance = ctx.economic_strain.value
                    result.social_grievance = ctx.social_fracture.value
                    result.economic_score = ctx.economic_dissatisfaction_score
                    result.social_score = ctx.social_dissatisfaction_score
                    result.shock_marker = ctx.shock_marker
                    result.polarization_marker = ctx.polarization_marker
                    result.csm = ctx.stress_multiplier
                    result.adjusted_risk = min(
                        100.0, result.base_risk * ctx.stress_multiplier
                    )

            # Phase 2: Parallel supplementary analyses
            tasks = {}
            
            if self.mode.time_to_action:
                tasks["tta"] = self.provider.analyze_tta(text)
            if self.mode.role_classification:
                tasks["role"] = self.provider.analyze_role_v3(text)
            if self.mode.indices_extraction:
                tasks["indices"] = self.provider.analyze_indices(text)
            if self.mode.resilience_analysis:
                tasks["resilience"] = self.provider.analyze_resilience(text)
            if self.mode.language_detection:
                tasks["language"] = self.provider.detect_language(text)
            if self.mode.policy_impact:
                tasks["policy"] = self.provider.analyze_policy_impact(text)

            if tasks:
                task_names = list(tasks.keys())
                task_coros = list(tasks.values())
                results = await asyncio.gather(*task_coros, return_exceptions=True)
                
                for name, res in zip(task_names, results):
                    if isinstance(res, Exception):
                        logger.warning(f"{name} failed for {source_id}: {res}")
                        continue
                    
                    if name == "tta" and isinstance(res, TimeToAction):
                        result.tta = res.value
                    elif name == "role" and isinstance(res, RoleType):
                        result.role = res.value
                    elif name == "indices" and res is not None:
                        result.lei_score = res.lei_score
                        result.si_score = res.si_score
                        result.maturation_score = res.maturation_score
                        result.maturation_stage = res.maturation_stage
                        result.aa_score = res.aa_score
                    elif name == "resilience" and res is not None:
                        result.resilience_counter = res.counter_narrative_score
                        result.resilience_community = res.community_resilience
                    elif name == "language" and isinstance(res, dict):
                        result.language_primary = res.get("primary_language", "english")
                        result.language_sheng_pct = float(res.get("sheng_pct", 0))
                        result.language_swahili_pct = float(res.get("swahili_pct", 0))
                    elif name == "policy" and isinstance(res, dict):
                        result.policy_relevance = float(res.get("policy_relevance", 0))
                        result.policy_stance = res.get("stance", "neutral")
                        result.policy_mobilization = float(
                            res.get("mobilization_potential", 0)
                        )

            # Compute gating status
            if result.base_risk >= 80:
                result.gating_status = "IMMEDIATE_ESCALATION"
            elif 60 <= result.base_risk < 80:
                result.gating_status = (
                    "ACTIVE_MONITORING" if result.csm >= 1.10 
                    else "ROUTINE_MONITORING"
                )
            elif 40 <= result.base_risk < 60:
                result.gating_status = (
                    "ANALYST_REVIEW" if result.csm >= 1.15 
                    else "LOG_ONLY"
                )
            else:
                result.gating_status = "MONITOR_CONTEXT"

            result.success = True

        except Exception as e:
            logger.error(f"Analysis failed for {source_id}: {e}")
            result.success = False
            result.error = str(e)

        result.processing_ms = (time.monotonic() - start) * 1000
        return result

    # =========================================================================
    # CSV Processing
    # =========================================================================

    async def process_csv(
        self,
        input_path: str,
        output_path: str,
        text_column: str = "text",
        id_column: Optional[str] = None,
        limit: Optional[int] = None,
        checkpoint_every: int = 100,
        progress_every: int = 50,
        resume: bool = True,
    ) -> BatchProgress:
        """
        Process a CSV file through the LLM pipeline.
        
        Args:
            input_path: Path to input CSV
            output_path: Path for output CSV (with analysis columns)
            text_column: Column containing text to analyze
            id_column: Column for source IDs (optional, auto-generates if None)
            limit: Max rows to process (None = all)
            checkpoint_every: Write checkpoint every N rows
            progress_every: Log progress every N rows
            resume: Resume from last checkpoint if output exists
            
        Returns:
            BatchProgress with final statistics
        """
        await self._ensure_provider()
        
        # Check Ollama is running
        healthy = await self.provider.health_check()
        if not healthy:
            logger.error("Ollama server is not reachable! Start with: ollama serve")
            raise ConnectionError("Ollama server not available")

        # Read input
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        rows = []
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        if limit:
            rows = rows[:limit]
        
        logger.info(f"Loaded {len(rows)} rows from {input_path}")

        # Check for resume
        already_processed = set()
        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    already_processed.add(row.get("source_id", ""))
            logger.info(f"Resume: {len(already_processed)} already processed")

        # Initialize progress
        self.progress = BatchProgress(
            total=len(rows),
            processed=len(already_processed),
            succeeded=len(already_processed),
            start_time=time.monotonic(),
        )

        # Prepare output file
        sample_result = AnalysisResult()
        fieldnames = list(sample_result.to_dict().keys())
        
        write_mode = "a" if already_processed else "w"
        output_file = open(output_path, write_mode, newline="", encoding="utf-8")
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        
        if not already_processed:
            writer.writeheader()

        # Process rows
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        buffer: List[Dict] = []

        try:
            for i, row in enumerate(rows):
                source_id = row.get(id_column, str(i)) if id_column else str(i)
                
                # Skip already processed
                if source_id in already_processed:
                    continue
                
                text = row.get(text_column, "")
                if not text or not text.strip():
                    continue

                # Build context from other CSV columns
                context = {
                    k: v for k, v in row.items()
                    if k not in [text_column, id_column] and v
                }

                # Process
                async with semaphore:
                    result = await self.analyze_one(text, source_id, context)

                # Update progress
                self.progress.processed += 1
                if result.success:
                    self.progress.succeeded += 1
                    tier = result.threat_tier
                    self.progress.tier_counts[tier] = (
                        self.progress.tier_counts.get(tier, 0) + 1
                    )
                else:
                    self.progress.failed += 1

                # Buffer for batch writing
                buffer.append(result.to_dict())

                # Checkpoint
                if len(buffer) >= checkpoint_every:
                    for row_dict in buffer:
                        writer.writerow(row_dict)
                    output_file.flush()
                    buffer.clear()

                # Progress logging
                if self.progress.processed % progress_every == 0:
                    self.progress.log_status()

                # Small delay to prevent overloading Ollama
                await asyncio.sleep(self.config.batch_delay)

        except KeyboardInterrupt:
            logger.warning("Processing interrupted! Saving checkpoint...")
        finally:
            # Write remaining buffer
            for row_dict in buffer:
                writer.writerow(row_dict)
            output_file.close()

        # Final stats
        self.progress.log_status()
        logger.info(f"Tier distribution: {self.progress.tier_counts}")
        logger.info(f"Output saved to: {output_path}")
        
        return self.progress

    # =========================================================================
    # Batch Analysis (in-memory)
    # =========================================================================

    async def process_texts(
        self,
        texts: List[str],
        source_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[AnalysisResult]:
        """
        Process a list of texts through the LLM pipeline.
        
        Args:
            texts: List of texts to analyze
            source_ids: Optional source identifiers
            progress_callback: Optional callback(processed, total)
            
        Returns:
            List of AnalysisResult objects
        """
        await self._ensure_provider()
        
        source_ids = source_ids or [str(i) for i in range(len(texts))]
        results: List[AnalysisResult] = []
        
        self.progress = BatchProgress(
            total=len(texts),
            start_time=time.monotonic(),
        )

        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def _process_one(text: str, sid: str) -> AnalysisResult:
            async with semaphore:
                result = await self.analyze_one(text, sid)
                await asyncio.sleep(self.config.batch_delay)
                return result

        # Process with controlled concurrency
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            batch_ids = source_ids[i : i + self.config.batch_size]
            
            batch_tasks = [
                _process_one(t, sid) for t, sid in zip(batch_texts, batch_ids)
            ]
            batch_results = await asyncio.gather(
                *batch_tasks, return_exceptions=True
            )
            
            for r in batch_results:
                if isinstance(r, Exception):
                    logger.warning(f"Batch processing error: {r}")
                    results.append(AnalysisResult(success=False, error=str(r)))
                else:
                    results.append(r)
            
            self.progress.processed += len(batch_results)
            self.progress.succeeded += sum(
                1 for r in batch_results if not isinstance(r, Exception) and r.success
            )
            
            if progress_callback:
                progress_callback(self.progress.processed, self.progress.total)

        return results

    # =========================================================================
    # Summary Statistics
    # =========================================================================

    @staticmethod
    def summarize_results(results: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results."""
        if not results:
            return {}

        successful = [r for r in results if r.success]
        
        # Tier distribution
        tier_dist = {}
        for r in successful:
            tier_dist[r.threat_tier] = tier_dist.get(r.threat_tier, 0) + 1

        # Risk statistics
        risks = [r.adjusted_risk for r in successful]
        
        # Role distribution
        role_dist = {}
        for r in successful:
            role_dist[r.role] = role_dist.get(r.role, 0) + 1

        # Language distribution
        lang_dist = {}
        for r in successful:
            lang_dist[r.language_primary] = lang_dist.get(r.language_primary, 0) + 1

        # Gating distribution
        gating_dist = {}
        for r in successful:
            gating_dist[r.gating_status] = gating_dist.get(r.gating_status, 0) + 1

        import numpy as np
        risk_arr = np.array(risks) if risks else np.array([0])

        return {
            "total_processed": len(results),
            "total_successful": len(successful),
            "total_failed": len(results) - len(successful),
            "success_rate": round(len(successful) / max(1, len(results)), 3),
            "risk_stats": {
                "mean": round(float(risk_arr.mean()), 2),
                "median": round(float(np.median(risk_arr)), 2),
                "std": round(float(risk_arr.std()), 2),
                "max": round(float(risk_arr.max()), 2),
                "p95": round(float(np.percentile(risk_arr, 95)), 2),
            },
            "tier_distribution": tier_dist,
            "role_distribution": role_dist,
            "language_distribution": lang_dist,
            "gating_distribution": gating_dist,
            "avg_processing_ms": round(
                sum(r.processing_ms for r in successful) / max(1, len(successful)), 1
            ),
        }

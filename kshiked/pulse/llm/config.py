"""
KShield Pulse LLM Configuration & Model Registry

Central configuration for Ollama-based LLM architecture.
Defines model profiles, task routing, and hardware-aware defaults.

Usage:
    from kshiked.pulse.llm.config import OllamaConfig, ModelProfile, TASK_ROUTING
    
    config = OllamaConfig()  # Auto-detects hardware
    model = config.get_model_for_task("threat_analysis")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Task Definitions
# =============================================================================

class AnalysisTask(str, Enum):
    """All LLM analysis tasks in the KShield pipeline."""
    THREAT_CLASSIFICATION = "threat_classification"
    CONTEXT_ANALYSIS = "context_analysis"
    INDICES_EXTRACTION = "indices_extraction"
    TIME_TO_ACTION = "time_to_action"
    RESILIENCE_ANALYSIS = "resilience_analysis"
    ROLE_CLASSIFICATION = "role_classification"
    NARRATIVE_ANALYSIS = "narrative_analysis"
    BATCH_CLASSIFICATION = "batch_classification"
    EMBEDDING = "embedding"
    SHENG_TRANSLATION = "sheng_translation"
    POLICY_IMPACT = "policy_impact"
    SUMMARY = "summary"


# =============================================================================
# Model Profiles
# =============================================================================

@dataclass
class ModelProfile:
    """
    Profile for a specific Ollama model.
    
    Attributes:
        name: Ollama model tag (e.g. "llama3.1:8b", "mistral:7b")
        context_window: Max tokens the model supports
        strengths: Tasks this model excels at
        temperature: Default temperature for this model
        num_ctx: Context window to request from Ollama
        num_gpu: GPU layers to offload (-1 = all)
        top_p: Nucleus sampling threshold
        repeat_penalty: Repetition penalty
        seed: Random seed for reproducibility
        vram_gb: Approximate VRAM needed (for auto-selection)
        speed_tier: 1=fastest, 3=slowest (for batch routing)
    """
    name: str
    context_window: int = 8192
    strengths: List[str] = field(default_factory=list)
    temperature: float = 0.1
    num_ctx: int = 4096
    num_gpu: int = -1
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    seed: int = 42
    vram_gb: float = 4.0
    speed_tier: int = 2  # 1=fast, 2=medium, 3=slow

    def to_ollama_options(self) -> Dict[str, Any]:
        """Convert to Ollama API options dict."""
        return {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
        }


# =============================================================================
# Pre-defined Model Profiles
# =============================================================================

# Primary analysis model — best balance of speed + quality
LLAMA3_8B = ModelProfile(
    name="llama3.1:8b",
    context_window=131072,
    strengths=["threat_classification", "context_analysis", "role_classification"],
    temperature=0.1,
    num_ctx=4096,
    vram_gb=4.7,
    speed_tier=1,
)

# Deep reasoning model — for complex multi-step analysis
LLAMA3_70B = ModelProfile(
    name="llama3.1:70b",
    context_window=131072,
    strengths=["narrative_analysis", "indices_extraction", "policy_impact"],
    temperature=0.05,
    num_ctx=8192,
    vram_gb=40.0,
    speed_tier=3,
)

# Fast classification model
MISTRAL_7B = ModelProfile(
    name="mistral:7b",
    context_window=32768,
    strengths=["batch_classification", "threat_classification"],
    temperature=0.1,
    num_ctx=4096,
    vram_gb=4.1,
    speed_tier=1,
)

# Multilingual model — Sheng/Swahili support (large)
QWEN_7B = ModelProfile(
    name="qwen2.5:7b",
    context_window=131072,
    strengths=["sheng_translation", "context_analysis", "embedding"],
    temperature=0.1,
    num_ctx=4096,
    vram_gb=4.7,
    speed_tier=1,
)

# Multilingual compact model — best JSON + multilingual for constrained VRAM
QWEN_3B = ModelProfile(
    name="qwen2.5:3b",
    context_window=131072,
    strengths=[
        "threat_classification", "context_analysis", "role_classification",
        "sheng_translation", "batch_classification", "narrative_analysis",
        "indices_extraction", "policy_impact",
    ],
    temperature=0.1,
    num_ctx=4096,
    vram_gb=1.9,
    speed_tier=1,
)

# Embedding model (for semantic similarity)
NOMIC_EMBED = ModelProfile(
    name="nomic-embed-text",
    context_window=8192,
    strengths=["embedding"],
    temperature=0.0,
    num_ctx=2048,
    vram_gb=0.3,
    speed_tier=1,
)

# All-MiniLM embedding (lighter)
MXBAI_EMBED = ModelProfile(
    name="mxbai-embed-large",
    context_window=512,
    strengths=["embedding"],
    temperature=0.0,
    num_ctx=512,
    vram_gb=0.7,
    speed_tier=1,
)

# Registry of all known models
MODEL_REGISTRY: Dict[str, ModelProfile] = {
    "llama3.1:8b": LLAMA3_8B,
    "llama3.1:70b": LLAMA3_70B,
    "mistral:7b": MISTRAL_7B,
    "qwen2.5:7b": QWEN_7B,
    "qwen2.5:3b": QWEN_3B,
    "nomic-embed-text": NOMIC_EMBED,
    "mxbai-embed-large": MXBAI_EMBED,
}


# =============================================================================
# Task → Model Routing
# =============================================================================

# Default task routing — uses qwen2.5:3b (best JSON + multilingual for ≤8GB VRAM)
DEFAULT_TASK_ROUTING: Dict[AnalysisTask, str] = {
    AnalysisTask.THREAT_CLASSIFICATION: "qwen2.5:3b",
    AnalysisTask.CONTEXT_ANALYSIS: "qwen2.5:3b",
    AnalysisTask.INDICES_EXTRACTION: "qwen2.5:3b",
    AnalysisTask.TIME_TO_ACTION: "qwen2.5:3b",
    AnalysisTask.RESILIENCE_ANALYSIS: "qwen2.5:3b",
    AnalysisTask.ROLE_CLASSIFICATION: "qwen2.5:3b",
    AnalysisTask.NARRATIVE_ANALYSIS: "qwen2.5:3b",
    AnalysisTask.BATCH_CLASSIFICATION: "qwen2.5:3b",
    AnalysisTask.EMBEDDING: "nomic-embed-text",
    AnalysisTask.SHENG_TRANSLATION: "qwen2.5:3b",
    AnalysisTask.POLICY_IMPACT: "qwen2.5:3b",
    AnalysisTask.SUMMARY: "qwen2.5:3b",
}

# Light routing for constrained hardware (< 6GB VRAM) — qwen2.5:3b fits in ~2GB
LIGHT_TASK_ROUTING: Dict[AnalysisTask, str] = {
    AnalysisTask.THREAT_CLASSIFICATION: "qwen2.5:3b",
    AnalysisTask.CONTEXT_ANALYSIS: "qwen2.5:3b",
    AnalysisTask.INDICES_EXTRACTION: "qwen2.5:3b",
    AnalysisTask.TIME_TO_ACTION: "qwen2.5:3b",
    AnalysisTask.RESILIENCE_ANALYSIS: "qwen2.5:3b",
    AnalysisTask.ROLE_CLASSIFICATION: "qwen2.5:3b",
    AnalysisTask.NARRATIVE_ANALYSIS: "qwen2.5:3b",
    AnalysisTask.BATCH_CLASSIFICATION: "qwen2.5:3b",
    AnalysisTask.EMBEDDING: "nomic-embed-text",
    AnalysisTask.SHENG_TRANSLATION: "qwen2.5:3b",
    AnalysisTask.POLICY_IMPACT: "qwen2.5:3b",
    AnalysisTask.SUMMARY: "qwen2.5:3b",
}


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class OllamaConfig:
    """
    Central Ollama LLM configuration.
    
    Supports:
    - Multiple models for different tasks (task routing)
    - Hardware-aware model selection
    - Retry and timeout settings
    - Batch processing parameters
    
    Usage:
        config = OllamaConfig()
        config = OllamaConfig(base_url="http://gpu-server:11434")
        config = OllamaConfig(default_model="mistral:7b")  # Override all tasks
    """
    # Connection
    base_url: str = "http://localhost:11434"
    
    # Default model (overrides task routing when set)
    default_model: Optional[str] = None
    
    # Task routing (which model handles which task)
    task_routing: Dict[AnalysisTask, str] = field(default_factory=lambda: dict(DEFAULT_TASK_ROUTING))
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    retry_backoff: float = 2.0  # exponential backoff multiplier
    
    # Timeouts (seconds)
    connect_timeout: float = 10.0
    read_timeout: float = 300.0  # LLM inference can be slow, esp. first load
    
    # Batch processing
    batch_size: int = 10  # Texts per batch call
    max_concurrent: int = 2  # Concurrent Ollama requests (limited by VRAM)
    batch_delay: float = 0.1  # Delay between batches (seconds)
    
    # Embedding settings
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768  # nomic-embed-text default
    
    # Kenya-specific
    enable_sheng_detection: bool = True
    enable_policy_context: bool = True
    kenyan_context_window: int = 4096
    
    # Monitoring
    log_prompts: bool = False  # Log full prompts (debug only)
    log_responses: bool = False  # Log full responses (debug only)
    track_latency: bool = True
    track_token_usage: bool = True
    
    def get_model_for_task(self, task: AnalysisTask) -> str:
        """Get the model assigned to a specific task."""
        if self.default_model:
            return self.default_model
        return self.task_routing.get(task, "qwen2.5:3b")
    
    def get_model_profile(self, task: AnalysisTask) -> ModelProfile:
        """Get the full model profile for a task."""
        model_name = self.get_model_for_task(task)
        if model_name in MODEL_REGISTRY:
            return MODEL_REGISTRY[model_name]
        # Unknown model — return sensible defaults
        return ModelProfile(name=model_name)
    
    def get_options_for_task(self, task: AnalysisTask) -> Dict[str, Any]:
        """Get Ollama options dict for a specific task."""
        profile = self.get_model_profile(task)
        return profile.to_ollama_options()
    
    @classmethod
    def for_hardware(cls, vram_gb: float = 8.0, **kwargs) -> "OllamaConfig":
        """
        Create config optimized for available hardware.
        
        Args:
            vram_gb: Available VRAM in GB
            **kwargs: Additional overrides
        """
        if vram_gb < 6:
            routing = dict(LIGHT_TASK_ROUTING)
            max_concurrent = 1
            embedding_model = "nomic-embed-text"
        elif vram_gb < 12:
            routing = dict(DEFAULT_TASK_ROUTING)
            max_concurrent = 1
            embedding_model = "nomic-embed-text"
        else:
            routing = dict(DEFAULT_TASK_ROUTING)
            max_concurrent = 2
            embedding_model = "nomic-embed-text"
        
        return cls(
            task_routing=routing,
            max_concurrent=max_concurrent,
            embedding_model=embedding_model,
            **kwargs,
        )
    
    @classmethod
    def single_model(cls, model: str = "qwen2.5:3b", **kwargs) -> "OllamaConfig":
        """
        Create config using a single model for everything.
        Simplest setup — good for getting started.
        """
        return cls(default_model=model, **kwargs)


# =============================================================================
# Latency Tracking
# =============================================================================

@dataclass
class InferenceMetrics:
    """Track per-call inference metrics."""
    task: AnalysisTask
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: str = ""
    
    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return (self.completion_tokens / self.latency_ms) * 1000


@dataclass 
class SessionMetrics:
    """Aggregate metrics across a session/batch run."""
    calls: List[InferenceMetrics] = field(default_factory=list)
    
    def record(self, metric: InferenceMetrics):
        self.calls.append(metric)
    
    @property
    def total_calls(self) -> int:
        return len(self.calls)
    
    @property
    def success_rate(self) -> float:
        if not self.calls:
            return 0.0
        return sum(1 for c in self.calls if c.success) / len(self.calls)
    
    @property
    def avg_latency_ms(self) -> float:
        successful = [c.latency_ms for c in self.calls if c.success]
        return sum(successful) / len(successful) if successful else 0.0
    
    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)
    
    def summary(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_tokens": self.total_tokens,
            "by_task": self._by_task(),
        }
    
    def _by_task(self) -> Dict[str, Dict[str, Any]]:
        from collections import defaultdict
        task_groups = defaultdict(list)
        for c in self.calls:
            task_groups[c.task.value].append(c)
        
        result = {}
        for task, calls in task_groups.items():
            successful = [c for c in calls if c.success]
            result[task] = {
                "count": len(calls),
                "success": len(successful),
                "avg_latency_ms": round(
                    sum(c.latency_ms for c in successful) / len(successful), 1
                ) if successful else 0.0,
            }
        return result

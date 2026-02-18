"""
KShield Pulse LLM Integration Layer

Provides a complete Ollama-based LLM architecture for threat intelligence:

Core:
    - OllamaProvider: Production Ollama client with retries, multi-model routing
    - OllamaConfig: Central configuration (models, tasks, hardware profiles)
    - KShieldAnalyzer: End-to-end analysis orchestrator

Analysis:
    - 14-Category Threat Taxonomy with Dual-Layer Risk (BaseRisk × CSM)
    - Kenya-specific prompts with Sheng/Swahili awareness
    - Policy impact analysis for Kenyan governance events
    - Semantic embeddings via Ollama /api/embed
    - Narrative clustering and anomaly detection

Operations:
    - BatchProcessor: Process 100K+ texts with checkpointing
    - ModelManager: Pull/verify/manage Ollama models
    - SessionMetrics: Track latency, tokens, success rates

Legacy:
    - GeminiProvider: Google Gemini implementation
    - FineTuningDataPreparer: Training data export

Usage:
    from kshiked.pulse.llm import KShieldAnalyzer, OllamaConfig

    config = OllamaConfig.single_model("llama3.1:8b")
    async with KShieldAnalyzer(config=config) as analyzer:
        report = await analyzer.analyze("Serikali wezi! Twende streets!")
        print(report.summary())
"""

# Base interfaces
from .base import (
    LLMProvider,
    ThreatClassification,
    RoleClassification,
    NarrativeAnalysis,
    ThreatTier,
    RoleType,
    NarrativeMaturity,
)

# Configuration
from .config import (
    OllamaConfig,
    AnalysisTask,
    ModelProfile,
    InferenceMetrics,
    SessionMetrics,
    MODEL_REGISTRY,
)

# V3 Signal models
from .signals import (
    KShieldSignal,
    ThreatSignal,
    ContextAnalysis,
    AdvancedIndices,
    ThreatCategory,
    ThreatTier as V3ThreatTier,
    EconomicGrievance,
    SocialGrievance,
    TimeToAction,
    ResilienceIndex,
    RoleType as V3RoleType,
    MonitoringTarget,
)

# Optional runtime modules (some environments intentionally omit heavy deps like aiohttp)
try:
    from .ollama import OllamaProvider
except Exception:  # pragma: no cover - environment dependent
    OllamaProvider = None

try:
    from .embeddings import OllamaEmbeddings
except Exception:  # pragma: no cover - environment dependent
    OllamaEmbeddings = None

try:
    from .batch_processor import BatchProcessor, ProcessingMode, AnalysisResult
except Exception:  # pragma: no cover - environment dependent
    BatchProcessor = None
    ProcessingMode = None
    AnalysisResult = None

try:
    from .analyzer import KShieldAnalyzer, AnalysisReport
except Exception:  # pragma: no cover - environment dependent
    KShieldAnalyzer = None
    AnalysisReport = None

try:
    from .models import ModelManager, SystemStatus
except Exception:  # pragma: no cover - environment dependent
    ModelManager = None
    SystemStatus = None

try:
    from .policy_extractor import PolicyExtractor, BillAnalysis, BillProvision
    from .policy_search import PolicySearchEngine, SearchResults, SearchResult
    from .policy_predictor import PolicyPredictor, ImpactPrediction, ProvisionImpact
    from .policy_chatbot import PolicyChatbot, ChatSession, ChatMessage
except Exception:  # pragma: no cover - environment dependent
    PolicyExtractor = None
    BillAnalysis = None
    BillProvision = None
    PolicySearchEngine = None
    SearchResults = None
    SearchResult = None
    PolicyPredictor = None
    ImpactPrediction = None
    ProvisionImpact = None
    PolicyChatbot = None
    ChatSession = None
    ChatMessage = None

try:
    from .gemini import GeminiProvider, GeminiConfig, create_gemini_provider
except Exception:  # pragma: no cover - environment dependent
    GeminiProvider = None
    GeminiConfig = None
    create_gemini_provider = None

try:
    from .fine_tuning import FineTuningDataPreparer, create_fine_tuning_preparer
except Exception:  # pragma: no cover - environment dependent
    FineTuningDataPreparer = None
    create_fine_tuning_preparer = None

__all__ = [
    # Core
    "LLMProvider",
    "ThreatClassification",
    "RoleClassification",
    "NarrativeAnalysis",
    "ThreatTier",
    "RoleType",
    "NarrativeMaturity",
    # Config
    "OllamaConfig",
    "AnalysisTask",
    "ModelProfile",
    "InferenceMetrics",
    "SessionMetrics",
    "MODEL_REGISTRY",
    # V3 Signals
    "KShieldSignal",
    "ThreatSignal",
    "ContextAnalysis",
    "AdvancedIndices",
    "ThreatCategory",
    "V3ThreatTier",
    "EconomicGrievance",
    "SocialGrievance",
    "TimeToAction",
    "ResilienceIndex",
    "V3RoleType",
    "MonitoringTarget",
    # Providers
    "OllamaProvider",
    "OllamaEmbeddings",
    "GeminiProvider",
    "GeminiConfig",
    "create_gemini_provider",
    # Processing
    "BatchProcessor",
    "ProcessingMode",
    "AnalysisResult",
    # Analyzer
    "KShieldAnalyzer",
    "AnalysisReport",
    # Model management
    "ModelManager",
    "SystemStatus",
    # Fine-tuning
    "FineTuningDataPreparer",
    "create_fine_tuning_preparer",
    # Policy chatbot
    "PolicyExtractor",
    "BillAnalysis",
    "BillProvision",
    "PolicySearchEngine",
    "SearchResults",
    "SearchResult",
    "PolicyPredictor",
    "ImpactPrediction",
    "ProvisionImpact",
    "PolicyChatbot",
    "ChatSession",
    "ChatMessage",
]

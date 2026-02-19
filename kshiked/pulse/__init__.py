# KShield Pulse Engine
"""
SIGINT layer for detecting social signals and mapping them to economic primitives.

The Pulse Engine provides:
- 15 intelligence signal detectors (distress, anger, institutional, identity, info warfare)
- 4 primitive data models (ScarcityVector, ActorStress, BondStrength, ShockPropagation)
- NLP pipeline (sentiment, emotion, NER, embeddings)
- Social media integration (Twitter, TikTok, Instagram)
- Time-weighted co-occurrence analysis
- Signal -> Primitive mapping with risk scoring
- Clean decoupling from simulation layer
"""

from .primitives import (
    PulseState,
    ScarcityVector,
    ActorStress,
    BondStrength,
    ShockPropagation,
    SignalCategory,
    ActorType,
    ResourceDomain,
    ScarcityUpdate,
    StressUpdate,
    BondUpdate,
)

from .mapper import (
    SignalMapper,
    SignalDetection,
    SignalID,
    SIGNAL_CATEGORIES,
)

# from .sensor import (
#     PulseSensor,
#     PulseSensorConfig,
#     AsyncPulseSensor,
#     SignalDetector,
#     KeywordDetector,
# )

# from .nlp import (
#     NLPPipeline,
#     SentimentAnalyzer,
#     EmotionDetector,
#     EntityRecognizer,
#     TextEmbedder,
#     TextPreprocessor,
#     SentimentResult,
#     EmotionResult,
#     Entity,
#     NLPResult,
# )

# from .detectors import (
#     NLPSignalDetector,
#     create_nlp_detectors,
# )

from .social import (
    Platform,
    SocialPost,
    StreamConfig,
    RateLimiter,
    SocialMediaClient,
    TwitterClient,
    TwitterConfig,
    TikTokClient,
    TikTokConfig,
    InstagramClient,
    InstagramConfig,
    SocialMediaManager,
    SocialPulseIngester,
)

from .cooccurrence import (
    ExponentialDecay,
    LinearDecay,
    StepDecay,
    RollingWindow,
    SignalEvent,
    SignalCorrelationMatrix,
    RiskScore,
    RiskScorer,
    AnomalyAlert,
    AnomalyDetector,
)

from .bridge import (
    ShockType,
    ShockEvent,
    ShockMapping,
    ShockMagnitudeCalculator,
    ShockScheduler,
    SchedulerConfig,
    SimulationBridge,
    create_kshield_bridge,
    create_backtest_handler,
)

# Dashboard is optional (requires streamlit)
try:
    from .dashboard import render_pulse_dashboard, run_dashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

# Threat Indices (new)
from .indices import (
    ThreatIndexReport,
    PolarizationIndex,
    LegitimacyErosionIndex,
    MobilizationReadinessScore,
    EliteCohesionIndex,
    InformationWarfareIndex,
    SecurityFrictionIndex,
    EconomicCascadeRisk,
    EthnicTensionMatrix,
    KenyanEthnicGroup,
    compute_threat_report,
)

__all__ = [
    # Primitives
    "PulseState", "ScarcityVector", "ActorStress", "BondStrength",
    "ShockPropagation", "SignalCategory", "ActorType", "ResourceDomain",
    "ScarcityUpdate", "StressUpdate", "BondUpdate",
    # Mapper
    "SignalMapper", "SignalDetection", "SignalID", "SIGNAL_CATEGORIES",
    # Sensor
    "PulseSensor", "PulseSensorConfig", "AsyncPulseSensor",
    "SignalDetector", "KeywordDetector",
    # NLP
    "NLPPipeline", "SentimentAnalyzer", "EmotionDetector", "EntityRecognizer",
    "TextEmbedder", "TextPreprocessor", "SentimentResult", "EmotionResult",
    "Entity", "NLPResult", "NLPSignalDetector", "create_nlp_detectors",
    # Social Media
    "Platform", "SocialPost", "StreamConfig", "RateLimiter",
    "SocialMediaClient", "TwitterClient", "TwitterConfig",
    "TikTokClient", "TikTokConfig", "InstagramClient", "InstagramConfig",
    "SocialMediaManager", "SocialPulseIngester",
    # Co-occurrence
    "ExponentialDecay", "LinearDecay", "StepDecay", "RollingWindow",
    "SignalEvent", "SignalCorrelationMatrix", "RiskScore", "RiskScorer",
    "AnomalyAlert", "AnomalyDetector",
    # Bridge
    "ShockType", "ShockEvent", "ShockMapping", "ShockMagnitudeCalculator",
    "ShockScheduler", "SchedulerConfig", "SimulationBridge",
    "create_kshield_bridge", "create_backtest_handler",
    # Threat Indices (new)
    "ThreatIndexReport", "PolarizationIndex", "LegitimacyErosionIndex",
    "MobilizationReadinessScore", "EliteCohesionIndex", "InformationWarfareIndex",
    "SecurityFrictionIndex", "EconomicCascadeRisk", "EthnicTensionMatrix",
    "KenyanEthnicGroup", "compute_threat_report",
]


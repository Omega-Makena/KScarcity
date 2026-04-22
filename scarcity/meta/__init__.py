"""
Meta-learning interfaces for SCARCITY.
"""

from .encoder import ContextEncoder, ContextEncoderConfig
from .memory import EpisodicMemory, EpisodicMemoryConfig, EpisodicEntry, RetrievalResult
from .adaptation import AdaptationEngine, AdaptationConfig, AdaptationResult
from .domain_meta import DomainMetaLearner, DomainMetaConfig, DomainMetaUpdate
from .cross_meta import (
    CrossDomainMetaAggregator, CrossMetaConfig,
    CrossDomainMetaLearner, CrossDomainMetaLearnerConfig,
)
from .domain_server_meta import DomainServerMeta, DomainServerMetaConfig
from .optimizer import OnlineReptileOptimizer, MetaOptimizerConfig
from .scheduler import MetaScheduler, MetaSchedulerConfig
from .validator import MetaPacketValidator, MetaValidatorConfig
from .storage import MetaStorageManager, MetaStorageConfig
from .telemetry_hooks import (
    build_meta_metrics_snapshot,
    publish_meta_metrics,
)
from .meta_learning import MetaLearningAgent, MetaLearningConfig

__version__ = "1.1.0"
__author__ = "Omega Makena"

__all__ = [
    "ContextEncoder",
    "ContextEncoderConfig",
    "EpisodicMemory",
    "EpisodicMemoryConfig",
    "EpisodicEntry",
    "RetrievalResult",
    "AdaptationEngine",
    "AdaptationConfig",
    "AdaptationResult",
    "DomainMetaLearner",
    "DomainMetaConfig",
    "DomainMetaUpdate",
    "CrossDomainMetaAggregator",
    "CrossMetaConfig",
    "CrossDomainMetaLearner",
    "CrossDomainMetaLearnerConfig",
    "DomainServerMeta",
    "DomainServerMetaConfig",
    "OnlineReptileOptimizer",
    "MetaOptimizerConfig",
    "MetaScheduler",
    "MetaSchedulerConfig",
    "MetaPacketValidator",
    "MetaValidatorConfig",
    "MetaStorageManager",
    "MetaStorageConfig",
    "build_meta_metrics_snapshot",
    "publish_meta_metrics",
    "MetaLearningAgent",
    "MetaLearningConfig",
]


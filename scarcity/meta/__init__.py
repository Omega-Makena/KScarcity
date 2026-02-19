"""
Meta-learning interfaces for SCARCITY.
"""

from .domain_meta import DomainMetaLearner, DomainMetaConfig, DomainMetaUpdate
from .cross_meta import CrossDomainMetaAggregator, CrossMetaConfig
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
    "DomainMetaLearner",
    "DomainMetaConfig",
    "DomainMetaUpdate",
    "CrossDomainMetaAggregator",
    "CrossMetaConfig",
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


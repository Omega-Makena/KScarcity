"""
Stream Layer for SCARCITY.

This module provides data ingestion, transformation, caching, replay, 
and federation capabilities for online learning systems.
"""

from .source import StreamSource
from .window import WindowBuilder
from .schema import SchemaManager
from .sharder import StreamSharder
from .cache import CacheManager
from .replay import ReplayManager
from .federator import StreamFederator

__all__ = [
    'StreamSource',
    'WindowBuilder',
    'SchemaManager',
    'StreamSharder',
    'CacheManager',
    'ReplayManager',
    'StreamFederator',
]


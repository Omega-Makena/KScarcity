"""
Dynamic Resource Governor (DRG) implementation for SCARCITY.
"""

from .drg_core import DynamicResourceGovernor, DRGConfig
from .registry import SubsystemHandle

__all__ = [
    "DynamicResourceGovernor",
    "DRGConfig",
    "SubsystemHandle",
]


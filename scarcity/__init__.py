"""
SCARCITY — online causal discovery, stock-flow simulation, and
privacy-preserving federation.

A general-purpose toolkit for learning causal structure from streaming data,
running stock-flow-consistent (SFC) simulations, and aggregating knowledge
across institutions without sharing raw data.

Public entry points are lazily loaded, so ``import scarcity`` stays light and
does not require the optional extras (``stream``, ``dashboard``, ``gpu``,
``federation``, ...). The heavy dependencies are only imported when you touch
the relevant attribute or subpackage.

Examples
--------
>>> import scarcity
>>> scarcity.__version__                       # doctest: +SKIP
'1.0.0'
>>> engine = scarcity.OnlineDiscoveryEngine()  # doctest: +SKIP
>>> from scarcity.simulation import SFCEconomy  # subpackages import directly
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# --- version (single source of truth: installed package metadata) -----------
try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("scarcity")
    except PackageNotFoundError:  # running from a source checkout, not installed
        __version__ = "1.0.0"
except ImportError:  # pragma: no cover - Python < 3.8 has no importlib.metadata
    __version__ = "1.0.0"

__author__ = "Omega Makena"

# --- lazy public API --------------------------------------------------------
# Map each public name to the subpackage that provides it. Access is resolved
# on first use via PEP 562 ``__getattr__`` so optional dependencies stay opt-in.
_LAZY_ATTRS: dict[str, str] = {
    # engine — online relationship/causal discovery
    "OnlineDiscoveryEngine": "scarcity.engine",
    "Engine": "scarcity.engine",
    "Hypothesis": "scarcity.engine",
    "RelationshipType": "scarcity.engine",
    # causal — offline causal-inference pipeline (DoWhy/EconML style)
    "run_causal": "scarcity.causal",
    # simulation — stock-flow-consistent economies
    "SimulationEngine": "scarcity.simulation",
    "SFCEconomy": "scarcity.simulation",
    "MultiSectorSFCEngine": "scarcity.simulation",
    "LearnedSFCEconomy": "scarcity.simulation",
    # federation — privacy-preserving knowledge aggregation
    "HierarchicalFederation": "scarcity.federation",
    # meta — cross-domain meta-learning
    "MetaLearningAgent": "scarcity.meta",
    # governor — dynamic resource governance
    "DynamicResourceGovernor": "scarcity.governor",
    # runtime — event bus / telemetry
    "EventBus": "scarcity.runtime",
    "get_bus": "scarcity.runtime",
    # stream — online ingestion / replay
    "StreamSource": "scarcity.stream",
    # fmi — federation–meta interface service
    "FMIService": "scarcity.fmi",
    # synthetic — synthetic data generation
    "SyntheticPipeline": "scarcity.synthetic",
    # analytics — policy terrain
    "TerrainGenerator": "scarcity.analytics",
}

# Subpackages that should be reachable as ``scarcity.<name>`` without an
# explicit ``import scarcity.<name>`` first.
_SUBPACKAGES: tuple[str, ...] = (
    "engine",
    "causal",
    "simulation",
    "federation",
    "meta",
    "fmi",
    "stream",
    "synthetic",
    "governor",
    "runtime",
    "analytics",
)

if TYPE_CHECKING:  # let type checkers / IDEs see the real symbols
    from scarcity.analytics import TerrainGenerator
    from scarcity.causal import run_causal
    from scarcity.engine import (
        Engine,
        Hypothesis,
        OnlineDiscoveryEngine,
        RelationshipType,
    )
    from scarcity.federation import HierarchicalFederation
    from scarcity.fmi import FMIService
    from scarcity.governor import DynamicResourceGovernor
    from scarcity.meta import MetaLearningAgent
    from scarcity.runtime import EventBus, get_bus
    from scarcity.simulation import (
        LearnedSFCEconomy,
        MultiSectorSFCEngine,
        SFCEconomy,
        SimulationEngine,
    )
    from scarcity.stream import StreamSource
    from scarcity.synthetic import SyntheticPipeline


def __getattr__(name: str) -> Any:  # PEP 562 module-level lazy attribute access
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        return getattr(module, name)
    if name in _SUBPACKAGES:
        return importlib.import_module(f"scarcity.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ATTRS, *_SUBPACKAGES})


__all__ = [
    "__version__",
    *_SUBPACKAGES,
    *_LAZY_ATTRS,
]

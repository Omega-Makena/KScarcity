"""
Regression tests for OnlineDiscoveryEngine backend resolution.

Covers the fix where the default constructor used to hard-require torch:
- vectorized=None (default) auto-detects torch and degrades gracefully.
- vectorized=False forces the pure-Python loop (core install, no torch).
- vectorized=True without torch raises a clear, actionable ImportError.
"""

import types
import sys

import pytest

from scarcity.engine import engine_v2
from scarcity.engine.engine_v2 import OnlineDiscoveryEngine


ROWS = [
    {"price": 1.0, "demand": 9.8},
    {"price": 1.1, "demand": 9.2},
    {"price": 1.2, "demand": 8.9},
]


def test_vectorized_false_uses_python_backend():
    engine = OnlineDiscoveryEngine(vectorized=False, small_dataset_mode=True)
    assert engine.vectorized is False
    assert engine._vec_engine is None
    for row in ROWS:
        engine.process_row(row)  # pure-Python path must stream without torch


def test_auto_falls_back_when_torch_absent(monkeypatch):
    """Default (vectorized=None) must NOT crash when torch is unavailable."""
    monkeypatch.setattr(engine_v2, "_torch_available", lambda: False)
    engine = OnlineDiscoveryEngine(small_dataset_mode=True)  # default
    assert engine.vectorized is False
    assert engine._vec_engine is None
    for row in ROWS:
        engine.process_row(row)


def test_explicit_true_without_torch_raises_with_hint(monkeypatch):
    """vectorized=True must fail loudly with an install hint, not silently."""
    monkeypatch.setattr(engine_v2, "_torch_available", lambda: False)
    # Force the backend import to fail even though torch may be installed.
    monkeypatch.setitem(
        sys.modules,
        "scarcity.engine.gpu_engine",
        types.ModuleType("scarcity.engine.gpu_engine"),
    )
    with pytest.raises(ImportError, match=r"scarcity\[gpu\]"):
        OnlineDiscoveryEngine(vectorized=True)


@pytest.mark.skipif(
    not engine_v2._torch_available(), reason="torch not installed"
)
def test_auto_selects_vectorized_when_torch_present():
    engine = OnlineDiscoveryEngine(small_dataset_mode=True)  # default
    assert engine.vectorized is True
    assert engine._vec_engine is not None

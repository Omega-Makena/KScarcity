"""
Scarcity Test Configuration

Fixtures and common test utilities for all test modules.
"""

import pytest
import numpy as np
from typing import Dict, Any


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_row():
    """Sample data row for testing."""
    return {
        'X': 1.0,
        'Y': 2.0,
        'Z': 3.0,
        'year': 2020
    }


@pytest.fixture
def time_series_data(rng):
    """Generate simple time series data."""
    n = 100
    X = np.cumsum(rng.standard_normal(n))
    Y = 0.8 * X + 0.2 * rng.standard_normal(n)
    return {'X': X, 'Y': Y}

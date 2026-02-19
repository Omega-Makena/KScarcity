"""
Test: Federation Layer

Validates Byzantine-robust aggregation methods work correctly.
"""

import pytest
import numpy as np
from scarcity.federation.aggregator import (
    FederatedAggregator,
    AggregationConfig,
    AggregationMethod,
)


class TestFederatedAggregatorTrimmedMean:
    def test_rejects_outliers(self):
        """Trimmed mean should ignore outlier updates."""
        config = AggregationConfig(
            method=AggregationMethod.TRIMMED_MEAN,
            trim_alpha=0.2
        )
        aggregator = FederatedAggregator(config)
        
        # 4 honest clients close to [1, 1, 1]
        honest_updates = [
            [1.0, 1.0, 1.0],
            [1.1, 0.9, 1.0],
            [0.9, 1.1, 1.0],
            [1.0, 1.0, 1.1],
        ]
        
        # 1 malicious client with poison attack
        malicious = [100.0, 100.0, 100.0]
        
        all_updates = honest_updates + [malicious]
        
        result, meta = aggregator.aggregate(all_updates)
        
        # Result should be close to [1, 1, 1], not [21, 21, 21]
        assert np.allclose(result, [1.0, 1.0, 1.0], atol=0.3), f"Result: {result}"
        print(f"Trimmed mean result: {result}, meta: {meta}")


class TestFederatedAggregatorKrum:
    def test_krum_selects_honest(self):
        """Krum should select the honest client closest to majority."""
        config = AggregationConfig(method=AggregationMethod.KRUM)
        aggregator = FederatedAggregator(config)
        
        # Honest clients
        updates = [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [100.0, 100.0],  # Outlier
        ]
        
        result, meta = aggregator.aggregate(updates)
        
        # Krum should select one honest client
        assert np.allclose(result, [1.0, 1.0], atol=0.2)
        print(f"Krum result: {result}, selected: {meta.get('selected')}")


class TestFederatedAggregatorBulyan:
    def test_bulyan_robust(self):
        """Bulyan combines Krum selection with trimmed mean."""
        config = AggregationConfig(
            method=AggregationMethod.BULYAN,
            multi_krum_m=3,
            trim_alpha=0.1
        )
        aggregator = FederatedAggregator(config)
        
        updates = [
            [1.0, 1.0, 1.0],
            [1.1, 0.9, 1.0],
            [0.9, 1.1, 1.0],
            [1.0, 1.0, 0.9],
            [1000.0, 1000.0, 1000.0],  # Poison
        ]
        
        result, meta = aggregator.aggregate(updates)
        
        assert np.allclose(result, [1.0, 1.0, 1.0], atol=0.2)
        print(f"Bulyan result: {result}")


class TestDetectOutliers:
    def test_identifies_outliers(self):
        """detect_outliers should flag divergent updates."""
        updates = [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [50.0, 50.0],  # Outlier
        ]
        
        reference = [1.0, 1.0]
        
        outliers = FederatedAggregator.detect_outliers(updates, reference, z_thresh=1.5)
        
        assert 3 in outliers, f"Outlier 3 not detected: {outliers}"
        print(f"Detected outliers: {outliers}")


class TestFederationVarianceReduction:
    def test_aggregation_reduces_variance(self):
        """Aggregation should produce lower variance than individual clients."""
        config = AggregationConfig(method=AggregationMethod.TRIMMED_MEAN)
        aggregator = FederatedAggregator(config)
        
        # Simulate 5 clients with noisy estimates
        np.random.seed(42)
        true_value = np.array([10.0, 20.0, 30.0])
        
        client_updates = [
            true_value + np.random.randn(3) * 2.0
            for _ in range(5)
        ]
        
        result, meta = aggregator.aggregate(client_updates)
        
        # Aggregated should be closer to true value than average individual error
        agg_error = np.linalg.norm(result - true_value)
        individual_errors = [np.linalg.norm(u - true_value) for u in client_updates]
        avg_individual_error = np.mean(individual_errors)
        
        assert agg_error < avg_individual_error, \
            f"Aggregation error {agg_error} not less than individual {avg_individual_error}"
        print(f"Aggregated error: {agg_error:.3f}, Avg individual: {avg_individual_error:.3f}")

import numpy as np

from scarcity.federation.aggregator import AggregationConfig, AggregationMethod, FederatedAggregator


def test_federated_aggregation_weighted():
    agg = FederatedAggregator(AggregationConfig(method=AggregationMethod.WEIGHTED))
    updates = [{"vector": [1.0, 1.0], "weight": 1}, {"vector": [3.0, 3.0], "weight": 3}]
    result, _ = agg.aggregate(updates)
    assert np.allclose(result, [2.5, 2.5])


def test_federated_aggregation_adaptive():
    agg = FederatedAggregator(AggregationConfig(method=AggregationMethod.ADAPTIVE))
    updates = [{"vector": [1.0, 1.0], "loss": 1.0}, {"vector": [3.0, 3.0], "loss": 3.0}]
    result, _ = agg.aggregate(updates)
    assert np.allclose(result, [1.5, 1.5])

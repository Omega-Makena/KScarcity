import numpy as np
from unittest import mock

from scarcity.fmi.aggregator import AggregationConfig, FMIAggregator
from scarcity.fmi.contracts import MetaSignalPack, PacketType


def test_fmi_aggregation_applies_dp_noise():
    cfg = AggregationConfig(metrics_aggregation="mean", dp_noise_sigma=0.1)
    agg = FMIAggregator(cfg)
    pkt = MetaSignalPack(
        type=PacketType.MSP,
        schema_hash="x",
        rev=3,
        domain_id="d",
        profile_class="p",
        metrics={"latency_ms": 1.0},
        controller={},
        evaluator={},
        operators={},
    )
    with mock.patch("numpy.random.normal", return_value=np.array(0.5)):
        result = agg.aggregate("cohort", [pkt])
    value = result.prior_update.prior["metrics"]["latency_ms"]
    assert value == 1.5

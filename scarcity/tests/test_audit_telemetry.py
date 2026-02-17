import asyncio

from scarcity.runtime.telemetry import Telemetry


def test_telemetry_snapshot_has_aliases():
    telemetry = Telemetry()
    telemetry.record_latency(10.0)
    snapshot = asyncio.run(telemetry._collect_snapshot())
    assert "latency_ms" in snapshot and "fps" in snapshot

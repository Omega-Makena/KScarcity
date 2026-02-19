import asyncio

from scarcity.fmi.emitter import FMIEmitter, EmitterConfig
from scarcity.fmi.contracts import MetaPriorUpdate
from scarcity.runtime.bus import EventBus


def test_fmi_emitter_bridges_meta_prior():
    bus = EventBus()
    seen = []

    async def handler(topic, data):
        seen.append(topic)

    bus.subscribe("meta_prior_update", handler)
    emitter = FMIEmitter(EmitterConfig(), bus=bus)
    update = MetaPriorUpdate(rev=1, prior={"controller": {}}, contexts=[], confidence=0.1, cohorts=["c"])

    async def run():
        await emitter.emit_prior_update(update, window=1)
        await bus.wait_for_idle()

    asyncio.run(run())
    assert "meta_prior_update" in seen

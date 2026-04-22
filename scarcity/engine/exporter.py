"""
Exporter — Insight and path-pack emission to EventBus.

Emits an insight payload every window and batched path packs periodically.
Previously had TODO stubs instead of actual EventBus publishing — now fixed.
"""

import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class Exporter:
    """
    Outbound gateway for the inference engine.

    Publishes two event types:
    - ``engine.insight``   — emitted every window with the accepted edges.
    - ``inference.path_pack`` — batched pack emitted every export_interval windows.

    The EventBus is resolved lazily (via get_bus()) the first time emit_insights()
    is called so that the Exporter can be constructed before the runtime is up.
    """

    def __init__(self):
        self.export_count = 0
        self.last_pack_time = 0.0
        self._bus = None
        logger.info("Exporter initialized")

    def _get_bus(self):
        """Lazy bus resolution to avoid circular imports at construction time."""
        if self._bus is None:
            try:
                from scarcity.runtime import get_bus
                self._bus = get_bus()
            except Exception as exc:
                logger.warning(f"Exporter: cannot resolve EventBus — {exc}")
        return self._bus

    def emit_insights(self, accepted_edges: List[Dict[str, Any]],
                      resource_profile: Dict[str, Any]) -> None:
        """
        Broadcast accepted edges every window and emit path packs periodically.

        Args:
            accepted_edges: Edges accepted in the current inference cycle.
            resource_profile: Active resource config (reads export_interval).
        """
        export_interval = int(resource_profile.get('export_interval', 10))
        now = time.time()
        bus = self._get_bus()

        # Per-window insight payload
        insight = {
            'edges': accepted_edges,
            'count': len(accepted_edges),
            'timestamp': now,
            'export_index': self.export_count,
        }
        if bus is not None:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(bus.publish("engine.insight", insight))
                else:
                    loop.run_until_complete(bus.publish("engine.insight", insight))
            except Exception as exc:
                logger.debug(f"Exporter insight publish failed: {exc}")
        else:
            logger.debug(f"Exporter insight (no bus): {len(accepted_edges)} edges")

        # Batched path pack
        if self.export_count > 0 and self.export_count % export_interval == 0 \
                and accepted_edges:
            self._emit_path_pack(accepted_edges, bus)
            self.last_pack_time = now

        self.export_count += 1

    def _emit_path_pack(self, edges: List[Dict[str, Any]], bus=None) -> None:
        """Publish a batched PathPack to inference.path_pack topic."""
        pack = {
            'edges': edges,
            'count': len(edges),
            'timestamp': time.time(),
            'pack_index': self.export_count,
        }
        if bus is not None:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(bus.publish("inference.path_pack", pack))
                else:
                    loop.run_until_complete(bus.publish("inference.path_pack", pack))
            except Exception as exc:
                logger.debug(f"Exporter path_pack publish failed: {exc}")

        logger.debug(f"Path pack emitted: {len(edges)} edges")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'export_count': self.export_count,
            'last_pack_time': self.last_pack_time,
        }

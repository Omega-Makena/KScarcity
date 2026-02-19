"""
Exporter — Insight and path pack emission.

Emits insights every window and path packs periodically.
"""

import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class Exporter:
    """
    Handles the broadcast of inference insights.

    The Exporter acts as the outbound gateway for the inference engine. It organizes
    discovered relationships into standard formats and emits them to:
    1. The Event Bus (for real-time dashboard updates and other components).
    2. Batched "Path Packs" (for efficient storage or bulk analysis).
    
    It manages buffering and periodic emission based on configured intervals.
    """
    
    def __init__(self):
        """
        Initializes the insight exporter.
        
        Sets up internal counters and timers for tracking export intervals.
        """
        self.export_count = 0
        self.last_pack_time = 0.0
        
        logger.info("Exporter initialized")
    
    def emit_insights(self, accepted_edges: List[Dict[str, Any]], resource_profile: Dict[str, Any]) -> None:
        """
        Processes and broadcasts a batch of accepted causal edges.

        This method is called every window (or whenever edges are accepted). It
        constructs an immediate "insight" payload for real-time consumers and
        accumulates edges for batched "path pack" emission.

        Args:
            accepted_edges: A list of dictionaries representing the edges accepted
                in the current cycle.
            resource_profile: The active resource profile configuration, used to
                determine the batch export interval.
        """
        export_interval = resource_profile.get('export_interval', 10)
        current_time = time.time()
        
        # Emit insight every window
        insight = {
            'edges': [{'accepted': True} for _ in accepted_edges],
            'count': len(accepted_edges),
            'timestamp': current_time
        }
        
        # TODO: Publish to bus
        
        # Check if it's time for a path pack
        if self.export_count % export_interval == 0 and len(accepted_edges) > 0:
            self._emit_path_pack(accepted_edges)
            self.last_pack_time = current_time
        
        self.export_count += 1
    
    def _emit_path_pack(self, edges: List[Dict[str, Any]]) -> None:
        """
        Emits a batched 'Path Pack' containing multiple edges.

        Path Packs are designed for bulk consumption, such as saving to disk or
        sending to a high-latency storage system.

        Args:
            edges: The list of edge dictionaries to include in the pack.
        """
        pack = {
            'edges': edges,
            'count': len(edges),
            'timestamp': time.time()
        }
        
        # TODO: Publish to bus
        logger.debug(f"Path pack emitted: {len(edges)} edges")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics for the exporter subsystem.

        Returns:
            A dictionary containing the total number of export cycles performed
            and the timestamp of the last batched path pack emission.
        """
        return {
            'export_count': self.export_count,
            'last_pack_time': self.last_pack_time
        }


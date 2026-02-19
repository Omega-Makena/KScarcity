"""
Network Analysis Module for KShield Pulse

Provides:
- Actor role detection (Mobilizer, Broker, Ideologue, etc.)
- Community detection (clusters of coordinated actors)
- Coordination pattern detection
- Information flow analysis
- Kenya-focused network visualization

Uses NetworkX for graph analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict
import logging

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from .db.models import RoleType

logger = logging.getLogger("kshield.pulse.network")


# =============================================================================
# Actor Roles (from KShield Taxonomy)
# =============================================================================

class ActorRole(str, Enum):
    """
    Actor roles in threat networks.
    
    Based on KShield's role taxonomy for identifying key players.
    """
    IDEOLOGUE = "ideologue"       # Produces justification narratives
    MOBILIZER = "mobilizer"       # Calls for action, coordinates
    AMPLIFIER = "amplifier"       # High-volume resharing
    BROKER = "broker"             # Connects communities
    LEGITIMIZER = "legitimizer"   # Adds authority cues
    GATEKEEPER = "gatekeeper"     # Controls channels
    INFLUENCER = "influencer"     # High follower count
    BOT = "bot"                   # Automated account
    UNKNOWN = "unknown"


@dataclass
class ActorProfile:
    """
    Profile of a social media actor.
    
    Tracks behavior patterns for role classification.
    """
    actor_id: str
    platform: str
    username: Optional[str] = None
    
    # Activity metrics
    post_count: int = 0
    repost_count: int = 0
    reply_count: int = 0
    mention_count: int = 0
    
    # Engagement
    total_likes_received: int = 0
    total_shares_received: int = 0
    avg_engagement_rate: float = 0.0
    
    # Network position
    in_degree: int = 0   # Others mentioning/replying to this actor
    out_degree: int = 0  # This actor mentioning/replying to others
    betweenness: float = 0.0  # Bridge between communities
    
    # Content patterns
    threat_tier_counts: Dict[str, int] = field(default_factory=dict)
    mobilization_posts: int = 0
    original_content_ratio: float = 0.0
    
    # Timing
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    posting_frequency: float = 0.0  # Posts per day
    
    # Classification
    role: ActorRole = ActorRole.UNKNOWN
    role_confidence: float = 0.0
    suspicion_score: float = 0.0
    
    def classify_role(self) -> Tuple[ActorRole, float]:
        """
        Classify actor role based on behavior patterns.
        
        Returns:
            Tuple of (role, confidence)
        """
        scores = {}
        
        # MOBILIZER: High mobilization posts, calls to action
        mobilizer_score = min(1.0, self.mobilization_posts / 10) * 0.6
        if self.out_degree > 20:
            mobilizer_score += 0.2
        if self.original_content_ratio > 0.7:
            mobilizer_score += 0.2
        scores[ActorRole.MOBILIZER] = mobilizer_score
        
        # AMPLIFIER: High repost ratio, low original content
        amplifier_score = 0.0
        if self.post_count > 0:
            repost_ratio = self.repost_count / self.post_count
            amplifier_score = repost_ratio * 0.7
            if repost_ratio > 0.8:
                amplifier_score += 0.3
        scores[ActorRole.AMPLIFIER] = amplifier_score
        
        # BROKER: High betweenness centrality
        broker_score = min(1.0, self.betweenness * 10) * 0.8
        if self.in_degree > 10 and self.out_degree > 10:
            broker_score += 0.2
        scores[ActorRole.BROKER] = broker_score
        
        # IDEOLOGUE: High original content, threat-tier posts
        ideologue_score = 0.0
        high_tier = self.threat_tier_counts.get("tier_1", 0) + self.threat_tier_counts.get("tier_2", 0)
        if high_tier > 3:
            ideologue_score = 0.6
        if self.original_content_ratio > 0.8:
            ideologue_score += 0.2
        scores[ActorRole.IDEOLOGUE] = ideologue_score
        
        # INFLUENCER: High engagement
        influencer_score = 0.0
        if self.total_likes_received > 1000:
            influencer_score = 0.5
        if self.total_shares_received > 500:
            influencer_score += 0.3
        if self.in_degree > 50:
            influencer_score += 0.2
        scores[ActorRole.INFLUENCER] = min(1.0, influencer_score)
        
        # BOT: Suspicious patterns
        bot_score = 0.0
        if self.posting_frequency > 50:  # >50 posts/day
            bot_score = 0.7
        if self.original_content_ratio < 0.1 and self.post_count > 100:
            bot_score += 0.3
        scores[ActorRole.BOT] = min(1.0, bot_score)
        
        # Get highest scoring role
        best_role = max(scores, key=scores.get)
        confidence = scores[best_role]
        
        if confidence < 0.3:
            best_role = ActorRole.UNKNOWN
            confidence = 0.0
        
        self.role = best_role
        self.role_confidence = confidence
        
        return best_role, confidence


# =============================================================================
# Network Graph
# =============================================================================

@dataclass
class NetworkEdge:
    """Edge in the actor network."""
    source_id: str
    target_id: str
    edge_type: str  # reply, retweet, mention, quote
    weight: float = 1.0
    timestamp: Optional[datetime] = None


class ActorNetwork:
    """
    Network graph of social media actors.
    
    Provides:
    - Community detection
    - Centrality analysis
    - Coordination detection
    - Role classification
    """
    
    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("networkx required: pip install networkx")
        
        self.graph = nx.DiGraph()
        self.actors: Dict[str, ActorProfile] = {}
        self.communities: List[Set[str]] = []
        
    def add_actor(self, profile: ActorProfile) -> None:
        """Add or update an actor."""
        self.actors[profile.actor_id] = profile
        self.graph.add_node(
            profile.actor_id,
            username=profile.username,
            platform=profile.platform,
            role=profile.role.value,
        )
    
    def add_edge(self, edge: NetworkEdge) -> None:
        """Add an edge between actors."""
        if edge.source_id not in self.graph:
            self.graph.add_node(edge.source_id)
        if edge.target_id not in self.graph:
            self.graph.add_node(edge.target_id)
        
        # Accumulate edge weight
        if self.graph.has_edge(edge.source_id, edge.target_id):
            self.graph[edge.source_id][edge.target_id]["weight"] += edge.weight
        else:
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                weight=edge.weight,
                edge_type=edge.edge_type,
            )
    
    def compute_centrality(self) -> Dict[str, Dict[str, float]]:
        """
        Compute centrality metrics for all actors.
        
        Returns:
            Dict mapping actor_id -> {metric: value}
        """
        if len(self.graph) == 0:
            return {}
        
        metrics = {}
        
        # Degree centrality
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        # Betweenness centrality (expensive, sample for large graphs)
        try:
            if len(self.graph) > 1000:
                betweenness = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
            else:
                betweenness = nx.betweenness_centrality(self.graph)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence) as e:
            logger.warning(f"Betweenness centrality failed: {e}")
            betweenness = {n: 0.0 for n in self.graph.nodes()}
        
        # PageRank
        try:
            pagerank = nx.pagerank(self.graph, weight="weight")
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence) as e:
            logger.warning(f"PageRank computation failed: {e}")
            pagerank = {n: 0.0 for n in self.graph.nodes()}
        
        for node in self.graph.nodes():
            metrics[node] = {
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "betweenness": betweenness.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
            }
            
            # Update actor profile if exists
            if node in self.actors:
                self.actors[node].in_degree = in_degree.get(node, 0)
                self.actors[node].out_degree = out_degree.get(node, 0)
                self.actors[node].betweenness = betweenness.get(node, 0.0)
        
        return metrics
    
    def detect_communities(self, resolution: float = 1.0) -> List[Set[str]]:
        """
        Detect communities using Louvain algorithm.
        
        Args:
            resolution: Higher = more communities
            
        Returns:
            List of actor_id sets (communities)
        """
        if len(self.graph) < 2:
            return []
        
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()
            communities = nx.community.louvain_communities(
                undirected, 
                weight="weight",
                resolution=resolution,
            )
            self.communities = [set(c) for c in communities]
            return self.communities
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return []
    
    def classify_all_roles(self) -> Dict[str, Tuple[ActorRole, float]]:
        """Classify roles for all actors."""
        # First compute centrality to update profiles
        self.compute_centrality()
        
        results = {}
        for actor_id, profile in self.actors.items():
            role, confidence = profile.classify_role()
            results[actor_id] = (role, confidence)
        
        return results
    
    def detect_coordination(
        self,
        time_window_seconds: float = 60,
        min_cluster_size: int = 3,
    ) -> List[Dict]:
        """
        Detect coordinated behavior patterns.
        
        Looks for:
        - Synchronized posting times
        - Same content/hashtags
        - Unusual engagement patterns
        
        Returns:
            List of coordination clusters
        """
        # Placeholder - would need post data with timestamps
        return []
    
    def get_key_actors(self, top_n: int = 10) -> List[ActorProfile]:
        """
        Get the most important actors by combined metrics.
        
        Returns:
            Top N actors by importance score
        """
        if not self.actors:
            return []
        
        # Compute importance score
        scored = []
        for actor_id, profile in self.actors.items():
            importance = (
                0.3 * min(1.0, profile.in_degree / 100) +
                0.2 * min(1.0, profile.out_degree / 50) +
                0.3 * profile.betweenness +
                0.2 * min(1.0, profile.suspicion_score)
            )
            scored.append((importance, profile))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [p for _, p in scored[:top_n]]
    
    def to_dict(self) -> Dict:
        """Export network summary as dictionary."""
        return {
            "node_count": len(self.graph),
            "edge_count": self.graph.number_of_edges(),
            "community_count": len(self.communities),
            "actors_by_role": self._count_by_role(),
            "key_actors": [
                {
                    "id": a.actor_id,
                    "username": a.username,
                    "role": a.role.value,
                    "suspicion": a.suspicion_score,
                }
                for a in self.get_key_actors(5)
            ],
        }
    
    def _count_by_role(self) -> Dict[str, int]:
        """Count actors by role."""
        counts = defaultdict(int)
        for profile in self.actors.values():
            counts[profile.role.value] += 1
        return dict(counts)


# =============================================================================
# Kenya Location Network
# =============================================================================

# Kenya counties with approximate coordinates
KENYA_COUNTIES: Dict[str, Tuple[float, float]] = {
    "nairobi": (-1.2921, 36.8219),
    "mombasa": (-4.0435, 39.6682),
    "kisumu": (-0.0917, 34.7680),
    "nakuru": (-0.3031, 36.0800),
    "eldoret": (0.5143, 35.2698),
    "machakos": (-1.5177, 37.2634),
    "meru": (0.0500, 37.6500),
    "nyeri": (-0.4197, 36.9553),
    "kakamega": (0.2827, 34.7519),
    "kisii": (-0.6817, 34.7667),
    "garissa": (-0.4536, 39.6401),
    "turkana": (3.3122, 35.5658),
    "mandera": (3.9366, 41.8670),
    "wajir": (1.7471, 40.0573),
    "kitui": (-1.3667, 38.0167),
    "nyanza": (-0.6000, 34.7500),
    "rift_valley": (0.5000, 36.0000),
    "coast": (-4.0000, 39.5000),
    "western": (0.5000, 34.5000),
    "central": (-0.5000, 37.0000),
    "eastern": (0.0000, 38.0000),
    "north_eastern": (2.0000, 40.0000),
}


@dataclass
class LocationActivity:
    """Activity in a Kenya location."""
    location: str
    lat: float
    lon: float
    
    # Metrics
    post_count: int = 0
    threat_count: int = 0
    avg_threat_tier: float = 5.0  # Lower = more severe
    
    # Signals detected
    signal_counts: Dict[str, int] = field(default_factory=dict)
    
    # Time
    last_activity: Optional[datetime] = None
    
    @property
    def threat_intensity(self) -> float:
        """Threat intensity score [0, 1]."""
        if self.post_count == 0:
            return 0.0
        threat_ratio = self.threat_count / self.post_count
        tier_factor = (5 - self.avg_threat_tier) / 4  # 0-1
        return min(1.0, threat_ratio * 0.5 + tier_factor * 0.5)


class KenyaLocationTracker:
    """
    Track threat activity by Kenya location.
    
    Provides:
    - Location-based threat heatmap
    - Regional trend analysis
    - Hotspot detection
    """
    
    def __init__(self):
        self.locations: Dict[str, LocationActivity] = {}
        self._initialize_locations()
    
    def _initialize_locations(self) -> None:
        """Initialize all Kenya locations."""
        for name, (lat, lon) in KENYA_COUNTIES.items():
            self.locations[name] = LocationActivity(
                location=name,
                lat=lat,
                lon=lon,
            )
    
    def record_activity(
        self,
        location: str,
        is_threat: bool = False,
        threat_tier: Optional[int] = None,
        signal_ids: Optional[List[str]] = None,
    ) -> None:
        """Record activity in a location."""
        location_key = location.lower().replace(" ", "_")
        
        if location_key not in self.locations:
            # Unknown location - skip
            return
        
        loc = self.locations[location_key]
        loc.post_count += 1
        loc.last_activity = datetime.utcnow()
        
        if is_threat:
            loc.threat_count += 1
        
        if threat_tier is not None:
            # Running average
            n = loc.post_count
            loc.avg_threat_tier = ((n - 1) * loc.avg_threat_tier + threat_tier) / n
        
        if signal_ids:
            for sig in signal_ids:
                loc.signal_counts[sig] = loc.signal_counts.get(sig, 0) + 1
    
    def get_hotspots(self, top_n: int = 5) -> List[LocationActivity]:
        """Get locations with highest threat intensity."""
        sorted_locs = sorted(
            self.locations.values(),
            key=lambda x: x.threat_intensity,
            reverse=True,
        )
        return sorted_locs[:top_n]
    
    def get_heatmap_data(self) -> List[Dict]:
        """
        Get data for heatmap visualization.
        
        Returns:
            List of {lat, lon, intensity, name} dicts
        """
        return [
            {
                "lat": loc.lat,
                "lon": loc.lon,
                "intensity": loc.threat_intensity,
                "name": loc.location,
                "posts": loc.post_count,
                "threats": loc.threat_count,
            }
            for loc in self.locations.values()
            if loc.post_count > 0
        ]
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        hotspots = self.get_hotspots(5)
        return {
            "total_locations": len([l for l in self.locations.values() if l.post_count > 0]),
            "hotspots": [
                {
                    "location": h.location,
                    "intensity": h.threat_intensity,
                    "threat_count": h.threat_count,
                }
                for h in hotspots
            ],
            "heatmap": self.get_heatmap_data(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_actor_network() -> ActorNetwork:
    """Create a new actor network."""
    return ActorNetwork()


def create_location_tracker() -> KenyaLocationTracker:
    """Create a Kenya location tracker."""
    return KenyaLocationTracker()

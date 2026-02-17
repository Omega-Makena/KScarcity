"""
Memory Buffer Layer for Hierarchical Federated Learning.

This module implements staleness-aware update storage with configurable triggers
for Layer 1 (intra-basket) and Layer 2 (cross-basket) aggregation.

Features:
- Timestamped updates with exponential time decay
- Replay attack detection via sequence validation
- Client participation caps for privacy
- Configurable triggers: count, time, quality, and privacy-based
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np


@dataclass
class BufferConfig:
    """Configuration for update buffer."""
    # Staleness handling
    max_age_seconds: float = 300.0        # Hard drop after 5 min
    decay_half_life: float = 60.0         # Exponential decay τ
    max_buffer_size: int = 1000           # Max stored updates per basket
    
    # Layer 1 triggers (any-of)
    trigger_count: int = 200              # After B samples
    trigger_interval: float = 2.0         # Every T seconds
    trigger_drift_threshold: float = 0.05 # Quality threshold
    
    # Layer 2 triggers (gated periodic)
    global_round_interval: int = 10       # Every R rounds
    min_basket_coverage: float = 0.6      # ≥60% baskets required
    
    # Privacy limits
    max_daily_participations: int = 100   # Per-client cap


@dataclass
class BufferedUpdate:
    """A timestamped update with metadata."""
    client_id: str
    basket_id: str
    vector: np.ndarray
    sequence_number: int
    round_id: int
    timestamp: float = field(default_factory=time.time)
    
    def age(self) -> float:
        """Compute age in seconds."""
        return time.time() - self.timestamp
    
    def decay_weight(self, half_life: float) -> float:
        """Compute exponential decay weight."""
        age = self.age()
        return math.exp(-age * math.log(2) / half_life)
    
    def is_stale(self, max_age: float) -> bool:
        """Check if update exceeds max age."""
        return self.age() > max_age


class ReplayGuard:
    """
    Prevents replay attacks and enforces participation caps.
    
    Tracks:
    - Per-client sequence numbers to detect replays
    - Per-client daily participation counts
    """
    
    def __init__(self, max_daily_participations: int = 100):
        """
        Initialize the replay guard.
        
        Args:
            max_daily_participations: Maximum participations per client per day
        """
        self.max_daily_participations = max_daily_participations
        
        # client_id -> last seen sequence number
        self._last_sequence: Dict[str, int] = {}
        
        # client_id -> participation count today
        self._participation_counts: Dict[str, int] = defaultdict(int)
        self._day_start: float = time.time()
    
    def validate(self, client_id: str, sequence_number: int) -> bool:
        """
        Check if update is valid (not replay, not over cap).
        
        Args:
            client_id: Client identifier
            sequence_number: Sequence number of the update
            
        Returns:
            True if update is valid
        """
        self._maybe_reset_day()
        
        # Check sequence number (must be strictly increasing)
        last_seq = self._last_sequence.get(client_id, -1)
        if sequence_number <= last_seq:
            return False  # Replay detected
        
        # Check participation cap
        if self._participation_counts[client_id] >= self.max_daily_participations:
            return False  # Cap exceeded
        
        return True
    
    def record_participation(self, client_id: str, sequence_number: int) -> None:
        """
        Record successful participation.
        
        Args:
            client_id: Client identifier
            sequence_number: Sequence number of the update
        """
        self._maybe_reset_day()
        self._last_sequence[client_id] = sequence_number
        self._participation_counts[client_id] += 1
    
    def get_participation_count(self, client_id: str) -> int:
        """Get today's participation count for a client."""
        self._maybe_reset_day()
        return self._participation_counts.get(client_id, 0)
    
    def _maybe_reset_day(self) -> None:
        """Reset counts if a new day has started."""
        now = time.time()
        if now - self._day_start >= 86400:  # 24 hours
            self._participation_counts.clear()
            self._day_start = now


class UpdateBuffer:
    """
    Staleness-aware update storage.
    
    Features:
    - Organizes updates by basket
    - Applies exponential time decay for weighting
    - Prunes stale updates automatically
    - Enforces buffer size limits
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """
        Initialize the buffer.
        
        Args:
            config: Buffer configuration
        """
        self.config = config or BufferConfig()
        self.replay_guard = ReplayGuard(
            max_daily_participations=self.config.max_daily_participations
        )
        
        # basket_id -> list of updates
        self._buffers: Dict[str, List[BufferedUpdate]] = defaultdict(list)
        
        # Tracking for triggers
        self._basket_counts: Dict[str, int] = defaultdict(int)
        self._basket_last_trigger: Dict[str, float] = {}
    
    def add(self, update: BufferedUpdate) -> bool:
        """
        Add an update to the buffer.
        
        Args:
            update: Update to add
            
        Returns:
            False if replay detected or cap exceeded
        """
        # Validate with replay guard
        if not self.replay_guard.validate(update.client_id, update.sequence_number):
            return False
        
        # Record participation
        self.replay_guard.record_participation(update.client_id, update.sequence_number)
        
        # Add to buffer
        basket_id = update.basket_id
        self._buffers[basket_id].append(update)
        self._basket_counts[basket_id] += 1
        
        # Enforce buffer size limit
        if len(self._buffers[basket_id]) > self.config.max_buffer_size:
            # Remove oldest
            self._buffers[basket_id] = self._buffers[basket_id][-self.config.max_buffer_size:]
        
        return True
    
    def get_weighted(
        self, 
        basket_id: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Get decay-weighted aggregate of updates.
        
        Args:
            basket_id: Optional basket to filter by (None = all)
            
        Returns:
            Tuple of (weighted average vector, number of updates used)
        """
        updates = []
        
        if basket_id is not None:
            updates = list(self._buffers.get(basket_id, []))
        else:
            for buf in self._buffers.values():
                updates.extend(buf)
        
        if not updates:
            return np.array([]), 0
        
        # Filter stale updates
        updates = [u for u in updates if not u.is_stale(self.config.max_age_seconds)]
        
        if not updates:
            return np.array([]), 0
        
        # Compute weights
        weights = np.array([
            u.decay_weight(self.config.decay_half_life) 
            for u in updates
        ])
        
        if weights.sum() <= 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / weights.sum()
        
        # Weighted average
        dim = len(updates[0].vector)
        result = np.zeros(dim, dtype=np.float32)
        for w, u in zip(weights, updates):
            result += w * u.vector
        
        return result, len(updates)
    
    def prune_stale(self) -> int:
        """
        Remove updates exceeding max_age.
        
        Returns:
            Number of updates pruned
        """
        total_pruned = 0
        
        for basket_id in list(self._buffers.keys()):
            before = len(self._buffers[basket_id])
            self._buffers[basket_id] = [
                u for u in self._buffers[basket_id]
                if not u.is_stale(self.config.max_age_seconds)
            ]
            total_pruned += before - len(self._buffers[basket_id])
        
        return total_pruned
    
    def clear_basket(self, basket_id: str) -> int:
        """
        Clear buffer for a basket after aggregation.
        
        Args:
            basket_id: Basket to clear
            
        Returns:
            Number of updates cleared
        """
        count = len(self._buffers.get(basket_id, []))
        self._buffers[basket_id] = []
        self._basket_counts[basket_id] = 0
        self._basket_last_trigger[basket_id] = time.time()
        return count
    
    def get_basket_update_count(self, basket_id: str) -> int:
        """Get current update count for a basket (since last clear)."""
        return self._basket_counts.get(basket_id, 0)
    
    def get_buffered_baskets(self) -> List[str]:
        """Get list of baskets with buffered updates."""
        return [bid for bid, buf in self._buffers.items() if buf]


class TriggerEngine:
    """
    Determines when to trigger Layer 1 and Layer 2 aggregation.
    
    Layer 1 triggers (any-of):
    - Count-based: after B samples
    - Time-based: every T seconds
    - Quality/drift-based: if drift exceeds threshold
    - Privacy-based: if local DP budget exhausted
    
    Layer 2 triggers (gated periodic):
    - Periodic: every R rounds
    - Coverage gate: ≥m% baskets contributed
    - DP gate: remaining budget allows release
    """
    
    def __init__(
        self, 
        config: BufferConfig,
        privacy_accountant: Optional['PrivacyAccountant'] = None
    ):
        """
        Initialize the trigger engine.
        
        Args:
            config: Buffer configuration with trigger thresholds
            privacy_accountant: Optional DP budget tracker
        """
        self.config = config
        self.privacy_accountant = privacy_accountant
        
        # Tracking state
        self._basket_last_trigger: Dict[str, float] = {}
        self._basket_sample_counts: Dict[str, int] = defaultdict(int)
        self._last_layer2_round: int = 0
        self._current_round: int = 0
        self._basket_contributed: Set[str] = set()
    
    def check_layer1(
        self, 
        basket_id: str, 
        sample_count: Optional[int] = None,
        drift_score: Optional[float] = None
    ) -> bool:
        """
        Check if Layer 1 should trigger for a basket.
        
        Args:
            basket_id: Basket to check
            sample_count: Current sample count (or use internal tracking)
            drift_score: Optional quality metric
            
        Returns:
            True if aggregation should trigger
        """
        now = time.time()
        
        # Update tracking
        if sample_count is not None:
            self._basket_sample_counts[basket_id] = sample_count
        
        count = self._basket_sample_counts[basket_id]
        last_trigger = self._basket_last_trigger.get(basket_id, now - 1000)
        
        # Count trigger
        if count >= self.config.trigger_count:
            return True
        
        # Time trigger
        if now - last_trigger >= self.config.trigger_interval:
            return True
        
        # Drift trigger
        if drift_score is not None and drift_score >= self.config.trigger_drift_threshold:
            return True
        
        return False
    
    def check_layer2(
        self, 
        total_baskets: int,
        contributed_baskets: Optional[int] = None
    ) -> bool:
        """
        Check if Layer 2 should trigger globally.
        
        Args:
            total_baskets: Total number of active baskets
            contributed_baskets: Number of baskets that contributed (or use internal)
            
        Returns:
            True if global aggregation should trigger
        """
        # Round check
        rounds_since = self._current_round - self._last_layer2_round
        if rounds_since < self.config.global_round_interval:
            return False
        
        # Coverage check
        if contributed_baskets is None:
            contributed_baskets = len(self._basket_contributed)
        
        if total_baskets > 0:
            coverage = contributed_baskets / total_baskets
            if coverage < self.config.min_basket_coverage:
                return False
        
        # Privacy check
        if self.privacy_accountant is not None:
            if not self.privacy_accountant.can_release():
                return False
        
        return True
    
    def mark_triggered(
        self, 
        layer: int, 
        basket_id: Optional[str] = None
    ) -> None:
        """
        Record trigger event.
        
        Args:
            layer: 1 or 2
            basket_id: Required for Layer 1
        """
        now = time.time()
        
        if layer == 1 and basket_id is not None:
            self._basket_last_trigger[basket_id] = now
            self._basket_sample_counts[basket_id] = 0
            self._basket_contributed.add(basket_id)
        
        elif layer == 2:
            self._last_layer2_round = self._current_round
            self._basket_contributed.clear()
    
    def advance_round(self) -> None:
        """Advance to next round."""
        self._current_round += 1
    
    def increment_count(self, basket_id: str, count: int = 1) -> None:
        """Increment sample count for a basket."""
        self._basket_sample_counts[basket_id] += count


class PrivacyAccountant:
    """
    Tracks ε/δ budget across rounds.
    
    Uses simple composition (conservative) for tracking.
    Advanced composition (e.g., Renyi DP) could be added later.
    """
    
    def __init__(
        self, 
        total_epsilon: float, 
        total_delta: float
    ):
        """
        Initialize the accountant.
        
        Args:
            total_epsilon: Total privacy budget (ε)
            total_delta: Total privacy failure probability (δ)
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self._spent_epsilon: float = 0.0
        self._spent_delta: float = 0.0
        self._release_count: int = 0
    
    def spend(self, epsilon: float, delta: float) -> bool:
        """
        Attempt to spend budget.
        
        Args:
            epsilon: Privacy cost (ε)
            delta: Privacy failure probability (δ)
            
        Returns:
            False if budget exhausted
        """
        if self._spent_epsilon + epsilon > self.total_epsilon:
            return False
        if self._spent_delta + delta > self.total_delta:
            return False
        
        self._spent_epsilon += epsilon
        self._spent_delta += delta
        self._release_count += 1
        return True
    
    def remaining(self) -> Tuple[float, float]:
        """Return remaining (ε, δ) budget."""
        return (
            max(0.0, self.total_epsilon - self._spent_epsilon),
            max(0.0, self.total_delta - self._spent_delta),
        )
    
    def can_release(self, epsilon: float = 0.1, delta: float = 1e-6) -> bool:
        """
        Check if budget allows another release.
        
        Args:
            epsilon: Expected cost of next release
            delta: Expected delta of next release
            
        Returns:
            True if release is allowed
        """
        rem_eps, rem_delta = self.remaining()
        return rem_eps >= epsilon and rem_delta >= delta
    
    @property
    def spent_epsilon(self) -> float:
        """Get total spent epsilon."""
        return self._spent_epsilon
    
    @property
    def spent_delta(self) -> float:
        """Get total spent delta."""
        return self._spent_delta
    
    @property
    def release_count(self) -> int:
        """Get number of releases made."""
        return self._release_count
    
    def reset(self) -> None:
        """Reset the accountant (use with caution)."""
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._release_count = 0

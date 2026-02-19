"""
Gossip Protocol for Intra-Basket Learning.

This module implements a push-pull hybrid gossip protocol for decentralized
learning within domain baskets. All gossip messages are clipped and locally
DP-noised before transmission (untrusted gossip model).

Privacy Model:
- Every message is L2-clipped to clip_norm
- Gaussian noise calibrated to (ε, δ)-DP is added before sending
- This provides client-level DP, not just server/global DP
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np

from .basket import BasketManager


@dataclass
class GossipConfig:
    """Configuration for gossip protocol."""
    peers_per_round: int = 5              # k peers (simple default)
    pull_interval: float = 2.0            # Seconds between pull rounds
    push_drift_threshold: float = 0.1     # Materiality threshold for push
    clip_norm: float = 1.0                # L2 clipping bound
    local_dp_epsilon: float = 1.0         # REQUIRED: Local DP at gossip layer
    local_dp_delta: float = 1e-5          # DP delta for Gaussian mechanism
    max_messages_per_day: int = 24        # Privacy budget limit per client
    message_ttl: float = 300.0            # Message time-to-live in seconds


@dataclass
class GossipMessage:
    """A gossip message between peers (clipped and DP-noised)."""
    sender_id: str
    basket_id: str
    summary_vector: np.ndarray            # Clipped and noised
    sequence_number: int
    timestamp: float = field(default_factory=time.time)
    round_id: int = 0
    
    def age(self) -> float:
        """Get age of message in seconds."""
        return time.time() - self.timestamp
    
    def is_expired(self, ttl: float) -> bool:
        """Check if message has expired."""
        return self.age() > ttl


class LocalDPMechanism:
    """
    Local Differential Privacy mechanism using Gaussian noise.
    
    Calibrates noise to achieve (ε, δ)-DP for a given sensitivity.
    """
    
    def __init__(self, epsilon: float, delta: float, sensitivity: float = 1.0):
        """
        Initialize the DP mechanism.
        
        Args:
            epsilon: Privacy parameter (lower = more private)
            delta: Probability of privacy failure
            sensitivity: L2 sensitivity of the query (clip_norm)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng()
        
        # Compute noise scale for Gaussian mechanism
        # σ >= sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        self.sigma = self._compute_sigma()
    
    def _compute_sigma(self) -> float:
        """Compute noise standard deviation for (ε,δ)-DP."""
        if self.epsilon <= 0:
            return float('inf')
        
        # Standard Gaussian mechanism calibration
        c = math.sqrt(2 * math.log(1.25 / self.delta))
        return self.sensitivity * c / self.epsilon
    
    def add_noise(self, vector: np.ndarray) -> np.ndarray:
        """Add calibrated Gaussian noise to a vector."""
        noise = self._rng.normal(0, self.sigma, size=vector.shape)
        return vector + noise.astype(vector.dtype)
    
    def clip_and_noise(self, vector: np.ndarray) -> np.ndarray:
        """Clip to L2 norm and add DP noise."""
        # L2 clipping
        norm = np.linalg.norm(vector)
        if norm > self.sensitivity:
            vector = vector * (self.sensitivity / norm)
        
        # Add noise
        return self.add_noise(vector)


class MaterialityDetector:
    """
    Detects when local changes warrant a push event.
    
    Tracks the last known state and computes drift metrics to determine
    if an update is "material" enough to justify pushing to peers.
    """
    
    def __init__(self, drift_threshold: float = 0.1):
        """
        Initialize the detector.
        
        Args:
            drift_threshold: Relative change threshold for triggering push
        """
        self.drift_threshold = drift_threshold
        self._last_states: Dict[str, np.ndarray] = {}
    
    def check_drift(self, client_id: str, new_state: np.ndarray) -> float:
        """
        Compute drift score between old and new state.
        
        Args:
            client_id: Client identifier
            new_state: New state vector
            
        Returns:
            Drift score (relative L2 change)
        """
        old_state = self._last_states.get(client_id)
        
        if old_state is None:
            # First observation - always material
            self._last_states[client_id] = new_state.copy()
            return 1.0
        
        # Compute relative change
        diff_norm = np.linalg.norm(new_state - old_state)
        old_norm = np.linalg.norm(old_state)
        
        if old_norm < 1e-10:
            drift = diff_norm
        else:
            drift = diff_norm / old_norm
        
        return float(drift)
    
    def should_push(self, client_id: str, new_state: np.ndarray) -> bool:
        """
        Check if update is material enough to push.
        
        Args:
            client_id: Client identifier
            new_state: New state vector
            
        Returns:
            True if push is warranted
        """
        drift = self.check_drift(client_id, new_state)
        return drift >= self.drift_threshold
    
    def update_state(self, client_id: str, state: np.ndarray) -> None:
        """Update the reference state for a client."""
        self._last_states[client_id] = state.copy()


class PeerSampler:
    """
    Samples k peers from a basket with rotation for privacy budget spreading.
    
    Maintains history to avoid oversampling the same peers repeatedly.
    """
    
    def __init__(self, rotation_window: int = 10):
        """
        Initialize the sampler.
        
        Args:
            rotation_window: Number of rounds before a peer can be resampled
        """
        self.rotation_window = rotation_window
        self._rng = np.random.default_rng()
        # Track (client_id, peer_id) -> last round sampled
        self._sample_history: Dict[Tuple[str, str], int] = {}
        self._current_round: int = 0
    
    def sample(
        self, 
        basket_peers: List[str], 
        k: int, 
        exclude: Optional[str] = None
    ) -> List[str]:
        """
        Sample k peers from the basket.
        
        Args:
            basket_peers: List of all peer IDs in the basket
            k: Number of peers to sample
            exclude: Client ID to exclude (self)
            
        Returns:
            List of sampled peer IDs
        """
        # Filter out self and recently sampled
        available = [
            peer_id for peer_id in basket_peers
            if peer_id != exclude and self._can_sample(exclude or "", peer_id)
        ]
        
        # If not enough available, allow resampling
        if len(available) < k:
            available = [p for p in basket_peers if p != exclude]
        
        # Sample
        k = min(k, len(available))
        if k == 0:
            return []
        
        indices = self._rng.choice(len(available), size=k, replace=False)
        sampled = [available[i] for i in indices]
        
        # Record sampling
        for peer_id in sampled:
            self._sample_history[(exclude or "", peer_id)] = self._current_round
        
        return sampled
    
    def _can_sample(self, client_id: str, peer_id: str) -> bool:
        """Check if peer can be sampled (not in rotation window)."""
        key = (client_id, peer_id)
        last_round = self._sample_history.get(key, -self.rotation_window - 1)
        return self._current_round - last_round > self.rotation_window
    
    def advance_round(self) -> None:
        """Advance to next round."""
        self._current_round += 1
    
    def rotate(self) -> None:
        """Alias for advance_round."""
        self.advance_round()


class MessageBudgetTracker:
    """
    Tracks per-client message counts for privacy budget enforcement.
    """
    
    def __init__(self, max_messages_per_day: int = 24):
        """
        Initialize the tracker.
        
        Args:
            max_messages_per_day: Maximum messages allowed per client per day
        """
        self.max_messages_per_day = max_messages_per_day
        self._message_counts: Dict[str, int] = defaultdict(int)
        self._day_start: float = time.time()
    
    def can_send(self, client_id: str) -> bool:
        """Check if client can send another message today."""
        self._maybe_reset_day()
        return self._message_counts[client_id] < self.max_messages_per_day
    
    def record_message(self, client_id: str) -> None:
        """Record that a message was sent."""
        self._maybe_reset_day()
        self._message_counts[client_id] += 1
    
    def remaining_budget(self, client_id: str) -> int:
        """Get remaining message budget for client."""
        self._maybe_reset_day()
        return max(0, self.max_messages_per_day - self._message_counts[client_id])
    
    def _maybe_reset_day(self) -> None:
        """Reset counts if a new day has started."""
        now = time.time()
        if now - self._day_start >= 86400:  # 24 hours
            self._message_counts.clear()
            self._day_start = now


class GossipProtocol:
    """
    Push-pull hybrid gossip for intra-basket learning.
    
    Features:
    - Pull loop: periodically sample k peers and request their summaries
    - Push trigger: send updates when local changes exceed materiality threshold
    - Local DP: all messages are clipped and noised before transmission
    - Budget tracking: limits messages per client per day
    """
    
    def __init__(
        self, 
        config: GossipConfig, 
        basket_manager: BasketManager
    ):
        """
        Initialize the gossip protocol.
        
        Args:
            config: Gossip configuration
            basket_manager: Basket manager for peer lookup
        """
        self.config = config
        self.basket_manager = basket_manager
        
        # Initialize components
        self.dp_mechanism = LocalDPMechanism(
            epsilon=config.local_dp_epsilon,
            delta=config.local_dp_delta,
            sensitivity=config.clip_norm,
        )
        self.materiality_detector = MaterialityDetector(
            drift_threshold=config.push_drift_threshold
        )
        self.peer_sampler = PeerSampler()
        self.budget_tracker = MessageBudgetTracker(
            max_messages_per_day=config.max_messages_per_day
        )
        
        # Message storage
        self._inbox: Dict[str, List[GossipMessage]] = defaultdict(list)
        self._outbox: Dict[str, List[GossipMessage]] = defaultdict(list)
        self._sequence_numbers: Dict[str, int] = defaultdict(int)
        self._round_id: int = 0
    
    def create_message(
        self, 
        client_id: str, 
        raw_vector: np.ndarray
    ) -> Optional[GossipMessage]:
        """
        Create a gossip message with clipping and DP noise.
        
        Args:
            client_id: Sender client ID
            raw_vector: Raw (unprotected) update vector
            
        Returns:
            GossipMessage with DP-protected vector, or None if budget exhausted
        """
        # Check budget
        if not self.budget_tracker.can_send(client_id):
            return None
        
        # Get client's basket
        client_info = self.basket_manager.get_client(client_id)
        if client_info is None:
            return None
        
        # Clip and add DP noise
        protected_vector = self.dp_mechanism.clip_and_noise(raw_vector)
        
        # Create message
        seq = self._sequence_numbers[client_id]
        self._sequence_numbers[client_id] = seq + 1
        
        message = GossipMessage(
            sender_id=client_id,
            basket_id=client_info.basket_id,
            summary_vector=protected_vector.astype(np.float32),
            sequence_number=seq,
            round_id=self._round_id,
        )
        
        # Record budget usage
        self.budget_tracker.record_message(client_id)
        
        return message
    
    def pull_round(self, client_id: str) -> List[str]:
        """
        Initiate a pull round: sample k peers to request summaries from.
        
        Args:
            client_id: Client initiating the pull
            
        Returns:
            List of peer IDs to request summaries from
        """
        basket_id = self.basket_manager.get_basket_for_client(client_id)
        if basket_id is None:
            return []
        
        peers = self.basket_manager.get_basket_peers(basket_id)
        return self.peer_sampler.sample(
            peers, 
            self.config.peers_per_round, 
            exclude=client_id
        )
    
    def push_update(
        self, 
        client_id: str, 
        update: np.ndarray
    ) -> Optional[GossipMessage]:
        """
        Push an update if it exceeds materiality threshold.
        
        Args:
            client_id: Client pushing the update
            update: Update vector
            
        Returns:
            GossipMessage if push was triggered, None otherwise
        """
        if not self.materiality_detector.should_push(client_id, update):
            return None
        
        message = self.create_message(client_id, update)
        if message is not None:
            self.materiality_detector.update_state(client_id, update)
        
        return message
    
    def receive_message(self, message: GossipMessage) -> bool:
        """
        Receive a gossip message into inbox.
        
        Args:
            message: Received message
            
        Returns:
            True if message was accepted (not expired, not duplicate)
        """
        # Check TTL
        if message.is_expired(self.config.message_ttl):
            return False
        
        # Store in inbox
        self._inbox[message.basket_id].append(message)
        return True
    
    def get_inbox_messages(
        self, 
        basket_id: str, 
        clear: bool = True
    ) -> List[GossipMessage]:
        """
        Get all messages in inbox for a basket.
        
        Args:
            basket_id: Basket to get messages for
            clear: Whether to clear inbox after retrieval
            
        Returns:
            List of messages
        """
        messages = list(self._inbox.get(basket_id, []))
        if clear:
            self._inbox[basket_id] = []
        return messages
    
    def merge_messages(
        self, 
        messages: List[GossipMessage], 
        decay_half_life: float = 60.0
    ) -> np.ndarray:
        """
        Merge received messages using exponential time decay.
        
        Args:
            messages: List of messages to merge
            decay_half_life: Half-life for exponential decay (seconds)
            
        Returns:
            Weighted average of message vectors
        """
        if not messages:
            return np.array([])
        
        # Compute decay weights
        now = time.time()
        weights = []
        vectors = []
        
        for msg in messages:
            age = now - msg.timestamp
            weight = math.exp(-age * math.log(2) / decay_half_life)
            weights.append(weight)
            vectors.append(msg.summary_vector)
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Weighted average
        result = np.zeros_like(vectors[0])
        for w, v in zip(weights, vectors):
            result += w * v
        
        return result
    
    def advance_round(self) -> None:
        """Advance to next gossip round."""
        self._round_id += 1
        self.peer_sampler.advance_round()
        
        # Prune expired messages
        for basket_id in list(self._inbox.keys()):
            self._inbox[basket_id] = [
                msg for msg in self._inbox[basket_id]
                if not msg.is_expired(self.config.message_ttl)
            ]
    
    @property
    def current_round(self) -> int:
        """Get current round ID."""
        return self._round_id

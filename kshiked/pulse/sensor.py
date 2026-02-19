"""
Pulse Sensor — Main Orchestrator for Signal Detection and Primitive Updates

The PulseSensor is the central coordinator that:
1. Maintains a registry of signal detectors
2. Processes incoming social media data through detectors
3. Maps detected signals to primitive updates via SignalMapper
4. Applies updates to the PulseState
5. Provides the interface for simulation integration

This module ensures clean decoupling: Pulse never calls simulation directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
import numpy as np

from .primitives import (
    PulseState,
    ScarcityUpdate,
    StressUpdate,
    BondUpdate,
    ResourceDomain,
    ActorType,
)
from .mapper import SignalMapper, SignalDetection, SignalID, SignalCategory

logger = logging.getLogger("kshield.pulse.sensor")


# =============================================================================
# Signal Detector Protocol
# =============================================================================

class SignalDetector(ABC):
    """
    Abstract base class for all signal detectors.
    
    Each detector is responsible for analyzing input data (text, metadata)
    and producing a SignalDetection with intensity and confidence.
    """
    
    @property
    @abstractmethod
    def signal_id(self) -> SignalID:
        """The signal this detector identifies."""
        pass
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        return self.signal_id.name
    
    @abstractmethod
    def detect(self, text: str, metadata: Dict[str, Any] = None) -> Optional[SignalDetection]:
        """
        Analyze text and return a detection if the signal is present.
        
        Args:
            text: The text content to analyze
            metadata: Optional context (author, location, timestamp, etc.)
            
        Returns:
            SignalDetection if signal detected, None otherwise
        """
        pass
    
    def batch_detect(self, texts: List[str], metadata_list: List[Dict] = None) -> List[SignalDetection]:
        """Process multiple texts."""
        metadata_list = metadata_list or [{}] * len(texts)
        results = []
        for text, meta in zip(texts, metadata_list):
            detection = self.detect(text, meta)
            if detection:
                results.append(detection)
        return results


# =============================================================================
# Placeholder Detectors (to be replaced with NLP implementations)
# =============================================================================

class KeywordDetector(SignalDetector):
    """
    Simple keyword-based detector for initial testing.
    Will be replaced with NLP models in Phase 3.
    """
    
    def __init__(
        self, 
        signal_id: SignalID, 
        keywords: List[str], 
        weight_map: Dict[str, float] = None
    ):
        self._signal_id = signal_id
        self.keywords = [k.lower() for k in keywords]
        self.weight_map = weight_map or {k: 1.0 for k in self.keywords}
    
    @property
    def signal_id(self) -> SignalID:
        return self._signal_id
    
    def detect(self, text: str, metadata: Dict[str, Any] = None) -> Optional[SignalDetection]:
        text_lower = text.lower()
        
        # Count weighted keyword matches
        score = 0.0
        matched_keywords = []
        for keyword in self.keywords:
            if keyword in text_lower:
                score += self.weight_map.get(keyword, 1.0)
                matched_keywords.append(keyword)
        
        if score < 0.5:  # Threshold
            return None
        
        # Normalize intensity [0, 1]
        intensity = min(1.0, score / 5.0)
        
        return SignalDetection(
            signal_id=self._signal_id,
            intensity=intensity,
            confidence=0.6,  # Keyword matching has moderate confidence
            raw_score=score,
            context={"matched_keywords": matched_keywords},
            timestamp=time.time()
        )


# =============================================================================
# Default Detector Registry
# =============================================================================

def create_default_detectors() -> Dict[SignalID, SignalDetector]:
    """
    Create default keyword-based detectors for all 15 signals.
    These are placeholders until NLP models are implemented.
    """
    detectors = {}
    
    # Signal 1: Survival Cost Stress
    detectors[SignalID.SURVIVAL_COST_STRESS] = KeywordDetector(
        SignalID.SURVIVAL_COST_STRESS,
        ["expensive", "can't afford", "too costly", "prices up", "inflation", 
         "rent increase", "food prices", "fuel prices", "unemployment", "no jobs"]
    )
    
    # Signal 2: Distress Framing
    detectors[SignalID.DISTRESS_FRAMING] = KeywordDetector(
        SignalID.DISTRESS_FRAMING,
        ["we're suffering", "people are dying", "crisis", "emergency", 
         "starvation", "disaster", "collapse", "catastrophe", "desperate"]
    )
    
    # Signal 3: Emotional Exhaustion
    detectors[SignalID.EMOTIONAL_EXHAUSTION] = KeywordDetector(
        SignalID.EMOTIONAL_EXHAUSTION,
        ["tired of this", "exhausted", "hopeless", "given up", "no hope",
         "can't take it anymore", "burnt out", "fed up", "enough is enough"]
    )
    
    # Signal 4: Directed Rage
    detectors[SignalID.DIRECTED_RAGE] = KeywordDetector(
        SignalID.DIRECTED_RAGE,
        ["hate", "destroy", "kill", "death to", "punish", "revenge",
         "traitor", "enemy", "corrupt leader", "must pay"]
    )
    
    # Signal 5: Rotating Regime Slang
    detectors[SignalID.ROTATING_REGIME_SLANG] = KeywordDetector(
        SignalID.ROTATING_REGIME_SLANG,
        ["#systemfail", "@corrupt", "thiefgovernment", "clownleader",
         "dictator", "puppet", "regime", "tyrant", "oppressor"]
    )
    
    # Signal 6: Dehumanization Language
    detectors[SignalID.DEHUMANIZATION_LANGUAGE] = KeywordDetector(
        SignalID.DEHUMANIZATION_LANGUAGE,
        ["cockroaches", "snakes", "rats", "vermin", "animals", "savages",
         "plague", "infestation", "cleanse", "eliminate them"]
    )
    
    # Signal 7: Legitimacy Rejection
    detectors[SignalID.LEGITIMACY_REJECTION] = KeywordDetector(
        SignalID.LEGITIMACY_REJECTION,
        ["fake election", "stolen vote", "illegitimate", "not my president",
         "rigged", "corrupt system", "fraud", "sham democracy"]
    )
    
    # Signal 8: Security Force Friction
    detectors[SignalID.SECURITY_FORCE_FRICTION] = KeywordDetector(
        SignalID.SECURITY_FORCE_FRICTION,
        ["police brutality", "abused by police", "military violence",
         "soldiers refuse", "police protest", "security defect"]
    )
    
    # Signal 9: Economic Cascade Failure
    detectors[SignalID.ECONOMIC_CASCADE_FAILURE] = KeywordDetector(
        SignalID.ECONOMIC_CASCADE_FAILURE,
        ["bank run", "currency collapse", "business closing", "bankruptcy",
         "market crash", "economic freefall", "hyperinflation", "no money"]
    )
    
    # Signal 10: Elite Fracture
    detectors[SignalID.ELITE_FRACTURE] = KeywordDetector(
        SignalID.ELITE_FRACTURE,
        ["businessman defects", "minister resigns", "general speaks out",
         "elite division", "oligarch leaves", "inner circle split"]
    )
    
    # Signal 11: Ethno-Regional Framing
    detectors[SignalID.ETHNO_REGIONAL_FRAMING] = KeywordDetector(
        SignalID.ETHNO_REGIONAL_FRAMING,
        ["our people", "our tribe", "our region", "they're taking from us",
         "their kind", "those people", "ethnic", "tribal", "ancestral land"]
    )
    
    # Signal 12: Mobilization Language
    detectors[SignalID.MOBILIZATION_LANGUAGE] = KeywordDetector(
        SignalID.MOBILIZATION_LANGUAGE,
        ["rise up", "take the streets", "protest now", "march together",
         "general strike", "shut it down", "join us", "everyone come"]
    )
    
    # Signal 13: Coordination Infrastructure
    detectors[SignalID.COORDINATION_INFRASTRUCTURE] = KeywordDetector(
        SignalID.COORDINATION_INFRASTRUCTURE,
        ["telegram group", "join our channel", "protest location",
         "meeting point", "bring supplies", "coordinate", "organize"]
    )
    
    # Signal 14: Rumor Velocity & Panic
    detectors[SignalID.RUMOR_VELOCITY_PANIC] = KeywordDetector(
        SignalID.RUMOR_VELOCITY_PANIC,
        ["heard that", "they say", "unconfirmed", "spreading fast",
         "panic buying", "stockpile", "run on", "emergency broadcast"]
    )
    
    # Signal 15: Counter-Narrative Activation
    detectors[SignalID.COUNTER_NARRATIVE_ACTIVATION] = KeywordDetector(
        SignalID.COUNTER_NARRATIVE_ACTIVATION,
        ["propaganda", "fake news", "don't believe", "truth is",
         "they're lying", "real story", "cover up", "hidden truth"]
    )
    
    return detectors


# =============================================================================
# Pulse Sensor Orchestrator
# =============================================================================

@dataclass
class PulseSensorConfig:
    """Configuration for the Pulse Sensor."""
    # Detection thresholds
    min_intensity_threshold: float = 0.1
    min_confidence_threshold: float = 0.3
    
    # Time decay for signal aggregation (lambda for exponential decay)
    time_decay_lambda: float = 0.01  # per second
    
    # Aggregation window (seconds)
    aggregation_window: float = 3600.0  # 1 hour
    
    # Update throttling (minimum seconds between state updates)
    update_interval: float = 60.0


class PulseSensor:
    """
    Main orchestrator for the Pulse Engine signal detection.
    
    Responsibilities:
    1. Registry of signal detectors
    2. Processing pipeline for incoming data
    3. Signal aggregation and time-weighting
    4. Primitive state management
    5. Interface for simulation layer
    """
    
    def __init__(self, config: PulseSensorConfig = None, use_nlp: bool = False):
        self.config = config or PulseSensorConfig()
        self.mapper = SignalMapper()
        self.state = PulseState()

        # Advanced Risk Scoring (Co-occurrence & Anomaly)
        from .cooccurrence import RiskScorer, RollingWindow, SignalCorrelationMatrix
        self.window = RollingWindow(window_seconds=config.aggregation_window)
        self.correlation_matrix = SignalCorrelationMatrix()
        self.scorer = RiskScorer(window=self.window, correlation_matrix=self.correlation_matrix)

        
        # Detector registry
        self._detectors: Dict[SignalID, SignalDetector] = {}
        
        # Signal history for time-weighted aggregation
        self._signal_history: List[SignalDetection] = []
        self._last_update_time: float = 0.0
        
        # Metrics
        self._total_processed: int = 0
        self._total_detections: int = 0
        
        # Initialize detectors based on mode
        if use_nlp:
            self._register_nlp_detectors()
        else:
            self._register_default_detectors()
    
    def _register_default_detectors(self) -> None:
        """Register the default keyword-based detectors."""
        defaults = create_default_detectors()
        for signal_id, detector in defaults.items():
            self.register_detector(detector)
    
    def _register_nlp_detectors(self) -> None:
        """Register NLP-enhanced detectors."""
        try:
            from .detectors import create_nlp_detectors
            nlp_detectors = create_nlp_detectors()
            for signal_id, detector in nlp_detectors.items():
                self.register_detector(detector)
            logger.info("Registered NLP-enhanced detectors")
        except ImportError as e:
            logger.warning(f"Failed to import NLP detectors: {e}. Falling back to keyword detectors.")
            self._register_default_detectors()
    
    def upgrade_to_nlp(self) -> None:
        """Upgrade from keyword to NLP detectors at runtime."""
        self._detectors.clear()
        self._register_nlp_detectors()
    
    def register_detector(self, detector: SignalDetector) -> None:
        """Register a signal detector."""
        self._detectors[detector.signal_id] = detector
        logger.debug(f"Registered detector for {detector.signal_id.name}")
    
    def get_detector(self, signal_id: SignalID) -> Optional[SignalDetector]:
        """Get a registered detector."""
        return self._detectors.get(signal_id)
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> List[SignalDetection]:
        """
        Process a single text through all detectors.
        
        Args:
            text: Text content to analyze
            metadata: Optional context
            
        Returns:
            List of detected signals
        """
        self._total_processed += 1
        metadata = metadata or {}
        detections = []
        
        for signal_id, detector in self._detectors.items():
            try:
                detection = detector.detect(text, metadata)
                if detection and self._passes_thresholds(detection):
                    detections.append(detection)
                    self._signal_history.append(detection)
                    # Feed advanced scorer
                    if hasattr(self, 'scorer'):
                        self.scorer.add_detection(detection)
                    self._total_detections += 1
            except Exception as e:
                logger.error(f"Detector {signal_id.name} failed: {e}")
        
        return detections
    
    def process_batch(
        self, 
        texts: List[str], 
        metadata_list: List[Dict] = None
    ) -> List[SignalDetection]:
        """Process multiple texts."""
        metadata_list = metadata_list or [{}] * len(texts)
        all_detections = []
        for text, meta in zip(texts, metadata_list):
            all_detections.extend(self.process_text(text, meta))
        return all_detections
    
    def _passes_thresholds(self, detection: SignalDetection) -> bool:
        """Check if detection passes minimum thresholds."""
        return (
            detection.intensity >= self.config.min_intensity_threshold and
            detection.confidence >= self.config.min_confidence_threshold
        )
    
    def update_state(self) -> PulseState:
        """
        Apply pending signal detections to primitive state.
        
        This is the main update loop that:
        1. Aggregates recent signals with time-weighting
        2. Maps signals to primitive updates
        3. Applies updates to state
        4. Computes risk metrics
        """
        now = time.time()
        
        # Throttle updates
        if now - self._last_update_time < self.config.update_interval:
            return self.state
        
        # Prune old signals
        self._prune_old_signals(now)
        
        # Update Risk Scorer
        if hasattr(self, 'scorer'):
            self.scorer.compute(now)
        
        # Aggregate signals with time decay
        aggregated = self._aggregate_signals(now)
        
        # Map to updates
        all_updates = []
        for detection in aggregated:
            updates = self.mapper.map_signal(detection, self.state)
            all_updates.extend(updates)
        
        # Apply updates
        self._apply_updates(all_updates)
        
        # Compute risk metrics
        self.state.compute_risk_metrics()
        self.state.timestamp = now
        self._last_update_time = now
        
        logger.info(f"State updated: instability={self.state.instability_index:.3f}, "
                   f"crisis_prob={self.state.crisis_probability:.3f}")
        
        return self.state
    
    def _prune_old_signals(self, now: float) -> None:
        """Remove signals outside the aggregation window."""
        cutoff = now - self.config.aggregation_window
        self._signal_history = [
            s for s in self._signal_history 
            if s.timestamp > cutoff
        ]
    
    def _aggregate_signals(self, now: float) -> List[SignalDetection]:
        """
        Aggregate signals with time-decay weighting.
        
        Returns a representative detection per signal ID with weighted intensity.
        """
        if not self._signal_history:
            return []
        
        # Group by signal ID
        by_signal: Dict[SignalID, List[SignalDetection]] = {}
        for det in self._signal_history:
            by_signal.setdefault(det.signal_id, []).append(det)
        
        aggregated = []
        for signal_id, detections in by_signal.items():
            # Time-weighted aggregation
            total_weight = 0.0
            weighted_intensity = 0.0
            
            for det in detections:
                age = now - det.timestamp
                weight = np.exp(-self.config.time_decay_lambda * age) * det.confidence
                total_weight += weight
                weighted_intensity += weight * det.intensity
            
            if total_weight > 0:
                avg_intensity = weighted_intensity / total_weight
                # Create aggregated detection
                aggregated.append(SignalDetection(
                    signal_id=signal_id,
                    intensity=avg_intensity,
                    confidence=min(1.0, total_weight / len(detections)),
                    raw_score=sum(d.raw_score for d in detections),
                    context={"aggregated_count": len(detections)},
                    timestamp=now
                ))
        
        return aggregated
    
    def _apply_updates(self, updates: List[ScarcityUpdate | StressUpdate | BondUpdate]) -> None:
        """Apply updates to the primitive state."""
        for update in updates:
            if isinstance(update, ScarcityUpdate):
                self.state.scarcity.set(
                    update.domain,
                    self.state.scarcity.get(update.domain) + update.delta
                )
            elif isinstance(update, StressUpdate):
                self.state.stress.apply_stress(update.actor, update.delta)
            elif isinstance(update, BondUpdate):
                self.state.bonds.apply_fracture(update.bond_type, update.delta)
    
    def get_shock_vector(self, variables: List[str]) -> Dict[str, float]:
        """
        Get shock magnitudes for simulation integration.
        
        This is the interface between Pulse and the simulation layer.
        Pulse never calls simulation directly — it only provides shocks.
        
        Args:
            variables: Economic variable names from simulation
            
        Returns:
            Dict of variable -> shock magnitude
        """
        return self.state.to_shock_vector(variables)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sensor metrics for monitoring."""
        return {
            "total_processed": self._total_processed,
            "total_detections": self._total_detections,
            "active_signals": len(self._signal_history),
            "instability_index": self.state.instability_index,
            "crisis_probability": self.state.crisis_probability,
            "scarcity_aggregate": self.state.scarcity.aggregate_score(),
            "system_stress": self.state.stress.total_system_stress(),
            "cohesion": self.state.bonds.overall_cohesion(),
        }
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the current signal correlation matrix."""
        if hasattr(self, 'correlation_matrix'):
            return self.correlation_matrix.get_matrix()
        return np.zeros((15, 15))

    def reset(self) -> None:
        """Reset sensor state."""
        self.state = PulseState()
        self._signal_history.clear()
        self._total_processed = 0
        self._total_detections = 0
        self._last_update_time = 0.0
        
        # Reset scorer
        if hasattr(self, 'scorer'):
             from .cooccurrence import RollingWindow, SignalCorrelationMatrix, RiskScorer
             self.window = RollingWindow(window_seconds=self.config.aggregation_window)
             self.correlation_matrix = SignalCorrelationMatrix()
             self.scorer = RiskScorer(window=self.window, correlation_matrix=self.correlation_matrix)



# =============================================================================
# Async Streaming Interface
# =============================================================================

class AsyncPulseSensor(PulseSensor):
    """
    Async version of PulseSensor for streaming data sources.
    """
    
    def __init__(self, config: PulseSensorConfig = None):
        super().__init__(config)
        self._running = False
        self._queue: asyncio.Queue = None
    
    async def start(self) -> None:
        """Start the async sensor."""
        self._running = True
        self._queue = asyncio.Queue()
        logger.info("AsyncPulseSensor started")
    
    async def stop(self) -> None:
        """Stop the async sensor."""
        self._running = False
        logger.info("AsyncPulseSensor stopped")
    
    async def ingest(self, text: str, metadata: Dict = None) -> None:
        """Add text to processing queue."""
        if self._queue:
            await self._queue.put((text, metadata or {}))
    
    async def process_queue(self) -> None:
        """Process items from the queue."""
        while self._running:
            try:
                text, metadata = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=1.0
                )
                self.process_text(text, metadata)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

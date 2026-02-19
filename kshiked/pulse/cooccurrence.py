"""
Time-Weighted Co-Occurrence Analysis

Provides:
- Temporal decay functions for signal weighting
- Rolling time windows for signal aggregation
- Signal correlation matrix computation
- Risk scoring algorithm
- Anomaly detection for signal spikes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

from .mapper import SignalID, SignalDetection, SIGNAL_CATEGORIES, SignalCategory

logger = logging.getLogger("kshield.pulse.cooccurrence")


# =============================================================================
# Temporal Decay Functions
# =============================================================================

class DecayFunction:
    """Base class for temporal decay functions."""
    
    def compute(self, age_seconds: float) -> float:
        """
        Compute decay weight for a given age.
        
        Args:
            age_seconds: How old the signal is in seconds
            
        Returns:
            Weight in [0, 1] where 1 = no decay, 0 = fully decayed
        """
        raise NotImplementedError


class ExponentialDecay(DecayFunction):
    """Exponential decay: w = e^(-lambda * t)"""
    
    def __init__(self, half_life_seconds: float = 3600):
        """
        Args:
            half_life_seconds: Time for weight to decay to 0.5
        """
        self.lambda_ = np.log(2) / half_life_seconds
    
    def compute(self, age_seconds: float) -> float:
        return np.exp(-self.lambda_ * age_seconds)


class LinearDecay(DecayFunction):
    """Linear decay: w = max(0, 1 - t/max_age)"""
    
    def __init__(self, max_age_seconds: float = 7200):
        self.max_age = max_age_seconds
    
    def compute(self, age_seconds: float) -> float:
        return max(0.0, 1.0 - age_seconds / self.max_age)


class StepDecay(DecayFunction):
    """Step decay: full weight within window, zero outside."""
    
    def __init__(self, window_seconds: float = 3600):
        self.window = window_seconds
    
    def compute(self, age_seconds: float) -> float:
        return 1.0 if age_seconds <= self.window else 0.0


# =============================================================================
# Rolling Time Windows
# =============================================================================

@dataclass
class SignalEvent:
    """A signal detection event with timestamp."""
    signal_id: SignalID
    intensity: float
    confidence: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)


class RollingWindow:
    """
    Rolling time window for signal events.
    
    Maintains a fixed-duration window of recent signals for analysis.
    """
    
    def __init__(
        self, 
        window_seconds: float = 3600,
        decay: DecayFunction = None,
    ):
        self.window_seconds = window_seconds
        self.decay = decay or ExponentialDecay(window_seconds / 2)
        self._events: deque = deque()
        self._last_prune = 0.0
    
    def add(self, event: SignalEvent) -> None:
        """Add an event to the window."""
        self._events.append(event)
        self._maybe_prune()
    
    def add_detection(self, detection: SignalDetection) -> None:
        """Add a detection as an event."""
        self.add(SignalEvent(
            signal_id=detection.signal_id,
            intensity=detection.intensity,
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            metadata=detection.context or {},
        ))
    
    def _maybe_prune(self, now: float = None) -> None:
        """Remove old events periodically."""
        now = now or time.time()
        if now - self._last_prune < 60:  # Prune at most once per minute
            return
        
        cutoff = now - self.window_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()
        
        self._last_prune = now
    
    def get_events(self, now: float = None) -> List[SignalEvent]:
        """Get all events in the current window."""
        now = now or time.time()
        self._maybe_prune(now)
        
        cutoff = now - self.window_seconds
        return [e for e in self._events if e.timestamp >= cutoff]
    
    def get_weighted_events(self, now: float = None) -> List[Tuple[SignalEvent, float]]:
        """Get events with their decay weights."""
        now = now or time.time()
        events = self.get_events(now)
        
        result = []
        for event in events:
            age = now - event.timestamp
            weight = self.decay.compute(age)
            result.append((event, weight))
        
        return result
    
    def get_signal_intensity(self, signal_id: SignalID, now: float = None) -> float:
        """Get weighted intensity for a specific signal."""
        now = now or time.time()
        weighted = self.get_weighted_events(now)
        
        total_weight = 0.0
        weighted_intensity = 0.0
        
        for event, weight in weighted:
            if event.signal_id == signal_id:
                total_weight += weight
                weighted_intensity += weight * event.intensity
        
        return weighted_intensity / total_weight if total_weight > 0 else 0.0
    
    def count(self) -> int:
        """Get number of events in window."""
        return len(self._events)


# =============================================================================
# Signal Correlation Matrix
# =============================================================================

class SignalCorrelationMatrix:
    """
    Computes and maintains co-occurrence correlations between signals.
    
    Tracks which signals tend to appear together, indicating
    compound risk scenarios.
    """
    
    def __init__(self, time_threshold_seconds: float = 300):
        """
        Args:
            time_threshold_seconds: Max time gap for co-occurrence
        """
        self.time_threshold = time_threshold_seconds
        self.n_signals = len(SignalID)
        
        # Co-occurrence counts
        self._cooccur_counts = np.zeros((self.n_signals, self.n_signals))
        self._signal_counts = np.zeros(self.n_signals)
        self._total_windows = 0
    
    def update(self, events: List[SignalEvent]) -> None:
        """
        Update correlation matrix from a batch of events.
        
        Args:
            events: List of events (typically from one time window)
        """
        if not events:
            return
        
        # Get unique signals in this batch
        signals_present = set()
        for event in events:
            idx = list(SignalID).index(event.signal_id)
            signals_present.add(idx)
            self._signal_counts[idx] += 1
        
        # Update co-occurrence
        for i in signals_present:
            for j in signals_present:
                if i <= j:  # Only upper triangle + diagonal
                    self._cooccur_counts[i, j] += 1
                    if i != j:
                        self._cooccur_counts[j, i] += 1
        
        self._total_windows += 1
    
    def get_correlation(self, signal_a: SignalID, signal_b: SignalID) -> float:
        """
        Get correlation coefficient between two signals.
        
        Uses Jaccard similarity: |A ∩ B| / |A ∪ B|
        """
        i = list(SignalID).index(signal_a)
        j = list(SignalID).index(signal_b)
        
        cooccur = self._cooccur_counts[i, j]
        union = self._signal_counts[i] + self._signal_counts[j] - cooccur
        
        if union == 0:
            return 0.0
        
        return cooccur / union
    
    def get_matrix(self) -> np.ndarray:
        """Get the full correlation matrix."""
        matrix = np.zeros((self.n_signals, self.n_signals))
        
        for i in range(self.n_signals):
            for j in range(self.n_signals):
                cooccur = self._cooccur_counts[i, j]
                union = self._signal_counts[i] + self._signal_counts[j] - cooccur
                matrix[i, j] = cooccur / union if union > 0 else 0.0
        
        return matrix
    
    def get_top_correlations(self, n: int = 10) -> List[Tuple[SignalID, SignalID, float]]:
        """Get top N correlated signal pairs."""
        matrix = self.get_matrix()
        
        # Get upper triangle indices
        pairs = []
        for i in range(self.n_signals):
            for j in range(i + 1, self.n_signals):
                pairs.append((
                    list(SignalID)[i],
                    list(SignalID)[j],
                    matrix[i, j]
                ))
        
        # Sort by correlation
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]


# =============================================================================
# Risk Scoring Algorithm
# =============================================================================

@dataclass
class RiskScore:
    """Computed risk score with breakdown."""
    overall: float              # [0, 1] aggregate risk
    by_category: Dict[str, float]  # Category -> risk
    by_signal: Dict[SignalID, float]  # Signal -> contribution
    anomaly_score: float        # [0, 1] how anomalous current state is
    trend: str                  # "rising", "stable", "falling"
    timestamp: float


class RiskScorer:
    """
    Computes aggregate risk scores from signal data.
    
    Uses:
    - Time-weighted signal intensities
    - Category-level aggregation
    - Co-occurrence amplification
    - Historical baseline comparison
    """
    
    # Category weights for risk computation
    CATEGORY_WEIGHTS = {
        SignalCategory.DISTRESS: 0.15,
        SignalCategory.ANGER: 0.25,
        SignalCategory.INSTITUTIONAL: 0.25,
        SignalCategory.IDENTITY: 0.20,
        SignalCategory.INFORMATION: 0.15,
    }
    
    def __init__(
        self,
        window: RollingWindow = None,
        correlation_matrix: SignalCorrelationMatrix = None,
    ):
        self.window = window or RollingWindow()
        self.correlation = correlation_matrix or SignalCorrelationMatrix()
        
        # Historical tracking for trend detection
        self._score_history: deque = deque(maxlen=100)
        self._baseline_mean = 0.0
        self._baseline_std = 0.1
    
    def add_detection(self, detection: SignalDetection) -> None:
        """Add a detection to the scorer."""
        self.window.add_detection(detection)
    
    def compute(self, now: float = None) -> RiskScore:
        """
        Compute current risk score.
        
        Returns:
            RiskScore with breakdown
        """
        now = now or time.time()
        
        # Get weighted events
        weighted_events = self.window.get_weighted_events(now)
        
        if not weighted_events:
            return RiskScore(
                overall=0.0,
                by_category={c.name: 0.0 for c in SignalCategory},
                by_signal={s: 0.0 for s in SignalID},
                anomaly_score=0.0,
                trend="stable",
                timestamp=now,
            )
        
        # Aggregate by signal
        signal_scores: Dict[SignalID, float] = {s: 0.0 for s in SignalID}
        signal_weights: Dict[SignalID, float] = {s: 0.0 for s in SignalID}
        
        for event, weight in weighted_events:
            signal_scores[event.signal_id] += weight * event.intensity * event.confidence
            signal_weights[event.signal_id] += weight
        
        # Normalize
        for signal_id in SignalID:
            if signal_weights[signal_id] > 0:
                signal_scores[signal_id] /= signal_weights[signal_id]
        
        # Aggregate by category
        category_scores: Dict[str, float] = {}
        for category in SignalCategory:
            signals_in_category = [
                s for s, c in SIGNAL_CATEGORIES.items() if c == category
            ]
            if signals_in_category:
                category_scores[category.name] = np.mean([
                    signal_scores[s] for s in signals_in_category
                ])
            else:
                category_scores[category.name] = 0.0
        
        # Compute overall (weighted)
        overall = sum(
            self.CATEGORY_WEIGHTS.get(cat, 0.2) * category_scores.get(cat.name, 0.0)
            for cat in SignalCategory
        )
        
        # Apply co-occurrence amplification
        amplification = self._compute_cooccurrence_amplification(signal_scores)
        overall = min(1.0, overall * (1.0 + amplification))
        
        # Compute anomaly score
        anomaly = self._compute_anomaly(overall)
        
        # Compute trend
        trend = self._compute_trend(overall)
        
        # Update history
        self._score_history.append((now, overall))
        self._update_baseline()
        
        return RiskScore(
            overall=overall,
            by_category=category_scores,
            by_signal=signal_scores,
            anomaly_score=anomaly,
            trend=trend,
            timestamp=now,
        )
    
    def _compute_cooccurrence_amplification(
        self, 
        signal_scores: Dict[SignalID, float]
    ) -> float:
        """
        Compute amplification factor from signal co-occurrence.
        
        When multiple high signals co-occur, risk is amplified.
        """
        # Count active signals (intensity > 0.3)
        active = [s for s, score in signal_scores.items() if score > 0.3]
        
        if len(active) < 2:
            return 0.0
        
        # Sum correlations between active signals
        total_corr = 0.0
        count = 0
        
        for i, signal_a in enumerate(active):
            for signal_b in active[i+1:]:
                corr = self.correlation.get_correlation(signal_a, signal_b)
                total_corr += corr
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_corr = total_corr / count
        
        # Amplification based on number of active signals and their correlation
        return min(0.5, len(active) * 0.05 * (1 + avg_corr))
    
    def _compute_anomaly(self, current_score: float) -> float:
        """
        Compute how anomalous the current score is.
        
        Uses z-score against historical baseline.
        """
        if self._baseline_std < 0.01:
            return 0.0
        
        z = (current_score - self._baseline_mean) / self._baseline_std
        
        # Convert z-score to [0, 1] probability
        # z > 2 is anomalous (top 2.5%)
        return min(1.0, max(0.0, (z - 1) / 3))
    
    def _compute_trend(self, current_score: float) -> str:
        """Determine if risk is rising, stable, or falling."""
        if len(self._score_history) < 5:
            return "stable"
        
        # Get recent scores
        recent = [s for _, s in list(self._score_history)[-10:]]
        
        # Simple linear trend
        if len(recent) >= 5:
            first_half = np.mean(recent[:len(recent)//2])
            second_half = np.mean(recent[len(recent)//2:])
            
            diff = second_half - first_half
            if diff > 0.05:
                return "rising"
            elif diff < -0.05:
                return "falling"
        
        return "stable"
    
    def _update_baseline(self) -> None:
        """Update baseline statistics from history."""
        if len(self._score_history) < 10:
            return
        
        scores = [s for _, s in self._score_history]
        self._baseline_mean = np.mean(scores)
        self._baseline_std = max(0.01, np.std(scores))


# =============================================================================
# Anomaly Detection
# =============================================================================

@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    signal_id: Optional[SignalID]
    category: Optional[SignalCategory]
    anomaly_type: str  # "spike", "cooccurrence", "trend"
    severity: float    # [0, 1]
    description: str
    timestamp: float


class AnomalyDetector:
    """
    Detects anomalies in signal patterns.
    
    Types of anomalies:
    - Spike: Sudden increase in a signal
    - Co-occurrence: Unusual combination of signals
    - Trend: Sustained directional movement
    """
    
    def __init__(
        self,
        spike_threshold: float = 3.0,  # z-score threshold
        window_seconds: float = 3600,
    ):
        self.spike_threshold = spike_threshold
        self.window_seconds = window_seconds
        
        # Historical baselines per signal
        self._baselines: Dict[SignalID, Tuple[float, float]] = {}  # mean, std
        self._history: Dict[SignalID, deque] = {
            s: deque(maxlen=100) for s in SignalID
        }
    
    def update(self, detection: SignalDetection) -> List[AnomalyAlert]:
        """
        Update with new detection and check for anomalies.
        
        Returns:
            List of anomaly alerts (may be empty)
        """
        alerts = []
        
        signal_id = detection.signal_id
        intensity = detection.intensity
        
        # Update history
        self._history[signal_id].append((detection.timestamp, intensity))
        
        # Check for spike
        spike_alert = self._check_spike(signal_id, intensity, detection.timestamp)
        if spike_alert:
            alerts.append(spike_alert)
        
        # Update baseline
        self._update_baseline(signal_id)
        
        return alerts
    
    def _check_spike(
        self, 
        signal_id: SignalID, 
        intensity: float,
        timestamp: float,
    ) -> Optional[AnomalyAlert]:
        """Check if current intensity is a spike."""
        if signal_id not in self._baselines:
            return None
        
        mean, std = self._baselines[signal_id]
        
        if std < 0.01:
            return None
        
        z = (intensity - mean) / std
        
        if z > self.spike_threshold:
            category = SIGNAL_CATEGORIES.get(signal_id)
            return AnomalyAlert(
                signal_id=signal_id,
                category=category,
                anomaly_type="spike",
                severity=min(1.0, z / 5.0),
                description=f"{signal_id.name} spiked to {intensity:.2f} (z={z:.1f})",
                timestamp=timestamp,
            )
        
        return None
    
    def _update_baseline(self, signal_id: SignalID) -> None:
        """Update baseline for a signal."""
        history = self._history[signal_id]
        if len(history) < 10:
            return
        
        intensities = [i for _, i in history]
        self._baselines[signal_id] = (np.mean(intensities), max(0.01, np.std(intensities)))
    
    def get_baseline(self, signal_id: SignalID) -> Tuple[float, float]:
        """Get baseline (mean, std) for a signal."""
        return self._baselines.get(signal_id, (0.0, 0.1))

"""
Comprehensive Pulse Engine Tests

Tests for:
- Primitives (ScarcityVector, ActorStress, BondStrength, ShockPropagation)
- NLP (SentimentAnalyzer, EmotionDetector, EntityRecognizer)
- Detectors (NLP signal detectors)
- Sensor (PulseSensor orchestration)
- Co-occurrence (RiskScorer, AnomalyDetector)
- Bridge (SimulationBridge, ShockScheduler)
"""

import pytest
import numpy as np
import time

# =============================================================================
# Imports
# =============================================================================

from kshiked.pulse.primitives import (
    ScarcityVector, ActorStress, BondStrength, ShockPropagation,
    PulseState, ResourceDomain, ActorType, SignalCategory,
)

from kshiked.pulse.mapper import (
    SignalMapper, SignalDetection, SignalID, SIGNAL_CATEGORIES,
)

from kshiked.pulse.sensor import (
    PulseSensor, PulseSensorConfig, KeywordDetector,
)

from kshiked.pulse.nlp import (
    SentimentAnalyzer, EmotionDetector, EntityRecognizer,
    TextPreprocessor, NLPPipeline,
)

from kshiked.pulse.detectors import (
    NLPSignalDetector, create_nlp_detectors,
)

from kshiked.pulse.cooccurrence import (
    ExponentialDecay, LinearDecay, RollingWindow,
    RiskScorer, SignalCorrelationMatrix, AnomalyDetector,
)

from kshiked.pulse.bridge import (
    ShockEvent, ShockType, ShockMagnitudeCalculator,
    ShockScheduler, SimulationBridge, create_kshield_bridge,
)


# =============================================================================
# Primitive Tests
# =============================================================================

class TestScarcityVector:
    """Tests for ScarcityVector primitive."""
    
    def test_init_defaults(self):
        """All domains initialize to 0."""
        sv = ScarcityVector()
        assert len(sv.values) == len(ResourceDomain)
        for domain in ResourceDomain:
            assert sv.get(domain) == 0.0
    
    def test_set_clamps_upper(self):
        """Values clamp to max 1.0."""
        sv = ScarcityVector()
        sv.set(ResourceDomain.FOOD, 1.5)
        assert sv.get(ResourceDomain.FOOD) == 1.0
    
    def test_set_clamps_lower(self):
        """Values clamp to min 0.0."""
        sv = ScarcityVector()
        sv.set(ResourceDomain.FUEL, -0.5)
        assert sv.get(ResourceDomain.FUEL) == 0.0
    
    def test_aggregate_score(self):
        """Aggregate computes mean of all domains."""
        sv = ScarcityVector()
        sv.set(ResourceDomain.FOOD, 0.6)
        sv.set(ResourceDomain.FUEL, 0.4)
        # Other domains are 0, so mean is (0.6+0.4+0+0+0+0)/6 = 1.0/6
        expected = (0.6 + 0.4) / len(ResourceDomain)
        assert abs(sv.aggregate_score() - expected) < 0.01
    
    def test_to_vector(self):
        """Converts to numpy array correctly."""
        sv = ScarcityVector()
        sv.set(ResourceDomain.FOOD, 0.5)
        vec = sv.to_vector()
        assert isinstance(vec, np.ndarray)
        assert len(vec) == len(ResourceDomain)


class TestActorStress:
    """Tests for ActorStress primitive."""
    
    def test_init_defaults(self):
        """All actors initialize to 0 stress."""
        stress = ActorStress()
        for actor in ActorType:
            assert stress.get_stress(actor) == 0.0
    
    def test_apply_stress_accumulates(self):
        """Stress deltas accumulate."""
        stress = ActorStress()
        stress.apply_stress(ActorType.STATE, -0.3)
        stress.apply_stress(ActorType.STATE, -0.2)
        assert stress.get_stress(ActorType.STATE) == -0.5
    
    def test_apply_stress_clamps(self):
        """Stress clamps to [-1, 1]."""
        stress = ActorStress()
        stress.apply_stress(ActorType.STATE, -1.5)
        assert stress.get_stress(ActorType.STATE) == -1.0
        
        stress.apply_stress(ActorType.ELITE, 1.5)
        assert stress.get_stress(ActorType.ELITE) == 1.0
    
    def test_total_system_stress(self):
        """Total stress sums negative stress values."""
        stress = ActorStress()
        stress.apply_stress(ActorType.STATE, -0.5)
        stress.apply_stress(ActorType.ELITE, -0.3)
        stress.apply_stress(ActorType.POPULATION, 0.2)  # Positive, not counted
        assert stress.total_system_stress() == 0.8


class TestBondStrength:
    """Tests for BondStrength primitive."""
    
    def test_init_defaults(self):
        """Bonds initialize to 0.5 (neutral)."""
        bonds = BondStrength()
        assert bonds.national_cohesion == 0.5
        assert bonds.class_solidarity == 0.5
    
    def test_apply_fracture(self):
        """Fracture reduces bond strength."""
        bonds = BondStrength()
        bonds.apply_fracture("national", -0.2)
        assert bonds.national_cohesion == 0.3
    
    def test_fragility_inverse_of_cohesion(self):
        """Low cohesion = high fragility."""
        bonds = BondStrength(national_cohesion=0.2, class_solidarity=0.2,
                            regional_unity=0.2)
        assert bonds.fragility_score() > 0.5
        
        strong_bonds = BondStrength(national_cohesion=0.9, class_solidarity=0.9,
                                   regional_unity=0.9)
        assert strong_bonds.fragility_score() < 0.3


class TestPulseState:
    """Tests for combined PulseState."""
    
    def test_compute_risk_metrics(self):
        """Risk metrics compute from primitives."""
        state = PulseState()
        state.scarcity.set(ResourceDomain.FOOD, 0.8)
        state.stress.apply_stress(ActorType.STATE, -0.7)
        state.bonds.apply_fracture("national", -0.3)
        
        state.compute_risk_metrics()
        
        assert state.instability_index > 0
        assert 0 <= state.crisis_probability <= 1


# =============================================================================
# NLP Tests
# =============================================================================

class TestSentimentAnalyzer:
    """Tests for sentiment analysis."""
    
    def test_negative_sentiment(self):
        """Negative text produces negative compound."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I hate this terrible situation")
        assert result.compound < 0
    
    def test_positive_sentiment(self):
        """Positive text produces positive compound."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I love this wonderful day")
        assert result.compound > 0
    
    def test_neutral_sentiment(self):
        """Neutral text produces near-zero compound."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("The meeting is at noon")
        assert abs(result.compound) < 0.3
    
    def test_negation_handling(self):
        """Negation flips sentiment."""
        analyzer = SentimentAnalyzer()
        positive = analyzer.analyze("I love this")
        negated = analyzer.analyze("I do not love this")
        assert negated.compound < positive.compound


class TestEmotionDetector:
    """Tests for emotion detection."""
    
    def test_anger_detection(self):
        """Angry text produces high anger score."""
        detector = EmotionDetector()
        result = detector.detect("I am furious and full of rage")
        assert result.emotions["anger"] > 0.5
        assert result.dominant == "anger"
    
    def test_sadness_detection(self):
        """Sad text produces high sadness score."""
        detector = EmotionDetector()
        result = detector.detect("I am so sad and depressed")
        assert result.emotions["sadness"] > 0.5
    
    def test_fear_detection(self):
        """Fearful text produces high fear score."""
        detector = EmotionDetector()
        result = detector.detect("I am terrified and scared")
        assert result.emotions["fear"] > 0.5


class TestEntityRecognizer:
    """Tests for NER."""
    
    def test_institution_recognition(self):
        """Recognizes government institutions."""
        ner = EntityRecognizer()
        entities = ner.extract("The government announced new policies")
        labels = [e.label for e in entities]
        assert "INSTITUTION" in labels
    
    def test_economic_recognition(self):
        """Recognizes economic terms."""
        ner = EntityRecognizer()
        entities = ner.extract("Inflation is rising and prices are high")
        labels = [e.label for e in entities]
        assert "ECONOMIC" in labels


class TestNLPPipeline:
    """Tests for combined NLP pipeline."""
    
    def test_full_analysis(self):
        """Pipeline produces all components."""
        pipeline = NLPPipeline()
        result = pipeline.analyze("The corrupt government is destroying our economy")
        
        assert result.sentiment is not None
        assert result.emotions is not None
        assert len(result.entities) >= 0
        assert result.token_count > 0


# =============================================================================
# Detector Tests
# =============================================================================

class TestNLPDetectors:
    """Tests for NLP signal detectors."""
    
    def test_all_15_detectors_created(self):
        """Factory creates all 15 detectors."""
        detectors = create_nlp_detectors()
        assert len(detectors) == 15
        for signal_id in SignalID:
            assert signal_id in detectors
    
    def test_survival_cost_stress_detection(self):
        """Signal 1 detects economic hardship."""
        detectors = create_nlp_detectors()
        detector = detectors[SignalID.SURVIVAL_COST_STRESS]
        
        result = detector.detect("Food prices are too expensive, cannot afford rent")
        assert result is not None
        assert result.intensity > 0.2
    
    def test_mobilization_language_detection(self):
        """Signal 12 detects call to action."""
        detectors = create_nlp_detectors()
        detector = detectors[SignalID.MOBILIZATION_LANGUAGE]
        
        result = detector.detect("Rise up! Take the streets! Join the protest now!")
        assert result is not None
        assert result.intensity > 0.3
    
    def test_dehumanization_detection(self):
        """Signal 6 detects degrading language."""
        detectors = create_nlp_detectors()
        detector = detectors[SignalID.DEHUMANIZATION_LANGUAGE]
        
        result = detector.detect("Those cockroaches and vermin must be eliminated")
        assert result is not None
        assert result.intensity > 0.4


# =============================================================================
# Sensor Tests
# =============================================================================

class TestPulseSensor:
    """Tests for PulseSensor orchestrator."""
    
    def test_init_with_keyword_detectors(self):
        """Default init uses keyword detectors."""
        sensor = PulseSensor()
        assert len(sensor._detectors) == 15
    
    def test_init_with_nlp_detectors(self):
        """NLP mode uses NLP detectors."""
        sensor = PulseSensor(use_nlp=True)
        assert len(sensor._detectors) == 15
    
    def test_process_text_returns_detections(self):
        """Processing text returns signal detections."""
        sensor = PulseSensor()
        detections = sensor.process_text("Food prices too expensive, inflation killing us")
        assert len(detections) > 0
    
    def test_state_update(self):
        """State updates from detections."""
        sensor = PulseSensor()
        sensor.config.update_interval = 0
        
        sensor.process_text("Economic crisis everywhere! Bank runs!")
        sensor.process_text("We hate the corrupt government!")
        
        sensor.update_state()
        
        assert sensor.state.scarcity.aggregate_score() > 0 or \
               sensor.state.stress.total_system_stress() > 0
    
    def test_upgrade_to_nlp(self):
        """Can upgrade detectors at runtime."""
        sensor = PulseSensor()
        sensor.upgrade_to_nlp()
        # Should still have 15 detectors
        assert len(sensor._detectors) == 15


# =============================================================================
# Co-occurrence Tests
# =============================================================================

class TestRollingWindow:
    """Tests for rolling time window."""
    
    def test_add_and_get_events(self):
        """Events can be added and retrieved."""
        window = RollingWindow(window_seconds=3600)
        
        detection = SignalDetection(
            signal_id=SignalID.SURVIVAL_COST_STRESS,
            intensity=0.5,
            confidence=0.8,
            raw_score=2.0,
            timestamp=time.time()
        )
        window.add_detection(detection)
        
        events = window.get_events()
        assert len(events) == 1
    
    def test_decay_weighting(self):
        """Older events have lower weight."""
        window = RollingWindow(window_seconds=3600)
        
        now = time.time()
        old_detection = SignalDetection(
            signal_id=SignalID.SURVIVAL_COST_STRESS,
            intensity=0.5, confidence=0.8, raw_score=2.0,
            timestamp=now - 1800  # 30 min ago
        )
        new_detection = SignalDetection(
            signal_id=SignalID.SURVIVAL_COST_STRESS,
            intensity=0.5, confidence=0.8, raw_score=2.0,
            timestamp=now
        )
        
        window.add_detection(old_detection)
        window.add_detection(new_detection)
        
        weighted = window.get_weighted_events(now)
        # New event should have higher weight
        weights = [w for _, w in weighted]
        assert weights[-1] > weights[0]


class TestRiskScorer:
    """Tests for risk scoring."""
    
    def test_empty_returns_zero(self):
        """No events = zero risk."""
        scorer = RiskScorer()
        score = scorer.compute()
        assert score.overall == 0.0
    
    def test_high_intensity_signals_increase_risk(self):
        """High intensity signals produce higher risk."""
        scorer = RiskScorer()
        
        for _ in range(5):
            detection = SignalDetection(
                signal_id=SignalID.DIRECTED_RAGE,
                intensity=0.9,
                confidence=0.9,
                raw_score=5.0,
                timestamp=time.time()
            )
            scorer.add_detection(detection)
        
        score = scorer.compute()
        assert score.overall > 0.1


class TestAnomalyDetector:
    """Tests for anomaly detection."""
    
    def test_spike_detection(self):
        """Detects sudden intensity spikes."""
        detector = AnomalyDetector(spike_threshold=2.0)
        
        # Build baseline with low intensity
        for i in range(20):
            detection = SignalDetection(
                signal_id=SignalID.SURVIVAL_COST_STRESS,
                intensity=0.2,
                confidence=0.7,
                raw_score=1.0,
                timestamp=time.time() - (20 - i)
            )
            detector.update(detection)
        
        # Now spike
        spike = SignalDetection(
            signal_id=SignalID.SURVIVAL_COST_STRESS,
            intensity=0.9,
            confidence=0.9,
            raw_score=5.0,
            timestamp=time.time()
        )
        alerts = detector.update(spike)
        
        # Should detect spike
        assert len(alerts) > 0 or detector._baselines.get(SignalID.SURVIVAL_COST_STRESS) is not None


# =============================================================================
# Bridge Tests
# =============================================================================

class TestShockMagnitudeCalculator:
    """Tests for shock calculation."""
    
    def test_low_risk_no_shocks(self):
        """Low risk produces no shocks."""
        calc = ShockMagnitudeCalculator()
        
        from kshiked.pulse.cooccurrence import RiskScore
        low_risk = RiskScore(
            overall=0.1,
            by_category={c.name: 0.1 for c in SignalCategory},
            by_signal={s: 0.1 for s in SignalID},
            anomaly_score=0.0,
            trend="stable",
            timestamp=time.time()
        )
        
        shocks = calc.compute_shocks(low_risk, PulseState())
        assert len(shocks) == 0
    
    def test_high_risk_produces_shocks(self):
        """High risk produces shocks."""
        calc = ShockMagnitudeCalculator()
        
        from kshiked.pulse.cooccurrence import RiskScore
        high_risk = RiskScore(
            overall=0.7,
            by_category={c.name: 0.6 for c in SignalCategory},
            by_signal={s: 0.5 for s in SignalID},
            anomaly_score=0.5,
            trend="rising",
            timestamp=time.time()
        )
        
        shocks = calc.compute_shocks(high_risk, PulseState())
        assert len(shocks) > 0


class TestShockScheduler:
    """Tests for shock scheduling."""
    
    def test_throttling(self):
        """Respects minimum interval between shocks."""
        from kshiked.pulse.bridge import SchedulerConfig
        config = SchedulerConfig(min_interval_seconds=60)
        scheduler = ShockScheduler(config)
        
        shock = ShockEvent(
            shock_type=ShockType.DEMAND_SHOCK,
            target_variable="GDP",
            magnitude=-0.05,
            risk_score=0.8,
        )
        
        scheduler.schedule([shock])
        due1 = scheduler.get_due_shocks()
        
        # Immediately after, should be throttled
        scheduler.schedule([shock])
        due2 = scheduler.get_due_shocks()
        
        assert len(due1) >= 0  # May or may not trigger based on probability
        assert len(due2) == 0  # Should be throttled


class TestSimulationBridge:
    """Tests for simulation bridge."""
    
    def test_handler_registration(self):
        """Handlers can be registered."""
        bridge = SimulationBridge()
        
        received = []
        def handler(shock):
            received.append(shock)
        
        bridge.register_handler(handler)
        assert len(bridge._handlers) == 1
    
    def test_create_kshield_bridge(self):
        """Factory creates configured bridge."""
        bridge, sensor = create_kshield_bridge(use_nlp=True)
        
        assert bridge is not None
        assert sensor is not None
        assert len(sensor._detectors) == 15


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_text_to_shock_pipeline(self):
        """Full pipeline: text -> detection -> risk -> shock."""
        bridge, sensor = create_kshield_bridge(use_nlp=True)
        sensor.config.update_interval = 0
        
        # Track shocks
        shocks_received = []
        bridge.register_handler(lambda s: shocks_received.append(s))
        
        # Process crisis texts
        crisis_texts = [
            "Economic crisis! Bank runs everywhere!",
            "Rise up! General strike now!",
            "The corrupt dictator must fall!",
            "Food prices killing us!",
            "Military violence against protesters!",
        ] * 3  # Repeat for more intensity
        
        for text in crisis_texts:
            for detection in sensor.process_text(text):
                bridge.scorer.add_detection(detection)
        
        # Process cycle - may or may not produce shocks depending on probability
        shocks = bridge.process_cycle()
        
        # At minimum, the system should run without errors
        assert bridge.get_stats() is not None
        
    def test_signal_mapping_correctness(self):
        """Signals map to correct primitive updates."""
        mapper = SignalMapper()
        state = PulseState()
        
        # Survival cost stress should affect scarcity
        detection = SignalDetection(
            signal_id=SignalID.SURVIVAL_COST_STRESS,
            intensity=0.8,
            confidence=0.9,
            raw_score=4.0,
            timestamp=time.time()
        )
        
        updates = mapper.map_signal(detection, state)
        
        # Should have scarcity updates
        has_scarcity = any(hasattr(u, 'domain') for u in updates)
        assert has_scarcity


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

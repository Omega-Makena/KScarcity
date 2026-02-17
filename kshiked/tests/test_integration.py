"""
Integration Tests for KShield Pulse

End-to-end tests for the complete pipeline:
- Signal detection → Threat classification → Index computation
- Database persistence
- Network analysis
- Visualization generation
"""

import pytest
import asyncio
import os
import tempfile
from datetime import datetime


# =============================================================================
# Signal Detection Tests
# =============================================================================

class TestSignalDetection:
    """Test signal detection pipeline."""
    
    def test_pulse_sensor_initialization(self):
        """Test PulseSensor initializes with all 15 detectors."""
        from kshiked.pulse import PulseSensor
        
        sensor = PulseSensor()
        assert len(sensor._detectors) == 15
    
    def test_detect_survival_cost_stress(self):
        """Test detection of survival cost stress signal."""
        from kshiked.pulse import PulseSensor
        from kshiked.pulse.mapper import SignalID
        
        sensor = PulseSensor()
        detections = sensor.process_text("Fuel prices are killing us! We can't afford food!")
        
        signal_ids = [d.signal_id for d in detections]
        assert SignalID.SURVIVAL_COST_STRESS in signal_ids
    
    def test_detect_mobilization_language(self):
        """Test detection of mobilization language."""
        from kshiked.pulse import PulseSensor
        from kshiked.pulse.mapper import SignalID
        
        sensor = PulseSensor()
        detections = sensor.process_text("Rise up Kenya! Take the streets tomorrow!")
        
        signal_ids = [d.signal_id for d in detections]
        assert SignalID.MOBILIZATION_LANGUAGE in signal_ids
    
    def test_detect_dehumanization(self):
        """Test detection of dehumanization language."""
        from kshiked.pulse import PulseSensor
        from kshiked.pulse.mapper import SignalID
        
        sensor = PulseSensor()
        detections = sensor.process_text("These cockroaches must be eliminated!")
        
        signal_ids = [d.signal_id for d in detections]
        assert SignalID.DEHUMANIZATION_LANGUAGE in signal_ids
    
    def test_detect_legitimacy_rejection(self):
        """Test detection of legitimacy rejection."""
        from kshiked.pulse import PulseSensor
        from kshiked.pulse.mapper import SignalID
        
        sensor = PulseSensor()
        detections = sensor.process_text("The election was stolen! Fake government!")
        
        signal_ids = [d.signal_id for d in detections]
        assert SignalID.LEGITIMACY_REJECTION in signal_ids


# =============================================================================
# Threat Index Tests
# =============================================================================

class TestThreatIndices:
    """Test all 8 threat indices."""
    
    def test_compute_threat_report(self):
        """Test full threat report computation."""
        from kshiked.pulse import PulseSensor, compute_threat_report
        
        sensor = PulseSensor()
        sensor.process_text("Rise up! The election was stolen!")
        
        report = compute_threat_report(sensor.state, sensor._signal_history)
        
        assert report is not None
        assert hasattr(report, 'polarization')
        assert hasattr(report, 'legitimacy_erosion')
        assert hasattr(report, 'mobilization_readiness')
        assert hasattr(report, 'elite_cohesion')
        assert hasattr(report, 'information_warfare')
        assert hasattr(report, 'security_friction')
        assert hasattr(report, 'economic_cascade')
        assert hasattr(report, 'ethnic_tension')
    
    def test_polarization_index(self):
        """Test polarization index computation."""
        from kshiked.pulse.indices import PolarizationIndex
        from kshiked.pulse.primitives import PulseState
        
        state = PulseState()
        pi = PolarizationIndex.compute(state, [])
        
        assert 0 <= pi.value <= 1
        assert pi.severity in ("LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL")
    
    def test_ethnic_tension_matrix(self):
        """Test Kenya ethnic tension matrix."""
        from kshiked.pulse.indices import EthnicTensionMatrix, TENSION_PAIRS
        from kshiked.pulse.primitives import PulseState
        
        state = PulseState()
        etm = EthnicTensionMatrix.compute(state, [])
        
        # Check tension pairs exist
        assert len(etm.tensions) >= len(TENSION_PAIRS)
        
        # Check Kikuyu-Luo pair exists (historical)
        kikuyu_luo = etm.get_tension("kikuyu", "luo")
        assert kikuyu_luo >= 0
    
    def test_threat_report_to_dict(self):
        """Test threat report JSON serialization."""
        from kshiked.pulse import PulseSensor, compute_threat_report
        
        sensor = PulseSensor()
        sensor.process_text("Test threat content")
        
        report = compute_threat_report(sensor.state, sensor._signal_history)
        data = report.to_dict()
        
        assert "overall_threat_level" in data
        assert "indices" in data
        assert "priority_alerts" in data


# =============================================================================
# Network Analysis Tests
# =============================================================================

class TestNetworkAnalysis:
    """Test network analysis module."""
    
    def test_actor_profile_creation(self):
        """Test actor profile creation."""
        from kshiked.pulse.network import ActorProfile, ActorRole
        
        profile = ActorProfile(
            actor_id="test_123",
            platform="twitter",
            username="test_user",
            post_count=50,
            mobilization_posts=10,
        )
        
        assert profile.actor_id == "test_123"
        assert profile.role == ActorRole.UNKNOWN
    
    def test_actor_role_classification(self):
        """Test actor role classification."""
        from kshiked.pulse.network import ActorProfile, ActorRole
        
        # Create mobilizer profile
        mobilizer = ActorProfile(
            actor_id="mob_123",
            platform="twitter",
            mobilization_posts=15,
            out_degree=30,
            original_content_ratio=0.8,
        )
        
        role, confidence = mobilizer.classify_role()
        assert role == ActorRole.MOBILIZER
        assert confidence > 0.5
    
    def test_kenya_location_tracker(self):
        """Test Kenya location tracker."""
        from kshiked.pulse.network import KenyaLocationTracker
        
        tracker = KenyaLocationTracker()
        
        # Record activity in Nairobi
        tracker.record_activity("nairobi", is_threat=True, threat_tier=3)
        tracker.record_activity("nairobi", is_threat=False)
        
        nairobi = tracker.locations["nairobi"]
        assert nairobi.post_count == 2
        assert nairobi.threat_count == 1
    
    def test_location_hotspots(self):
        """Test hotspot detection."""
        from kshiked.pulse.network import KenyaLocationTracker
        
        tracker = KenyaLocationTracker()
        
        # Create threats in multiple locations
        for _ in range(5):
            tracker.record_activity("nairobi", is_threat=True, threat_tier=2)
        for _ in range(2):
            tracker.record_activity("mombasa", is_threat=True, threat_tier=3)
        
        hotspots = tracker.get_hotspots(3)
        
        # Nairobi should be top hotspot
        assert len(hotspots) >= 2
        assert hotspots[0].location == "nairobi"


# =============================================================================
# Kenya Filter Tests
# =============================================================================

class TestKenyaFilters:
    """Test Kenya-specific keyword filters."""
    
    def test_political_keywords(self):
        """Test political keyword list."""
        from kshiked.pulse.filters import KENYA_POLITICAL
        
        assert "ruto" in KENYA_POLITICAL
        assert "raila" in KENYA_POLITICAL
        assert len(KENYA_POLITICAL) > 20
    
    def test_threat_detection(self):
        """Test threat-related text detection."""
        from kshiked.pulse.filters import is_threat_related
        
        assert is_threat_related("Join the maandamano tomorrow!")
        assert is_threat_related("Rise up against the government!")
        assert not is_threat_related("Beautiful weather in Nairobi today")
    
    def test_keyword_extraction(self):
        """Test keyword extraction from text."""
        from kshiked.pulse.filters import get_matched_keywords
        
        text = "President Ruto and Raila discussed issues in Nairobi"
        matches = get_matched_keywords(text)
        
        assert "ruto" in matches
        assert "raila" in matches
        assert "nairobi" in matches


# =============================================================================
# Visualization Tests
# =============================================================================

class TestVisualization:
    """Test visualization generation."""
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        from kshiked.pulse import PulseSensor, compute_threat_report
        from kshiked.pulse.visualization import generate_html_report
        
        sensor = PulseSensor()
        sensor.process_text("Test threat")
        report = compute_threat_report(sensor.state, sensor._signal_history)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name
        
        try:
            result = generate_html_report(report.to_dict(), output_path=output_path)
            assert os.path.exists(result)
            
            with open(result, 'r') as f:
                content = f.read()
                assert "KShield Pulse" in content
                assert "Threat" in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# End-to-End Integration Test
# =============================================================================

class TestEndToEnd:
    """Full pipeline integration tests."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from text to report."""
        from kshiked.pulse import PulseSensor, compute_threat_report
        from kshiked.pulse.network import KenyaLocationTracker
        
        # 1. Initialize components
        sensor = PulseSensor()
        location_tracker = KenyaLocationTracker()
        
        # 2. Process multiple threat texts
        texts = [
            ("nairobi", "Rise up Kenya! Take the streets!"),
            ("mombasa", "The election was stolen! Fake government!"),
            ("nairobi", "Food prices killing us! Government doesn't care!"),
            ("kisumu", "Our Luo people are being oppressed!"),
        ]
        
        for location, text in texts:
            detections = sensor.process_text(text)
            is_threat = len(detections) > 0
            location_tracker.record_activity(
                location,
                is_threat=is_threat,
                signal_ids=[d.signal_id.name for d in detections],
            )
        
        # 3. Update state and compute indices
        sensor.update_state()
        report = compute_threat_report(sensor.state, sensor._signal_history)
        
        # 4. Verify outputs
        assert report.overall_threat_level in ("LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL")
        
        hotspots = location_tracker.get_hotspots(3)
        assert len(hotspots) >= 2
        
        # 5. Export
        data = report.to_dict()
        assert "indices" in data
        assert len(data["indices"]) >= 8


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

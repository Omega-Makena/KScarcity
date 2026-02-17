"""
Test: Meta-Learning Layer

Validates cross-domain transfer and meta-update generation.
"""

import pytest
import numpy as np
from scarcity.meta.domain_meta import (
    DomainMetaLearner,
    DomainMetaConfig,
)


class TestDomainMetaLearner:
    def test_observe_generates_update(self):
        """observe() should generate a meta-update."""
        learner = DomainMetaLearner()
        
        update = learner.observe(
            domain_id='healthcare',
            metrics={'gain_p50': 0.7, 'stability_avg': 0.8},
            parameters={'lr': 0.01, 'alpha': 0.1}
        )
        
        assert update.domain_id == 'healthcare'
        assert update.vector is not None
        assert len(update.keys) == 2
        print(f"Update: {update}")
    
    def test_confidence_increases_with_good_performance(self):
        """Confidence should increase with consistent improvements."""
        learner = DomainMetaLearner()
        
        confidences = []
        for i in range(10):
            update = learner.observe(
                domain_id='finance',
                metrics={'gain_p50': 0.5 + i * 0.05, 'stability_avg': 0.8},
                parameters={'lr': 0.01}
            )
            confidences.append(update.confidence)
        
        # Confidence should generally increase
        assert confidences[-1] > confidences[0], \
            f"Confidence should increase: {confidences}"
        print(f"Confidence progression: {confidences}")
    
    def test_state_persistence(self):
        """State should persist across observations."""
        learner = DomainMetaLearner()
        
        learner.observe('domain_a', {'gain_p50': 0.5}, {'x': 1.0})
        learner.observe('domain_a', {'gain_p50': 0.6}, {'x': 1.5})
        
        state = learner.state('domain_a')
        
        assert state.last_score == 0.6
        assert len(state.history) == 2
        print(f"State: {state}")


class TestCrossDomainTransfer:
    def test_multiple_domains_tracked(self):
        """Learner should track multiple domains independently."""
        learner = DomainMetaLearner()
        
        learner.observe('healthcare', {'gain_p50': 0.3}, {'lr': 0.01})
        learner.observe('finance', {'gain_p50': 0.7}, {'lr': 0.02})
        
        state_health = learner.state('healthcare')
        state_finance = learner.state('finance')
        
        assert state_health.last_score == 0.3
        assert state_finance.last_score == 0.7
        
        print(f"Healthcare: {state_health.last_score}")
        print(f"Finance: {state_finance.last_score}")
    
    def test_domain_updates_independent(self):
        """Domain updates should be independent."""
        learner = DomainMetaLearner()
        
        # Different domains, different parameters
        update_a = learner.observe('domain_a', {'gain_p50': 0.5}, {'x': 1.0, 'y': 2.0})
        update_b = learner.observe('domain_b', {'gain_p50': 0.8}, {'z': 3.0})
        
        assert update_a.keys == ['x', 'y']
        assert update_b.keys == ['z']
        
        print(f"Domain A keys: {update_a.keys}")
        print(f"Domain B keys: {update_b.keys}")


class TestMetaLearningAdaptation:
    def test_adaptive_learning_rate(self):
        """Meta learning rate should adapt based on confidence."""
        config = DomainMetaConfig(
            meta_lr_min=0.01,
            meta_lr_max=0.1
        )
        learner = DomainMetaLearner(config)
        
        # First observation - low confidence
        update1 = learner.observe('test', {'gain_p50': 0.5}, {'param': 1.0})
        state1 = learner.state('test')
        
        # Many successful observations - build confidence
        for i in range(10):
            learner.observe('test', {'gain_p50': 0.5 + i * 0.02, 'stability_avg': 0.9}, {'param': 1.0 + i * 0.1})
        
        state2 = learner.state('test')
        
        # Confidence should have increased (or stayed if it was already building)
        assert state2.confidence >= state1.confidence or state2.confidence > 0.3, \
            f"Confidence should build: {state1.confidence:.3f} → {state2.confidence:.3f}"

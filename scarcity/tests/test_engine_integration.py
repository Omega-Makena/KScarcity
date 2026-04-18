"""
Test: OnlineDiscoveryEngine with New Hypothesis Classes

Validates that the V2 initialization works and processes data correctly.
"""

import pytest
import numpy as np
from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.discovery import HypothesisPool, HypothesisState
from scarcity.engine.controller import MetaController
from scarcity.engine.arbitration import HypothesisArbiter
from scarcity.engine.relationships import CorrelationalHypothesis, CausalHypothesis
from scarcity.tests.fixtures import generate_causal, generate_correlational


class TestEngineV2Integration:
    def test_initialize_v2_creates_hypotheses(self):
        """initialize_v2 should populate the hypothesis pool."""
        engine = OnlineDiscoveryEngine()
        
        schema = {
            'fields': [
                {'name': 'GDP'},
                {'name': 'Inflation'},
                {'name': 'Exports'},
            ]
        }
        
        engine.initialize_v2(schema, use_causal=True)
        
        # Should have created hypotheses
        assert len(engine.hypotheses.population) > 0
        print(f"Created {len(engine.hypotheses.population)} hypotheses")
    
    def test_process_rows_updates_hypotheses(self):
        """process_row should update hypothesis metrics."""
        engine = OnlineDiscoveryEngine()
        
        schema = {
            'fields': [
                {'name': 'X'},
                {'name': 'Y'},
            ]
        }
        
        engine.initialize_v2(schema, use_causal=True)
        
        # Generate correlated data
        dataset = generate_correlational(n=100)
        
        # Process rows
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            result = engine.process_row(row)
        
        assert result['step'] == 100
        assert result['active_hypotheses'] > 0
        print(f"After 100 steps: {result}")
    
    def test_get_knowledge_graph(self):
        """Should return learned relationships."""
        engine = OnlineDiscoveryEngine()
        
        schema = {
            'fields': [
                {'name': 'X'},
                {'name': 'Y'},
            ]
        }
        
        engine.initialize_v2(schema, use_causal=False)
        
        # Generate causal data
        dataset = generate_causal(n=200, lag=2)
        
        for i in range(len(dataset.data['X'])):
            row = {'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]}
            engine.process_row(row)
        
        # Get knowledge graph
        kg = engine.get_knowledge_graph()
        
        assert len(kg) > 0
        print(f"Knowledge graph has {len(kg)} relationships")
        
        # Print top relationships
        for edge in kg[:5]:
            print(f"  {edge.get('variables', 'N/A')}: {edge.get('rel_type', 'N/A')} "
                  f"(fit={edge.get('fit_score', 0):.3f})")

    def test_performance_mode_reports_mode_and_processes_stream(self):
        """Pilot simulation: performance mode should process stream and report mode explicitly."""
        engine = OnlineDiscoveryEngine(mode='performance')
        schema = {
            'fields': [
                {'name': 'X'},
                {'name': 'Y'},
            ]
        }
        engine.initialize_v2(schema, use_causal=True)

        dataset = generate_correlational(n=300)
        for i in range(len(dataset.data['X'])):
            result = engine.process_row({'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]})

        assert result['engine_mode'] == 'performance'
        assert result['step'] == 300
        assert result['total_hypotheses'] > 0

    def test_performance_mode_disables_group_monitoring(self, monkeypatch):
        """Performance mode should reduce overhead by skipping grouping monitor calls."""
        engine = OnlineDiscoveryEngine(mode='performance')
        schema = {'fields': [{'name': 'X'}, {'name': 'Y'}]}
        engine.initialize_v2(schema, use_causal=False)

        calls = {'n': 0}

        def _count_monitor(row, errors):
            calls['n'] += 1

        monkeypatch.setattr(engine.grouper, 'monitor', _count_monitor)

        dataset = generate_correlational(n=80)
        for i in range(len(dataset.data['X'])):
            engine.process_row({'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]})

        assert calls['n'] == 0

    def test_balanced_mode_keeps_group_monitoring_enabled(self, monkeypatch):
        """Balanced mode should retain full grouping behavior."""
        engine = OnlineDiscoveryEngine(mode='balanced')
        schema = {'fields': [{'name': 'X'}, {'name': 'Y'}]}
        engine.initialize_v2(schema, use_causal=False)

        calls = {'n': 0}

        def _count_monitor(row, errors):
            calls['n'] += 1

        monkeypatch.setattr(engine.grouper, 'monitor', _count_monitor)

        dataset = generate_correlational(n=60)
        for i in range(len(dataset.data['X'])):
            engine.process_row({'X': dataset.data['X'][i], 'Y': dataset.data['Y'][i]})

        assert calls['n'] == 60

    def test_invalid_mode_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            OnlineDiscoveryEngine(mode="turbo")

    def test_set_mode_switches_runtime_flags(self):
        engine = OnlineDiscoveryEngine(mode="balanced")
        assert engine.grouping_enabled is True
        assert engine.exploration_enabled is True

        engine.set_mode("performance")
        assert engine.grouping_enabled is False
        assert engine.exploration_enabled is False
        assert engine.lifecycle_interval == 25
        assert engine.arbitration_interval == 100

    def test_process_row_handles_malformed_payload_without_crash(self):
        engine = OnlineDiscoveryEngine(mode="performance")
        engine.initialize_v2({'fields': [{'name': 'X'}, {'name': 'Y'}]}, use_causal=False)

        # Non-dict payload should be ignored but processing should continue.
        result1 = engine.process_row("not-a-dict")
        result2 = engine.process_row(None)

        assert result1["step"] == 1
        assert result2["step"] == 2
        assert result2["engine_mode"] == "performance"

    def test_process_row_continues_when_single_hypothesis_fails(self, monkeypatch):
        engine = OnlineDiscoveryEngine(mode='balanced')
        engine.initialize_v2({'fields': [{'name': 'X'}, {'name': 'Y'}]}, use_causal=False)

        bad_hyp = next(iter(engine.hypotheses.population.values()))

        def _raise_on_update(row):
            raise RuntimeError("forced failure for robustness test")

        monkeypatch.setattr(bad_hyp, 'update', _raise_on_update)

        result1 = engine.process_row({'X': 1.0, 'Y': 1.1})
        result2 = engine.process_row({'X': 1.2, 'Y': 1.0})

        assert result1['update_errors'] >= 1
        assert result1['update_error_total'] >= 1
        assert isinstance(result1.get('update_error_details'), list)
        assert result2['step'] == 2
        assert result2['update_error_total'] >= result1['update_error_total']

    def test_drift_signal_detects_regime_shift_and_triggers_group_split(self):
        engine = OnlineDiscoveryEngine(mode='balanced')
        engine.initialize_v2({'fields': [{'name': 'X'}, {'name': 'Y'}]}, use_causal=False)

        # Force one coarse group to make drift-triggered shattering observable.
        engine.grouper.groups.clear()
        engine.grouper.var_to_group.clear()
        engine.grouper._create_group({'X', 'Y'})
        engine.grouper.split_threshold = 1.0

        stable_pressures = []
        drift_pressures = []

        # Stable phase: Y ~= X
        for t in range(120):
            x = float(t) / 50.0
            result = engine.process_row({'X': x, 'Y': x + 0.01})
            stable_pressures.append(float(result.get('drift_pressure', 0.0)))

        stable_mean = float(np.mean(stable_pressures[-60:]))
        # Set a calibrated threshold just above stable pressure so only true drift crosses it.
        engine.grouper.split_threshold = stable_mean + 0.03
        groups_before_drift = result['groups']

        # Drift phase: abrupt regime change, Y decouples and inverts.
        max_groups_during_drift = groups_before_drift
        for t in range(120, 240):
            x = float(t) / 50.0
            result = engine.process_row({'X': x, 'Y': -x + 5.0})
            drift_pressures.append(float(result.get('drift_pressure', 0.0)))
            max_groups_during_drift = max(max_groups_during_drift, int(result.get('groups', 0)))

        drift_mean = float(np.mean(drift_pressures[-60:]))

        assert drift_mean > stable_mean + 0.05
        assert groups_before_drift == 1
        assert max_groups_during_drift >= 2


class TestLifecycleAndArbitration:
    def test_meta_controller_promotes_tentative_to_active(self):
        pool = HypothesisPool()
        hyp = CorrelationalHypothesis('A', 'B')
        hyp.meta.state = HypothesisState.TENTATIVE
        hyp.evidence = 25
        hyp.confidence = 0.82
        hyp.stability = 0.79
        pool.add(hyp)

        controller = MetaController(confidence_threshold=0.7, stability_threshold=0.6, min_evidence=20)
        controller.manage_lifecycle(pool)

        assert hyp.meta.state == HypothesisState.ACTIVE

    def test_meta_controller_decays_then_kills_weak_hypothesis(self):
        pool = HypothesisPool()
        hyp = CorrelationalHypothesis('A', 'B')
        hyp.meta.state = HypothesisState.ACTIVE
        hyp.evidence = 50
        hyp.confidence = 0.4
        hyp.stability = 0.45
        pool.add(hyp)

        controller = MetaController(confidence_threshold=0.7, stability_threshold=0.6, min_evidence=20)
        controller.manage_lifecycle(pool)
        assert hyp.meta.state == HypothesisState.DECAYING

        # Force critical decay and run lifecycle again; hypothesis should be killed.
        hyp.confidence = 0.1
        hyp.stability = 0.2
        controller.manage_lifecycle(pool)

        assert hyp.meta.id not in pool.population
        assert len(pool.graveyard) >= 1

    def test_arbiter_prefers_causal_over_correlational_for_same_pair(self):
        arbiter = HypothesisArbiter()

        weak_corr = CorrelationalHypothesis('X', 'Y')
        weak_corr.confidence = 0.95
        strong_type = CausalHypothesis('X', 'Y', lag=2)
        strong_type.confidence = 0.40

        survivors = arbiter.arbitrate([weak_corr, strong_type])
        assert len(survivors) == 1
        assert survivors[0].rel_type.value == 'causal'

    def test_arbiter_detects_bidirectional_causality_conflict(self):
        arbiter = HypothesisArbiter()
        forward = CausalHypothesis('X', 'Y', lag=2)
        backward = CausalHypothesis('Y', 'X', lag=2)

        conflicts = arbiter.detect_conflicts([forward, backward])
        kinds = {c.get('conflict_type') for c in conflicts}
        assert 'bidirectional_causality' in kinds

    def test_pool_prune_weakest_removes_lowest_confidence(self):
        pool = HypothesisPool(capacity=5)
        h1 = CorrelationalHypothesis('A', 'B')
        h2 = CorrelationalHypothesis('A', 'C')
        h3 = CorrelationalHypothesis('B', 'C')
        h1.confidence = 0.9
        h2.confidence = 0.15
        h3.confidence = 0.6
        pool.add(h1)
        pool.add(h2)
        pool.add(h3)

        pool._prune_weakest(force=True)

        remaining = set(pool.population.keys())
        assert h2.meta.id not in remaining
        assert h1.meta.id in remaining and h3.meta.id in remaining

    def test_arbiter_collapses_mixed_type_interference_to_one_survivor(self):
        arbiter = HypothesisArbiter()
        corr = CorrelationalHypothesis('X', 'Y')
        causal = CausalHypothesis('X', 'Y', lag=2)
        reverse_causal = CausalHypothesis('Y', 'X', lag=2)

        survivors = arbiter.arbitrate([corr, causal, reverse_causal])
        assert len(survivors) == 1
        assert survivors[0].rel_type.value == 'causal'

    def test_arbiter_stable_under_large_mixed_pool(self):
        arbiter = HypothesisArbiter()
        vars_ = [f"V{i}" for i in range(12)]

        hypotheses = []
        for i in range(len(vars_)):
            for j in range(i + 1, len(vars_)):
                a = vars_[i]
                b = vars_[j]

                # Mix weaker and stronger claims on same pair.
                corr = CorrelationalHypothesis(a, b)
                corr.confidence = 0.95
                c1 = CausalHypothesis(a, b, lag=2)
                c1.confidence = 0.40
                c2 = CausalHypothesis(b, a, lag=2)
                c2.confidence = 0.45
                hypotheses.extend([corr, c1, c2])

        survivors = arbiter.arbitrate(hypotheses)

        # One survivor per undirected pair.
        expected_pairs = (len(vars_) * (len(vars_) - 1)) // 2
        assert len(survivors) == expected_pairs

        pair_keys = [tuple(sorted(h.variables)) for h in survivors]
        assert len(set(pair_keys)) == expected_pairs
        assert all(h.rel_type.value == 'causal' for h in survivors)

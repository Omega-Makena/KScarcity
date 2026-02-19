"""
Test: OnlineDiscoveryEngine with New Hypothesis Classes

Validates that the V2 initialization works and processes data correctly.
"""

import pytest
import numpy as np
from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
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

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class FederationEvaluator:
    def __init__(self, data: pd.DataFrame, num_nodes: int = 5):
        self.data = data
        self.num_nodes = num_nodes
        
    def evaluate_in_memory(self, target_variable: str, train_years: int) -> Dict[str, Any]:
        """
        Simulate an in-memory federation by splitting data and aggregating weights.
        """
        train_data = self.data.iloc[:train_years]
        test_data = self.data.iloc[train_years:]
        
        if len(train_data) == 0 or len(test_data) == 0:
            return {}
            
        # Split data equally among nodes
        chunks = np.array_split(train_data, self.num_nodes)
        
        local_weights = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            from sklearn.linear_model import Ridge
            chunk_filled = chunk.fillna(chunk.mean()).fillna(0)
            y = chunk_filled[target_variable].values[1:]
            X = chunk_filled.drop(columns=[target_variable]).values[:-1]
            if len(X) == 0:
                continue
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            local_weights.append(model.coef_)
            
        if not local_weights:
            return {}
            
        # Global aggregation (FedAvg)
        global_weights = np.mean(local_weights, axis=0)
        
        # Test performance
        test_filled = test_data.fillna(test_data.mean()).fillna(0)
        y_test = test_filled[target_variable].values[1:]
        X_test = test_filled.drop(columns=[target_variable]).values[:-1]

        if len(X_test) > 0:
            preds = X_test @ global_weights
            mse = np.mean((y_test - preds)**2)
        else:
            mse = np.nan
            
        return {
            'mode': 'in_memory',
            'nodes': self.num_nodes,
            'mse': float(mse),
            'communication_bytes': sum(w.nbytes for w in local_weights) * 2 # round trip
        }
        
    def evaluate_physical(self) -> Dict[str, Any]:
        """
        Uses the actual federated_databases infrastructure.
        """
        try:
            from federated_databases.scarcity_federation import get_scarcity_federation
            manager = get_scarcity_federation("benchmark/federation_runtime")
            
            # Setup nodes
            nodes = manager.list_nodes()
            if len(nodes) < self.num_nodes:
                for i in range(len(nodes), self.num_nodes):
                    manager.register_node(f"benchmark_node_{i}")
                    
            start_time = time.time()
            # Run a sync round
            try:
                res = manager.run_sync_round(model_name="logistic")
                sync_time = time.time() - start_time
                return {
                    'mode': 'physical',
                    'nodes': self.num_nodes,
                    'global_loss': res.global_loss,
                    'sync_time_seconds': sync_time,
                    'participants': res.participants
                }
            except RuntimeError as e:
                return {'mode': 'physical', 'error': str(e)}
        except ImportError:
            return {'mode': 'physical', 'error': 'federated_databases not found'}


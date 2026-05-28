import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from benchmark.evaluation.forecasting import ForecastingEvaluator

class RollingOriginBacktest:
    def __init__(self, data: pd.DataFrame, target_variables: List[str], initial_train_years: int = 15):
        self.data = data.sort_index()
        self.target_variables = target_variables
        self.initial_train_years = initial_train_years
        
    def run_backtest(self, scarcity_graph_over_time: Dict[int, Dict[str, list]]) -> pd.DataFrame:
        """
        scarcity_graph_over_time: dict mapping test_year -> {target: [parents]}
        """
        results = []
        years = self.data.index.values
        if len(years) <= self.initial_train_years:
            raise ValueError("Data length shorter than initial train years.")
            
        test_years = years[self.initial_train_years:]
        
        for test_year in test_years:
            train_data = self.data[self.data.index < test_year]
            test_data = self.data[self.data.index == test_year]
            
            # Use the graph discovered exactly using data < test_year
            # (which means no future leakage)
            current_graph = scarcity_graph_over_time.get(test_year, {})
            
            for target in self.target_variables:
                evaluator = ForecastingEvaluator(target_variable=target, horizon=1)
                
                # Persistence
                pers_res = evaluator.evaluate_persistence(train_data, test_data)
                
                # ARIMA
                arima_res = evaluator.evaluate_arima(train_data, test_data)
                
                # VAR
                var_res = evaluator.evaluate_var(train_data, test_data)
                
                # Prophet
                proph_res = evaluator.evaluate_prophet(train_data, test_data)
                
                # Scarcity Graph
                scarcity_res = evaluator.evaluate_scarcity_graph(train_data, test_data, current_graph)
                
                results.append({
                    'test_year': test_year,
                    'target': target,
                    'persistence_mae': pers_res['mae'],
                    'arima_mae': arima_res['mae'],
                    'var_mae': var_res['mae'],
                    'prophet_mae': proph_res['mae'],
                    'scarcity_mae': scarcity_res['mae'],
                    'persistence_dir': pers_res['dir_acc'],
                    'arima_dir': arima_res['dir_acc'],
                    'var_dir': var_res['dir_acc'],
                    'prophet_dir': proph_res['dir_acc'],
                    'scarcity_dir': scarcity_res['dir_acc'],
                })
                
        return pd.DataFrame(results)

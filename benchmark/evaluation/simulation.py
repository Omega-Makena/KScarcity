import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class HistoricalShockSimulator:
    """
    Evaluates observational consistency by propagating historical shocks through the 
    learned Scarcity graph and comparing with actual trajectories.
    """
    def __init__(self, data: pd.DataFrame, shock_definitions: List[Dict[str, Any]]):
        """
        shock_definitions format: 
        [{'year': 2008, 'variable': 'inflation_cpi', 'magnitude': 15.0, 'duration': 1}]
        """
        self.data = data.sort_index()
        self.shock_definitions = shock_definitions

    def evaluate_shocks(self, scarcity_graph_over_time: Dict[int, Dict[str, list]]) -> pd.DataFrame:
        results = []
        for shock in self.shock_definitions:
            year = shock['year']
            var = shock['variable']
            mag = shock['magnitude']
            
            # Learn graph up to T-1
            graph_T_minus_1 = scarcity_graph_over_time.get(year - 1, {})
            
            # Identify descendants of the shocked variable in the graph (1-step and 2-step)
            descendants = self._get_descendants(graph_T_minus_1, var)
            
            # Simulate response 
            # We compare actual observed difference vs simulated difference
            for target in descendants:
                actual_diff = self._get_actual_diff(target, year, year + 1)
                
                # Estimate simulation direction
                # For simplicity, if target is a direct child of var, we assume positive
                # correlation if the historical data has positive correlation, else negative
                simulated_dir = self._estimate_shock_direction(var, target, year - 1)
                
                dir_agreement = 1.0 if np.sign(actual_diff) == np.sign(simulated_dir * mag) else 0.0
                
                results.append({
                    'shock_year': year,
                    'source_variable': var,
                    'target_variable': target,
                    'actual_diff': actual_diff,
                    'directional_agreement': dir_agreement
                })
                
        return pd.DataFrame(results)
        
    def _get_descendants(self, graph: Dict[str, list], source: str) -> List[str]:
        descendants = set()
        for target, parents in graph.items():
            if source in parents:
                descendants.add(target)
                # 2-step
                for target2, parents2 in graph.items():
                    if target in parents2:
                        descendants.add(target2)
        return list(descendants)

    def _get_actual_diff(self, variable: str, year_start: int, year_end: int) -> float:
        try:
            v_start = self.data.loc[year_start, variable]
            v_end = self.data.loc[year_end, variable]
            return float(v_end - v_start)
        except KeyError:
            return np.nan

    def _estimate_shock_direction(self, source: str, target: str, year_end: int) -> float:
        """Estimate the coefficient direction from historical data up to year_end."""
        train = self.data[self.data.index <= year_end].dropna(subset=[source, target])
        if len(train) < 3:
            return 1.0
        corr = train[source].corr(train[target])
        return 1.0 if corr > 0 else -1.0


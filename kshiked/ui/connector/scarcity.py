"""
Scarcity Engine Connector.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .models import HypothesisData

logger = logging.getLogger("sentinel.connector.scarcity")

class ScarcityConnector:
    """Connect to Scarcity Discovery Engine."""
    
    def __init__(self):
        self._engine = None
        self._connected = False
        self._training_complete = False
        self._granger_cache: Optional[List[Dict[str, Any]]] = None
    
    def connect(self) -> bool:
        """Try to connect to scarcity engine."""
        try:
            from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
            self._engine = OnlineDiscoveryEngine(explore_interval=10)
            self._connected = True
            
            # Initialize and train on historical data if available
            self._train_on_historical_data()
            
            logger.info("Connected to Scarcity Engine and trained on historical data")
            return True
        except ImportError as e:
            logger.warning(f"Scarcity Engine not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing Scarcity Engine: {e}")
            return False
    
    def _train_on_historical_data(self):
        """Train the engine on historical Kenya economic data."""
        if not self._engine or self._training_complete:
            return

        try:
            # Load data
            try:
                from kshiked.ui.kenya_data_loader import get_kenya_data_loader
            except ImportError:
                from kenya_data_loader import get_kenya_data_loader
            
            loader = get_kenya_data_loader()
            
            # Broader macro set
            indicators = [
                # Real / prices / labor
                "gdp_current", "gdp_growth", "gdp_per_capita",
                "inflation", "inflation_gdp_deflator", "food_price_index",
                "unemployment", "employment_ratio",
                # External
                "exports_gdp", "imports_gdp", "trade_gdp", "current_account",
                # Fiscal
                "govt_consumption", "tax_revenue", "govt_debt",
                # Monetary / credit
                "real_interest_rate", "broad_money", "private_credit",
                # Social / infra proxies (economy-wide context)
                "population", "urban_population",
                "electricity_access", "internet_users", "mobile_subscriptions",
            ]

            # Map short indicator keys into human-readable labels
            display_name = {
                "gdp_current": "GDP",
                "gdp_growth": "GDP Growth",
                "gdp_per_capita": "GDP Per Capita",
                "inflation": "Inflation",
                "inflation_gdp_deflator": "GDP Deflator Inflation",
                "food_price_index": "Food Prices",
                "unemployment": "Unemployment",
                "employment_ratio": "Employment",
                "exports_gdp": "Exports",
                "imports_gdp": "Imports",
                "trade_gdp": "Trade",
                "current_account": "Current Account",
                "govt_consumption": "Gov Spending",
                "tax_revenue": "Taxes",
                "govt_debt": "Public Debt",
                "real_interest_rate": "Interest Rate",
                "broad_money": "Money Supply",
                "private_credit": "Credit Supply",
                "population": "Population",
                "urban_population": "Urban Population",
                "electricity_access": "Electricity Access",
                "internet_users": "Internet Users",
                "mobile_subscriptions": "Mobile Subscriptions",
            }
            
            # Get variable names
            var_names = [display_name.get(v, v) for v in indicators]
            
            # Initialize engine schema
            schema = {"fields": [{"name": v, "type": "float"} for v in var_names]}
            self._engine.initialize_v2(schema, use_causal=True)
            
            # Feed historical rows
            df = loader.get_historical_trajectory(indicators, start_year=1990)
            # Rename to match schema labels.
            df = df.rename(columns={k: display_name.get(k, k) for k in df.columns})
            
            count = 0
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Clean NaNs
                clean_row = {k: v for k, v in row_dict.items() if str(v) != 'nan'}
                if clean_row:
                    self._engine.process_row(clean_row)
                    count += 1
            
            self._training_complete = True
            self._granger_cache = None  # Invalidate cached causal results
            logger.info(f"Trained Scarcity Engine on {count} historical data points")
            
        except Exception as e:
            logger.error(f"Failed to train on historical data: {e}")
    
    def get_hypotheses(self, limit: int = 50) -> List[HypothesisData]:
        """Get top hypotheses from engine."""
        if not self._connected or not self._engine:
            return []
        
        try:
            # Get graph from engine
            graph = self._engine.get_knowledge_graph()
            
            # Convert to dashboard format
            results = []
            for item in graph:
                # Handle different dict structures from engine versions
                h_type = item.get('type', 'Unknown')
                vars_ = list(item.get('variables', []) or [])
                metrics = item.get('metrics', {}) or {}

                if h_type == "causal" and len(vars_) >= 2:
                    direction = int(metrics.get("direction", 1) or 1)
                    if direction == -1:
                        vars_ = [vars_[1], vars_[0]]
                
                # Filter out single-variable hypotheses if we have enough pairs
                if len(vars_) < 2 and len(graph) > 10:
                    continue
                    
                results.append(HypothesisData(
                    id=str(item.get('id', 'unknown')),
                    relationship_type=str(h_type).replace('Hypothesis', ''),
                    variables=vars_,
                    confidence=float(metrics.get('confidence', item.get('confidence', 0.0))),
                    fit_score=float(metrics.get('fit_score', item.get('fit_score', 0.0))),
                    state=str(item.get('state', 'active')),
                    created_at=float(item.get('created_at', datetime.now().timestamp())),
                ))
            
            if not results:
                return []
                
            return sorted(results, key=lambda x: x.confidence, reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching hypotheses: {e}")
            return []

    def get_granger_results(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Extract Granger-style causal results."""
        if not self._connected or not self._engine or not self._training_complete:
            return []

        if self._granger_cache is not None:
            return self._granger_cache[:limit]

        try:
            from scarcity.engine.discovery import RelationshipType  # type: ignore

            candidates = []
            for hyp in getattr(self._engine, "hypotheses", None).population.values():
                if getattr(hyp, "rel_type", None) != RelationshipType.CAUSAL:
                    continue
                if getattr(hyp, "direction", 0) == 0:
                    continue
                if getattr(hyp, "evidence", 0) < 8:
                    continue
                candidates.append(hyp)

            # Sort by engine confidence
            candidates.sort(key=lambda h: float(getattr(h, "confidence", 0.0)), reverse=True)

            results: List[Dict[str, Any]] = []
            for h in candidates[: max(limit, 100)]:
                src = getattr(h, "source", None)
                tgt = getattr(h, "target", None)
                if not src or not tgt:
                    continue

                direction = int(getattr(h, "direction", 0))
                cause, effect = (src, tgt) if direction == 1 else (tgt, src)

                gain_fwd = float(getattr(h, "gain_forward", 0.0))
                gain_bwd = float(getattr(h, "gain_backward", 0.0))
                strength = max(gain_fwd, gain_bwd)

                conf = float(getattr(h, "confidence", 0.0))
                lag = int(getattr(h, "lag", 2))

                results.append({
                    "cause": str(cause),
                    "effect": str(effect),
                    "lag": lag,
                    "f_stat": strength * 100.0,
                    "p_value": max(0.0, min(1.0, 1.0 - conf)),
                    "significant": conf >= 0.7,
                    "strength": strength,
                    "confidence": conf,
                })

            self._granger_cache = results
            return results[:limit]

        except Exception as e:
            logger.error(f"Error computing Granger results: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status metadata."""
        status = {
            "connected": self._connected,
            "training_complete": self._training_complete,
            "engine_type": "OnlineDiscoveryEngine" if self._engine else "None",
            "nodes": 0,
            "edges": 0,
            "hypotheses": 0
        }
        
        if self._engine and self._connected:
            try:
                graph = self._engine.get_knowledge_graph()
                status["nodes"] = len(set(
                    [n.get('id') for n in graph] if isinstance(graph, list) else []
                ))
                status["edges"] = len(graph)
                
                if hasattr(self._engine, "hypotheses") and hasattr(self._engine.hypotheses, "population"):
                    status["hypotheses"] = len(self._engine.hypotheses.population)
            except Exception:
                pass
                
        return status

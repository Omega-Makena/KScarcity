"""
Simulation Engine Connector.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from .models import SimulationState

logger = logging.getLogger("sentinel.connector.simulation")

class SimulationConnector:
    """Connect to Simulation Engine."""
    
    def __init__(self):
        self._sim = None
        self._connected = False
    
    def connect(self) -> bool:
        """Try to connect to simulation engine."""
        try:
            from scarcity.simulation.sfc import SFCEconomy, SFCConfig
            # Just test import
            self._connected = True
            logger.info("Connected to Simulation Engine (SFC)")
            return True
        except ImportError:
            logger.warning("Simulation Engine not available")
            return False
    
    def run_simulation(
        self, 
        shock_type: str, 
        magnitude: float, 
        policy_mode: str = "on"
    ) -> SimulationState:
        """Legacy Entry Point."""
        return self._run_legacy(shock_type, magnitude, policy_mode)

    def _run_legacy(self, shock_type, magnitude, policy_mode):
        if not self._connected:
            return self._get_demo_state()
            
        try:
            from scarcity.simulation.sfc import SFCEconomy, SFCConfig
            # NEW: Import Refactored Engine Components
            from kshiked.simulation.compiler import ShockCompiler
            from kshiked.simulation.controller import PolicyController
            from kshiked.core.shocks import ImpulseShock, OUProcessShock
            from kshiked.core.policies import default_economic_policies
            
            # 1. Compile Shocks (Dynamic / Stochastic)
            # Map legacy simplified inputs to advanced Shock objects
            shocks = []
            
            if shock_type == "demand_shock":
                 # Use OU Process for "realistic" volatility + impulse
                 s = ImpulseShock(name="Demand Hit", target_metric="demand", magnitude=magnitude)
                 shocks.append(s)
            elif shock_type == "supply_shock":
                 s = ImpulseShock(name="Supply Hit", target_metric="supply", magnitude=magnitude)
                 shocks.append(s)
            
            # Setup Compiler
            compiler = ShockCompiler(steps=50, seed=42)
            vectors = compiler.compile(shocks)
            
            # 2. Setup Config
            config = SFCConfig(
                steps=50,
                shock_vectors=vectors, # Use compiled vectors
                policy_mode="on" # Always on, Controller manages overrides
            )
            
            economy = SFCEconomy(config)
            economy.initialize()
            
            # 3. Setup Controller (The Brain)
            # Use default policies for now (Inflation targeting etc.)
            policies = default_economic_policies()
            controller = PolicyController(economy, policies)
            
            # 4. Run via Controller
            trajectory = controller.run(50)
            
            return self._wrap_result(trajectory, {"shock": shock_type, "mag": magnitude, "mode": policy_mode})
            
        except Exception as e:
            logger.error(f"Simulation run failed: {e}")
            return self._get_demo_state()

    # =========================================================
    # Professional Scenario Platform API
    # =========================================================
    
    def list_scenarios(self) -> List[Dict]:
        """List all saved scenarios."""
        try:
            from scarcity.simulation.scenario import ScenarioManager
            return ScenarioManager.list_scenarios()
        except ImportError:
            return []
            
    def load_scenario(self, scen_id: str) -> Optional[Any]:
        """Load a full scenario object."""
        try:
            from scarcity.simulation.scenario import ScenarioManager
            return ScenarioManager.load_scenario(scen_id)
        except ImportError:
            return None
            
    def save_scenario(self, scenario_data: Dict) -> str:
        """Create/Update a scenario from dict."""
        try:
            from scarcity.simulation.scenario import ScenarioManager, Scenario
            scen = Scenario.from_dict(scenario_data)
            path = ScenarioManager.save_scenario(scen)
            return scen.id
        except Exception as e:
            logger.error(f"Failed to save scenario: {e}")
            return ""

    def run_scenario_object(self, scenario: Any) -> SimulationState:
        """Run a Scenario object."""
        if not self._connected:
            return self._get_demo_state()
            
        try:
            from scarcity.simulation.sfc import SFCEconomy
            
            # Compile
            config = scenario.compile_to_config()
            
            # Run
            trajectory = SFCEconomy.run_scenario(config)
            
            return self._wrap_result(trajectory, {"scenario_id": scenario.id, "name": scenario.name})
            
        except Exception as e:
            logger.error(f"Scenario run failed: {e}")
            return self._get_demo_state()

    def _wrap_result(self, trajectory: List[Dict], meta: Dict) -> SimulationState:
        """Helper to wrap trajectory into SimulationState."""
        if not trajectory:
             return self._get_demo_state()
             
        latest = trajectory[-1]
        outcomes = latest.get("outcomes", {})
        
        # Try to get real baseline data
        baseline = self._get_real_baseline()
        
        # Calculate absolute values by applying simulation deltas to real baseline
        # Simulation often works in growth rates (e.g. 0.02 = 2%).
        # If baseline GDP is 100B, and sim outcome is 0.02:
        # We assume sim output 'gdp_growth' is total growth from t0.
        
        # Base values
        base_gdp_growth = baseline.get("gdp_growth", 0.0) / 100.0
        base_inf = baseline.get("inflation", 0.0)
        base_unemp = baseline.get("unemployment", 0.0)
        
        # Sim deltas (assuming sim returns absolute levels or deviations, here we treat as levels for simplicity or absolute rates)
        # Note: If sim returns 0.06 for inflation, that is 6%.
        
        sim_inf = outcomes.get("inflation", 0.0) * 100
        sim_unemp = outcomes.get("unemployment", 0.0) * 100
        
        return SimulationState(
            gdp=100.0 * (1.0 + outcomes.get("gdp_growth", 0.0)), # Index
            inflation=sim_inf if sim_inf != 0 else base_inf,
            unemployment=sim_unemp if sim_unemp != 0 else base_unemp,
            interest_rate=latest.get("policy_vector", {}).get("policy_rate", 0.0) * 100,
            exchange_rate=110.0,
            trajectory=trajectory,
            latest=latest,
            meta=meta
        )

    def get_state(self) -> SimulationState:
        """Get current/cached simulation state."""
        # For MVP, try to fetch real data state directly
        return self._get_demo_state()
        
    def _get_demo_state(self) -> SimulationState:
        """Get baseline state from real data or default to 0."""
        data = self._get_real_baseline()
        
        return SimulationState(
            gdp=data.get("gdp_current", 0.0) / 1e9, # Billions
            inflation=data.get("inflation", 0.0),
            unemployment=data.get("unemployment", 0.0),
            interest_rate=data.get("real_interest_rate", 0.0),
            exchange_rate=0.0,
            is_demo=True # EXPLICIT FLAG
        )

    def _get_real_baseline(self) -> Dict[str, float]:
        """Fetch latest real data."""
        try:
            from kshiked.ui.kenya_data_loader import get_latest_economic_state
            return get_latest_economic_state()
        except ImportError:
            try:
                from kenya_data_loader import get_latest_economic_state
                return get_latest_economic_state()
            except ImportError:
                return {}

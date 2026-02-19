"""
Governance logic for Kshield Economic Simulation (V4).
Phase 4: Tensor Engine & Event Bus Logic.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Imports
from scarcity.simulation.environment import SimulationEnvironment
from scarcity.governor.profiler import ResourceProfiler, ProfilerConfig
from scarcity.governor.monitor import DRGMonitor, MonitorConfig
from scarcity.runtime.bus import EventBus, get_bus

from .policies import default_economic_policies
from .tensor_policies import PolicyTensorEngine

logger = logging.getLogger("kshield.governance")

@dataclass
class Event:
    topic: str
    data: Dict

class SimSensor:
    def __init__(self, env: SimulationEnvironment):
        self.env = env
        
    def get_vector(self) -> Tuple[np.ndarray, List[str]]:
        # Case A: SimulationEnvironment (Real-time)
        if hasattr(self.env, 'state') and callable(self.env.state):
            state = self.env.state()
            return np.array(state.values, dtype=np.float32), state.node_ids
            
        # Case B: SimulationHandle (Dashboard Dict)
        if hasattr(self.env, 'state') and isinstance(self.env.state, dict):
            # Sort for deterministic vectorization
            keys = sorted(list(self.env.state.keys()))
            vals = [self.env.state[k] for k in keys]
            return np.array(vals, dtype=np.float32), keys
            
        return np.array([]), []

from scarcity.simulation.sfc import SFCEconomy, SFCConfig

class EventActuator:
    def __init__(self, env: SimulationEnvironment, sfc: SFCEconomy, bus: EventBus):
        self.env = env
        self.sfc = sfc
        self.bus = bus

    async def execute_signals(self, signals: Dict[str, float], metrics: Dict[str, float]):
        """
        Execute governance signals by modifying the underlying SFC Economy parameters.
        This replaces 'mock' logic with real structural adjustments.
        """
        # 1. Monetary Policy (Interest Rates)
        # Signal > 0: Tighten (Raise Rate). Signal < 0: Loosen (Lower Rate).
        hike_mag = signals.get("tighten_policy", 0) - signals.get("loosen_policy", 0)
        
        if abs(hike_mag) > 1e-4:
            # Transmission: Governor overrides Central Bank reaction function
            # We treat the signal as a bias to the Taylor Rule or direct intervention
            # 0.1 signal = 100bps change? Let's scale it.
            rate_delta = float(hike_mag) * 0.01 
            self.sfc.apply_shock("monetary", rate_delta)
            
            await self.bus.publish("monetary_policy_update", {
                "instrument": "interest_rate",
                "delta": rate_delta,
                "new_rate": self.sfc.interest_rate,
                "reason": "governance_consensus"
            })

        # 2. Fiscal Policy (Government Spending)
        # Signal > 0: Stimulus. Signal < 0: Austerity.
        stim_mag = signals.get("stimulus_package", 0) - signals.get("austerity", 0)
        
        if abs(stim_mag) > 1e-4:
            # Transmission: Modify Spending/GDP ratio
            # 1.0 signal = 1% increase in G/Y ratio
            fiscal_delta = float(stim_mag) * 0.01
            self.sfc.apply_shock("fiscal", fiscal_delta)
            
            await self.bus.publish("fiscal_policy_update", {
                "instrument": "spending_ratio",
                "delta": fiscal_delta,
                "deficit": self.sfc.government.net_lending,
                "reason": "governance_consensus"
            })

    def sync_to_environment(self):
        """
        Syncs the sophisticated SFC state back to the graph environment
        so the dashboard and graph-based tools see the effects.
        """
        state = self.sfc.get_state()
        
        # Helper to safely update simple graph nodes
        def _update(name, val):
            # Case A: SimulationEnvironment
            if hasattr(self.env, 'state') and callable(self.env.state):
                s = self.env.state()
                if name in s.node_ids:
                    idx = s.node_ids.index(name)
                    s.values[idx] = float(val)
            
            # Case B: SimulationHandle (Dashboard)
            elif hasattr(self.env, 'state') and isinstance(self.env.state, dict):
                self.env.state[name] = float(val)

        # Mapping SFC -> Knowledge Graph Nodes
        _update("GDP (current US$)", state['gdp'])
        _update("Inflation, consumer prices (annual %)", state['inflation'] * 100) # SFC uses 0.02, Graph uses %
        _update("Real interest rate (%)", (state['interest_rate'] - state['inflation']) * 100)
        _update("Unemployment, total (% of total labor force)", state['unemployment'] * 100)
        _update("Central government debt, total (% of GDP)", (state['government_debt'] / state['gdp']) * 100)
        # _update("Household Net Worth", state['household_net_worth'])

@dataclass
class EconomicGovernorConfig:
    control_interval: int = 1
    policies: Dict = field(default_factory=default_economic_policies)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

class EconomicGovernor:
    def __init__(self, config: EconomicGovernorConfig, env: SimulationEnvironment):
        self.config = config
        self.env = env
        self.bus = get_bus() 
        self.sensors = SimSensor(env)
        self.engine = PolicyTensorEngine()
        
        # Initialize Production-Grade SFC Model
        self.sfc = SFCEconomy()
        self.sfc.initialize(gdp=100.0)
        
        # Link Actuator
        self.actuator = EventActuator(env, self.sfc, self.bus)
        self._compiled = False
        
        logger.info("Economic Governor initialized with SFC Dynamics")

    def _ensure_compiled(self):
        if self._compiled: return
        vals, names = self.sensors.get_vector()
        self.engine.compile(self.config.policies, names)
        self._compiled = True

    async def step(self, current_tick: int):
        # 1. Run Economic Dynamics (Real physics)
        # We step the economy every tick, regardless of control interval
        self.sfc.step()
        
        # 2. Sync macro state to graph for visibility
        self.actuator.sync_to_environment()
        
        # 3. Governance Control Loop (Policy Regulation)
        if current_tick % self.config.control_interval != 0:
            return

        vals, names = self.sensors.get_vector()
        
        # Guard against empty state vector
        if len(vals) == 0 or len(names) == 0:
            logger.warning(f"Tick {current_tick}: SimSensor returned empty vector, skipping governance step")
            return
        
        # Ensure policy engine is compiled with current schema
        if not self._compiled:
            self.engine.compile(self.config.policies, names)
            self._compiled = True
        
        # Evaluate Policies (PID Control)
        # dt = interval because PID needs time delta
        action_signals = self.engine.evaluate(vals, dt=self.config.control_interval)
        
        if action_signals:
            logger.debug(f"Gov Tick {current_tick}: Control Signals {action_signals}")
        
        # 4. Actuation
        m_dict = dict(zip(names, vals))
        await self.actuator.execute_signals(action_signals, m_dict)

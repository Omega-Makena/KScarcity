"""
KShield Hub - Central Orchestrator

Unified interface that brings together:
1. Pulse Engine (Threat Detection)
2. Scarcity Engine (Full Architecture — Discovery, Meta-Learning, Governor, Simulation)
3. Governance/Policy Modules
4. 3D Manifold Visualizations

This module serves as the single source of truth for the Unified Dashboard.

K-SHIELD accesses ALL of scarcity through ScarcityBridge:
    - Discovery Engine: learn 306+ economic relationships from data
    - Meta-Learning: Reptile optimizer, global priors
    - Governor: resource stability
    - PolicySimulator: shock propagation through learned graph
    - LearnedSFCEconomy: SFC with learned (not hardcoded) relationships
"""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

# KShield Components
from kshiked.pulse import PulseSensor, compute_threat_report
from kshiked.pulse.simulation_connector import (
    KShieldEconomicBridge, 
    RealTimeSimulationConnector,
    ShockConfig
)
from kshiked.pulse.indices import ThreatIndexReport

# ScarcityBridge: Full access to all scarcity subsystems
try:
    from kshiked.core.scarcity_bridge import ScarcityBridge, TrainingReport
    from scarcity.simulation.learned_sfc import LearnedSFCEconomy, LearnedSFCConfig
    from scarcity.runtime import EventBus, get_bus
    HAS_SCARCITY = True
except ImportError:
    HAS_SCARCITY = False
    ScarcityBridge = None
    LearnedSFCEconomy = None
    _logger = logging.getLogger("kshield.hub")
    _logger.warning("Scarcity engine not found. Simulation features will be disabled.")

# Legacy fallback (kept for backwards compatibility)
try:
    from scarcity.simulation.engine import SimulationEngine, SimulationConfig
    from scarcity.simulation.agents import AgentRegistry
    HAS_LEGACY_SIM = True
except ImportError:
    HAS_LEGACY_SIM = False
    SimulationEngine = None


class KShieldHub:
    """
    Central hub unifying Pulse, Scarcity, and K-SHIELD.
    
    Acts as a singleton accessor for the running KShield instance.
    Access to the entire scarcity architecture is provided through
    self.bridge (ScarcityBridge).
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KShieldHub, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("kshield.hub")
        
        # 1. Initialize Pulse Engine
        self.pulse_sensor = PulseSensor()
        self.economic_bridge = KShieldEconomicBridge()
        self.last_threat_report: Optional[ThreatIndexReport] = None
        
        # 2. ScarcityBridge — full access to all scarcity subsystems
        self.bridge: Optional[ScarcityBridge] = None
        self.training_report: Optional[TrainingReport] = None
        self.learned_economy: Optional[LearnedSFCEconomy] = None

        # 3. Legacy simulation engine (fallback)
        self.sim_engine = None
        self.sim_bus = None
        
        if HAS_SCARCITY:
            self._init_scarcity_bridge()
        elif HAS_LEGACY_SIM:
            self._init_legacy_simulation()
            
        self._initialized = True
        self.logger.info("KShield Hub initialized")

    # -----------------------------------------------------------------
    # Initialization: ScarcityBridge (preferred) vs Legacy (fallback)
    # -----------------------------------------------------------------

    def _init_scarcity_bridge(self):
        """Initialize full scarcity access via ScarcityBridge."""
        try:
            self.bridge = ScarcityBridge()
            self.sim_bus = self.bridge.bus
            self.logger.info(
                "ScarcityBridge connected — "
                "discovery, meta-learning, governor accessible"
            )
        except Exception as e:
            self.logger.error(f"ScarcityBridge init failed: {e}")
            if HAS_LEGACY_SIM:
                self.logger.info("Falling back to legacy simulation engine")
                self._init_legacy_simulation()

    def _init_legacy_simulation(self):
        """Legacy: direct SimulationEngine initialization."""
        try:
            from scarcity.runtime import get_bus
            self.sim_bus = get_bus()
            config = SimulationConfig()
            registry = AgentRegistry()
            self.sim_engine = SimulationEngine(registry, config, self.sim_bus)
            self.logger.info("Legacy simulation engine linked")
        except Exception as e:
            self.logger.error(f"Legacy simulation init failed: {e}")

    # -----------------------------------------------------------------
    # Training: Feed historical data through discovery engine
    # -----------------------------------------------------------------

    def train(self, data_path: Optional[Path] = None) -> Optional[TrainingReport]:
        """
        Train the discovery engine on historical World Bank data.
        
        This teaches scarcity the economic relationships of Kenya
        (Phillips curve, Okun's law, etc.) from real data instead
        of hardcoding them.

        Args:
            data_path: Path to World Bank CSV. Defaults to project data path.

        Returns:
            TrainingReport or None if bridge unavailable.
        """
        if not self.bridge:
            self.logger.warning("ScarcityBridge not available — cannot train")
            return None

        self.training_report = self.bridge.train(data_path)
        self.logger.info(
            f"Training complete: {self.training_report.years_fed} years, "
            f"{self.training_report.hypotheses_created} hypotheses, "
            f"confidence: {self.training_report.overall_confidence:.1%}"
        )
        return self.training_report

    # -----------------------------------------------------------------
    # Simulation: Learned economy with confidence-weighted fallback
    # -----------------------------------------------------------------

    def create_learned_economy(self, sfc_config=None) -> Optional[LearnedSFCEconomy]:
        """
        Create a LearnedSFCEconomy — SFC backed by discovered relationships.
        
        Fallback blending is automatic: where scarcity's learned confidence
        is high, learned values dominate. Where low, parametric SFC fills in.

        Args:
            sfc_config: SFCConfig for parametric fallback and shock schedules.

        Returns:
            LearnedSFCEconomy instance or None.
        """
        if not self.bridge:
            self.logger.warning("No bridge — cannot create learned economy")
            return None

        if not self.bridge.trained:
            self.logger.info("Bridge not trained — training now...")
            self.train()

        self.learned_economy = LearnedSFCEconomy(self.bridge, sfc_config)
        self.learned_economy.initialize()
        return self.learned_economy

    # -----------------------------------------------------------------
    # Validation: Historical accuracy scoring
    # -----------------------------------------------------------------

    def validate(self, data_path: Optional[Path] = None):
        """
        Run data-driven validation: detect historical episodes,
        replay through simulation, score accuracy.

        Returns:
            ValidationReport or None.
        """
        if not self.bridge or not self.bridge.trained:
            self.logger.warning("Must train before validating")
            return None

        try:
            from kshiked.simulation.validation import ValidationRunner
            runner = ValidationRunner(self.bridge)
            return runner.validate(data_path)
        except ImportError:
            self.logger.warning("Validation module not available")
            return None

    # -----------------------------------------------------------------
    # Inspection: What did scarcity learn?
    # -----------------------------------------------------------------

    def get_top_relationships(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get the top-k strongest discovered economic relationships."""
        if self.bridge:
            return self.bridge.get_top_relationships(k)
        return []

    def get_confidence_map(self) -> Dict[str, float]:
        """Per-variable confidence (0–1). Higher = less fallback."""
        if self.bridge:
            return self.bridge.get_confidence_map()
        return {}

    def get_knowledge_graph(self) -> List[Dict[str, Any]]:
        """Full knowledge graph as edge list."""
        if self.bridge:
            return self.bridge.get_knowledge_graph()
        return []

    # -----------------------------------------------------------------
    # Pulse processing (existing)
    # -----------------------------------------------------------------

    async def _connect_bridge_to_sim(self) -> bool:
        """Callback for bridge connection."""
        return self.bridge is not None or self.sim_engine is not None

    async def _apply_shock_to_sim(self, shock) -> bool:
        """Callback to apply Pulse shock to simulation."""
        if not self.sim_bus:
            return False
            
        payload = {
            "type": shock.shock_type.name,
            "variable": shock.target_variable,
            "magnitude": shock.magnitude,
            "source": "KShield Pulse",
            "signals": shock.source_signals
        }
        
        await self.sim_bus.publish("simulation.shock", payload)
        self.logger.info(f"injected shock into simulation: {payload}")
        return True

    def process_pulse_update(self, text_input: Optional[str] = None):
        """
        Process new data through the Pulse pipeline.
        
        Args:
            text_input: Optional new text to process
        """
        if text_input:
            self.pulse_sensor.process_text(text_input)
            
        self.pulse_sensor.update_state()
        
        self.last_threat_report = compute_threat_report(
            self.pulse_sensor.state, 
            self.pulse_sensor._signal_history
        )
        
        if self.last_threat_report:
            self.economic_bridge.process_signals(
                self.last_threat_report, 
                self.pulse_sensor.state
            )
            
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Aggregate data for the unified dashboard."""
        
        data = {
            "pulse": {
                "threat_level": "UNKNOWN",
                "indices": {},
                "alerts": [],
                "recent_signals": []
            },
            "simulation": {
                "active": bool(self.bridge and self.bridge.trained),
                "mode": "learned" if self.bridge and self.bridge.trained else "legacy",
                "shocks_history": self.economic_bridge.connector.shock_generator.get_history(10),
            },
            "scarcity": {
                "bridge_available": self.bridge is not None,
                "trained": self.bridge.trained if self.bridge else False,
                "confidence": (self.training_report.overall_confidence 
                              if self.training_report else 0.0),
                "hypotheses": (self.training_report.hypotheses_created 
                              if self.training_report else 0),
            }
        }
        
        if self.last_threat_report:
            report = self.last_threat_report
            data["pulse"]["threat_level"] = report.overall_threat_level
            data["pulse"]["indices"] = report.to_dict()["indices"]
            data["pulse"]["alerts"] = report.priority_alerts
            
        return data


def get_hub() -> KShieldHub:
    """Get the singleton hub instance."""
    return KShieldHub()

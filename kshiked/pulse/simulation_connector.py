"""
KShield Simulation Connector

Connects Pulse Engine signals to the Economic Simulation Engine.

This module:
1. Translates Pulse threat indices into simulation shocks
2. Manages the signal → shock → simulation pipeline
3. Provides real-time shock streaming to the simulation
4. Tracks shock history for backtesting

The simulation engine receives:
- GDP shocks (from instability, elite fracture)
- Inflation shocks (from scarcity, panic)
- Trade shocks (from security friction, cascade)
- Currency shocks (from rumor velocity, external pressure)
- Confidence shocks (from legitimacy erosion, polarization)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import time

from .primitives import PulseState, SignalCategory, ActorType, ResourceDomain
from .mapper import SignalID, SignalDetection
from .indices import ThreatIndexReport, compute_threat_report
from .cooccurrence import RiskScore, RiskScorer
from .bridge import ShockType, ShockEvent, ShockMagnitudeCalculator, ShockScheduler, SchedulerConfig

logger = logging.getLogger("kshield.simulation_connector")


# =============================================================================
# Shock Mapping Configuration
# =============================================================================

@dataclass
class ShockConfig:
    """Configuration for signal → shock mapping."""
    
    # Thresholds for shock generation
    min_instability: float = 0.25
    min_crisis_probability: float = 0.20
    min_index_value: float = 0.30
    
    # Magnitude scaling
    gdp_coefficient: float = -0.05  # Negative = GDP decrease
    inflation_coefficient: float = 0.10
    trade_coefficient: float = -0.08
    currency_coefficient: float = -0.06
    confidence_coefficient: float = -0.10
    
    # Timing
    shock_duration_steps: int = 4
    decay_rate: float = 0.5
    
    # Simulation variable names (match your economic model)
    gdp_variable: str = "GDP (current US$)"
    inflation_variable: str = "Inflation, consumer prices (annual %)"
    trade_variable: str = "Exports of goods and services (BoP, current US$)"
    currency_variable: str = "Official exchange rate (LCU per US$, period average)"
    confidence_variable: str = "Consumer confidence index"


# =============================================================================
# Shock Generator
# =============================================================================

class SimulationShockGenerator:
    """
    Generates economic shocks from Pulse threat indices.
    
    Maps each threat index to specific economic variables:
    
    | Threat Index | Primary Shock | Secondary Shock |
    |--------------|---------------|-----------------|
    | Polarization | Confidence | Trade |
    | Legitimacy Erosion | Confidence | GDP |
    | Mobilization Readiness | GDP | Inflation |
    | Elite Cohesion | GDP | Trade |
    | Information Warfare | Inflation | Confidence |
    | Security Friction | GDP | Trade |
    | Economic Cascade | GDP, Inflation | Currency |
    | Ethnic Tension | Confidence | GDP |
    """
    
    def __init__(self, config: ShockConfig = None):
        self.config = config or ShockConfig()
        self.shock_history: List[ShockEvent] = []
        self._last_shock_time = 0.0
    
    def generate_shocks(
        self,
        threat_report: ThreatIndexReport,
        pulse_state: PulseState,
    ) -> List[ShockEvent]:
        """
        Generate economic shocks from threat report.
        
        Args:
            threat_report: Current threat index report
            pulse_state: Current Pulse engine state
            
        Returns:
            List of shock events for simulation
        """
        shocks = []
        now = time.time()
        
        # Check instability threshold
        if pulse_state.instability_index < self.config.min_instability:
            return shocks
        
        c = self.config
        
        # 1. Polarization → Confidence shock
        if threat_report.polarization.value >= c.min_index_value:
            shocks.append(self._create_shock(
                ShockType.CONFIDENCE_SHOCK,
                c.confidence_variable,
                threat_report.polarization.value * c.confidence_coefficient,
                source_signals=[SignalID.DEHUMANIZATION_LANGUAGE.value],
            ))
        
        # 2. Legitimacy Erosion → Confidence + GDP shock
        if threat_report.legitimacy_erosion.value >= c.min_index_value:
            shocks.append(self._create_shock(
                ShockType.CONFIDENCE_SHOCK,
                c.confidence_variable,
                threat_report.legitimacy_erosion.value * c.confidence_coefficient * 1.5,
                source_signals=[SignalID.LEGITIMACY_REJECTION.value],
            ))
            shocks.append(self._create_shock(
                ShockType.DEMAND_SHOCK,
                c.gdp_variable,
                threat_report.legitimacy_erosion.value * c.gdp_coefficient * 0.5,
                source_signals=[SignalID.LEGITIMACY_REJECTION.value],
            ))
        
        # 3. Mobilization → GDP + Inflation shock
        if threat_report.mobilization_readiness.value >= c.min_index_value:
            intensity = threat_report.mobilization_readiness.value
            shocks.append(self._create_shock(
                ShockType.DEMAND_SHOCK,
                c.gdp_variable,
                intensity * c.gdp_coefficient * 1.5,
                source_signals=[SignalID.MOBILIZATION_LANGUAGE.value],
            ))
            shocks.append(self._create_shock(
                ShockType.SUPPLY_SHOCK,
                c.inflation_variable,
                intensity * c.inflation_coefficient,
            ))
        
        # 4. Elite Cohesion (inverted - low cohesion = shock)
        elite_fragility = 1.0 - threat_report.elite_cohesion.value
        if elite_fragility >= c.min_index_value:
            shocks.append(self._create_shock(
                ShockType.DEMAND_SHOCK,
                c.gdp_variable,
                elite_fragility * c.gdp_coefficient,
                source_signals=[SignalID.ELITE_FRACTURE.value],
            ))
            shocks.append(self._create_shock(
                ShockType.TRADE_SHOCK,
                c.trade_variable,
                elite_fragility * c.trade_coefficient,
            ))
        
        # 5. Information Warfare → Inflation + Confidence
        if threat_report.information_warfare.value >= c.min_index_value:
            shocks.append(self._create_shock(
                ShockType.SUPPLY_SHOCK,
                c.inflation_variable,
                threat_report.information_warfare.value * c.inflation_coefficient * 1.2,
                source_signals=[SignalID.RUMOR_VELOCITY_PANIC.value],
            ))
        
        # 6. Security Friction → GDP + Trade
        if threat_report.security_friction.value >= c.min_index_value:
            shocks.append(self._create_shock(
                ShockType.DEMAND_SHOCK,
                c.gdp_variable,
                threat_report.security_friction.value * c.gdp_coefficient * 2.0,
                source_signals=[SignalID.SECURITY_FORCE_FRICTION.value],
            ))
        
        # 7. Economic Cascade → GDP, Inflation, Currency
        if threat_report.economic_cascade.value >= c.min_index_value:
            intensity = threat_report.economic_cascade.value
            shocks.append(self._create_shock(
                ShockType.DEMAND_SHOCK,
                c.gdp_variable,
                intensity * c.gdp_coefficient * 2.0,
                source_signals=[SignalID.ECONOMIC_CASCADE_FAILURE.value],
            ))
            shocks.append(self._create_shock(
                ShockType.SUPPLY_SHOCK,
                c.inflation_variable,
                intensity * c.inflation_coefficient * 1.5,
            ))
            shocks.append(self._create_shock(
                ShockType.CURRENCY_SHOCK,
                c.currency_variable,
                intensity * c.currency_coefficient,
            ))
        
        # 8. Ethnic Tension → Confidence + Trade
        if threat_report.ethnic_tension.avg_tension >= c.min_index_value:
            shocks.append(self._create_shock(
                ShockType.CONFIDENCE_SHOCK,
                c.confidence_variable,
                threat_report.ethnic_tension.avg_tension * c.confidence_coefficient,
                source_signals=[SignalID.ETHNO_REGIONAL_FRAMING.value],
            ))
        
        # Record history
        self.shock_history.extend(shocks)
        self._last_shock_time = now
        
        return shocks
    
    def _create_shock(
        self,
        shock_type: ShockType,
        variable: str,
        magnitude: float,
        source_signals: List[int] = None,
    ) -> ShockEvent:
        """Create a shock event."""
        return ShockEvent(
            shock_type=shock_type,
            target_variable=variable,
            magnitude=magnitude,
            duration_steps=self.config.shock_duration_steps,
            decay_rate=self.config.decay_rate,
            source_signals=source_signals or [],
            trigger_time=time.time(),
        )
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get recent shock history as dicts."""
        return [
            {
                "type": s.shock_type.name,
                "variable": s.target_variable,
                "magnitude": s.magnitude,
                "time": datetime.fromtimestamp(s.trigger_time).isoformat(),
            }
            for s in self.shock_history[-limit:]
        ]


# =============================================================================
# Simulation Engine Interface
# =============================================================================

class SimulationEngineInterface:
    """
    Interface to the KShield Economic Simulation Engine.
    
    This is the bridge between Pulse signals and the simulation.
    Can be subclassed to connect to specific simulation implementations.
    """
    
    def __init__(self):
        self.connected = False
        self.pending_shocks: List[ShockEvent] = []
        self.applied_shocks: List[ShockEvent] = []
        
    async def connect(self) -> bool:
        """Connect to the simulation engine."""
        # Override in subclass to connect to real simulation
        self.connected = True
        logger.info("Simulation engine connected")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from simulation engine."""
        self.connected = False
        logger.info("Simulation engine disconnected")
    
    async def apply_shock(self, shock: ShockEvent) -> bool:
        """
        Apply a shock to the simulation.
        
        Args:
            shock: Shock event to apply
            
        Returns:
            True if successfully applied
        """
        if not self.connected:
            logger.warning("Simulation not connected")
            return False
        
        # Mark as applied
        shock.applied = True
        self.applied_shocks.append(shock)
        
        logger.info(
            f"Applied shock: {shock.shock_type.name} to {shock.target_variable} "
            f"(magnitude: {shock.magnitude:.4f})"
        )
        
        return True
    
    async def apply_shocks(self, shocks: List[ShockEvent]) -> int:
        """Apply multiple shocks. Returns count of successfully applied."""
        applied = 0
        for shock in shocks:
            if await self.apply_shock(shock):
                applied += 1
        return applied
    
    def get_stats(self) -> Dict:
        """Get simulation stats."""
        return {
            "connected": self.connected,
            "pending": len(self.pending_shocks),
            "applied": len(self.applied_shocks),
            "last_shock": self.applied_shocks[-1].trigger_time if self.applied_shocks else None,
        }


# =============================================================================
# Real-Time Connector
# =============================================================================

class RealTimeSimulationConnector:
    """
    Real-time connector between Pulse Engine and Simulation.
    
    Continuously monitors threat indices and transmits shocks
    to the simulation engine when thresholds are crossed.
    """
    
    def __init__(
        self,
        shock_config: ShockConfig = None,
        scheduler_config: SchedulerConfig = None,
    ):
        self.shock_generator = SimulationShockGenerator(shock_config)
        self.scheduler = ShockScheduler(scheduler_config)
        self.engine = SimulationEngineInterface()
        
        self._running = False
        self._task = None
        
        # Callbacks
        self.on_shock_generated: Optional[Callable[[List[ShockEvent]], None]] = None
        self.on_shock_applied: Optional[Callable[[ShockEvent], None]] = None
    
    async def start(self) -> None:
        """Start the real-time connector."""
        self._running = True
        await self.engine.connect()
        logger.info("Real-time simulation connector started")
    
    async def stop(self) -> None:
        """Stop the connector."""
        self._running = False
        await self.engine.disconnect()
        logger.info("Real-time simulation connector stopped")
    
    async def process_threat_report(
        self,
        threat_report: ThreatIndexReport,
        pulse_state: PulseState,
    ) -> int:
        """
        Process a threat report and apply shocks.
        
        Args:
            threat_report: Current threat indices
            pulse_state: Current Pulse state
            
        Returns:
            Number of shocks applied
        """
        # Generate shocks
        shocks = self.shock_generator.generate_shocks(threat_report, pulse_state)
        
        if shocks and self.on_shock_generated:
            self.on_shock_generated(shocks)
        
        # Schedule
        self.scheduler.schedule(shocks)
        
        # Get due shocks
        due = self.scheduler.get_due_shocks()
        
        # Apply
        applied = 0
        for shock in due:
            if await self.engine.apply_shock(shock):
                applied += 1
                if self.on_shock_applied:
                    self.on_shock_applied(shock)
        
        return applied
    
    async def run_continuous(
        self,
        sensor,
        interval_seconds: float = 30.0,
    ) -> None:
        """
        Run continuous monitoring and shock transmission.
        
        Args:
            sensor: PulseSensor instance
            interval_seconds: Update interval
        """
        await self.start()
        
        try:
            while self._running:
                # Update sensor state
                sensor.update_state()
                
                # Compute threat report
                report = compute_threat_report(
                    sensor.state,
                    sensor._signal_history,
                )
                
                # Process and apply shocks
                applied = await self.process_threat_report(report, sensor.state)
                
                if applied > 0:
                    logger.info(f"Applied {applied} shocks to simulation")
                
                await asyncio.sleep(interval_seconds)
        
        finally:
            await self.stop()
    
    def get_stats(self) -> Dict:
        """Get connector statistics."""
        return {
            "running": self._running,
            "engine": self.engine.get_stats(),
            "scheduler": self.scheduler.get_stats(),
            "shock_history_count": len(self.shock_generator.shock_history),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_simulation_connector(
    shock_config: ShockConfig = None,
) -> RealTimeSimulationConnector:
    """Create a configured simulation connector."""
    return RealTimeSimulationConnector(shock_config=shock_config)


async def run_pulse_to_simulation(
    sensor,
    interval: float = 30.0,
) -> None:
    """
    Run the Pulse → Simulation pipeline.
    
    Args:
        sensor: Initialized PulseSensor
        interval: Update interval in seconds
    """
    connector = create_simulation_connector()
    await connector.run_continuous(sensor, interval)


# =============================================================================
# Integration with KShield Economic Model
# =============================================================================

class KShieldEconomicBridge:
    """
    Full integration bridge for KShield Economic Simulation.
    
    Combines:
    - Pulse threat indices
    - Shock generation
    - Economic variable updates
    - Terrain state management
    """
    
    def __init__(self, simulation_callback: Callable = None):
        """
        Args:
            simulation_callback: Function to call with shock updates.
                                 Signature: callback(variable: str, magnitude: float, metadata: dict)
        """
        self.connector = create_simulation_connector()
        self.simulation_callback = simulation_callback
        
        # Track economic state
        self.economic_state: Dict[str, float] = {
            "gdp_impact": 0.0,
            "inflation_impact": 0.0,
            "trade_impact": 0.0,
            "currency_impact": 0.0,
            "confidence_impact": 0.0,
        }
    
    def process_signals(
        self,
        threat_report: ThreatIndexReport,
        pulse_state: PulseState,
    ) -> Dict[str, float]:
        """
        Process signals and return economic impacts.
        
        Returns:
            Dict of variable → cumulative impact
        """
        shocks = self.connector.shock_generator.generate_shocks(
            threat_report, pulse_state
        )
        
        for shock in shocks:
            # Update cumulative impacts
            if "GDP" in shock.target_variable:
                self.economic_state["gdp_impact"] += shock.magnitude
            elif "Inflation" in shock.target_variable:
                self.economic_state["inflation_impact"] += shock.magnitude
            elif "Export" in shock.target_variable or "Trade" in shock.target_variable:
                self.economic_state["trade_impact"] += shock.magnitude
            elif "exchange" in shock.target_variable.lower():
                self.economic_state["currency_impact"] += shock.magnitude
            elif "confidence" in shock.target_variable.lower():
                self.economic_state["confidence_impact"] += shock.magnitude
            
            # Call simulation callback if provided
            if self.simulation_callback:
                self.simulation_callback(
                    shock.target_variable,
                    shock.magnitude,
                    {"shock_type": shock.shock_type.name},
                )
        
        return self.economic_state.copy()
    
    def get_terrain_state(self) -> Dict:
        """
        Get current terrain state for simulation.
        
        Returns:
            Dict compatible with KShield terrain model
        """
        return {
            "pulse_impacts": self.economic_state,
            "shock_history": self.connector.shock_generator.get_history(20),
            "last_update": datetime.now().isoformat(),
        }

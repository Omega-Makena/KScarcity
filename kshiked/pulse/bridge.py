"""
KShield Simulation Integration — Bridge between Pulse and Economic Simulation

Provides:
- Clean interface for Pulse -> Simulation shock transmission
- Probabilistic shock triggering based on risk scores
- Primitive -> Shock magnitude conversion
- Shock scheduling and throttling
- Event logging for audit trail

This module ensures Pulse never calls simulation directly - it only
provides shocks through this bridge layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto
import numpy as np

from .primitives import (
    PulseState,
    SignalCategory,
    ResourceDomain,
    ActorType,
)
from .cooccurrence import RiskScore, RiskScorer
from .sensor import PulseSensor

logger = logging.getLogger("kshield.pulse.bridge")


# =============================================================================
# Shock Types and Magnitude Mapping
# =============================================================================

class ShockType(Enum):
    """Types of shocks that can be generated."""
    DEMAND_SHOCK = auto()       # Consumer demand changes
    SUPPLY_SHOCK = auto()       # Production/supply disruption
    CONFIDENCE_SHOCK = auto()   # Investor/consumer confidence
    CURRENCY_SHOCK = auto()     # Exchange rate pressure
    TRADE_SHOCK = auto()        # Import/export disruption
    FISCAL_SHOCK = auto()       # Government spending/tax changes


@dataclass
class ShockEvent:
    """
    A shock event to be applied to the simulation.
    
    Decoupled from simulation internals - just describes
    what should happen.
    """
    shock_type: ShockType
    target_variable: str
    magnitude: float            # Percentage change (e.g., -0.05 = -5%)
    duration_steps: int = 1     # How many simulation steps the shock lasts
    decay_rate: float = 0.5     # How quickly shock effect decays
    
    # Source attribution
    source_category: Optional[SignalCategory] = None
    source_signals: List[int] = field(default_factory=list)
    
    # Timing
    trigger_time: float = 0.0
    applied: bool = False
    
    # Metadata
    risk_score: float = 0.0
    confidence: float = 0.5


# =============================================================================
# Shock Magnitude Mapping
# =============================================================================

@dataclass
class ShockMapping:
    """Maps risk categories to economic variable shocks."""
    
    # Variable -> (base_magnitude, scaling_factor)
    mappings: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.mappings:
            self.mappings = {
                # GDP impacts
                "GDP (current US$)": (-0.02, 1.5),
                "GDP growth (annual %)": (-0.01, 1.0),
                
                # Trade impacts
                "Exports of goods and services (BoP, current US$)": (-0.03, 1.2),
                "Imports of goods and services (BoP, current US$)": (-0.02, 1.0),
                
                # Price impacts  
                "Inflation, consumer prices (annual %)": (0.05, 2.0),
                
                # Investment impacts
                "Foreign direct investment, net inflows (BoP, current US$)": (-0.05, 1.5),
                
                # Debt impacts
                "External debt stocks, total (DOD, current US$)": (0.02, 0.8),
            }


class ShockMagnitudeCalculator:
    """
    Calculates shock magnitudes from Pulse state and risk scores.
    
    Implements the conversion: Risk Metrics -> Economic Shocks
    """
    
    # Category -> ShockType mapping
    CATEGORY_SHOCK_TYPES: Dict[SignalCategory, List[ShockType]] = {
        SignalCategory.DISTRESS: [ShockType.DEMAND_SHOCK, ShockType.CONFIDENCE_SHOCK],
        SignalCategory.ANGER: [ShockType.CONFIDENCE_SHOCK, ShockType.TRADE_SHOCK],
        SignalCategory.INSTITUTIONAL: [ShockType.SUPPLY_SHOCK, ShockType.FISCAL_SHOCK],
        SignalCategory.IDENTITY: [ShockType.DEMAND_SHOCK, ShockType.TRADE_SHOCK],
        SignalCategory.INFORMATION: [ShockType.CONFIDENCE_SHOCK, ShockType.CURRENCY_SHOCK],
    }
    
    def __init__(self, mapping: ShockMapping = None):
        self.mapping = mapping or ShockMapping()
        
        # Risk thresholds for shock triggering
        self.low_risk_threshold = 0.2
        self.medium_risk_threshold = 0.4
        self.high_risk_threshold = 0.6
        self.critical_threshold = 0.8
    
    def compute_shocks(
        self, 
        risk_score: RiskScore,
        pulse_state: PulseState,
    ) -> List[ShockEvent]:
        """
        Compute shock events from current risk state.
        
        Args:
            risk_score: Current risk assessment
            pulse_state: Current Pulse primitives state
            
        Returns:
            List of shock events to apply
        """
        shocks = []
        
        # Only generate shocks if risk exceeds threshold
        if risk_score.overall < self.low_risk_threshold:
            return shocks
        
        # Compute magnitude multiplier based on overall risk
        risk_multiplier = self._risk_multiplier(risk_score.overall)
        
        # Generate shocks per category
        for category in SignalCategory:
            category_risk = risk_score.by_category.get(category.name, 0.0)
            
            if category_risk < self.low_risk_threshold:
                continue
            
            # Get shock types for this category
            shock_types = self.CATEGORY_SHOCK_TYPES.get(category, [ShockType.DEMAND_SHOCK])
            
            # Generate shocks for each affected variable
            for variable, (base_mag, scale) in self.mapping.mappings.items():
                magnitude = base_mag * category_risk * risk_multiplier * scale
                
                if abs(magnitude) < 0.001:  # Skip negligible shocks
                    continue
                
                shocks.append(ShockEvent(
                    shock_type=shock_types[0] if shock_types else ShockType.DEMAND_SHOCK,
                    target_variable=variable,
                    magnitude=magnitude,
                    duration_steps=self._shock_duration(risk_score.overall),
                    decay_rate=0.5,
                    source_category=category,
                    risk_score=category_risk,
                    confidence=min(0.9, 0.5 + category_risk),
                    trigger_time=time.time(),
                ))
        
        return shocks
    
    def _risk_multiplier(self, overall_risk: float) -> float:
        """Non-linear risk multiplier for shock magnitude."""
        if overall_risk < self.medium_risk_threshold:
            return 0.5
        elif overall_risk < self.high_risk_threshold:
            return 1.0
        elif overall_risk < self.critical_threshold:
            return 1.5
        else:
            return 2.5  # Crisis mode
    
    def _shock_duration(self, overall_risk: float) -> int:
        """Determine shock duration based on risk level."""
        if overall_risk < self.medium_risk_threshold:
            return 1
        elif overall_risk < self.high_risk_threshold:
            return 2
        else:
            return 3


# =============================================================================
# Shock Scheduler
# =============================================================================

@dataclass
class SchedulerConfig:
    """Configuration for shock scheduling."""
    min_interval_seconds: float = 300       # Min time between shocks
    max_pending_shocks: int = 10            # Max shocks in queue
    probability_scale: float = 1.0          # Scale for probabilistic triggering
    enable_probabilistic: bool = True       # Use probabilistic triggering


class ShockScheduler:
    """
    Schedules and throttles shock application.
    
    Prevents shock spam while ensuring high-risk situations
    get appropriate responses.
    """
    
    def __init__(self, config: SchedulerConfig = None):
        self.config = config or SchedulerConfig()
        self._pending_shocks: List[ShockEvent] = []
        self._last_shock_time: float = 0.0
        self._applied_count: int = 0
        self._rng = np.random.default_rng()
    
    def schedule(self, shocks: List[ShockEvent]) -> None:
        """Add shocks to the pending queue."""
        for shock in shocks:
            if len(self._pending_shocks) < self.config.max_pending_shocks:
                self._pending_shocks.append(shock)
            else:
                # Replace lowest priority if new shock is higher
                self._pending_shocks.sort(key=lambda s: s.risk_score)
                if shock.risk_score > self._pending_shocks[0].risk_score:
                    self._pending_shocks[0] = shock
    
    def get_due_shocks(self, now: float = None) -> List[ShockEvent]:
        """
        Get shocks that are due for application.
        
        Applies throttling and probabilistic filtering.
        """
        now = now or time.time()
        
        # Check throttling
        if now - self._last_shock_time < self.config.min_interval_seconds:
            return []
        
        if not self._pending_shocks:
            return []
        
        # Get highest priority shocks
        self._pending_shocks.sort(key=lambda s: s.risk_score, reverse=True)
        
        due = []
        for shock in self._pending_shocks[:]:
            # Probabilistic filtering
            if self.config.enable_probabilistic:
                trigger_prob = shock.risk_score * self.config.probability_scale
                if self._rng.random() > trigger_prob:
                    continue
            
            due.append(shock)
            self._pending_shocks.remove(shock)
            shock.applied = True
        
        if due:
            self._last_shock_time = now
            self._applied_count += len(due)
        
        return due
    
    def clear(self) -> None:
        """Clear pending shocks."""
        self._pending_shocks.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "pending_count": len(self._pending_shocks),
            "applied_count": self._applied_count,
            "last_shock_time": self._last_shock_time,
        }


# =============================================================================
# Simulation Bridge
# =============================================================================

# Type for shock handler callback
ShockHandler = Callable[[ShockEvent], None]


class SimulationBridge:
    """
    Main bridge between Pulse Engine and economic simulation.
    
    This is the ONLY interface through which Pulse affects simulation.
    Ensures clean decoupling as per architecture requirements.
    """
    
    def __init__(
        self,
        sensor: PulseSensor = None,
        scorer: RiskScorer = None,
        calculator: ShockMagnitudeCalculator = None,
        scheduler: ShockScheduler = None,
    ):
        self.sensor = sensor
        self.scorer = scorer or RiskScorer()
        self.calculator = calculator or ShockMagnitudeCalculator()
        self.scheduler = scheduler or ShockScheduler()
        
        # Registered shock handlers (simulation callbacks)
        self._handlers: List[ShockHandler] = []
        
        # Event log for audit
        self._event_log: List[Dict[str, Any]] = []
        self._max_log_size = 1000
        
        # Running state
        self._running = False
    
    def register_handler(self, handler: ShockHandler) -> None:
        """
        Register a shock handler callback.
        
        The simulation layer registers its shock application function here.
        """
        self._handlers.append(handler)
        logger.info(f"Registered shock handler: {handler.__name__ if hasattr(handler, '__name__') else 'anonymous'}")
    
    def process_cycle(self) -> List[ShockEvent]:
        """
        Run one processing cycle.
        
        1. Update sensor state
        2. Compute risk score
        3. Calculate shocks
        4. Schedule and apply due shocks
        
        Returns:
            List of applied shocks
        """
        # Update sensor state
        if self.sensor:
            self.sensor.update_state()
            pulse_state = self.sensor.state
        else:
            pulse_state = PulseState()
        
        # Compute risk
        risk_score = self.scorer.compute()
        
        # Calculate shocks
        shocks = self.calculator.compute_shocks(risk_score, pulse_state)
        
        # Schedule
        if shocks:
            self.scheduler.schedule(shocks)
        
        # Get due shocks
        due_shocks = self.scheduler.get_due_shocks()
        
        # Apply via handlers
        for shock in due_shocks:
            self._apply_shock(shock)
        
        return due_shocks
    
    def _apply_shock(self, shock: ShockEvent) -> None:
        """Apply a shock through registered handlers."""
        for handler in self._handlers:
            try:
                handler(shock)
            except Exception as e:
                logger.error(f"Shock handler error: {e}")
        
        # Log event
        self._log_event({
            "type": "shock_applied",
            "shock_type": shock.shock_type.name,
            "target": shock.target_variable,
            "magnitude": shock.magnitude,
            "risk_score": shock.risk_score,
            "source_category": shock.source_category.name if shock.source_category else None,
            "timestamp": shock.trigger_time,
        })
    
    def _log_event(self, event: Dict[str, Any]) -> None:
        """Add event to log with size management."""
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size // 2:]
    
    async def run_continuous(self, interval_seconds: float = 60.0) -> None:
        """
        Run continuous processing loop.
        
        Args:
            interval_seconds: Time between processing cycles
        """
        self._running = True
        logger.info(f"Starting continuous bridge processing (interval={interval_seconds}s)")
        
        while self._running:
            try:
                shocks = self.process_cycle()
                if shocks:
                    logger.info(f"Applied {len(shocks)} shocks")
            except Exception as e:
                logger.error(f"Bridge cycle error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop(self) -> None:
        """Stop continuous processing."""
        self._running = False
    
    def get_event_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events from log."""
        return self._event_log[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "scheduler": self.scheduler.get_stats(),
            "handlers_registered": len(self._handlers),
            "event_log_size": len(self._event_log),
            "running": self._running,
        }


# =============================================================================
# KShield Integration Helper
# =============================================================================

def create_kshield_bridge(
    sensor: PulseSensor = None,
    use_nlp: bool = True,
) -> Tuple[SimulationBridge, PulseSensor]:
    """
    Factory function to create a fully configured Pulse-KShield bridge.
    
    Args:
        sensor: Existing sensor (creates new if None)
        use_nlp: Use NLP detectors
        
    Returns:
        Tuple of (bridge, sensor)
    """
    if sensor is None:
        sensor = PulseSensor(use_nlp=use_nlp)
    
    scorer = RiskScorer()
    calculator = ShockMagnitudeCalculator()
    scheduler = ShockScheduler()
    
    bridge = SimulationBridge(
        sensor=sensor,
        scorer=scorer,
        calculator=calculator,
        scheduler=scheduler,
    )
    
    return bridge, sensor


# =============================================================================
# Example Shock Handler for KShield BacktestEngine
# =============================================================================

def create_backtest_handler(env_state) -> ShockHandler:
    """
    Create a shock handler for BacktestEngine's environment state.
    
    Args:
        env_state: The simulation environment state object
        
    Returns:
        Handler function that applies shocks to env_state
    """
    def handler(shock: ShockEvent) -> None:
        """Apply shock to simulation state."""
        # Build node map
        if not hasattr(env_state, 'node_ids') or not hasattr(env_state, 'values'):
            logger.warning("Invalid env_state for shock application")
            return
        
        node_map = {name: i for i, name in enumerate(env_state.node_ids)}
        
        if shock.target_variable in node_map:
            idx = node_map[shock.target_variable]
            current_val = env_state.values[idx]
            delta = current_val * shock.magnitude
            env_state.values[idx] += delta
            
            logger.info(
                f"Applied {shock.shock_type.name} to {shock.target_variable}: "
                f"{shock.magnitude:+.2%} ({delta/1e9:+.2f}B)"
            )
    
    return handler

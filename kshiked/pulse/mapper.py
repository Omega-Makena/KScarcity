"""
Signal Mapper — Maps 15 Intelligence Signals to Pulse Primitives

This module defines the mapping logic between detected social signals
and the four KShield primitives (Scarcity, Stress, Bonds, Propagation).

Each signal has a dedicated handler that computes the appropriate updates
to primitives based on signal intensity and context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from enum import IntEnum

from .primitives import (
    PulseState,
    ScarcityUpdate,
    StressUpdate,
    BondUpdate,
    SignalCategory,
    ActorType,
    ResourceDomain,
)

logger = logging.getLogger("kshield.pulse.mapper")


# =============================================================================
# Signal Definitions (1-15)
# =============================================================================

class SignalID(IntEnum):
    """The 15 Intelligence Signals."""
    # Distress Signals (1-3)
    SURVIVAL_COST_STRESS = 1
    DISTRESS_FRAMING = 2
    EMOTIONAL_EXHAUSTION = 3
    
    # Anger & Delegitimization (4-7)
    DIRECTED_RAGE = 4
    ROTATING_REGIME_SLANG = 5
    DEHUMANIZATION_LANGUAGE = 6
    LEGITIMACY_REJECTION = 7
    
    # Institutional Friction (8-10)
    SECURITY_FORCE_FRICTION = 8
    ECONOMIC_CASCADE_FAILURE = 9
    ELITE_FRACTURE = 10
    
    # Identity & Mobilization (11-13)
    ETHNO_REGIONAL_FRAMING = 11
    MOBILIZATION_LANGUAGE = 12
    COORDINATION_INFRASTRUCTURE = 13
    
    # Information Warfare (14-15)
    RUMOR_VELOCITY_PANIC = 14
    COUNTER_NARRATIVE_ACTIVATION = 15


@dataclass
class SignalDetection:
    """Result from a signal detector."""
    signal_id: SignalID
    intensity: float        # [0, 1] normalized
    confidence: float       # [0, 1] detector confidence
    raw_score: float        # Unnormalized score
    context: Dict = None    # Additional context (entities, etc.)
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


# =============================================================================
# Signal Category Mapping
# =============================================================================

SIGNAL_CATEGORIES: Dict[SignalID, SignalCategory] = {
    SignalID.SURVIVAL_COST_STRESS: SignalCategory.DISTRESS,
    SignalID.DISTRESS_FRAMING: SignalCategory.DISTRESS,
    SignalID.EMOTIONAL_EXHAUSTION: SignalCategory.DISTRESS,
    
    SignalID.DIRECTED_RAGE: SignalCategory.ANGER,
    SignalID.ROTATING_REGIME_SLANG: SignalCategory.ANGER,
    SignalID.DEHUMANIZATION_LANGUAGE: SignalCategory.ANGER,
    SignalID.LEGITIMACY_REJECTION: SignalCategory.ANGER,
    
    SignalID.SECURITY_FORCE_FRICTION: SignalCategory.INSTITUTIONAL,
    SignalID.ECONOMIC_CASCADE_FAILURE: SignalCategory.INSTITUTIONAL,
    SignalID.ELITE_FRACTURE: SignalCategory.INSTITUTIONAL,
    
    SignalID.ETHNO_REGIONAL_FRAMING: SignalCategory.IDENTITY,
    SignalID.MOBILIZATION_LANGUAGE: SignalCategory.IDENTITY,
    SignalID.COORDINATION_INFRASTRUCTURE: SignalCategory.IDENTITY,
    
    SignalID.RUMOR_VELOCITY_PANIC: SignalCategory.INFORMATION,
    SignalID.COUNTER_NARRATIVE_ACTIVATION: SignalCategory.INFORMATION,
}


# =============================================================================
# Handler Functions — One per Signal
# =============================================================================

def _handle_survival_cost_stress(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 1: Survival Cost Stress
    
    Indicators: Complaints about food prices, fuel costs, rent, job loss.
    Affects: Scarcity (food, fuel, housing), Population stress.
    """
    updates = []
    intensity = detection.intensity
    
    # Increase scarcity perception
    updates.append(ScarcityUpdate(
        domain=ResourceDomain.FOOD,
        delta=0.15 * intensity,
        source_signal=detection.signal_id,
        confidence=detection.confidence
    ))
    updates.append(ScarcityUpdate(
        domain=ResourceDomain.FUEL,
        delta=0.12 * intensity,
        source_signal=detection.signal_id,
        confidence=detection.confidence
    ))
    
    # Increase population stress
    updates.append(StressUpdate(
        actor=ActorType.POPULATION,
        delta=-0.10 * intensity,  # Negative = under pressure
        source_signal=detection.signal_id,
        reason="survival_cost_stress"
    ))
    
    return updates


def _handle_distress_framing(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 2: Distress Framing
    
    Indicators: "We're suffering", "People are dying", crisis framing.
    Affects: Scarcity (general), Population stress, State pressure.
    """
    updates = []
    intensity = detection.intensity
    
    # General scarcity increase across domains
    for domain in [ResourceDomain.FOOD, ResourceDomain.HEALTHCARE, ResourceDomain.CURRENCY]:
        updates.append(ScarcityUpdate(
            domain=domain,
            delta=0.08 * intensity,
            source_signal=detection.signal_id,
            confidence=detection.confidence
        ))
    
    # State under pressure (blamed for suffering)
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.08 * intensity,
        source_signal=detection.signal_id,
        reason="blamed_for_distress"
    ))
    
    return updates


def _handle_emotional_exhaustion(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 3: Emotional Exhaustion
    
    Indicators: Hopelessness, fatigue with the system, "tired of this".
    Affects: Population stress, Bond weakening.
    """
    updates = []
    intensity = detection.intensity
    
    # Deep population stress
    updates.append(StressUpdate(
        actor=ActorType.POPULATION,
        delta=-0.15 * intensity,
        source_signal=detection.signal_id,
        reason="emotional_exhaustion"
    ))
    
    # Weakens national cohesion (disillusionment)
    updates.append(BondUpdate(
        bond_type="national",
        delta=-0.05 * intensity,
        source_signal=detection.signal_id,
        reason="disillusionment"
    ))
    
    return updates


def _handle_directed_rage(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 4: Directed Rage
    
    Indicators: Specific anger at leaders, named targets.
    Affects: State/Elite stress, Actor friction.
    """
    updates = []
    intensity = detection.intensity
    
    # State/Elite under attack
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.12 * intensity,
        source_signal=detection.signal_id,
        reason="directed_rage"
    ))
    updates.append(StressUpdate(
        actor=ActorType.ELITE,
        delta=-0.10 * intensity,
        source_signal=detection.signal_id,
        reason="directed_rage"
    ))
    
    return updates


def _handle_rotating_regime_slang(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 5: Rotating Regime Slang
    
    Indicators: Coded language, mockery of regime (e.g., nicknames, hashtags).
    Affects: State legitimacy stress.
    """
    updates = []
    intensity = detection.intensity
    
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.08 * intensity,
        source_signal=detection.signal_id,
        reason="delegitimizing_slang"
    ))
    
    return updates


def _handle_dehumanization_language(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 6: Dehumanization Language
    
    Indicators: Calling groups "cockroaches", "snakes", etc.
    Affects: Ethnic bonds, Polarization.
    """
    updates = []
    intensity = detection.intensity
    
    # Weakens ethnic bonds
    target_group = detection.context.get("target_group", "generic")
    updates.append(BondUpdate(
        bond_type=f"ethnic_{target_group}",
        delta=-0.15 * intensity,
        source_signal=detection.signal_id,
        reason="dehumanization"
    ))
    
    # Increases polarization (via state.bonds.polarization_index)
    updates.append(BondUpdate(
        bond_type="national",
        delta=-0.10 * intensity,
        source_signal=detection.signal_id,
        reason="polarization"
    ))
    
    return updates


def _handle_legitimacy_rejection(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 7: Legitimacy Rejection
    
    Indicators: "Fake election", "Corrupt system", rejecting authority.
    Affects: State stress, National cohesion.
    """
    updates = []
    intensity = detection.intensity
    
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.15 * intensity,
        source_signal=detection.signal_id,
        reason="legitimacy_rejection"
    ))
    
    updates.append(BondUpdate(
        bond_type="national",
        delta=-0.08 * intensity,
        source_signal=detection.signal_id,
        reason="rejection_of_social_contract"
    ))
    
    return updates


def _handle_security_force_friction(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 8: Security Force Friction
    
    Indicators: Police brutality reports, military dissent.
    Affects: Security actor stress, State-Population friction.
    """
    updates = []
    intensity = detection.intensity
    
    updates.append(StressUpdate(
        actor=ActorType.SECURITY,
        delta=-0.10 * intensity,
        source_signal=detection.signal_id,
        reason="security_friction"
    ))
    
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.08 * intensity,
        source_signal=detection.signal_id,
        reason="security_crisis"
    ))
    
    return updates


def _handle_economic_cascade_failure(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 9: Economic Cascade Failure
    
    Indicators: Bank runs, currency collapse fears, business closures.
    Affects: All scarcity domains, Elite stress.
    """
    updates = []
    intensity = detection.intensity
    
    # Broad scarcity spike
    for domain in ResourceDomain:
        updates.append(ScarcityUpdate(
            domain=domain,
            delta=0.12 * intensity,
            source_signal=detection.signal_id,
            confidence=detection.confidence
        ))
    
    updates.append(StressUpdate(
        actor=ActorType.ELITE,
        delta=-0.15 * intensity,
        source_signal=detection.signal_id,
        reason="economic_cascade"
    ))
    
    return updates


def _handle_elite_fracture(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 10: Elite Fracture
    
    Indicators: Public disagreements among elites, defections.
    Affects: Elite stress, State stability.
    """
    updates = []
    intensity = detection.intensity
    
    updates.append(StressUpdate(
        actor=ActorType.ELITE,
        delta=-0.20 * intensity,
        source_signal=detection.signal_id,
        reason="elite_fracture"
    ))
    
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.10 * intensity,
        source_signal=detection.signal_id,
        reason="elite_defection"
    ))
    
    return updates


def _handle_ethno_regional_framing(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 11: Ethno-Regional Framing
    
    Indicators: "Our people", tribal/regional identity language.
    Affects: Ethnic bonds, Regional unity.
    """
    updates = []
    intensity = detection.intensity
    
    # Strengthens in-group but weakens cross-group bonds
    updates.append(BondUpdate(
        bond_type="regional",
        delta=-0.10 * intensity,
        source_signal=detection.signal_id,
        reason="ethno_regional_division"
    ))
    
    updates.append(BondUpdate(
        bond_type="national",
        delta=-0.08 * intensity,
        source_signal=detection.signal_id,
        reason="ethnic_fragmentation"
    ))
    
    return updates


def _handle_mobilization_language(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 12: Mobilization Language
    
    Indicators: "Rise up", "Take to the streets", call-to-action.
    Affects: State stress (facing action), Opposition empowerment.
    """
    updates = []
    intensity = detection.intensity
    
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.15 * intensity,
        source_signal=detection.signal_id,
        reason="mobilization_pressure"
    ))
    
    updates.append(StressUpdate(
        actor=ActorType.OPPOSITION,
        delta=0.10 * intensity,  # Positive = empowered
        source_signal=detection.signal_id,
        reason="mobilized"
    ))
    
    return updates


def _handle_coordination_infrastructure(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 13: Coordination Infrastructure
    
    Indicators: Telegram groups growing, protest logistics shared.
    Affects: Opposition empowerment, State threat.
    """
    updates = []
    intensity = detection.intensity
    
    updates.append(StressUpdate(
        actor=ActorType.OPPOSITION,
        delta=0.15 * intensity,
        source_signal=detection.signal_id,
        reason="coordination_capacity"
    ))
    
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.08 * intensity,
        source_signal=detection.signal_id,
        reason="organized_threat"
    ))
    
    return updates


def _handle_rumor_velocity_panic(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 14: Rumor Velocity & Panic
    
    Indicators: Rapid spread of unverified claims, panic buying.
    Affects: Scarcity (perceived), Population stress.
    """
    updates = []
    intensity = detection.intensity
    
    # Perceived scarcity spike
    for domain in [ResourceDomain.FOOD, ResourceDomain.FUEL, ResourceDomain.CURRENCY]:
        updates.append(ScarcityUpdate(
            domain=domain,
            delta=0.20 * intensity,
            source_signal=detection.signal_id,
            confidence=detection.confidence * 0.5  # Lower confidence for rumors
        ))
    
    updates.append(StressUpdate(
        actor=ActorType.POPULATION,
        delta=-0.12 * intensity,
        source_signal=detection.signal_id,
        reason="panic"
    ))
    
    return updates


def _handle_counter_narrative_activation(
    detection: SignalDetection, 
    state: PulseState
) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
    """
    Signal 15: Counter-Narrative Activation
    
    Indicators: Government propaganda pushback, competing narratives.
    Affects: State stress (legitimacy battle), Polarization.
    """
    updates = []
    intensity = detection.intensity
    
    # Information warfare increases polarization
    updates.append(BondUpdate(
        bond_type="national",
        delta=-0.05 * intensity,
        source_signal=detection.signal_id,
        reason="narrative_warfare"
    ))
    
    # Both state and opposition under pressure
    updates.append(StressUpdate(
        actor=ActorType.STATE,
        delta=-0.05 * intensity,
        source_signal=detection.signal_id,
        reason="counter_narrative"
    ))
    
    return updates


# =============================================================================
# Signal Mapper — Master Registry
# =============================================================================

SignalHandler = Callable[[SignalDetection, PulseState], List[ScarcityUpdate | StressUpdate | BondUpdate]]

SIGNAL_HANDLERS: Dict[SignalID, SignalHandler] = {
    SignalID.SURVIVAL_COST_STRESS: _handle_survival_cost_stress,
    SignalID.DISTRESS_FRAMING: _handle_distress_framing,
    SignalID.EMOTIONAL_EXHAUSTION: _handle_emotional_exhaustion,
    SignalID.DIRECTED_RAGE: _handle_directed_rage,
    SignalID.ROTATING_REGIME_SLANG: _handle_rotating_regime_slang,
    SignalID.DEHUMANIZATION_LANGUAGE: _handle_dehumanization_language,
    SignalID.LEGITIMACY_REJECTION: _handle_legitimacy_rejection,
    SignalID.SECURITY_FORCE_FRICTION: _handle_security_force_friction,
    SignalID.ECONOMIC_CASCADE_FAILURE: _handle_economic_cascade_failure,
    SignalID.ELITE_FRACTURE: _handle_elite_fracture,
    SignalID.ETHNO_REGIONAL_FRAMING: _handle_ethno_regional_framing,
    SignalID.MOBILIZATION_LANGUAGE: _handle_mobilization_language,
    SignalID.COORDINATION_INFRASTRUCTURE: _handle_coordination_infrastructure,
    SignalID.RUMOR_VELOCITY_PANIC: _handle_rumor_velocity_panic,
    SignalID.COUNTER_NARRATIVE_ACTIVATION: _handle_counter_narrative_activation,
}


class SignalMapper:
    """
    Orchestrates mapping from signal detections to primitive updates.
    """
    
    def __init__(self):
        self.handlers = SIGNAL_HANDLERS.copy()
        self._update_log: List[Tuple[SignalID, List]] = []
    
    def map_signal(
        self, 
        detection: SignalDetection, 
        state: PulseState
    ) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
        """
        Map a single signal detection to primitive updates.
        
        Args:
            detection: The detected signal
            state: Current pulse state (for context-aware mapping)
            
        Returns:
            List of updates to apply to primitives
        """
        handler = self.handlers.get(detection.signal_id)
        if handler is None:
            logger.warning(f"No handler for signal {detection.signal_id}")
            return []
        
        try:
            updates = handler(detection, state)
            self._update_log.append((detection.signal_id, updates))
            return updates
        except Exception as e:
            logger.error(f"Error handling signal {detection.signal_id}: {e}")
            return []
    
    def map_batch(
        self, 
        detections: List[SignalDetection], 
        state: PulseState
    ) -> List[ScarcityUpdate | StressUpdate | BondUpdate]:
        """Map multiple signal detections to updates."""
        all_updates = []
        for detection in detections:
            all_updates.extend(self.map_signal(detection, state))
        return all_updates
    
    def get_category(self, signal_id: SignalID) -> SignalCategory:
        """Get the category for a signal."""
        return SIGNAL_CATEGORIES.get(signal_id, SignalCategory.DISTRESS)
    
    def clear_log(self) -> None:
        """Clear the update log."""
        self._update_log.clear()

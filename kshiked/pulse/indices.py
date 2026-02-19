"""
KShield Pulse - Comprehensive Threat Indices

Computes all threat-related indices from signal data:

Phase 1 (HIGH Priority):
- Polarization Index (PI)
- Legitimacy Erosion Index (LEI)  
- Mobilization Readiness Score (MRS)

Phase 2 (MEDIUM Priority):
- Elite Cohesion Index (ECI)
- Information Warfare Index (IWI)
- Security Friction Index (SFI)

Phase 3 (LOWER Priority):
- Economic Cascade Risk (ECR)
- Ethnic Tension Matrix (ETM) - Kenya-specific

All indices are computed from:
- PulseSensor signal detections
- PulseState primitives (Scarcity, Stress, Bonds)
- Time-weighted co-occurrence data
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .primitives import (
    PulseState, ScarcityVector, ActorStress, BondStrength,
    SignalCategory, ActorType, ResourceDomain,
)
from .mapper import SignalID, SignalDetection


# =============================================================================
# Kenya-Specific Ethnic Groups
# =============================================================================

class KenyanEthnicGroup(str, Enum):
    """Major ethnic groups in Kenya for tension tracking."""
    KIKUYU = "kikuyu"
    KALENJIN = "kalenjin"
    LUO = "luo"
    LUHYA = "luhya"
    KAMBA = "kamba"
    KISII = "kisii"
    MERU = "meru"
    MIJIKENDA = "mijikenda"
    MAASAI = "maasai"
    TURKANA = "turkana"
    SOMALI = "somali"
    OTHER = "other"


# Historical tension pairs (based on political history)
TENSION_PAIRS: List[Tuple[str, str]] = [
    ("kikuyu", "luo"),        # 2007-08 post-election violence
    ("kikuyu", "kalenjin"),   # Rift Valley tensions
    ("luo", "kalenjin"),      # Political rivalry
    ("luhya", "kalenjin"),    # Western-Rift competition
    ("somali", "other"),      # North-Eastern marginalization
]


# =============================================================================
# Phase 1: High Priority Indices
# =============================================================================

@dataclass
class PolarizationIndex:
    """
    Polarization Index (PI) - Measures group division and extremism.
    
    Components:
    - Language extremity (dehumanization, rage)
    - Us-vs-them framing
    - Bond fracturing signals
    
    Range: [0, 1] where 1 = extreme polarization
    """
    value: float = 0.0
    confidence: float = 0.5
    
    # Component scores
    language_extremity: float = 0.0
    identity_framing: float = 0.0
    bond_fracture: float = 0.0
    
    # Trend
    trend_24h: float = 0.0  # Change in last 24h
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "PolarizationIndex":
        """Compute polarization index from state and signals."""
        
        # Language extremity: signals 4, 6 (DIRECTED_RAGE, DEHUMANIZATION)
        extremity_signals = [
            s for s in recent_signals 
            if s.signal_id in (SignalID.DIRECTED_RAGE, SignalID.DEHUMANIZATION_LANGUAGE)
        ]
        language_extremity = np.mean([s.intensity for s in extremity_signals]) if extremity_signals else 0.0
        
        # Identity framing: signal 11 (ETHNO_REGIONAL_FRAMING)
        identity_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.ETHNO_REGIONAL_FRAMING
        ]
        identity_framing = np.mean([s.intensity for s in identity_signals]) if identity_signals else 0.0
        
        # Bond fracture: from state
        bond_fracture = state.bonds.fragility_score()
        
        # Weighted combination
        value = (
            0.40 * language_extremity +
            0.30 * identity_framing +
            0.30 * bond_fracture
        )
        
        confidence = 0.5 + 0.5 * min(1.0, len(recent_signals) / 20)
        
        return cls(
            value=min(1.0, value),
            confidence=confidence,
            language_extremity=language_extremity,
            identity_framing=identity_framing,
            bond_fracture=bond_fracture,
        )
    
    @property
    def severity(self) -> str:
        """Human-readable severity level."""
        if self.value >= 0.8:
            return "CRITICAL"
        elif self.value >= 0.6:
            return "HIGH"
        elif self.value >= 0.4:
            return "ELEVATED"
        elif self.value >= 0.2:
            return "MODERATE"
        else:
            return "LOW"


@dataclass
class LegitimacyErosionIndex:
    """
    Legitimacy Erosion Index (LEI) - State legitimacy degradation.
    
    Components:
    - Legitimacy rejection signals
    - Regime slang/mockery
    - Election fraud claims
    - State stress levels
    
    Range: [0, 1] where 1 = complete delegitimization
    """
    value: float = 0.0
    confidence: float = 0.5
    
    # Components
    rejection_rate: float = 0.0
    mockery_intensity: float = 0.0
    state_stress: float = 0.0
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "LegitimacyErosionIndex":
        """Compute legitimacy erosion from state and signals."""
        
        # Legitimacy rejection: signal 7
        rejection_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.LEGITIMACY_REJECTION
        ]
        rejection_rate = np.mean([s.intensity for s in rejection_signals]) if rejection_signals else 0.0
        
        # Regime mockery: signal 5
        mockery_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.ROTATING_REGIME_SLANG
        ]
        mockery_intensity = np.mean([s.intensity for s in mockery_signals]) if mockery_signals else 0.0
        
        # State stress (negative = under pressure)
        state_stress = max(0, -state.stress.get_stress(ActorType.STATE))
        
        # Weighted combination
        value = (
            0.45 * rejection_rate +
            0.25 * mockery_intensity +
            0.30 * state_stress
        )
        
        return cls(
            value=min(1.0, value),
            confidence=0.5 + 0.3 * min(1.0, len(recent_signals) / 15),
            rejection_rate=rejection_rate,
            mockery_intensity=mockery_intensity,
            state_stress=state_stress,
        )
    
    @property
    def severity(self) -> str:
        if self.value >= 0.7:
            return "CRITICAL"
        elif self.value >= 0.5:
            return "HIGH"
        elif self.value >= 0.3:
            return "ELEVATED"
        else:
            return "LOW"


@dataclass
class MobilizationReadinessScore:
    """
    Mobilization Readiness Score (MRS) - Likelihood of mass action.
    
    Components:
    - Mobilization language intensity
    - Coordination infrastructure activity
    - Population stress levels
    - Recent action calls
    
    Range: [0, 1] where 1 = imminent mass mobilization
    """
    value: float = 0.0
    confidence: float = 0.5
    
    # Components
    mobilization_language: float = 0.0
    coordination_activity: float = 0.0
    population_readiness: float = 0.0
    opposition_strength: float = 0.0
    
    # Time-sensitive
    calls_to_action_24h: int = 0
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "MobilizationReadinessScore":
        """Compute mobilization readiness from state and signals."""
        
        # Mobilization language: signal 12
        mob_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.MOBILIZATION_LANGUAGE
        ]
        mobilization_language = np.mean([s.intensity for s in mob_signals]) if mob_signals else 0.0
        calls_to_action = len(mob_signals)
        
        # Coordination infrastructure: signal 13
        coord_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.COORDINATION_INFRASTRUCTURE
        ]
        coordination_activity = np.mean([s.intensity for s in coord_signals]) if coord_signals else 0.0
        
        # Population stress (more stressed = more ready to act)
        pop_stress = max(0, -state.stress.get_stress(ActorType.POPULATION))
        population_readiness = pop_stress
        
        # Opposition empowerment
        opposition_strength = max(0, state.stress.get_stress(ActorType.OPPOSITION))
        
        # Weighted combination
        value = (
            0.35 * mobilization_language +
            0.25 * coordination_activity +
            0.25 * population_readiness +
            0.15 * opposition_strength
        )
        
        return cls(
            value=min(1.0, value),
            confidence=0.5 + 0.3 * min(1.0, len(recent_signals) / 10),
            mobilization_language=mobilization_language,
            coordination_activity=coordination_activity,
            population_readiness=population_readiness,
            opposition_strength=opposition_strength,
            calls_to_action_24h=calls_to_action,
        )
    
    @property
    def severity(self) -> str:
        if self.value >= 0.8:
            return "IMMINENT"
        elif self.value >= 0.6:
            return "HIGH"
        elif self.value >= 0.4:
            return "ELEVATED"
        elif self.value >= 0.2:
            return "MODERATE"
        else:
            return "LOW"


# =============================================================================
# Phase 2: Medium Priority Indices
# =============================================================================

@dataclass
class EliteCohesionIndex:
    """
    Elite Cohesion Index (ECI) - Measures elite solidarity.
    
    Lower value = elite fracture = higher instability risk.
    
    Components:
    - Elite fracture signals
    - Elite stress levels
    - Public elite disagreements
    
    Range: [0, 1] where 0 = complete fracture, 1 = unified
    """
    value: float = 0.5
    confidence: float = 0.5
    
    # Components
    fracture_signals: float = 0.0
    elite_stress: float = 0.0
    defection_rate: float = 0.0
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "EliteCohesionIndex":
        """Compute elite cohesion from state and signals."""
        
        # Elite fracture signals: signal 10
        fracture_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.ELITE_FRACTURE
        ]
        fracture_intensity = np.mean([s.intensity for s in fracture_signals]) if fracture_signals else 0.0
        
        # Elite stress (negative = under pressure)
        elite_stress = max(0, -state.stress.get_stress(ActorType.ELITE))
        
        # Defection rate (proxy from signal count)
        defection_rate = min(1.0, len(fracture_signals) / 10)
        
        # Cohesion = 1 - average fragmentation
        fragmentation = (
            0.50 * fracture_intensity +
            0.30 * elite_stress +
            0.20 * defection_rate
        )
        
        value = 1.0 - min(1.0, fragmentation)
        
        return cls(
            value=value,
            confidence=0.5,
            fracture_signals=fracture_intensity,
            elite_stress=elite_stress,
            defection_rate=defection_rate,
        )
    
    @property
    def severity(self) -> str:
        """Lower cohesion = higher severity."""
        if self.value <= 0.2:
            return "CRITICAL"
        elif self.value <= 0.4:
            return "HIGH"
        elif self.value <= 0.6:
            return "MODERATE"
        else:
            return "STABLE"


@dataclass
class InformationWarfareIndex:
    """
    Information Warfare Index (IWI) - Disinformation and narrative velocity.
    
    Components:
    - Counter-narrative activation
    - Rumor velocity/panic
    - Narrative coordination patterns
    
    Range: [0, 1] where 1 = active information warfare
    """
    value: float = 0.0
    confidence: float = 0.5
    
    # Components
    counter_narrative: float = 0.0
    rumor_velocity: float = 0.0
    panic_indicators: float = 0.0
    
    # Metrics
    viral_narratives_24h: int = 0
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "InformationWarfareIndex":
        """Compute information warfare index from signals."""
        
        # Counter-narrative: signal 15
        counter_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.COUNTER_NARRATIVE_ACTIVATION
        ]
        counter_narrative = np.mean([s.intensity for s in counter_signals]) if counter_signals else 0.0
        
        # Rumor velocity: signal 14
        rumor_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.RUMOR_VELOCITY_PANIC
        ]
        rumor_velocity = np.mean([s.intensity for s in rumor_signals]) if rumor_signals else 0.0
        
        # Panic indicators (from rumor signals with high intensity)
        panic_indicators = sum(1 for s in rumor_signals if s.intensity > 0.7) / max(1, len(rumor_signals))
        
        # Weighted combination
        value = (
            0.40 * counter_narrative +
            0.40 * rumor_velocity +
            0.20 * panic_indicators
        )
        
        return cls(
            value=min(1.0, value),
            confidence=0.5,
            counter_narrative=counter_narrative,
            rumor_velocity=rumor_velocity,
            panic_indicators=panic_indicators,
            viral_narratives_24h=len(counter_signals) + len(rumor_signals),
        )
    
    @property
    def severity(self) -> str:
        if self.value >= 0.7:
            return "ACTIVE_CAMPAIGN"
        elif self.value >= 0.5:
            return "HIGH"
        elif self.value >= 0.3:
            return "ELEVATED"
        else:
            return "NORMAL"


@dataclass
class SecurityFrictionIndex:
    """
    Security Friction Index (SFI) - State-security force tensions.
    
    Components:
    - Security force friction signals
    - Security actor stress
    - State-Security relationship
    
    Range: [0, 1] where 1 = security forces defecting/rebelling
    """
    value: float = 0.0
    confidence: float = 0.5
    
    # Components
    friction_signals: float = 0.0
    security_stress: float = 0.0
    state_security_gap: float = 0.0
    
    # Warning flags
    defection_risk: bool = False
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "SecurityFrictionIndex":
        """Compute security friction from state and signals."""
        
        # Security friction signals: signal 8
        friction_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.SECURITY_FORCE_FRICTION
        ]
        friction_intensity = np.mean([s.intensity for s in friction_signals]) if friction_signals else 0.0
        
        # Security stress
        security_stress = max(0, -state.stress.get_stress(ActorType.SECURITY))
        
        # State-Security relationship gap
        state_pressure = max(0, -state.stress.get_stress(ActorType.STATE))
        state_security_gap = abs(security_stress - state_pressure)
        
        # Weighted combination
        value = (
            0.50 * friction_intensity +
            0.30 * security_stress +
            0.20 * state_security_gap
        )
        
        # Defection risk flag
        defection_risk = value > 0.7 or (friction_intensity > 0.6 and security_stress > 0.5)
        
        return cls(
            value=min(1.0, value),
            confidence=0.5,
            friction_signals=friction_intensity,
            security_stress=security_stress,
            state_security_gap=state_security_gap,
            defection_risk=defection_risk,
        )
    
    @property
    def severity(self) -> str:
        if self.defection_risk:
            return "DEFECTION_RISK"
        elif self.value >= 0.6:
            return "HIGH"
        elif self.value >= 0.4:
            return "ELEVATED"
        else:
            return "STABLE"


# =============================================================================
# Phase 3: Economic and Ethnic Indices
# =============================================================================

@dataclass
class EconomicCascadeRisk:
    """
    Economic Cascade Risk (ECR) - Chain-failure probability.
    
    Components:
    - Economic cascade signals
    - Scarcity aggregates
    - Cross-domain spillover
    
    Range: [0, 1] where 1 = imminent economic collapse cascade
    """
    value: float = 0.0
    confidence: float = 0.5
    
    # Components
    cascade_signals: float = 0.0
    scarcity_aggregate: float = 0.0
    cross_domain_stress: float = 0.0
    
    # Domain breakdown
    food_risk: float = 0.0
    fuel_risk: float = 0.0
    currency_risk: float = 0.0
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "EconomicCascadeRisk":
        """Compute economic cascade risk from state and signals."""
        
        # Economic cascade signals: signal 9
        cascade_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.ECONOMIC_CASCADE_FAILURE
        ]
        cascade_intensity = np.mean([s.intensity for s in cascade_signals]) if cascade_signals else 0.0
        
        # Scarcity aggregate
        scarcity_aggregate = state.scarcity.aggregate_score()
        
        # Domain-specific risks
        food_risk = state.scarcity.get(ResourceDomain.FOOD)
        fuel_risk = state.scarcity.get(ResourceDomain.FUEL)
        currency_risk = state.scarcity.get(ResourceDomain.CURRENCY)
        
        # Cross-domain stress (if multiple domains stressed, cascade more likely)
        domains_stressed = sum(1 for d in ResourceDomain if state.scarcity.get(d) > 0.5)
        cross_domain_stress = min(1.0, domains_stressed / 3)
        
        # Weighted combination
        value = (
            0.40 * cascade_intensity +
            0.30 * scarcity_aggregate +
            0.30 * cross_domain_stress
        )
        
        return cls(
            value=min(1.0, value),
            confidence=0.5,
            cascade_signals=cascade_intensity,
            scarcity_aggregate=scarcity_aggregate,
            cross_domain_stress=cross_domain_stress,
            food_risk=food_risk,
            fuel_risk=fuel_risk,
            currency_risk=currency_risk,
        )
    
    @property
    def severity(self) -> str:
        if self.value >= 0.8:
            return "CASCADE_IMMINENT"
        elif self.value >= 0.6:
            return "HIGH"
        elif self.value >= 0.4:
            return "ELEVATED"
        else:
            return "STABLE"


@dataclass
class EthnicTensionMatrix:
    """
    Ethnic Tension Matrix (ETM) - Kenya-specific group tensions.
    
    Tracks tension between major Kenyan ethnic groups based on:
    - Ethno-regional framing signals
    - Dehumanization targeting groups
    - Historical tension patterns
    
    Matrix values: [0, 1] where 1 = extreme inter-group tension
    """
    # Tension between group pairs
    tensions: Dict[str, float] = field(default_factory=dict)
    
    # Overall metrics
    max_tension: float = 0.0
    avg_tension: float = 0.0
    
    # Most tense pair
    highest_tension_pair: Optional[Tuple[str, str]] = None
    
    confidence: float = 0.5
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
    ) -> "EthnicTensionMatrix":
        """Compute ethnic tension matrix from state and signals."""
        
        # Initialize tensions from historical pairs
        tensions = {}
        for g1, g2 in TENSION_PAIRS:
            key = f"{g1}-{g2}"
            tensions[key] = 0.1  # Baseline tension
        
        # Ethno-regional signals: signal 11
        ethnic_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.ETHNO_REGIONAL_FRAMING
        ]
        ethnic_intensity = np.mean([s.intensity for s in ethnic_signals]) if ethnic_signals else 0.0
        
        # Dehumanization signals: signal 6
        dehum_signals = [
            s for s in recent_signals 
            if s.signal_id == SignalID.DEHUMANIZATION_LANGUAGE
        ]
        dehum_intensity = np.mean([s.intensity for s in dehum_signals]) if dehum_signals else 0.0
        
        # Apply signal intensity to historical tension pairs (amplified)
        combined_intensity = 0.6 * ethnic_intensity + 0.4 * dehum_intensity
        
        for key in tensions:
            # Increase tension based on signals
            tensions[key] = min(1.0, tensions[key] + combined_intensity * 0.8)
        
        # Also use state's ethnic bonds if available
        for bond_key, strength in state.bonds.ethnic_bonds.items():
            # Lower bond strength = higher tension
            if bond_key in tensions:
                tensions[bond_key] = max(tensions[bond_key], 1.0 - strength)
        
        # Compute aggregates
        tension_values = list(tensions.values())
        max_tension = max(tension_values) if tension_values else 0.0
        avg_tension = np.mean(tension_values) if tension_values else 0.0
        
        # Find highest tension pair
        highest_pair = None
        if tensions:
            highest_key = max(tensions, key=tensions.get)
            g1, g2 = highest_key.split("-")
            highest_pair = (g1, g2)
        
        return cls(
            tensions=tensions,
            max_tension=max_tension,
            avg_tension=avg_tension,
            highest_tension_pair=highest_pair,
            confidence=0.4 + 0.3 * min(1.0, len(ethnic_signals) / 5),
        )
    
    @property
    def severity(self) -> str:
        if self.max_tension >= 0.8:
            return "CRITICAL"
        elif self.max_tension >= 0.6:
            return "HIGH"
        elif self.max_tension >= 0.4:
            return "ELEVATED"
        else:
            return "MODERATE"
    
    def get_tension(self, group1: str, group2: str) -> float:
        """Get tension between two groups."""
        key1 = f"{group1}-{group2}"
        key2 = f"{group2}-{group1}"
        return self.tensions.get(key1, self.tensions.get(key2, 0.0))


# =============================================================================
# Unified Threat Index Aggregator
# =============================================================================

@dataclass
class ThreatIndexReport:
    """
    Complete threat index report aggregating all indices.
    
    Provides:
    - All 8 computed indices
    - Overall threat level
    - Priority alerts
    - Trend analysis
    """
    # Phase 1: High Priority
    polarization: PolarizationIndex = field(default_factory=PolarizationIndex)
    legitimacy_erosion: LegitimacyErosionIndex = field(default_factory=LegitimacyErosionIndex)
    mobilization_readiness: MobilizationReadinessScore = field(default_factory=MobilizationReadinessScore)
    
    # Phase 2: Medium Priority
    elite_cohesion: EliteCohesionIndex = field(default_factory=EliteCohesionIndex)
    information_warfare: InformationWarfareIndex = field(default_factory=InformationWarfareIndex)
    security_friction: SecurityFrictionIndex = field(default_factory=SecurityFrictionIndex)
    
    # Phase 3: Economic/Ethnic
    economic_cascade: EconomicCascadeRisk = field(default_factory=EconomicCascadeRisk)
    ethnic_tension: EthnicTensionMatrix = field(default_factory=EthnicTensionMatrix)
    
    # ESI from price aggregator (added externally)
    economic_satisfaction: float = 0.5
    
    # Overall metrics
    overall_threat_level: str = "LOW"
    priority_alerts: List[str] = field(default_factory=list)
    
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def compute_all(
        cls,
        state: PulseState,
        recent_signals: List[SignalDetection],
        esi_score: float = 0.5,
    ) -> "ThreatIndexReport":
        """Compute all threat indices from state and signals."""
        
        report = cls(
            polarization=PolarizationIndex.compute(state, recent_signals),
            legitimacy_erosion=LegitimacyErosionIndex.compute(state, recent_signals),
            mobilization_readiness=MobilizationReadinessScore.compute(state, recent_signals),
            elite_cohesion=EliteCohesionIndex.compute(state, recent_signals),
            information_warfare=InformationWarfareIndex.compute(state, recent_signals),
            security_friction=SecurityFrictionIndex.compute(state, recent_signals),
            economic_cascade=EconomicCascadeRisk.compute(state, recent_signals),
            ethnic_tension=EthnicTensionMatrix.compute(state, recent_signals),
            economic_satisfaction=esi_score,
        )
        
        report._compute_overall_threat()
        report._generate_alerts()
        
        return report
    
    def _compute_overall_threat(self) -> None:
        """Compute overall threat level from indices."""
        # Weighted average of key indices
        threat_score = (
            0.20 * self.polarization.value +
            0.15 * self.legitimacy_erosion.value +
            0.20 * self.mobilization_readiness.value +
            0.10 * (1 - self.elite_cohesion.value) +
            0.10 * self.information_warfare.value +
            0.10 * self.security_friction.value +
            0.10 * self.economic_cascade.value +
            0.05 * self.ethnic_tension.avg_tension
        )
        
        # Classify threat level
        if threat_score >= 0.75:
            self.overall_threat_level = "CRITICAL"
        elif threat_score >= 0.55:
            self.overall_threat_level = "HIGH"
        elif threat_score >= 0.35:
            self.overall_threat_level = "ELEVATED"
        elif threat_score >= 0.20:
            self.overall_threat_level = "GUARDED"
        else:
            self.overall_threat_level = "LOW"
    
    def _generate_alerts(self) -> None:
        """Generate priority alerts from indices."""
        alerts = []
        
        if self.polarization.severity == "CRITICAL":
            alerts.append("⚠️ CRITICAL: Extreme polarization detected")
        
        if self.legitimacy_erosion.severity in ("CRITICAL", "HIGH"):
            alerts.append("⚠️ HIGH: State legitimacy under severe pressure")
        
        if self.mobilization_readiness.severity == "IMMINENT":
            alerts.append("🔴 IMMINENT: Mass mobilization likely")
        
        if self.security_friction.defection_risk:
            alerts.append("🔴 WARNING: Security force defection risk")
        
        if self.economic_cascade.severity == "CASCADE_IMMINENT":
            alerts.append("⚠️ ECONOMIC: Cascade failure imminent")
        
        if self.ethnic_tension.severity == "CRITICAL":
            pair = self.ethnic_tension.highest_tension_pair
            if pair:
                alerts.append(f"⚠️ ETHNIC: Critical tension between {pair[0]}-{pair[1]}")
        
        self.priority_alerts = alerts
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_threat_level": self.overall_threat_level,
            "priority_alerts": self.priority_alerts,
            "indices": {
                "polarization": {
                    "value": self.polarization.value,
                    "severity": self.polarization.severity,
                },
                "legitimacy_erosion": {
                    "value": self.legitimacy_erosion.value,
                    "severity": self.legitimacy_erosion.severity,
                },
                "mobilization_readiness": {
                    "value": self.mobilization_readiness.value,
                    "severity": self.mobilization_readiness.severity,
                    "calls_to_action_24h": self.mobilization_readiness.calls_to_action_24h,
                },
                "elite_cohesion": {
                    "value": self.elite_cohesion.value,
                    "severity": self.elite_cohesion.severity,
                },
                "information_warfare": {
                    "value": self.information_warfare.value,
                    "severity": self.information_warfare.severity,
                },
                "security_friction": {
                    "value": self.security_friction.value,
                    "severity": self.security_friction.severity,
                    "defection_risk": self.security_friction.defection_risk,
                },
                "economic_cascade": {
                    "value": self.economic_cascade.value,
                    "severity": self.economic_cascade.severity,
                },
                "ethnic_tension": {
                    "avg_tension": self.ethnic_tension.avg_tension,
                    "max_tension": self.ethnic_tension.max_tension,
                    "severity": self.ethnic_tension.severity,
                    "highest_pair": self.ethnic_tension.highest_tension_pair,
                },
                "economic_satisfaction": self.economic_satisfaction,
            },
            "computed_at": self.computed_at.isoformat(),
        }


# =============================================================================
# Factory Function
# =============================================================================

def compute_threat_report(
    state: PulseState,
    recent_signals: List[SignalDetection],
    esi_score: float = 0.5,
) -> ThreatIndexReport:
    """
    Compute complete threat index report.
    
    Args:
        state: Current PulseState with primitives.
        recent_signals: Recent signal detections (from PulseSensor).
        esi_score: Economic Satisfaction Index from price aggregator.
        
    Returns:
        Complete ThreatIndexReport with all indices.
    """
    return ThreatIndexReport.compute_all(state, recent_signals, esi_score)

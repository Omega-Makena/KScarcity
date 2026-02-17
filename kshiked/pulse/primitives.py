"""
Pulse Engine Primitives — Core Data Models

Defines the four fundamental primitives that social signals map to:
1. ScarcityVector: Resource availability perception
2. ActorStress: Stress levels on key actors (state, elite, population)
3. BondStrength: Social cohesion between groups
4. ShockPropagation: Coefficients for shock transmission

These primitives decouple the signal detection layer from the simulation engine.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
import logging

logger = logging.getLogger("kshield.pulse.primitives")


# =============================================================================
# Enums for Classification
# =============================================================================

class SignalCategory(Enum):
    """Categories of intelligence signals."""
    DISTRESS = auto()           # Signals 1-3: Survival stress indicators
    ANGER = auto()              # Signals 4-7: Delegitimization and rage
    INSTITUTIONAL = auto()      # Signals 8-10: System friction
    IDENTITY = auto()           # Signals 11-13: Mobilization
    INFORMATION = auto()        # Signals 14-15: Info warfare


class ActorType(Enum):
    """Key actors in the system."""
    STATE = "state"             # Government, institutions
    ELITE = "elite"             # Business, political elite
    POPULATION = "population"   # General public
    SECURITY = "security"       # Police, military
    OPPOSITION = "opposition"   # Political opposition


class ResourceDomain(Enum):
    """Resource domains for scarcity tracking."""
    FOOD = "food"
    FUEL = "fuel"
    HOUSING = "housing"
    HEALTHCARE = "healthcare"
    EMPLOYMENT = "employment"
    CURRENCY = "currency"


# =============================================================================
# Primitive 1: Scarcity Vector
# =============================================================================

@dataclass
class ScarcityVector:
    """
    Tracks perceived resource scarcity across domains.
    
    Values are normalized [0, 1] where:
    - 0.0 = No scarcity perceived (abundant)
    - 1.0 = Extreme scarcity (crisis level)
    
    Attributes:
        values: Dict mapping ResourceDomain -> scarcity level
        confidence: Confidence in the measurements [0, 1]
        timestamp: Unix timestamp of last update
    """
    values: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: float = 0.0
    
    def __post_init__(self):
        # Initialize all domains to neutral if not provided
        for domain in ResourceDomain:
            if domain.value not in self.values:
                self.values[domain.value] = 0.0
    
    def get(self, domain: ResourceDomain) -> float:
        """Get scarcity level for a domain."""
        return self.values.get(domain.value, 0.0)
    
    def set(self, domain: ResourceDomain, value: float) -> None:
        """Set scarcity level for a domain (clamped to [0, 1])."""
        self.values[domain.value] = max(0.0, min(1.0, value))
    
    def aggregate_score(self) -> float:
        """Compute weighted aggregate scarcity score."""
        if not self.values:
            return 0.0
        return np.mean(list(self.values.values()))
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array (ordered by ResourceDomain enum)."""
        return np.array([self.get(d) for d in ResourceDomain], dtype=np.float32)


# =============================================================================
# Primitive 2: Actor Stress
# =============================================================================

@dataclass
class ActorStress:
    """
    Tracks stress levels on key system actors.
    
    Values are normalized [-1, 1] where:
    - -1.0 = Actor under extreme negative pressure (failing)
    -  0.0 = Neutral/stable
    - +1.0 = Actor empowered/strengthened
    
    Negative stress indicates the actor is losing legitimacy/capacity.
    """
    stress_levels: Dict[str, float] = field(default_factory=dict)
    friction_matrix: Optional[np.ndarray] = None  # Actor-actor friction
    timestamp: float = 0.0
    
    def __post_init__(self):
        # Initialize all actors to neutral
        for actor in ActorType:
            if actor.value not in self.stress_levels:
                self.stress_levels[actor.value] = 0.0
        
        # Initialize friction matrix (5x5 for 5 actors)
        if self.friction_matrix is None:
            self.friction_matrix = np.zeros((len(ActorType), len(ActorType)), dtype=np.float32)
    
    def get_stress(self, actor: ActorType) -> float:
        """Get stress level for an actor."""
        return self.stress_levels.get(actor.value, 0.0)
    
    def apply_stress(self, actor: ActorType, delta: float) -> None:
        """Apply stress delta to an actor (clamped to [-1, 1])."""
        current = self.get_stress(actor)
        self.stress_levels[actor.value] = max(-1.0, min(1.0, current + delta))
    
    def get_friction(self, actor1: ActorType, actor2: ActorType) -> float:
        """Get friction between two actors."""
        i, j = list(ActorType).index(actor1), list(ActorType).index(actor2)
        return self.friction_matrix[i, j]
    
    def set_friction(self, actor1: ActorType, actor2: ActorType, value: float) -> None:
        """Set friction between two actors (symmetric)."""
        i, j = list(ActorType).index(actor1), list(ActorType).index(actor2)
        self.friction_matrix[i, j] = value
        self.friction_matrix[j, i] = value  # Symmetric
    
    def total_system_stress(self) -> float:
        """Compute total negative stress across system (for crisis detection)."""
        return sum(max(0.0, -s) for s in self.stress_levels.values())


# =============================================================================
# Primitive 3: Bond Strength
# =============================================================================

@dataclass
class BondStrength:
    """
    Tracks social cohesion between population groups.
    
    Models ethnic, regional, and class bonds that can fracture under stress.
    Values [0, 1] where:
    - 0.0 = Complete fracture/hostility
    - 0.5 = Neutral
    - 1.0 = Strong cohesion
    """
    # Primary bonds (can be extended per country)
    national_cohesion: float = 0.5      # Cross-group national identity
    ethnic_bonds: Dict[str, float] = field(default_factory=dict)  # "group_a-group_b" -> strength
    class_solidarity: float = 0.5       # Economic class cohesion
    regional_unity: float = 0.5         # Geographic regional bonds
    
    # Polarization metric
    polarization_index: float = 0.0     # 0 = unpolarized, 1 = extreme polarization
    
    timestamp: float = 0.0
    
    def apply_fracture(self, bond_type: str, delta: float) -> None:
        """Apply fracture (negative delta) or healing (positive) to a bond."""
        if bond_type == "national":
            self.national_cohesion = max(0.0, min(1.0, self.national_cohesion + delta))
        elif bond_type == "class":
            self.class_solidarity = max(0.0, min(1.0, self.class_solidarity + delta))
        elif bond_type == "regional":
            self.regional_unity = max(0.0, min(1.0, self.regional_unity + delta))
        else:
            # Ethnic bond (create if doesn't exist)
            current = self.ethnic_bonds.get(bond_type, 0.5)
            self.ethnic_bonds[bond_type] = max(0.0, min(1.0, current + delta))
    
    def overall_cohesion(self) -> float:
        """Compute overall social cohesion score."""
        ethnic_avg = np.mean(list(self.ethnic_bonds.values())) if self.ethnic_bonds else 0.5
        return np.mean([
            self.national_cohesion,
            ethnic_avg,
            self.class_solidarity,
            self.regional_unity
        ])
    
    def fragility_score(self) -> float:
        """Compute fragility (inverse of cohesion + polarization effect)."""
        base = 1.0 - self.overall_cohesion()
        return min(1.0, base + 0.5 * self.polarization_index)


# =============================================================================
# Primitive 4: Shock Propagation Coefficients
# =============================================================================

@dataclass
class ShockPropagation:
    """
    Coefficients determining how social signals translate to economic shocks.
    
    These are learned/calibrated based on historical data.
    Higher values = stronger transmission from signal to shock.
    """
    # Signal category -> economic variable -> coefficient
    coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Decay rate for shock intensity over time
    decay_rate: float = 0.9
    
    # Threshold before shock is triggered
    activation_threshold: float = 0.3
    
    # Probabilistic trigger (vs deterministic)
    probabilistic: bool = True
    trigger_probability_scale: float = 1.0
    
    def __post_init__(self):
        # Initialize default coefficients
        defaults = {
            "distress": {
                "GDP (current US$)": -0.02,
                "Inflation, consumer prices (annual %)": 0.10,
            },
            "anger": {
                "GDP (current US$)": -0.03,
                "Exports of goods and services (BoP, current US$)": -0.05,
            },
            "institutional": {
                "GDP (current US$)": -0.05,
                "External debt stocks, total (DOD, current US$)": 0.03,
            },
            "identity": {
                "GDP (current US$)": -0.02,
                "Inflation, consumer prices (annual %)": 0.05,
            },
            "information": {
                "Inflation, consumer prices (annual %)": 0.08,
                "Imports of goods and services (BoP, current US$)": -0.03,
            },
        }
        for cat, mapping in defaults.items():
            if cat not in self.coefficients:
                self.coefficients[cat] = mapping
    
    def get_shock_magnitude(
        self, 
        category: SignalCategory, 
        signal_intensity: float,
        target_variable: str
    ) -> float:
        """
        Compute shock magnitude for a given signal and target variable.
        
        Args:
            category: Signal category
            signal_intensity: Normalized intensity [0, 1]
            target_variable: Economic variable to shock
            
        Returns:
            Shock magnitude (can be negative)
        """
        cat_key = category.name.lower()
        if cat_key not in self.coefficients:
            return 0.0
        
        coeff = self.coefficients[cat_key].get(target_variable, 0.0)
        return coeff * signal_intensity
    
    def should_trigger(self, aggregate_intensity: float, rng: np.random.Generator = None) -> bool:
        """
        Determine if a shock should be triggered based on intensity.
        
        Args:
            aggregate_intensity: Combined signal intensity [0, 1]
            rng: Random number generator (for probabilistic triggering)
        """
        if aggregate_intensity < self.activation_threshold:
            return False
        
        if not self.probabilistic:
            return True
        
        # Probabilistic: higher intensity = higher trigger probability
        if rng is None:
            rng = np.random.default_rng()
        
        prob = min(1.0, aggregate_intensity * self.trigger_probability_scale)
        return rng.random() < prob


# =============================================================================
# Update Messages (for event bus communication)
# =============================================================================

@dataclass
class ScarcityUpdate:
    """Message to update scarcity vector."""
    domain: ResourceDomain
    delta: float
    source_signal: int  # Signal ID (1-15)
    confidence: float = 0.5


@dataclass
class StressUpdate:
    """Message to update actor stress."""
    actor: ActorType
    delta: float
    source_signal: int
    reason: str = ""


@dataclass
class BondUpdate:
    """Message to update bond strength."""
    bond_type: str  # "national", "class", "regional", or ethnic pair key
    delta: float
    source_signal: int
    reason: str = ""


@dataclass
class PulseState:
    """
    Complete state snapshot of the Pulse Engine primitives.
    This is the interface between Pulse and the simulation layer.
    """
    scarcity: ScarcityVector = field(default_factory=ScarcityVector)
    stress: ActorStress = field(default_factory=ActorStress)
    bonds: BondStrength = field(default_factory=BondStrength)
    propagation: ShockPropagation = field(default_factory=ShockPropagation)
    
    # Aggregate risk metrics
    crisis_probability: float = 0.0
    instability_index: float = 0.0
    
    timestamp: float = 0.0
    
    def compute_risk_metrics(self) -> None:
        """Update aggregate risk metrics based on current primitive state."""
        # Crisis probability: high scarcity + high stress + low cohesion
        scarcity_risk = self.scarcity.aggregate_score()
        stress_risk = self.stress.total_system_stress()
        fragility = self.bonds.fragility_score()
        
        # Weighted combination
        self.instability_index = (
            0.3 * scarcity_risk +
            0.4 * stress_risk +
            0.3 * fragility
        )
        
        # Crisis probability (sigmoid-like)
        self.crisis_probability = 1.0 / (1.0 + np.exp(-5 * (self.instability_index - 0.5)))
    
    def to_shock_vector(self, variables: List[str]) -> Dict[str, float]:
        """
        Convert current state to shock magnitudes for simulation variables.
        
        Args:
            variables: List of economic variable names
            
        Returns:
            Dict of variable -> shock magnitude
        """
        shocks = {}
        self.compute_risk_metrics()
        
        # Only generate shocks if instability exceeds threshold
        if self.instability_index < self.propagation.activation_threshold:
            return shocks
        
        # Compute shocks per category (simplified - uses instability as proxy)
        for category in SignalCategory:
            for var in variables:
                mag = self.propagation.get_shock_magnitude(
                    category, 
                    self.instability_index, 
                    var
                )
                if abs(mag) > 1e-6:
                    shocks[var] = shocks.get(var, 0.0) + mag
        
        return shocks

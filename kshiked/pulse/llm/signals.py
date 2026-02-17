"""
KShield Pulse Engine V3.0 - Signal Data Models

Strict implementation of the 'KShield Pulse Engine' Threat Architecture.
Defines the Dual-Layer Risk Model:
1. Threat Layer (Mobilization/Intent) -> BaseRisk
2. Context Layer (Dissatisfaction/Stress) -> Context Stress Multiplier (CSM)

References:
- Taxonomy Categories 1-14
- Economic Dissatisfaction E0-E4
- Social Dissatisfaction S0-S4
- Indices: LEI, SI, MS, AA
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

# =============================================================================
# 1. THREAT TAXONOMY (The "What")
# =============================================================================

class TimeToAction(str, Enum):
    """
    Temporal Urgency (PDF Page 45).
    """
    IMMEDIATE_24H = "immediate_24h" # Active mobilization
    NEAR_TERM_72H = "near_term_72h" # Planning/Staging
    CHRONIC_14D = "chronic_14d"     # Latent radicalization
    LONG_TERM = "long_term_strategic"

class ThreatTier(str, Enum):
    """Ranked Severity Levels (PDF Page 1)."""
    TIER_1 = "TIER_1_EXISTENTIAL"       # Mass violence, coup, genocide
    TIER_2 = "TIER_2_SEVERE_STABILITY"  # Insurrection, infra sabotage
    TIER_3 = "TIER_3_HIGH_RISK"         # Disinformation, mobilization
    TIER_4 = "TIER_4_EMERGING"          # Radicalization pipelines
    TIER_5 = "TIER_5_NON_THREAT"        # Criticism, satire

class ThreatCategory(str, Enum):
    """
    14-Point Ranked Taxonomy (PDF).
    Maps specifically to Tiers.
    """
    # Tier 1
    CAT_1_MASS_VIOLENCE = "mass_casualty_advocacy"
    CAT_2_TERRORISM_SUPPORT = "terrorism_support"
    CAT_3_INFRA_SABOTAGE = "critical_infrastructure_sabotage"
    
    # Tier 2
    CAT_4_INSURRECTION = "coordinated_insurrection"
    CAT_5_ELECTION_SUBVERSION = "election_interference"
    CAT_6_OFFICIAL_THREATS = "targeted_threats_officials"
    
    # Tier 3
    CAT_7_ETHNIC_MOBILIZATION = "ethnic_religious_mobilization"
    CAT_8_DISINFO_CAMPAIGNS = "large_scale_disinformation"
    CAT_9_FINANCIAL_WARFARE = "economic_warfare_destabilization"
    
    # Tier 4
    CAT_10_RADICALIZATION = "radicalization_pipelines"
    CAT_11_HATE_NETWORKS = "coordinated_hate_networks"
    CAT_12_FOREIGN_INFLUENCE = "foreign_influence_proxy"
    
    # Tier 5 (Protected/Low Risk)
    CAT_13_POLITICAL_CRITICISM = "political_criticism"
    CAT_14_SATIRE_PROTEST = "satire_art_protest"
    
    UNKNOWN = "unknown"

# =============================================================================
# 2. CONTEXT LAYER (The "Why/Where")
# =============================================================================

class EconomicGrievance(str, Enum):
    """Economic Dissatisfaction (PDF Page 10)."""
    E0_LEGITIMATE = "E0_legitimate_grievance"
    E1_DELEGITIMIZATION = "E1_anger_delegitimization"
    E2_MOBILIZATION = "E2_mobilization_pressure"
    E3_DESTABILIZATION = "E3_destabilization_narratives"
    E4_SABOTAGE = "E4_economic_sabotage"

class SocialGrievance(str, Enum):
    """Social Dissatisfaction (PDF Page 13)."""
    S0_DISCONTENT = "S0_normal_discontent"
    S1_POLARIZATION = "S1_polarization_narratives"
    S2_MOBILIZATION = "S2_group_mobilization"
    S3_FRACTURE = "S3_violence_risk"
    S4_BREAKDOWN = "S4_societal_breakdown"

# =============================================================================
# 3. ADVANCED INDICES (The "How")
# =============================================================================

class RoleType(str, Enum):
    """
    V3 Role Taxonomy (PDF Page 26).
    """
    IDEOLOGUE = "ideologue"             # Frame creators (The "Why")
    MOBILIZER = "mobilizer"             # Action coordinators (The "When")
    BROKER = "broker"                   # Network bridges (The "Who")
    OPERATIONAL_SIGNALER = "op_signaler" # Tactical directions
    UNWITTING_AMPLIFIER = "unwitting_amplifier" # Super-spreaders
    OBSERVER = "observer"

@dataclass
class ResilienceIndex:
    """
    Counter-Narrative & Stability Factors (The "Dampener").
    """
    counter_narrative_score: float # 0-1 (Strength of pushback)
    community_resilience: float    # 0-1 (Local rejection of threat)
    confusion_factor: float        # 0-1 (Is the narrative incoherent?)

@dataclass
class ThreatSignal:
    """
    Core Threat Analysis (BaseRisk).
    """
    category: ThreatCategory
    tier: ThreatTier
    
    # Raw Scores (0.0 - 1.0)
    intent: float
    capability: float
    specificity: float
    reach: float
    trajectory: float
    
    classification_reason: str
    
    @property
    def base_risk_score(self) -> float:
        """
        Composite Risk Formula (PDF Page 8).
        Risk = 0.30*Intent + 0.20*Cap + 0.15*Spec + 0.15*Reach + 0.10*Traj + 0.10*Net
        (Network density is calculated externally, defaulting to 0.5 here for single-post estimation)
        """
        score = (
            (0.30 * self.intent) +
            (0.20 * self.capability) +
            (0.15 * self.specificity) +
            (0.15 * self.reach) +
            (0.10 * self.trajectory) + 
            (0.10 * 0.5) # Network density placeholder
        ) * 100
        return min(100.0, score)

@dataclass
class ContextAnalysis:
    """
    Contextual Stressors (CSM).
    """
    economic_strain: EconomicGrievance
    social_fracture: SocialGrievance
    
    # Normalized Indices (0.0 - 1.0)
    economic_dissatisfaction_score: float # ED
    social_dissatisfaction_score: float   # SD
    shock_marker: float                   # Shock presence
    polarization_marker: float            # Polarization intensity
    
    @property
    def stress_multiplier(self) -> float:
        """
        CSM = 1 + (α*ED) + (β*SD) + (γ*Shock) + (δ*Polarization)
        Using conservative PDF coefficients (PDF Page 18).
        """
        alpha, beta = 0.15, 0.15
        gamma = 0.20
        delta = 0.15
        
        csm = 1.0 + \
              (alpha * self.economic_dissatisfaction_score) + \
              (beta * self.social_dissatisfaction_score) + \
              (gamma * self.shock_marker) + \
              (delta * self.polarization_marker)
        
        return round(csm, 2)

@dataclass
class AdvancedIndices:
    """
    Specialized Intelligence Layers.
    """
    # 4. Legitimacy Erosion Index (LEI) - PDF Page 31
    lei_score: float # 0-1
    institution_target: str # e.g. "judiciary", "police"
    
    # 1. Susceptibility Index (SI) - PDF Page 24
    si_score: float # 0-1
    cognitive_rigidity: float
    identity_fusion: float
    
    # 5. Narrative Maturation (MS) - PDF Page 34
    maturation_score: float # 0-100
    maturation_stage: str # "Rumor", "Narrative", "Campaign"
    
    # 9. Adversarial Adaptation (AA) - PDF Page 41
    aa_score: float # 0-1
    evasion_technique: str # "codeword", "irony"

# =============================================================================
# 4. AGGREGATE SIGNAL (The Result)
# =============================================================================

@dataclass
class KShieldSignal:
    """
    The Official V3.0 Signal Object.
    """
    source_id: str
    timestamp: datetime
    content_text: str
    
    # Components
    threat: ThreatSignal
    context: ContextAnalysis
    indices: AdvancedIndices
    
    # New V3 Layers
    tta: TimeToAction = TimeToAction.CHRONIC_14D
    resilience: Optional[ResilienceIndex] = None
    role: RoleType = RoleType.OBSERVER
    
    # Calculated Fields
    base_risk: float = 0.0
    adjusted_risk: float = 0.0
    
    def calculate_risk(self):
        """
        AdjustedRisk = min(100, BaseRisk * CSM)
        """
        self.base_risk = self.threat.base_risk_score
        csm = self.context.stress_multiplier
        self.adjusted_risk = min(100.0, self.base_risk * csm)

    @property
    def status(self) -> str:
        """Gating Rules (PDF Page 19)."""
        if self.base_risk < 40:
            return "MONITOR_CONTEXT"
        
        csm = self.context.stress_multiplier
        
        if self.base_risk >= 80:
            return "IMMEDIATE_ESCALATION"
        
        if 60 <= self.base_risk < 80:
            if csm >= 1.10: return "ACTIVE_MONITORING"
            return "ROUTINE_MONITORING"
            
        if 40 <= self.base_risk < 60:
            if csm >= 1.15: return "ANALYST_REVIEW"
            return "LOG_ONLY"
            
        return "UNKNOWN"

# =============================================================================
# 5. MONITORING (Legacy/Support)
# =============================================================================

@dataclass
class MonitoringTarget:
    """
    Persistent Intelligence Target (KIT).
    """
    identifier: str
    target_type: str
    reason: str
    created_at: datetime
    expires_at: datetime
    active: bool = True
    
    def renew(self, days: int = 7):
        from datetime import timedelta
        self.expires_at = datetime.now() + timedelta(days=days)
        
    def drop(self):
        self.active = False

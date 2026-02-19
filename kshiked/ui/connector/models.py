"""
Data Transfer Objects (DTOs) for SENTINEL UI.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

@dataclass
class SignalData:
    """Processed signal for dashboard display."""
    id: int
    name: str
    intensity: float  # 0.0 - 1.0
    trend: str  # "up", "down", "stable"
    count: int
    last_detection: Optional[datetime] = None


@dataclass
class IndexData:
    """Threat index for dashboard display."""
    name: str
    value: float  # 0.0 - 1.0
    severity: str  # "low", "moderate", "high", "critical"
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class CountyRisk:
    """County-level risk assessment."""
    name: str
    risk_score: float  # 0.0 - 1.0
    level: str  # "low", "moderate", "high", "critical"
    top_signals: List[str] = field(default_factory=list)
    trend: str = "stable"
    lat: float = 0.0
    lon: float = 0.0
    is_demo: bool = False


@dataclass
class HypothesisData:
    """Hypothesis for dashboard display."""
    id: str
    relationship_type: str
    variables: List[str]
    confidence: float
    fit_score: float
    state: str
    created_at: float


@dataclass
class AgencyStatus:
    """Federation agency status."""
    id: str
    name: str
    full_name: str
    status: str  # "active", "pending", "offline"
    contribution_score: float
    last_update: Optional[datetime] = None
    rounds_participated: int = 0


@dataclass
class SimulationState:
    """Current simulation state."""
    gdp: float
    inflation: float
    unemployment: float
    interest_rate: float
    exchange_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    # Rich 4D trajectory (list of strict frames)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata for the run
    meta: Dict[str, Any] = field(default_factory=dict)
    # Last frame shortcut
    latest: Optional[Dict[str, Any]] = None
    # Flag to indicate if this is generated demo data
    is_demo: bool = False


@dataclass
class DashboardData:
    """Complete dashboard data package."""
    # Threat status
    threat_level: str = "ELEVATED"
    time_to_escalation: float = 48.0
    
    # Signals and indices
    signals: List[SignalData] = field(default_factory=list)
    indices: List[IndexData] = field(default_factory=list)
    cooccurrence_matrix: Optional[List[List[float]]] = None
    
    # Threat indices (8-index gauge grid from unified dashboard)
    threat_indices: List[Dict[str, Any]] = field(default_factory=list)
    
    # Ethnic tension matrix
    ethnic_tensions: Dict[str, Any] = field(default_factory=dict)
    
    # Network analysis (actor roles)
    network_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Economic satisfaction indicators (ESI by domain)
    esi_indicators: Dict[str, float] = field(default_factory=dict)
    
    # System primitives (scarcity, stress, bonds)
    primitives: Dict[str, Any] = field(default_factory=dict)
    
    # Risk score history for timeline chart
    risk_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Geographic
    counties: Dict[str, CountyRisk] = field(default_factory=dict)
    
    # Causal
    hypotheses: List[HypothesisData] = field(default_factory=list)
    granger_results: List[Dict] = field(default_factory=list)
    causal_graph: Dict[str, Any] = field(default_factory=dict)
    
    # Simulation
    simulation: Optional[SimulationState] = None
    shock_history: List[Dict] = field(default_factory=list)
    
    # Federation
    agencies: List[AgencyStatus] = field(default_factory=list)
    federation_rounds: List[Dict] = field(default_factory=list)
    
    # Meta
    last_update: datetime = field(default_factory=datetime.now)
    data_freshness: str = "live"  # "live", "cached", "demo"

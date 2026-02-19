"""
SENTINEL Dashboard Data Connector

Bridge layer connecting the dashboard to real data sources:
- Scarcity Discovery Engine (hypotheses, relationships)
- KShield Pulse Engine (signals, indices, counties)
- Federation Layer (agency participation, rounds)
- Simulation Engine (economic state, shocks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("sentinel.data_connector")


# =============================================================================
# Data Transfer Objects
# =============================================================================

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


# =============================================================================
# Data Connectors
# =============================================================================

class ScarcityConnector:
    """Connect to Scarcity Discovery Engine."""
    
    def __init__(self):
        self._engine = None
        self._connected = False
        self._training_complete = False
        self._granger_cache: Optional[List[Dict[str, Any]]] = None
    
    def connect(self) -> bool:
        """Try to connect to scarcity engine."""
        try:
            from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
            self._engine = OnlineDiscoveryEngine(explore_interval=10)
            self._connected = True
            
            # Initialize and train on historical data if available
            self._train_on_historical_data()
            
            logger.info("Connected to Scarcity Engine and trained on historical data")
            return True
        except ImportError as e:
            logger.warning(f"Scarcity Engine not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing Scarcity Engine: {e}")
            return False
    
    def _train_on_historical_data(self):
        """Train the engine on historical Kenya economic data."""
        if not self._engine or self._training_complete:
            return

        try:
            # Load data
            try:
                from kenya_data_loader import get_kenya_data_loader
            except ImportError:
                from kenya_data_loader import get_kenya_data_loader
            
            loader = get_kenya_data_loader()
            
            # Broader macro set
            indicators = [
                # Real / prices / labor
                "gdp_current", "gdp_growth", "gdp_per_capita",
                "inflation", "inflation_gdp_deflator", "food_price_index",
                "unemployment", "employment_ratio",
                # External
                "exports_gdp", "imports_gdp", "trade_gdp", "current_account",
                # Fiscal
                "govt_consumption", "tax_revenue", "govt_debt",
                # Monetary / credit
                "real_interest_rate", "broad_money", "private_credit",
                # Social / infra proxies (economy-wide context)
                "population", "urban_population",
                "electricity_access", "internet_users", "mobile_subscriptions",
            ]

            # Map short indicator keys into human-readable labels
            display_name = {
                "gdp_current": "GDP",
                "gdp_growth": "GDP Growth",
                "gdp_per_capita": "GDP Per Capita",
                "inflation": "Inflation",
                "inflation_gdp_deflator": "GDP Deflator Inflation",
                "food_price_index": "Food Prices",
                "unemployment": "Unemployment",
                "employment_ratio": "Employment",
                "exports_gdp": "Exports",
                "imports_gdp": "Imports",
                "trade_gdp": "Trade",
                "current_account": "Current Account",
                "govt_consumption": "Gov Spending",
                "tax_revenue": "Taxes",
                "govt_debt": "Public Debt",
                "real_interest_rate": "Interest Rate",
                "broad_money": "Money Supply",
                "private_credit": "Credit Supply",
                "population": "Population",
                "urban_population": "Urban Population",
                "electricity_access": "Electricity Access",
                "internet_users": "Internet Users",
                "mobile_subscriptions": "Mobile Subscriptions",
            }
            
            # Get variable names
            var_names = [display_name.get(v, v) for v in indicators]
            
            # Initialize engine schema
            schema = {"fields": [{"name": v, "type": "float"} for v in var_names]}
            self._engine.initialize_v2(schema, use_causal=True)
            
            # Feed historical rows
            df = loader.get_historical_trajectory(indicators, start_year=1990)
            # Rename to match schema labels.
            df = df.rename(columns={k: display_name.get(k, k) for k in df.columns})
            
            count = 0
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Clean NaNs
                clean_row = {k: v for k, v in row_dict.items() if str(v) != 'nan'}
                if clean_row:
                    self._engine.process_row(clean_row)
                    count += 1
            
            self._training_complete = True
            self._granger_cache = None  # Invalidate cached causal results
            logger.info(f"Trained Scarcity Engine on {count} historical data points")
            
        except Exception as e:
            logger.error(f"Failed to train on historical data: {e}")
    
    def get_hypotheses(self, limit: int = 50) -> List[HypothesisData]:
        """Get top hypotheses from engine."""
        if not self._connected or not self._engine:
            return []
        
        try:
            # Get graph from engine
            graph = self._engine.get_knowledge_graph()
            
            # Convert to dashboard format
            results = []
            for item in graph:
                # Handle different dict structures from engine versions
                h_type = item.get('type', 'Unknown')
                vars_ = list(item.get('variables', []) or [])
                metrics = item.get('metrics', {}) or {}

                if h_type == "causal" and len(vars_) >= 2:
                    direction = int(metrics.get("direction", 1) or 1)
                    if direction == -1:
                        vars_ = [vars_[1], vars_[0]]
                
                # Filter out single-variable hypotheses if we have enough pairs
                if len(vars_) < 2 and len(graph) > 10:
                    continue
                    
                results.append(HypothesisData(
                    id=str(item.get('id', 'unknown')),
                    relationship_type=str(h_type).replace('Hypothesis', ''),
                    variables=vars_,
                    confidence=float(metrics.get('confidence', item.get('confidence', 0.0))),
                    fit_score=float(metrics.get('fit_score', item.get('fit_score', 0.0))),
                    state=str(item.get('state', 'active')),
                    created_at=float(item.get('created_at', datetime.now().timestamp())),
                ))
            
            if not results:
                return self._get_demo_hypotheses()
                
            return sorted(results, key=lambda x: x.confidence, reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching hypotheses: {e}")
            return []

    def get_granger_results(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Extract Granger-style causal results."""
        if not self._connected or not self._engine or not self._training_complete:
            return []

        if self._granger_cache is not None:
            return self._granger_cache[:limit]

        try:
            from scarcity.engine.discovery import RelationshipType  # type: ignore

            candidates = []
            for hyp in getattr(self._engine, "hypotheses", None).population.values():
                if getattr(hyp, "rel_type", None) != RelationshipType.CAUSAL:
                    continue
                if getattr(hyp, "direction", 0) == 0:
                    continue
                if getattr(hyp, "evidence", 0) < 8:
                    continue
                candidates.append(hyp)

            # Sort by engine confidence
            candidates.sort(key=lambda h: float(getattr(h, "confidence", 0.0)), reverse=True)

            results: List[Dict[str, Any]] = []
            for h in candidates[: max(limit, 100)]:
                src = getattr(h, "source", None)
                tgt = getattr(h, "target", None)
                if not src or not tgt:
                    continue

                direction = int(getattr(h, "direction", 0))
                cause, effect = (src, tgt) if direction == 1 else (tgt, src)

                gain_fwd = float(getattr(h, "gain_forward", 0.0))
                gain_bwd = float(getattr(h, "gain_backward", 0.0))
                strength = max(gain_fwd, gain_bwd)

                conf = float(getattr(h, "confidence", 0.0))
                lag = int(getattr(h, "lag", 2))

                results.append({
                    "cause": str(cause),
                    "effect": str(effect),
                    "lag": lag,
                    "f_stat": strength * 100.0,
                    "p_value": max(0.0, min(1.0, 1.0 - conf)),
                    "significant": conf >= 0.7,
                    "strength": strength,
                    "confidence": conf,
                })

            self._granger_cache = results
            return results[:limit]

        except Exception as e:
            logger.error(f"Error computing Granger results: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status metadata."""
        status = {
            "connected": self._connected,
            "training_complete": self._training_complete,
            "engine_type": "OnlineDiscoveryEngine" if self._engine else "None",
            "nodes": 0,
            "edges": 0,
            "hypotheses": 0
        }
        
        if self._engine and self._connected:
            try:
                graph = self._engine.get_knowledge_graph()
                status["nodes"] = len(set(
                    [n.get('id') for n in graph] if isinstance(graph, list) else []
                ))
                status["edges"] = len(graph)
                
                if hasattr(self._engine, "hypotheses") and hasattr(self._engine.hypotheses, "population"):
                    status["hypotheses"] = len(self._engine.hypotheses.population)
            except Exception:
                pass
                
        return status
    


class PulseConnector:
    """Connect to KShield Pulse Engine."""
    
    SIGNAL_NAMES = [
        "Survival Cost Stress", "Distress Framing", "Emotional Exhaustion", 
        "Directed Rage", "Scapegoating", "Legitimacy Rejection", 
        "Fear Amplification", "Mobilization Language", "Coordination Activity", 
        "Violence Justification", "Intergroup Blame", "Elite Fracture", 
        "Counter-Narrative", "Urgency Escalation", "Protection Seeking",
    ]
    
    INDEX_NAMES = [
        ("Polarization Index", "PI"),
        ("Legitimacy Erosion Index", "LEI"),
        ("Mobilization Readiness Score", "MRS"),
        ("Elite Cohesion Index", "ECI"),
        ("Information Warfare Index", "IWI"),
    ]
    
    # All 8 threat indices for the full gauge grid
    THREAT_INDEX_NAMES = [
        "Polarization", "Legitimacy Erosion", "Mobilization Readiness",
        "Elite Cohesion", "Info Warfare", "Security Friction",
        "Economic Cascade", "Ethnic Tension",
    ]
    
    def __init__(self):
        self._sensor = None
        self._connected = False
        self._streaming = False
    
    def connect(self) -> bool:
        try:
            from kshiked.pulse.sensor import PulseSensor, PulseSensorConfig
            config = PulseSensorConfig(
                min_intensity_threshold=0.05,
                time_decay_lambda=0.001,
                update_interval=5.0
            )
            self._sensor = PulseSensor(config=config, use_nlp=False)
            self._connected = True
            self._inject_live_stream()
            logger.info("Connected to Pulse Engine and started live stream injection")
            return True
        except ImportError as e:
            logger.warning(f"Pulse Engine not available: {e}")
            return False
    
    def _inject_live_stream(self):
        if not self._sensor or self._streaming:
            return
            
        import random
        import threading
        import time
        import asyncio
        from kshiked.pulse.social import TwitterClient, TwitterConfig
        
        # Initialize Synthetic Driver (Auto-detects missing keys -> Synthetic Mode)
        client = TwitterClient(TwitterConfig())
        
        # Async helper for the thread
        def run_async(coro):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        def streamer():
            self._streaming = True
            
            # MOVED: Initialize Data Loader in background thread (was blocking main thread)
            base_pool = []
            try:
                from kshiked.ui.pulse_data_loader import PulseDataLoader
                loader = PulseDataLoader()
                df = loader.load_combined_data()
                if not df.empty:
                     base_pool = list(zip(df['text'].astype(str), df['signal_type'].astype(str)))
                logger.info(f"Loaded {len(base_pool)} base posts for stream injection")
            except Exception as e:
                logger.warning(f"Could not load PulseDataLoader in background: {e}")

            # Authenticate once
            run_async(client.authenticate())
            
            topics = ["inflation", "unrest", "corruption", "police", "election"]
            
            while self._connected:
                # 1. Decide: Real Data or Synthetic Generation?
                if base_pool and random.random() < 0.4:
                    text, _ = random.choice(base_pool)
                else:
                    # 2. Generate detailed synthetic data via Driver
                    topic = random.choice(topics)
                    try:
                        posts = run_async(client.search(topic, max_results=1))
                        if posts:
                            text = posts[0].text
                        else:
                            text = f"Synthetic signal about {topic}" # Fallback
                    except Exception as e:
                        logger.error(f"Synthetic generation failed: {e}")
                        text = f"Error generating signal for {topic}"

                meta = {
                    "location": random.choice(["Nairobi", "Mombasa", "Kisumu", "Eldoret", "Nakuru"]),
                    "platform": "twitter",
                    "author_influence": random.uniform(0.1, 0.9)
                }
                
                self._sensor.process_text(text, meta)
                
                if random.random() < 0.3:
                    self._sensor.update_state()
                    
                time.sleep(random.uniform(0.5, 2.0))
        
        t = threading.Thread(target=streamer, daemon=True)
        t.start()
    
    def get_signals(self) -> List[SignalData]:
        if not self._connected or not self._sensor:
            return []
        
        self._sensor.update_state()
        import time
        now = time.time()
        aggregated = self._sensor._aggregate_signals(now)
        
        results = []
        seen_signals = set()
        for det in aggregated:
            name = det.signal_id.name.replace("_", " ").title()
            seen_signals.add(name)
            results.append(SignalData(
                id=int(det.signal_id.value) if hasattr(det.signal_id, 'value') else hash(name),
                name=name,
                intensity=float(det.intensity),
                trend="stable",
                count=det.context.get("aggregated_count", 1),
                last_detection=datetime.fromtimestamp(det.timestamp),
            ))
            
        for name in self.SIGNAL_NAMES:
            if name.title() not in seen_signals and name not in seen_signals:
                results.append(SignalData(
                    id=hash(name),
                    name=name,
                    intensity=0.0,
                    trend="stable",
                    count=0,
                    last_detection=None
                ))
        return sorted(results, key=lambda x: x.intensity, reverse=True)
    
    
    def get_indices(self) -> List[IndexData]:
        if not self._connected or not self._sensor:
            return []
            
        state = self._sensor.state
        metrics = {
            "PI": state.instability_index,
            "LEI": state.stress.total_system_stress() / 100.0,
            "MRS": state.crisis_probability,
            "ECI": state.bonds.overall_cohesion(),
            "IWI": 0.5
        }
        
        results = []
        for name, abbrev in self.INDEX_NAMES:
            val = float(metrics.get(abbrev, 0.5))
            val = max(0.0, min(1.0, val))
            severity = "critical" if val > 0.8 else "high" if val > 0.6 else "moderate" if val > 0.4 else "low"
            results.append(IndexData(name=name, value=val, severity=severity, components={abbrev: val}))
        return results

    
    def get_county_risks(self) -> Dict[str, CountyRisk]:
        if not self._connected or not self._sensor:
            # Fallback to demo data if not connected, to verify viz
            return self._get_demo_risks()
        
        # Aggregate signals by location
        risks = {}
        # Get all signals from sensor history (Real Data)
        if not hasattr(self._sensor, "_signal_history"):
             return self._get_demo_risks()
        
        detections = self._sensor._signal_history
             
        counts = {}
        top_signals_map = {}
        
        for d in detections:
            loc = d.context.get("location", "Unknown")
            if loc not in counts: 
                counts[loc] = 0.0
                top_signals_map[loc] = []
            
            counts[loc] += d.intensity
            top_signals_map[loc].append(str(d.signal_id.name))
            
        # Normalize
        for loc, total_intensity in counts.items():
            if loc == "Unknown": continue
            # Simple risk model: intensity sum * factor, capped at 1.0
            score = min(1.0, total_intensity * 0.2)
            level = "critical" if score > 0.8 else "high" if score > 0.6 else "moderate" if score > 0.4 else "low"
            signal_counts: Dict[str, int] = {}
            for sig in top_signals_map.get(loc, []):
                signal_counts[sig] = signal_counts.get(sig, 0) + 1
            top_signals = [k.replace("_", " ").title() for k, _ in sorted(signal_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]]
            risks[loc] = CountyRisk(
                name=loc,
                risk_score=score,
                level=level,
                top_signals=top_signals,
                is_demo=False
            )
            
        if not risks:
            return self._get_demo_risks()
            
        return risks

    def get_correlation_matrix(self) -> List[List[float]]:
        """Get the signal co-occurrence matrix."""
        if not self._connected or not self._sensor:
             return []
        
        # Use new getter on sensor
        if hasattr(self._sensor, "get_correlation_matrix"):
            matrix = self._sensor.get_correlation_matrix()
            # Convert numpy to list for JSON serialization
            import numpy as np
            if isinstance(matrix, np.ndarray):
                return matrix.tolist()
            return matrix
        return []

    def _get_demo_risks(self):
        """Return dummy data for viz verification."""
        return {
            "Nairobi": CountyRisk("Nairobi", 0.75, "high", ["Unrest"], trend="up", is_demo=True),
            "Mombasa": CountyRisk("Mombasa", 0.55, "moderate", ["Inflation"], trend="stable", is_demo=True),
            "Kisumu": CountyRisk("Kisumu", 0.82, "critical", ["Protests"], trend="up", is_demo=True),
            "Turkana": CountyRisk("Turkana", 0.45, "moderate", ["Drought"], trend="stable", is_demo=True),
            "Nakuru": CountyRisk("Nakuru", 0.35, "low", [], trend="down", is_demo=True),
        }
    
    def get_threat_indices(self) -> List[Dict[str, Any]]:
        """Get all 8 threat indices for the gauge grid."""
        if not self._connected or not self._sensor:
            return self._get_demo_threat_indices()
        
        try:
            from kshiked.pulse.threat_index import compute_threat_report
            
            state = self._sensor.state
            history = getattr(self._sensor, '_signal_history', [])
            report = compute_threat_report(state, history)
            
            return [
                {"name": "Polarization", "value": report.polarization.value, "severity": report.polarization.severity},
                {"name": "Legitimacy Erosion", "value": report.legitimacy_erosion.value, "severity": report.legitimacy_erosion.severity},
                {"name": "Mobilization", "value": report.mobilization_readiness.value, "severity": report.mobilization_readiness.severity},
                {"name": "Elite Cohesion", "value": report.elite_cohesion.value, "severity": report.elite_cohesion.severity},
                {"name": "Info Warfare", "value": report.information_warfare.value, "severity": report.information_warfare.severity},
                {"name": "Security", "value": report.security_friction.value, "severity": report.security_friction.severity},
                {"name": "Economic", "value": report.economic_cascade.value, "severity": report.economic_cascade.severity},
                {"name": "Ethnic Tension", "value": report.ethnic_tension.avg_tension, "severity": report.ethnic_tension.severity},
            ]
        except Exception as e:
            logger.warning(f"Could not compute threat indices: {e}")
            return self._get_demo_threat_indices()
    
    def get_ethnic_tensions(self) -> Dict[str, Any]:
        """Get ethnic tension matrix data."""
        if not self._connected or not self._sensor:
            return self._get_demo_ethnic_tensions()
        
        try:
            from kshiked.pulse.threat_index import compute_threat_report
            
            state = self._sensor.state
            history = getattr(self._sensor, '_signal_history', [])
            report = compute_threat_report(state, history)
            
            et = report.ethnic_tension
            return {
                "tensions": et.tensions,
                "highest_pair": et.highest_tension_pair,
                "avg_tension": et.avg_tension,
                "severity": et.severity,
            }
        except Exception as e:
            logger.warning(f"Could not compute ethnic tensions: {e}")
            return self._get_demo_ethnic_tensions()
    
    def get_network_analysis(self) -> Dict[str, Any]:
        """Get actor network analysis."""
        if not self._connected or not self._sensor:
            return {
                "roles": {},
                "node_count": 0,
                "edge_count": 0,
                "community_count": 0,
            }

        try:
            history = getattr(self._sensor, "_signal_history", [])
            if not history:
                return {
                    "roles": {},
                    "node_count": 0,
                    "edge_count": 0,
                    "community_count": 0,
                }

            roles: Dict[str, int] = {
                "Mobilizer": 0,
                "Amplifier": 0,
                "Broker": 0,
                "Ideologue": 0,
                "Influencer": 0,
                "Unknown": 0,
            }
            locations = set()
            actor_nodes = set()

            for det in history[-1000:]:
                ctx = getattr(det, "context", {}) or {}
                sid = str(getattr(getattr(det, "signal_id", None), "name", "")).lower()
                influence = float(ctx.get("author_influence", 0.0) or 0.0)
                location = str(ctx.get("location", "Unknown"))
                platform = str(ctx.get("platform", "unknown"))

                locations.add(location)
                actor_nodes.add((platform, location, round(influence, 1)))

                if influence >= 0.75:
                    roles["Influencer"] += 1
                elif "mobilization" in sid or "coordination" in sid:
                    roles["Mobilizer"] += 1
                elif "counter" in sid or "legitimacy" in sid:
                    roles["Ideologue"] += 1
                elif "rage" in sid or "blame" in sid or "fear" in sid:
                    roles["Amplifier"] += 1
                elif "broker" in sid or "elite_fracture" in sid:
                    roles["Broker"] += 1
                else:
                    roles["Unknown"] += 1

            roles = {k: v for k, v in roles.items() if v > 0}
            return {
                "roles": roles,
                "node_count": len(actor_nodes),
                "edge_count": max(0, len(history[-1000:]) - 1),
                "community_count": max(1, len([loc for loc in locations if loc and loc != "Unknown"])),
            }
        except Exception as e:
            logger.warning(f"Could not compute network analysis: {e}")
            return {
                "roles": {},
                "node_count": 0,
                "edge_count": 0,
                "community_count": 0,
            }
    
    def get_esi_indicators(self) -> Dict[str, float]:
        """Get economic satisfaction indicators by domain."""
        if not self._connected or not self._sensor:
            return self._get_demo_esi()
        
        # Use Pulse state scarcity values mapped to satisfaction
        state = self._sensor.state
        try:
            from kshiked.pulse.primitives import ResourceDomain
            
            # Check if we have data, otherwise fallback
            if not state.scarcity:
                 return self._get_demo_esi()

            return {
                "Food": max(0, 1.0 - state.scarcity.get(ResourceDomain.FOOD, 0.5)),
                "Fuel": max(0, 1.0 - state.scarcity.get(ResourceDomain.FUEL, 0.5)),
                "Housing": max(0, 1.0 - state.scarcity.get(ResourceDomain.HOUSING, 0.5)),
                "Healthcare": max(0, 1.0 - state.scarcity.get(ResourceDomain.HEALTHCARE, 0.5)),
                "Transport": max(0, 1.0 - state.scarcity.get(ResourceDomain.EMPLOYMENT, 0.5)),
            }
        except Exception as e:
            logger.warning(f"Error getting ESI: {e}")
            return self._get_demo_esi()

        return {
            "Food": 0.45,
            "Fuel": 0.30,
            "Housing": 0.60,
            "Healthcare": 0.55,
            "Transport": 0.40,
        }
    
    def get_primitives(self) -> Dict[str, Any]:
        """Get pulse engine primitives (scarcity, stress, bonds)."""
        if not self._connected or not self._sensor:
            return self._get_demo_primitives()
        
        state = self._sensor.state
        try:
            from kshiked.pulse.primitives import ResourceDomain, ActorType
            
            scarcity = {}
            for domain in ResourceDomain:
                scarcity[domain.value] = state.scarcity.get(domain)
            
            stress = {}
            for actor in ActorType:
                stress[actor.value] = state.stress.get_stress(actor)
            
            bonds = {
                "national_cohesion": state.bonds.national_cohesion,
                "class_solidarity": state.bonds.class_solidarity,
                "regional_unity": state.bonds.regional_unity,
                "fragility": state.bonds.fragility_score(),
            }
            
            state.compute_risk_metrics()
            
            return {
                "scarcity": scarcity,
                "aggregate_scarcity": state.scarcity.aggregate_score(),
                "stress": stress,
                "total_stress": state.stress.total_system_stress(),
                "bonds": bonds,
                "instability_index": state.instability_index,
                "crisis_probability": state.crisis_probability,
            }
        except Exception as e:
            logger.warning(f"Could not read primitives: {e}")
            return self._get_demo_primitives()
    
    def get_risk_history(self) -> List[Dict[str, Any]]:
        """Get risk score history for timeline chart."""
        import time as _time
        if not self._connected or not self._sensor:
            return []
        
        try:
            history = getattr(self._sensor, '_signal_history', [])
            if not history:
                return []
            
            # Build timeline from signal history buckets
            now = _time.time()
            buckets = {}
            for det in history:
                # 10-second buckets
                bucket = int(det.timestamp / 10) * 10
                if bucket not in buckets:
                    buckets[bucket] = {"intensities": [], "count": 0}
                buckets[bucket]["intensities"].append(det.intensity)
                buckets[bucket]["count"] += 1
            
            result = []
            for ts, data in sorted(buckets.items()):
                avg = sum(data["intensities"]) / len(data["intensities"])
                peak = max(data["intensities"])
                result.append({
                    "timestamp": ts,
                    "overall_risk": avg,
                    "peak_risk": peak,
                    "signal_count": data["count"],
                })
            
            return result[-50:]  # Last 50 buckets
        except Exception as e:
            logger.warning(f"Could not build risk history: {e}")
            return []
    
    # Demo data fallbacks
    @staticmethod
    def _get_demo_threat_indices() -> List[Dict[str, Any]]:
        return [
            {"name": "Polarization", "value": 0.42, "severity": "MODERATE"},
            {"name": "Legitimacy Erosion", "value": 0.55, "severity": "ELEVATED"},
            {"name": "Mobilization", "value": 0.38, "severity": "MODERATE"},
            {"name": "Elite Cohesion", "value": 0.30, "severity": "LOW"},
            {"name": "Info Warfare", "value": 0.61, "severity": "HIGH"},
            {"name": "Security", "value": 0.25, "severity": "LOW"},
            {"name": "Economic", "value": 0.70, "severity": "HIGH"},
            {"name": "Ethnic Tension", "value": 0.45, "severity": "MODERATE"},
        ]
    
    @staticmethod
    def _get_demo_ethnic_tensions() -> Dict[str, Any]:
        return {
            "tensions": {
                "Kikuyu-Luo": 0.65, "Kikuyu-Kalenjin": 0.45,
                "Luo-Kalenjin": 0.30, "Kikuyu-Luhya": 0.25,
                "Luo-Luhya": 0.20, "Kalenjin-Luhya": 0.15,
            },
            "highest_pair": ("Kikuyu", "Luo"),
            "avg_tension": 0.33,
            "severity": "MODERATE",
        }
    
    @staticmethod
    def _get_demo_esi() -> Dict[str, float]:
        return {"Food": 0.65, "Fuel": 0.45, "Housing": 0.55, "Healthcare": 0.60, "Transport": 0.50}
    
    @staticmethod
    def _get_demo_primitives() -> Dict[str, Any]:
        return {
            "scarcity": {"food": 0.35, "fuel": 0.55, "housing": 0.45, "medicine": 0.40, "transport": 0.50},
            "aggregate_scarcity": 0.45,
            "stress": {"government": -0.2, "military": 0.1, "civilians": 0.4, "media": 0.3},
            "total_stress": 0.60,
            "bonds": {"national_cohesion": 0.55, "class_solidarity": 0.40, "regional_unity": 0.50, "fragility": 0.45},
            "instability_index": 0.42,
            "crisis_probability": 0.18,
        }


class FederationConnector:
    """Connect to Federation Layer (Aegis Protocol)."""
    
    def __init__(self):
        self._simulator = None
        self._connected = False
    
    def connect(self) -> bool:
        try:
            from kshiked.federation.integration import DefenseFederationSimulator
            self._simulator = DefenseFederationSimulator()
            self._connected = True
            logger.info("Connected to Aegis Defense Federation")
            return True
        except ImportError as e:
            logger.warning(f"Aegis Federation not available: {e}")
            return False
    
    def get_agency_status(self) -> List[AgencyStatus]:
        if not self._connected or not self._simulator:
             return self._get_demo_agencies()
        
        # Pull live stats from Aegis
        state = self._simulator.tick()
        agencies = state["agencies"]
        
        results = []
        for a in agencies:
            # Map Aegis node data to dashboard DTO
            status_str = "active" if a["links_count"] > 0 else "pending"
            role = "Top Secret" if a.get("clearance") == "TOP_SECRET" else "Secret"
            
            results.append(AgencyStatus(
                id=a["id"],
                name=a["id"].replace("-Nexus", ""),
                full_name=f"{role} Clearance Node",
                status=status_str,
                contribution_score=min(1.0, a["links_count"] / 10.0), # Heuristic
                rounds_participated=state["round"],
                last_update=datetime.now()
            ))
            
        return results
    
    def get_rounds(self, limit: int = 20) -> List[Dict]:
        if not self._connected or not self._simulator:
            return []
            
        # In a real app we'd query history. Here we return the latest tick info.
        # This is a bit of a hack since tick() advances state.
        # Ideally, detailed logs would be fetched separately.
        return [
            {
                "round": self._simulator.round_id,
                "participants": 3,
                "convergence": 0.95, # Mock
                "delta_norm": 0.05,
                "timestamp": datetime.now().timestamp()
            }
        ]

    def _get_demo_agencies(self):
        # Fallback
        import random
        return [
             AgencyStatus("NIS", "NIS", "National Intelligence", "active", 0.9, 10, datetime.now()),
             AgencyStatus("KDF", "KDF", "Defense Forces", "active", 0.8, 10, datetime.now())
        ]


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
            from kenya_data_loader import get_latest_economic_state
            return get_latest_economic_state()
        except ImportError:
            return {}


def get_dashboard_data(force_causal: bool = False) -> DashboardData:
    """
    Aggregate all data sources into a single dashboard payload.
    
    Args:
        force_causal: If True, force retraining of causal models.
    """
    logger.info(f"Fetching dashboard data (force_causal={force_causal})")
    
    # 1. Initialize Connectors
    scarcity = ScarcityConnector()
    pulse = PulseConnector()
    federation = FederationConnector()
    simulation = SimulationConnector()
    
    # 2. Connect in parallel (each connector is independent)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    connectors = [scarcity, pulse, federation, simulation]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(c.connect): c for c in connectors}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.warning(f"Connector failed: {exc}")
    
    # 3. Fetch Data
    # Pulse (Real-time signals)
    try:
        signals = pulse.get_signals()
        indices = pulse.get_indices()
        counties = pulse.get_county_risks()
        matrix = pulse.get_correlation_matrix()
        
        # Calculate derived threat level
        max_idx = max([i.value for i in indices]) if indices else 0.5
        if max_idx > 0.8: threat = "CRITICAL"
        elif max_idx > 0.6: threat = "HIGH"
        elif max_idx > 0.4: threat = "ELEVATED"
        else: threat = "LOW"
        
        escalation_time = max(0.0, 48.0 - (max_idx * 40.0)) # heuristic
        
    except Exception as e:
        logger.error(f"Error fetching pulse data: {e}")
        signals, indices, counties = [], [], {}
        threat, escalation_time = "UNKNOWN", 48.0

    # Scarcity (Causal)
    try:
        hypotheses = scarcity.get_hypotheses()
        granger = scarcity.get_granger_results()
        graph = scarcity.get_status()
    except Exception as e:
        logger.error(f"Error fetching scarcity data: {e}")
        hypotheses, granger, graph = [], [], {}

    # Federation (Agencies)
    try:
        agencies = federation.get_agency_status()
        rounds = federation.get_rounds()
    except Exception as e:
        logger.error(f"Error fetching federation data: {e}")
        agencies, rounds = [], []

    # Simulation (Economic State)
    try:
        sim_state = simulation.get_state()
    except Exception as e:
        logger.error(f"Error fetching simulation data: {e}")
        sim_state = None

    # 4. Construct Data Package
    # Merged features: threat indices, ethnic tensions, network, ESI, primitives, risk history
    try:
        threat_indices = pulse.get_threat_indices()
        ethnic_tensions = pulse.get_ethnic_tensions()
        network_analysis = pulse.get_network_analysis()
        esi_indicators = pulse.get_esi_indicators()
        primitives = pulse.get_primitives()
        risk_history = pulse.get_risk_history()
    except Exception as e:
        logger.error(f"Error fetching merged data: {e}")
        threat_indices, ethnic_tensions = [], {}
        network_analysis, esi_indicators, primitives = {}, {}, {}
        risk_history = []

    return DashboardData(
        threat_level=threat,
        time_to_escalation=escalation_time,
        signals=signals,
        indices=indices,
        cooccurrence_matrix=matrix,
        threat_indices=threat_indices,
        ethnic_tensions=ethnic_tensions,
        network_analysis=network_analysis,
        esi_indicators=esi_indicators,
        primitives=primitives,
        risk_history=risk_history,
        counties=counties,
        granger_results=granger,
        causal_graph=graph,
        agencies=agencies,
        federation_rounds=rounds,
        simulation=sim_state,
        last_update=datetime.now()
    )

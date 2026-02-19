"""
K-Shield Pulse Engine Connector.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Dict, Any

from .models import SignalData, IndexData, CountyRisk

logger = logging.getLogger("sentinel.connector.pulse")

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

    def _get_demo_esi(self):
        """Return dummy ESI data."""
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
            # Try to start connection if accessed before connect()
            return self._get_demo_primitives()
            
        state = self._sensor.state
        
        # Check if we have data
        if not state.scarcity.values and not state.stress._values:
             return self._get_demo_primitives()

        return {
            "instability_index": state.instability_index,
            "system_stress": state.stress.total_system_stress(),
            "crisis_probability": state.crisis_probability,
            "network_cohesion": state.bonds.overall_cohesion(),
            "aggregate_scarcity": state.scarcity.aggregate_score(),
        }

    def _get_demo_primitives(self):
        return {
            "instability_index": 0.65,
            "system_stress": 72.5,
            "crisis_probability": 0.48,
            "network_cohesion": 0.35,
            "aggregate_scarcity": 0.55,
        }
    
    def get_risk_history(self) -> List[Dict[str, Any]]:
        """Get risk score history for timeline chart."""
        if not self._connected or not self._sensor:
            return self._get_demo_risk_history()
            
        # Use sensor state history or generate synthetic history around current point
        history = getattr(self._sensor, "_state_history", [])
        if not history:
            return self._get_demo_risk_history()
            
        return [
            {
                "timestamp": h.get("timestamp", 0),
                "instability": h.get("instability", 0.5),
                "stress": h.get("stress", 50.0) / 100.0
            } 
            for h in history
        ]

    def _get_demo_risk_history(self):
        import time
        now = time.time()
        return [
            {"timestamp": now - 3600*4, "instability": 0.4, "stress": 0.3},
            {"timestamp": now - 3600*3, "instability": 0.5, "stress": 0.4},
            {"timestamp": now - 3600*2, "instability": 0.65, "stress": 0.55},
            {"timestamp": now - 3600*1, "instability": 0.6, "stress": 0.5},
            {"timestamp": now, "instability": 0.7, "stress": 0.65},
        ]
    
    def _get_demo_threat_indices(self):
        return [
            {"name": "Polarization", "value": 0.7, "severity": "high"},
            {"name": "Legitimacy Erosion", "value": 0.6, "severity": "high"},
            {"name": "Mobilization", "value": 0.8, "severity": "critical"},
            {"name": "Elite Cohesion", "value": 0.4, "severity": "moderate"},
            {"name": "Info Warfare", "value": 0.5, "severity": "moderate"},
            {"name": "Security", "value": 0.3, "severity": "low"},
            {"name": "Economic", "value": 0.75, "severity": "high"},
            {"name": "Ethnic Tension", "value": 0.4, "severity": "moderate"},
        ]
        
    def _get_demo_ethnic_tensions(self):
        return {
            "tensions": {"Kikuyu-Luo": 0.6, "Kalenjin-Kikuyu": 0.4},
            "highest_pair": ("Kikuyu", "Luo"),
            "avg_tension": 0.3,
            "severity": "moderate",
        }

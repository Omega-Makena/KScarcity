"""
Data Loader for SENTINEL UI.
Aggregates data from multiple connectors into a DashboardData object.
"""
from __future__ import annotations
import logging
from datetime import datetime
import streamlit as st

from .models import DashboardData
from .scarcity import ScarcityConnector
from .pulse import PulseConnector
from .federation import FederationConnector
from .simulation import SimulationConnector

logger = logging.getLogger("sentinel.connector.loader")

# =============================================================================
# Singleton Connectors
# =============================================================================

@st.cache_resource
def get_scarcity_connector() -> ScarcityConnector:
    connector = ScarcityConnector()
    if connector.connect():
        return connector
    return connector  # Return even if not connected, for graceful degradation

@st.cache_resource
def get_pulse_connector() -> PulseConnector:
    connector = PulseConnector()
    if connector.connect():
        return connector
    return connector

@st.cache_resource
def get_federation_connector() -> FederationConnector:
    connector = FederationConnector()
    if connector.connect():
        return connector
    return connector

@st.cache_resource
def get_simulation_connector() -> SimulationConnector:
    connector = SimulationConnector()
    if connector.connect():
        return connector
    return connector

def get_dashboard_data(force_causal: bool = False) -> DashboardData:
    """
    Fetch and aggregate all dashboard data.
    
    Args:
        force_causal: If True, force re-training/calculation of causal graph.
    """
    logger.info(f"Fetching dashboard data (force_causal={force_causal})")
    
    # 1. Connect to engines (cached singletons)
    scarcity = get_scarcity_connector()
    pulse = get_pulse_connector()
    federation = get_federation_connector()
    simulation = get_simulation_connector()
    
    # 2. Build Dashboard Data Object (initially empty)
    # 3. Fetch Data in blocks (could be parallelized if needed, but cached connectors are fast)
    
    # Scarcity
    hypotheses = scarcity.get_hypotheses()
    granger = scarcity.get_granger_results()
    graph = scarcity.get_status()
    
    # Pulse
    signals = pulse.get_signals()
    indices = pulse.get_indices()
    counties = pulse.get_county_risks()
    matrix = pulse.get_correlation_matrix()
    threat_indices = pulse.get_threat_indices()
    ethnic_tensions = pulse.get_ethnic_tensions()
    network_analysis = pulse.get_network_analysis()
    esi_indicators = pulse.get_esi_indicators()
    primitives = pulse.get_primitives()
    risk_history = pulse.get_risk_history()
    
    # Calculate simple threat level
    max_idx = max([i.value for i in indices]) if indices else 0.5
    if max_idx > 0.8: threat = "CRITICAL"
    elif max_idx > 0.6: threat = "HIGH"
    elif max_idx > 0.4: threat = "ELEVATED"
    else: threat = "LOW"
    escalation_time = max(0.0, 48.0 - (max_idx * 40.0))
    
    # Federation
    agencies = federation.get_agency_status()
    rounds = federation.get_rounds()
    
    # Simulation
    sim_state = simulation.get_state()
    
    # 4. Construct Data Package
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

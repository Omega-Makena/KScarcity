import streamlit as st
import sys
import os
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import enforce_role
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.backend.ontology import OntologyEnforcer
from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.scarcity_bridge import ScarcityBridge
from kshiked.ui.theme import LIGHT_THEME
from kshiked.ui.institution.style import inject_enterprise_theme
from kshiked.ui.institution.backend.messaging import SecureMessaging
from kshiked.ui.kshield.causal.view import (
    _render_granger_section,
    _render_causal_network,
    _render_cross_corr
)

def render():
    enforce_role(Role.INSTITUTION.value)
    inject_enterprise_theme()
    
    inst_id = st.session_state.get('institution_id')
    basket_id = st.session_state.get('basket_id')
    
    st.markdown(f"<h2 style='text-align: center; color: #006747;'>Institution Silo [Spoke]</h2>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center; color: #1F2937;'>Node: {st.session_state.get('username')} | Sector Basket: {basket_id}</h5>", unsafe_allow_html=True)
    
    st.write("---")
    
    tab1, tab2, tab_granger, tab_network, tab_cross, tab3 = st.tabs([
        "Ontology & Intake", 
        "Local Engine Discovery", 
        "Granger Causality", 
        "Causal Network", 
        "Cross-Correlations",
        "Downstream Updates"
    ])
    
    with tab1:
        st.info("### Pre-Flight Data Verification")
        st.write("Before the Scarcity Engine can boot, your local data must strictly conform to the Basket's Semantic Dictionary.")
        
        schema = OntologyEnforcer.get_basket_schema(basket_id)
        if schema:
            st.write("**Global Basket Schema Rules:**")
            st.json(schema)
        else:
            st.error("No Global Schema defined for this Basket. Engine initialization is locked.")
            st.stop()
            
        uploaded_file = st.file_uploader("Upload Highly Sensitive Local Data (CSV) - Remainder: Data never leaves this node.", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Local Data Preview:")
            st.dataframe(df.head(3))
            
            is_valid, message = OntologyEnforcer.validate_dataset_signature(basket_id, list(df.columns))
            
            if is_valid:
                st.success(message)
                st.session_state['local_df'] = df
            else:
                st.error(message)
                st.session_state['local_df'] = None

    with tab2:
        st.success("### Your Local Scarcity Engine")
        st.write("Executing the Anomaly Detection and Prediction Engine over your mathematically-secured private data.")
        
        if st.session_state.get('local_df') is not None:
            if st.button("Run Deep Engine Analysis", use_container_width=True):
                with st.spinner("Initializing Mathematical Engine..."):
                    bridge = ScarcityBridge()
                    results = bridge.process_dataframe(st.session_state['local_df'], basket_schema=schema)
                    st.session_state['engine_results'] = results
                    
            if 'engine_results' in st.session_state:
                res = st.session_state['engine_results']
                anoms = res.get('anomalies', [])
                
                if anoms:
                    st.write("---")
                    st.write("### Real-Time Anomaly Detection")
                    
                    # Synthesize dynamic interpretation of the anomalies
                    max_anomaly = max(anoms) if anoms else 0.0
                    peak_idx = anoms.index(max_anomaly) if anoms else 0
                    if max_anomaly > 2.0:
                        interpretation = (
                            f"**Critical Structural Drift Detected.** The RRCF engine identified a severe anomaly "
                            f"(Peak Severity: **{max_anomaly:.2f}**) at time step **{peak_idx}** within the mathematical window. "
                            f"This indicates a severe breakdown of established topological relationships across the input dimensions."
                        )
                    elif max_anomaly > 1.0:
                        interpretation = (
                            f"**Moderate Volatility.** The data shows noticeable noise (Peak Severity: **{max_anomaly:.2f}**) "
                            f"but remains generally within the bounds of historical expectation."
                        )
                    else:
                        interpretation = (
                            f"**Baseline Stability.** The structural relationships are highly stable (Peak Severity: **{max_anomaly:.2f}**). "
                            f"No significant anomalies detected."
                        )
                    
                    st.markdown(
                        f'<div style="background:rgba(0,103,71,0.05); border-left:4px solid #006747; padding:1rem; margin-bottom:1rem; border-radius:0 8px 8px 0;">'
                        f'<div style="color:#006747; font-weight:bold; margin-bottom:0.5rem; font-size:0.9rem;">DYNAMIC INTERPRETATION</div>'
                        f'<div style="font-size:0.9rem; color:#1F2937;">{interpretation}</div></div>',
                        unsafe_allow_html=True
                    )
                    
                    st.line_chart(anoms)
                    
                    # Automate the async delta queue push if confidence > a threshold
                    max_anomaly = max(anoms) if anoms else 0.0
                    if max_anomaly > 2.0:
                        st.error(f"High-Confidence Structural Drift Detected (Peak Anomaly Severity: {max_anomaly:.2f}). Pushing cryptographic Delta to Basket Admin...")
                        
                        # Extract deep context without sending the entire raw dataset
                        shock_vector = {}
                        if st.session_state.get('local_df') is not None:
                            df = st.session_state['local_df']
                            # Grab the state of the world at the peak of the anomaly
                            # peak_idx is the row index in the anomaly array, which maps to the dataframe index
                            safe_idx = min(peak_idx, len(df)-1)
                            snapshot = df.iloc[safe_idx].to_dict()
                            baseline = df.iloc[max(0, safe_idx - 30)].to_dict() # State 30 steps prior for context
                            
                            for col in schema["required_columns"]:
                                if col in snapshot and col in baseline:
                                    shock_vector[col] = {
                                        "pre_shock_baseline": round(baseline[col], 3),
                                        "peak_shock_value": round(snapshot[col], 3),
                                        "delta_magnitude": round(snapshot[col] - baseline[col], 3)
                                    }
                        
                        # Grab the variance bounds from the forecaster if available
                        garch_context = {}
                        if res.get('forecasts'):
                            last_f = res['forecasts'][-1]
                            vars_matrix = last_f['variances']
                            num_dims = len(vars_matrix[0]) if vars_matrix else 0
                            cols = schema["required_columns"] if (schema and "required_columns" in schema) else [f"Dim_{i}" for i in range(num_dims)]
                            volatilities = [np.mean(step) for step in np.array(vars_matrix).T]
                            for i, col in enumerate(cols):
                                if i < len(volatilities):
                                    garch_context[col] = round(volatilities[i], 3)
                        
                        # --- Composite Intelligence Scoring ---
                        # A) Detection Score (Did something unusual happen?)
                        # Combine base anomaly score with an estimate of data distribution shift
                        base_anomaly = min(10.0, max_anomaly)
                        drift_proxy = np.std(anoms[-30:]) if len(anoms) > 30 else 0.5 
                        detection_score = round(min(10.0, (base_anomaly * 0.7) + (drift_proxy * 3.0)), 2)
                        
                        # B) Impact Score (If true, how bad?)
                        # Combine magnitude of shocks and forecasted volatility
                        avg_shock_mag = np.mean([abs(v["delta_magnitude"]) for v in shock_vector.values()]) if shock_vector else 0.0
                        avg_volatility = np.mean(list(garch_context.values())) if garch_context else 0.0
                        impact_score = round(min(10.0, (avg_shock_mag * 2.0) + (avg_volatility * 1.5)), 2)
                        
                        # C) Certainty Score (How confident are we?)
                        # Based on model agreement/integrity. We proxy this using the inverse of variance bounds width.
                        certainty_proxy = max(0.0, 10.0 - (avg_volatility * 2.0))
                        certainty_score = round(min(10.0, certainty_proxy), 2)
                        
                        composite_scores = {
                            "A_Detection": detection_score,
                            "B_Impact": impact_score,
                            "C_Certainty": certainty_score
                        }
                        
                        insight = {
                            "incident_type": "SEVERE_STRUCTURAL_DRIFT", 
                            "detection_engine": "Numba RRCF CoDispersion",
                            "severity_score": round(max_anomaly, 2),
                            "composite_scores": composite_scores,
                            "time_index": peak_idx,
                            "shock_vector": shock_vector,
                            "post_shock_volatility_forecast": garch_context,
                            "spoke_interpretation": interpretation,
                            "local_weights": list(volatilities) if 'volatilities' in locals() else [1.0] * len(schema["required_columns"])
                        }
                        sync_id = DeltaSyncManager.queue_insight(inst_id, basket_id, insight)
                        st.success(f"Massive Contextual Delta Queued securely to Basket Admin (Sync ID: {sync_id})")
                else:
                    st.write("No anomaly data generated.")
                    
                # Render GARCH VARX info
                if res.get('forecasts'):
                    st.write("---")
                    st.write("### Trajectory Prediction with Uncertainty Bounds")
                    
                    forecasts = res['forecasts']
                    last_f = forecasts[-1]
                    forecast_matrix = last_f['forecasts']
                    variance_matrix = last_f['variances']
                    num_dims = len(forecast_matrix[0]) if forecast_matrix else 0
                    cols = schema["required_columns"] if (schema and "required_columns" in schema) else [f"Dim_{i}" for i in range(num_dims)]
                    
                    # Synthesize dynamic interpretation of the VARX bounds
                    if num_dims > 0:
                        volatilities = [np.mean(step) for step in np.array(variance_matrix).T]
                        max_vol_idx = int(np.argmax(volatilities))
                        min_vol_idx = int(np.argmin(volatilities))
                        
                        trajectory_interpreters = []
                        for k in range(num_dims):
                            start_val = forecast_matrix[0][k]
                            end_val = forecast_matrix[-1][k]
                            trend = "rising" if end_val > start_val + 0.1 else "falling" if end_val < start_val - 0.1 else "stable"
                            trajectory_interpreters.append(trend)
                        
                        f_interpretation = (
                            f"Over the next 5-step projection, the Bayesian engine forecasts that **{cols[max_vol_idx]}** "
                            f"will experience the highest structural volatility (GARCH bound amplitude: {volatilities[max_vol_idx]:.2f}). "
                            f"Conversely, **{cols[min_vol_idx]}** remains the most predictable dimension. "
                            f"Overall, the structural momentum is actively driving a **{trajectory_interpreters[max_vol_idx]}** trend in the most volatile variables."
                        )
                        
                        st.markdown(
                            f'<div style="background:rgba(0,103,71,0.05); border-left:4px solid #006747; padding:1rem; margin-bottom:1rem; border-radius:0 8px 8px 0;">'
                            f'<div style="color:#006747; font-weight:bold; margin-bottom:0.5rem; font-size:0.9rem;">DYNAMIC INTERPRETATION</div>'
                            f'<div style="font-size:0.9rem; color:#1F2937;">{f_interpretation}</div></div>',
                            unsafe_allow_html=True
                        )
                    
                    # Display the multi-dimensional plots in columns
                    num_cols = min(3, len(cols))
                    plot_cols = st.columns(num_cols)
                    
                    local_df_history = st.session_state['local_df']
                    
                    for k, col_name in enumerate(cols):
                        with plot_cols[k % num_cols]:
                            if k < num_dims:
                                # Get historical data
                                history_y = local_df_history[col_name].values[-100:]  # last 100 historical steps for context
                                history_x = list(range(len(history_y)))
                                
                                # Forecast matrix is shape [steps, D]. Extract k-th dim across steps.
                                y_mean = [step[k] for step in forecast_matrix] # T+5 forecast array
                                y_var = [step[k] for step in variance_matrix]
                                
                                forecast_x = list(range(len(history_y), len(history_y) + len(y_mean)))
                                
                                # Confidence interval bounds (1 sigma)
                                upper = np.array(y_mean) + np.sqrt(y_var)
                                lower = np.array(y_mean) - np.sqrt(y_var)
                                
                                fig_dyn = go.Figure()
                                # Context History
                                fig_dyn.add_scatter(x=history_x, y=history_y, mode='lines', line=dict(color='#888888', width=2), name='Actual')
                                
                                # GARCH Bounds
                                fig_dyn.add_scatter(x=forecast_x, y=upper, mode='lines', line=dict(width=0), showlegend=False)
                                fig_dyn.add_scatter(x=forecast_x, y=lower, mode='lines', fill='tonexty', fillcolor='rgba(0,255,255,0.2)', line=dict(width=0), showlegend=False)
                                
                                # Forecast Mean
                                fig_dyn.add_scatter(x=forecast_x, y=y_mean, mode='lines', line=dict(color='#00FFFF', width=2), name='Forecast')
                                
                                fig_dyn.update_layout(
                                    height=250, 
                                    margin=dict(l=0,r=0,t=30,b=0), 
                                    title=f"**{col_name.replace('_', ' ').title()}**", 
                                    paper_bgcolor="rgba(0,0,0,0)", 
                                    plot_bgcolor="rgba(0,0,0,0)", 
                                    showlegend=False
                                )
                                st.plotly_chart(fig_dyn, use_container_width=True)
                                
                    st.write("---")
                    st.write("### Data Feature Correlations")
                    st.caption("Interpretation: This matrix reveals the hidden relationships between your variables. Dark Red indicates they move together (positive correlation). Dark Blue implies they move in opposite directions. White means there is no measurable relationship.")
                    corr_matrix = local_df_history[cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix, 
                        color_continuous_scale="RdBu", 
                        zmin=-1, zmax=1,
                        x=cols, y=cols
                    )
                    fig_corr.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Engine Offline: Please complete Pre-Flight Ontology Verification in Tab 1.")

    if st.session_state.get('local_df') is not None and schema is not None:
        causal_df = st.session_state['local_df']
        causal_cols = schema.get("required_columns", list(causal_df.columns))
        
        with tab_granger:
            _render_granger_section(causal_df, causal_cols, LIGHT_THEME)
            
        with tab_network:
            _render_causal_network(causal_df, causal_cols, LIGHT_THEME)
            
        with tab_cross:
            _render_cross_corr(causal_df, causal_cols, LIGHT_THEME)
    else:
        with tab_granger:
            st.warning("Upload data first to run causal analysis.")
        with tab_network:
            st.warning("Upload data first to run causal analysis.")
        with tab_cross:
            st.warning("Upload data first to run causal analysis.")

    with tab3:
        with st.container(border=True):
            st.warning("### Secure Intelligence Channel (Spoke -> Sector Admin)")
            st.write("Communicate directly with your Sector Administrator.")
            
            msg_col1, msg_col2 = st.columns([1, 1])
            with msg_col1:
                st.write("**Send Encrypted Dispatch**")
                msg_content = st.text_area("Message Payload", height=100, placeholder="Detail anomalies, request resource allocation, or flag model drift...", key="spoke_msg_out")
                if st.button("Dispatch to Sector Admin", type="primary", use_container_width=True):
                    if msg_content:
                        SecureMessaging.send_message(
                            sender_role=Role.INSTITUTION.value,
                            sender_id=st.session_state.get('username'),
                            receiver_role=Role.BASKET_ADMIN.value,
                            receiver_id=str(basket_id),
                            content=msg_content
                        )
                        st.success("Dispatch routed successfully.")
                    else:
                        st.error("Cannot dispatch an empty payload.")
            
            with msg_col2:
                st.write("**Command & Control (Inbox)**")
                inbox = SecureMessaging.get_inbox(Role.INSTITUTION.value, st.session_state.get('username'))
                if not inbox:
                    st.caption("No pending commands from Command & Control.")
                else:
                    for msg in inbox:
                        sender_badge = "EXECUTIVE DIRECTIVE" if msg['sender_role'] == Role.EXECUTIVE.value else "Admin Directive"
                        with st.expander(f"{sender_badge} | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
                            st.write(msg['content'])
                            if not msg['is_read']:
                                if st.button("Acknowledge Command", key=f"ack_{msg['id']}"):
                                    SecureMessaging.mark_read(msg['id'])
                                    st.rerun()

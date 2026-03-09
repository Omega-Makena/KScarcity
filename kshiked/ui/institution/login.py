import streamlit as st
import sys
import os
from pathlib import Path

# Ensure backend package is accessible and Scarcity engine handles imports correctly.
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import login_user, logout_user
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.style import inject_enterprise_theme

def render_landing_page():
    inject_enterprise_theme(include_watermark=True)
    st.markdown("<h1 style='text-align: center; color: #1F2937; padding-top: 2rem;'>K-Scarcity</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #475569; padding-bottom: 2rem;'>National Intelligence & Systemic Risk Gateway</h3>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 10, 1])
    
    with center_col:
        # 5 Ws + 1 H bundled into a single html block for faster rendering using CSS Grid
        st.markdown(
            '''
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 2rem;">
                <div style="background:#F8FAFC; border-radius:8px; padding:20px; border-top:4px solid #14B8A6; box-shadow:0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                    <h4 style="color:#14B8A6; margin-top:0;">Who is it for?</h4>
                    <p style="color:#475569; font-size:1rem; margin-bottom:0;">Government executives, sector administrators, and institutional leaders managing national stability and security.</p>
                </div>
                
                <div style="background:#F8FAFC; border-radius:8px; padding:20px; border-top:4px solid #BB0000; box-shadow:0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                    <h4 style="color:#BB0000; margin-top:0;">What is K-Scarcity?</h4>
                    <p style="color:#475569; font-size:1rem; margin-bottom:0;">An early warning system that detects structural anomalies, trend degradations, and emerging risks. It is explicitly designed to operate across <b>all national sectors</b> (finance, agriculture, health, security, energy, etc.) before issues escalate into systemic crises.</p>
                </div>
                
                <div style="background:#F8FAFC; border-radius:8px; padding:20px; border-top:4px solid #3B82F6; box-shadow:0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                    <h4 style="color:#3B82F6; margin-top:0;">Where does it operate?</h4>
                    <p style="color:#475569; font-size:1rem; margin-bottom:0;">Across all geographic domains. From localized, county-level reporting to macro-level national indicators, providing geographic specificity to pinpoint exactly where systemic risks are materializing across all sectors.</p>
                </div>
                
                <div style="background:#F8FAFC; border-radius:8px; padding:20px; border-top:4px solid #8B5CF6; box-shadow:0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                    <h4 style="color:#8B5CF6; margin-top:0;">When should you act?</h4>
                    <p style="color:#475569; font-size:1rem; margin-bottom:0;">Continuous, real-time monitoring categorizes risks by urgency—providing projected consequences of inaction so you know exactly when an intervention is required to prevent a cascading failure.</p>
                </div>
                
                <div style="background:#F8FAFC; border-radius:8px; padding:20px; border-top:4px solid #F59E0B; box-shadow:0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                    <h4 style="color:#F59E0B; margin-top:0;">Why use it?</h4>
                    <p style="color:#475569; font-size:1rem; margin-bottom:0;">To transform complex, fragmented data into plain-language executive reports, shock propagation forecasts, and clear policy recommendations. It moves national decision-making from reactive to proactive.</p>
                </div>
                
                <div style="background:#F8FAFC; border-radius:8px; padding:20px; border-top:4px solid #006600; box-shadow:0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                    <h4 style="color:#006600; margin-top:0;">How does it work?</h4>
                    <p style="color:#475569; font-size:1rem; margin-bottom:0;">Through <b>Secure Federated Intelligence</b>. Institutions collaborate and train analytical models collectively, ensuring raw sensitive data never leaves their premises while still contributing to the national risk baseline.</p>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1, 1])
        with col_btn_2:
            if st.button("Enter Secure Portal", type="primary", use_container_width=True):
                st.session_state['show_login'] = True
                st.rerun()

def render_login_page():
    inject_enterprise_theme(include_watermark=True)
    st.markdown("<h1 style='text-align: center; color: #BB0000;'>National Intelligence Gateway</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #006600;'>Restricted Access Protocol</h3>", unsafe_allow_html=True)
    
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Return to Home", use_container_width=True):
            st.session_state['show_login'] = False
            st.rerun()
            
        st.write("")
        with st.form("login_form"):
            username = st.text_input("Username / Spoke ID")
            password = st.text_input("Passkey", type="password")
            submitted = st.form_submit_button("Authenticate", use_container_width=True)
            
            if submitted:
                if login_user(username, password):
                    st.success(f"Authentication Successful: Clearance {st.session_state['role']}")
                    st.rerun()
                else:
                    st.error("Invalid Credentials or Biometrics Failure.")

def route_authenticated_user():
    role = st.session_state.get('role')
    
    # Render global sidebar for authenticated users
    with st.sidebar:
        st.markdown("### Profile")
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border-left: 5px solid #006600; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="margin: 0; color: #1F2937; font-size: 0.9em;"><strong>Access ID:</strong><br/>
                <span style="font-size: 1.1em; color: #BB0000;">{st.session_state.get('username')}</span></p>
                <hr style="margin: 10px 0; border: none; border-top: 1px solid #e2e8f0;" />
                <p style="margin: 0; color: #1F2937; font-size: 0.9em;"><strong>Clearance Level:</strong><br/>
                <span style="font-size: 1.1em; color: #006600;">{role}</span></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button("Disconnect Session", use_container_width=True):
            logout_user()
            st.rerun()

        st.write("---")
        st.markdown("**Reporting Mode**")
        fl_on = st.toggle(
            "Enable Federated Learning (Mode B)",
            value=st.session_state.get('fl_mode_enabled', False),
            help="Mode A (default): Governance reporting — aggregated data and anomaly types flow upward. "
                 "Mode B: Federated Learning — local model gradients are also aggregated (raw data never leaves nodes)."
        )
        st.session_state['fl_mode_enabled'] = fl_on
        if fl_on:
            st.caption("Mode B active: FL training tabs are visible to admins and spokes.")
        else:
            st.caption("Mode A: Governance reporting only.")

    # Route based on explicit Role Based Access Controls
    if role == Role.EXECUTIVE.value:
        from kshiked.ui.institution.executive_dashboard import render as render_executive
        render_executive()
    elif role == Role.BASKET_ADMIN.value:
        from kshiked.ui.institution.admin_governance import render as render_basket_admin
        render_basket_admin()
    elif role == Role.INSTITUTION.value:
        from kshiked.ui.institution.local_dashboard import render as render_spoke
        render_spoke()
    else:
        st.error("Authentication Corrupted: Identity Context Lost.")
        logout_user()

def main():
    st.set_page_config(page_title="Scarcity: National Intelligence", layout="wide")
    
    # Check if this user is already authenticated via Streamlit Session State
    if not st.session_state.get("authenticated", False):
        if st.session_state.get('show_login', False):
            render_login_page()
        else:
            render_landing_page()
    else:
        route_authenticated_user()

if __name__ == "__main__":
    main()

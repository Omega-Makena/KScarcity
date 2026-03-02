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

def render_login_page():
    inject_enterprise_theme(include_watermark=True)
    st.markdown("<h1 style='text-align: center; color: #BB0000;'>National Intelligence Gateway</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #006600;'>Restricted Access Protocol</h3>", unsafe_allow_html=True)
    
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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
        render_login_page()
    else:
        route_authenticated_user()

if __name__ == "__main__":
    main()

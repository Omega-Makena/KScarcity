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
    inject_enterprise_theme()
    st.markdown("<h1 style='text-align: center; color: #C60C30;'>National Intelligence Gateway</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #006747;'>Restricted Access Protocol</h3>", unsafe_allow_html=True)
    
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
        st.info(f"**ID:** {st.session_state.get('username')}\n\n**Role:** {role}")
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

"""
Standalone Causal Analysis — Kenya World Bank Data
Run: streamlit run run_causal.py --server.port 8510
"""
import sys
from pathlib import Path

# Ensure kshield package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
import site
import os

# FORCE: Add user site-packages to path so Streamlit sees pip installed libs
try:
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
except Exception:
    pass

import streamlit as st
from theme import DARK_THEME, generate_css

st.set_page_config(
    page_title="Kenya Causal Analysis",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Use the real SENTINEL theme
theme = DARK_THEME
st.markdown(generate_css(theme, True), unsafe_allow_html=True)

from kshield.causal import render_causal
render_causal(theme)

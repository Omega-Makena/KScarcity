"""
Global CSS injection for the SENTINEL dashboard.

Injects the base theme CSS + card button pre-load CSS at the top
of every page to prevent flash of unstyled content.
"""

from __future__ import annotations
import streamlit as st


def inject_global_css(theme, dark_mode: bool, generate_css_fn) -> None:
    """
    Inject all global CSS at the top of the page.
    
    Args:
        theme: ThemeColors instance
        dark_mode: Whether dark mode is active
        generate_css_fn: The generate_css function from theme.py
    """
    # Base theme CSS
    st.markdown(generate_css_fn(theme, dark_mode), unsafe_allow_html=True)
    
    # Card button CSS pre-load (prevents FOUC)
    st.markdown(f"""<style>
    div[data-testid="stVerticalBlock"] button {{
        width: 100% !important;
        height: 320px !important;
        margin: 0 auto !important;
        white-space: pre-wrap !important;
        text-align: left !important;
        padding: 1.8rem !important;
        background: linear-gradient(160deg, rgba(31, 51, 29, 0.55), rgba(20, 38, 22, 0.45)) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-top: 1px solid rgba(0, 255, 136, 0.25) !important;
        border-left: 1px solid rgba(0, 255, 136, 0.15) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.4) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.3) !important;
        border-radius: 16px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        gap: 0.3rem !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35), 0 0 1px rgba(0, 255, 136, 0.1) !important;
    }}
    div[data-testid="stVerticalBlock"] button p:first-child {{
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, {theme.text_primary}, {theme.accent_primary}) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }}
    div[data-testid="stVerticalBlock"] button p:not(:first-child) {{
        font-size: 0.88rem !important;
        color: {theme.text_secondary} !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
        opacity: 0.85 !important;
    }}
    div[data-testid="stVerticalBlock"] button:hover {{
        transform: translateY(-5px) !important;
        border-color: rgba(0, 255, 136, 0.35) !important;
        background: linear-gradient(160deg, rgba(31, 51, 29, 0.7), rgba(0, 255, 136, 0.06)) !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 255, 136, 0.12),
                    inset 0 0 15px rgba(0, 255, 136, 0.03) !important;
    }}
    </style>""", unsafe_allow_html=True)

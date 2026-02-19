"""
Sidebar metrics and policy input controls.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from .context import process_bill_input, clear_session

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def render_sidebar(theme, data=None):
    """Render the sidebar with metrics and context controls."""
    
    # 1. Impact Metrics (if data is provided) - Moved from old tabs
    if data:
        st.sidebar.markdown("### Live Impact Metrics")
        
        # Mini ESI Chart
        esi = data.esi_indicators
        if esi:
            # Simple bar chart for sidebar
            st.sidebar.caption("Economic Satisfaction")
            try:
                # Create a simple dataframe for the chart
                df = pd.DataFrame({"Sector": list(esi.keys()), "Value": list(esi.values())})
                st.sidebar.bar_chart(df.set_index("Sector"), color=theme.accent_primary, height=150)
            except Exception:
                pass

        # Key Primitives
        prims = data.primitives
        if prims:
            st.sidebar.markdown("---")
            st.sidebar.caption("System Stability")
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Instability", f"{prims.get('instability_index',0):.0%}")
            col2.metric("Crisis Prob", f"{prims.get('crisis_probability',0):.0%}")
            
            scarcity = prims.get('aggregate_scarcity', 0)
            st.sidebar.progress(scarcity, text=f"Aggregate Scarcity: {scarcity:.0%}")

    st.sidebar.markdown("---")
    
    # 2. Policy Input (Collapsed by default to keep it clean)
    with st.sidebar.expander("📝 Policy Context / Upload", expanded=False):
        input_mode = st.radio(
            "Input method",
            ["Type/Ask", "Paste Text", "Upload PDF", "Enter URL"],
            key="policy_input_mode",
        )

        if input_mode == "Paste Text":
            bill_title = st.text_input("Bill title", key="policy_sidebar_title")
            bill_text = st.text_area("Paste text", key="policy_sidebar_text", height=150)
            if st.button("Analyze", key="policy_analyze_paste"):
                if bill_text.strip():
                    st.session_state.policy_bill_text = bill_text
                    st.session_state.policy_bill_title = bill_title
                    process_bill_input(text=bill_text, title=bill_title, theme=theme)

        elif input_mode == "Upload PDF":
            uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="policy_pdf_upload")
            bill_title = st.text_input("Title (opt)", key="policy_pdf_title")
            if st.button("Analyze PDF", key="policy_analyze_pdf"):
                if uploaded:
                    process_bill_input(
                        pdf_bytes=uploaded.getvalue(),
                        title=bill_title or uploaded.name,
                        theme=theme,
                    )

        elif input_mode == "Enter URL":
            url = st.text_input("Bill URL", key="policy_url_input")
            bill_title = st.text_input("Title", key="policy_url_title")
            if st.button("Analyze URL", key="policy_analyze_url"):
                if url.strip():
                    process_bill_input(url=url, title=bill_title, theme=theme)

    # Active bill info
    session = st.session_state.get("policy_session")
    if session and session.has_bill:
        bill = session.bill
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Active Analysis**: {bill.title}")
        if st.sidebar.button("Clear Context", key="policy_clear"):
            clear_session()
            st.rerun()

    # Quick policies
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Show Quick Examples"):
        quick_policies = ["Finance Bill 2026", "SHIF Phase 2", "Housing Levy Increase"]
        for policy in quick_policies:
            if st.sidebar.button(f"Analyze {policy}", key=f"quick_{policy}"):
                process_bill_input(title=policy, theme=theme)

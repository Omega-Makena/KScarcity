import streamlit as st
from kshiked.ui.theme import DARK_THEME
from kshiked.ui.institution.executive_simulator import STRATEGIC_SCENARIOS, SEVERITY_MULT, POLICY_RESPONSES

def test_ui():
    print("Testing scenario selectbox...")
    sc = st.selectbox("Crisis scenario", list(STRATEGIC_SCENARIOS.keys()), key="strat_sc_v3")
    print(f"Scenario selected: {sc}")
    
    print("Testing severity selectbox...")
    sev = st.selectbox("Severity", list(SEVERITY_MULT.keys()), index=1, key="strat_sev_v3")
    print(f"Severity selected: {sev}")
    
    print("Testing policy radio...")
    resp = st.radio("Policy:", list(POLICY_RESPONSES.keys()), horizontal=True, key="strat_resp_v3")
    print(f"Policy selected: {resp}")

test_ui()

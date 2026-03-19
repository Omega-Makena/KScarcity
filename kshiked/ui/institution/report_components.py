import streamlit as st
import pandas as pd
import datetime
from typing import Any

def render_sector_report(report_data: dict, theme_colors: Any):
  """
  Renders the Comprehensive Sector Report natively in Streamlit.
  """
  
  st.markdown(f"## Comprehensive Sector Intelligence Report: {report_data['sector_name']}")
  st.caption(f"Generated at: {datetime.datetime.fromtimestamp(report_data['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
  st.write("---")
  
  # 1. Executive Summary
  summary = report_data['summary']
  dq = report_data['data_quality']
  
  _health_color = "#BB0000" if summary['health_score'] == "Critical" else "#E05000" if summary['health_score'] == "Deteriorating" else "#059669"
  
  st.markdown(f"### 1. Executive Summary: <span style='color:{_health_color}'>{summary['health_score']}</span>", unsafe_allow_html=True)
  st.markdown(f"""
  <div style="background:#F8FAFC; border-left:4px solid {_health_color}; padding:15px; border-radius:4px; font-size:1.1rem; margin-bottom: 20px;">
    {summary['narrative']}
  </div>
  """, unsafe_allow_html=True)
  
  col1, col2, col3 = st.columns(3)
  col1.metric("System Health", summary['health_score'])
  col2.metric("Key Callouts", summary['callouts'])
  col3.metric("Recommended Focus", summary['recommended_action'])
  
  # Data Quality Context
  if dq['is_incomplete']:
    st.warning(f" **Data Quality Notice**: This report has {dq['coverage_pct']:.0%} coverage. Missing data from: {', '.join(dq['missing_spokes'][:3])}{' and others' if len(dq['missing_spokes'])>3 else ''}.")
  else:
    st.success(f" **Data Quality**: High confidence. {dq['coverage_pct']:.0%} coverage across sector nodes.")
    
  st.write("---")
  
  # 2. Critical & Deteriorating Areas
  st.markdown("### 2. Critical & Deteriorating Risk Areas")
  risks = report_data['risk_areas']
  if not risks:
    st.success("No active critical or deteriorating risks tracked at this time.")
  else:
    for r in risks:
      _color = "#BB0000" if r['severity'] == "Critical" else "#F59E0B"
      with st.expander(f" {r['title']} | Severity: {r['severity']}", expanded=(r['severity']=="Critical")):
        c1, c2 = st.columns([2,1])
        with c1:
          st.markdown(f"**Root Cause / Context:** {r['root_cause']}")
          st.markdown(f"**Current Impact:** {r['current_impact']}")
          st.markdown(f"**Projected Impact (Inaction):** {r['projected_impact']}")
        with c2:
          st.markdown(f"<span style='background:{_color}; color:#fff; padding:3px 8px; border-radius:4px;'>Trend: {r['trend']}</span>", unsafe_allow_html=True)
          st.write("")
          st.markdown(f"**Confidence:** {r['confidence']}")
          st.markdown(f"**Intervention:** {r['intervention']}")

  st.write("---")
  
  # 3 & 4. Stable and Improving Areas
  c_stable, c_improv = st.columns(2)
  with c_stable:
    st.markdown("### 3. Stable Operations")
    stable = report_data['stable_areas']
    if not stable:
       st.info("No explicitly stable operations indexed.")
    else:
      for s in stable:
        st.markdown(f"""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; padding:10px; border-radius:6px; margin-bottom:10px;">
          <div style="font-weight:600; color:#166534;">✓ {s['title']}</div>
          <div style="font-size:0.85rem; color:#14532D; margin-top:4px;">{s['reason']}</div>
        </div>
        """, unsafe_allow_html=True)
        
  with c_improv:
    st.markdown("### 4. Improving Trajectories")
    improv = report_data['improving_areas']
    if not improv:
      st.info("No recently recovered or improving areas indexed.")
    else:
      for i in improv:
        st.markdown(f"""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; padding:10px; border-radius:6px; margin-bottom:10px;">
          <div style="font-weight:600; color:#1E3A8A;"> {i['title']}</div>
          <div style="font-size:0.85rem; color:#1E40AF; margin-top:4px;">Driver: {i['driver']}</div>
          <div style="font-size:0.8rem; color:#3B82F6; margin-top:4px;">Status: {i['sustained']}</div>
        </div>
        """, unsafe_allow_html=True)

  st.write("---")
  
  # 5. Opportunities
  st.markdown("### 5. Sector Opportunities")
  opps = report_data['opportunities']
  if not opps:
    st.write("No major sector re-allocation opportunities detected currently.")
  else:
    for o in opps:
      st.markdown(f"- **{o['title']}**: {o['action']} *(Validity: {o['validity']})*")

  st.write("---")
  
  # 6. Action Table
  st.markdown("### 6. Decision-Ready Action Plan")
  actions = report_data['action_table']
  if actions:
    df_actions = pd.DataFrame(actions)
    st.dataframe(df_actions, use_container_width=True, hide_index=True)
  else:
    st.success("No immediate tactical actions required. Monitor incoming telemetry.")

  st.write("---")
  st.caption("End of Comprehensive Intelligence Report.")
  
  # Add Export Buttons
  col_dl1, col_dl2 = st.columns([1,5])
  with col_dl1:
    st.download_button(
      label=" Export to CSV (Action Plan)",
      data=pd.DataFrame(actions).to_csv(index=False) if actions else "Empty",
      file_name=f"{report_data['sector_name'].replace(' ', '_')}_Actions.csv",
      mime="text/csv"
    )

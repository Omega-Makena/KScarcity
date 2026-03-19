import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

from kshiked.ui.institution.backend.history_middleware import get_analysis_history, get_full_analysis_result
from kshiked.ui.theme import LIGHT_THEME, DARK_THEME

def render_history_tab(theme):
  """Renders the historical analysis tab with search, filter, and RBAC."""
  st.markdown("<h2 style='color:#14B8A6;'>Historical Analysis & Audit Trail</h2>", unsafe_allow_html=True)
  st.markdown(f"<p style='color:{theme.text_muted};'>View and export previously executed simulations, causal tests, and systemic risk analyses.</p>", unsafe_allow_html=True)
  st.write("---")
  
  # 1. Fetch Data (RBAC applied in middleware)
  role = st.session_state.get("role", "Spoke")
  user_id = st.session_state.get("user_id", 0)
  basket_id = st.session_state.get("basket_id", 0)
  
  with st.spinner("Loading history records..."):
    raw_history = get_analysis_history(role, user_id, basket_id)
    
  if not raw_history:
    st.info("No historical analyses found for your access level.")
    return
    
  df = pd.DataFrame(raw_history)
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  
  # 2. Filters & Search Bar
  c1, c2, c3 = st.columns([2, 1, 1])
  with c1:
    search_query = st.text_input(" Search (ID, User, Summary, Sector)", "")
  with c2:
    types = ["All"] + list(df["analysis_type"].unique())
    selected_type = st.selectbox("Analysis Type", types)
  with c3:
    if role == "Executive":
      sectors = ["All"] + list(df["sector"].unique())
      selected_sector = st.selectbox("Sector", sectors)
    else:
      selected_sector = "All"
      
  # Apply Filters
  if search_query:
    query_l = search_query.lower()
    df = df[df.apply(lambda row: query_l in str(row["id"]).lower() or 
                   query_l in str(row["username"]).lower() or 
                   query_l in str(row["result_summary"]).lower() or 
                   query_l in str(row["sector"]).lower(), axis=1)]
                   
  if selected_type != "All":
    df = df[df["analysis_type"] == selected_type]
    
  if selected_sector != "All":
    df = df[df["sector"] == selected_sector]
    
  # Formatting the table for display
  display_df = df[["timestamp", "analysis_type", "sector", "username", "result_summary", "id"]].copy()
  display_df = display_df.rename(columns={
    "timestamp": "Date (UTC)", 
    "analysis_type": "Type", 
    "sector": "Sector", 
    "username": "Run By", 
    "result_summary": "Summary",
    "id": "Record ID"
  })
  
  st.markdown(f"**Showing {len(display_df)} records**")
  
  # 3. Main Display + Drill-down interaction
  if len(display_df) > 0:
    # Provide interactive selection
    selected_record_id = st.selectbox(
      "Select a record ID to drill-down and view full results:", 
      options=["-- Select --"] + display_df["Record ID"].tolist()
    )
    
    # Display the table
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # 4. Drill-Down View
    if selected_record_id != "-- Select --":
      st.write("---")
      record = df[df["id"] == selected_record_id].iloc[0]
      st.markdown(f"### Drill-down: {record['analysis_type']}")
      st.caption(f"Run by {record['username']} ({record['role']}) on {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
      
      c_meta, c_actions = st.columns([3, 1])
      with c_meta:
        st.markdown("**Input Parameters:**")
        st.json(json.loads(record["input_parameters"]))
        
      with c_actions:
        # Export functionality
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
          label=" Export Filtered View (CSV)",
          data=csv_data,
          file_name=f"historical_analysis_{datetime.utcnow().strftime('%Y%m%d')}.csv",
          mime="text/csv",
          use_container_width=True
        )
      
      payload = get_full_analysis_result(record["full_result_path"])
      if payload:
        st.markdown("**Full Result Payload:**")
        # We can dynamically route to custom renderers if needed
        if record["analysis_type"] == "Granger Causality":
          try:
            results_df = pd.DataFrame(payload)
            st.dataframe(results_df, use_container_width=True)
          except:
            st.json(payload)
        elif record["analysis_type"] == "Executive Simulation":
          # Show trajectory summary (avoiding huge raw JSON block directly on screen)
          if "trajectory" in payload:
            traj = payload["trajectory"]
            st.write(f"Trajectory Length: {len(traj)} steps")
            if len(traj) > 0 and "outcomes" in traj[-1]:
              st.write("**Final Outcomes:**")
              st.json(traj[-1]["outcomes"])
          else:
            st.json(payload)
        else:
          st.json(payload)
      else:
        st.error("Full payload file could not be found. It may have been archived or deleted.")

  else:
    st.warning("No records match your filters.")

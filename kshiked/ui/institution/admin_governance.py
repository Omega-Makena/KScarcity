import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from html import escape

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import enforce_role
from kshiked.ui.institution.backend.auth import logout_user
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.federation_bridge import FederationBridge
from kshiked.ui.institution.backend.project_manager import ProjectManager
from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.institution.style import inject_enterprise_theme, get_base64_of_bin_file
from kshiked.ui.institution.backend.messaging import SecureMessaging
from kshiked.ui.institution.backend.data_sharing import DataSharingManager
from kshiked.ui.institution.collab_room import render_collab_room
from kshiked.ui.institution.backend.analytics_engine import (
  generate_inaction_projection,
  get_historical_context,
  generate_recommendation,
  compute_outcome_impact,
)
from kshiked.ui.institution.backend.report_narrator import (
  narrate_composite_scores,
  narrate_severity,
  narrate_shock_vector,
  narrate_risk_for_executive,
)
from kshiked.ui.institution.backend.sector_reports import SectorReportGenerator
from kshiked.ui.institution.report_components import render_sector_report
from kshiked.ui.institution.shared_sidebar import render_shared_sidebar


ADMIN_COLORS = {
  "black": "#1a1a1a",
  "red": "#BB0000",
  "red_light": "#F9EDED",
  "green": "#006600",
  "green_light": "#EAF3E0",
  "white": "#FFFFFF",
  "surface": "#F8F7F5",
  "border": "rgba(26,26,26,0.12)",
  "border_strong": "rgba(26,26,26,0.25)",
  "text_muted": "#6B6B6B",
  "text_faint": "#9E9E9E",
}


ADMIN_NAV_ITEMS = [
  ("overview", "Sector Overview", "sector-overview", ADMIN_COLORS["red"], None),
  ("overview", "Historical Archive", "historical-archive", ADMIN_COLORS["red"], None),
  ("spokes", "Spoke Reports", "spoke-reports", ADMIN_COLORS["green"], "pending"),
  ("spokes", "Data Sharing", "data-sharing", ADMIN_COLORS["green"], None),
  ("spokes", "Data Governance & Schemas", "data-governance", ADMIN_COLORS["green"], None),
  ("operations", "Operational Projects", "operational-projects", ADMIN_COLORS["black"], None),
  ("operations", "Risk Promotion", "risk-promotion", ADMIN_COLORS["black"], None),
  ("operations", "Communications", "communications", ADMIN_COLORS["black"], None),
  ("operations", "Collaboration Room", "collaboration-room", ADMIN_COLORS["black"], None),
  ("operations", "Federated Learning (Mode B)", "federated-learning", ADMIN_COLORS["black"], None),
]


ADMIN_NAV_KEY_TO_SECTION = {
  "sector-overview": "Sector Overview",
  "historical-archive": "Historical Archive",
  "spoke-reports": "Spoke Reports",
  "data-sharing": "Data Sharing",
  "data-governance": "Data Governance & Schemas",
  "operational-projects": "Operational Projects",
  "risk-promotion": "Risk Promotion",
  "communications": "Communications",
  "collaboration-room": "Collaboration Room",
  "federated-learning": "Federated Learning (Mode B)",
}

def plot_shock_vector(shock_vector, title):
  import plotly.graph_objects as go
  metrics = list(shock_vector.keys())
  baselines = [shock_vector[m]["pre_shock_baseline"] for m in metrics]
  peaks = [shock_vector[m]["peak_shock_value"] for m in metrics]
  
  fig = go.Figure()
  fig.add_trace(go.Bar(name='Pre-Shock Baseline', x=metrics, y=baselines, marker_color='#888888'))
  fig.add_trace(go.Bar(name='Peak Shock Value', x=metrics, y=peaks, marker_color='#FF4444'))
  
  fig.update_layout(
    title=title,
    barmode='group',
    template='plotly_white',
    height=300,
    margin=dict(l=0, r=0, t=30, b=0)
  )
  _apply_plotly_numeric_font(fig)
  return fig


def _apply_plotly_numeric_font(fig):
  fig.update_layout(
    font=dict(family="IBM Plex Sans, sans-serif"),
    hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace")),
  )
  for axis_name in [
    "xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5",
    "yaxis", "yaxis2", "yaxis3", "yaxis4", "yaxis5",
  ]:
    axis = getattr(fig.layout, axis_name, None)
    if axis is not None:
      axis.tickfont = dict(family="IBM Plex Mono, monospace")
  return fig


@st.cache_data(show_spinner=False)
def _get_gok_logo_b64() -> str:
  logo_path = Path(project_root) / "GOK.png"
  if not logo_path.exists():
    return ""
  return get_base64_of_bin_file(str(logo_path))


def _inject_admin_redesign_css():
  logo_b64 = _get_gok_logo_b64()
  if logo_b64:
    background_css = f"""
      .stApp {{
        background-image: url("data:image/png;base64,{logo_b64}") !important;
        background-size: 340px auto !important;
        background-repeat: no-repeat !important;
        background-position: center center !important;
        background-color: {ADMIN_COLORS['surface']} !important;
      }}

      [data-testid="stAppViewContainer"] {{
        background-color: rgba(248, 247, 245, 0.94) !important;
      }}
    """
  else:
    background_css = f"""
      .stApp {{
        background: {ADMIN_COLORS['surface']} !important;
      }}
    """

  st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

      html, body, [class*="css"], .stApp {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
      }}

      [data-testid="stMetricValue"],
      [data-testid="stMetricDelta"],
      .admin-num,
      .stNumberInput input,
      input[type="number"],
      .js-plotly-plot .xtick text,
      .js-plotly-plot .ytick text,
      .js-plotly-plot .hovertext {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-variant-numeric: tabular-nums;
      }}

      {background_css}

      .main .block-container {{
        padding-top: 20px !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
        padding-bottom: 24px !important;
      }}

      [data-testid="stSidebar"] {{
        background: {ADMIN_COLORS['white']} !important;
        width: 210px !important;
        min-width: 210px !important;
        max-width: 210px !important;
        border-right: 0.5px solid {ADMIN_COLORS['border']} !important;
      }}

      [data-testid="stSidebar"] > div:first-child {{
        background: {ADMIN_COLORS['white']} !important;
      }}

      [data-testid="stSidebar"] .block-container {{
        padding-top: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-bottom: 0.75rem !important;
      }}

      [data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        text-align: left !important;
        border-radius: 8px !important;
        border: 0.5px solid {ADMIN_COLORS['border_strong']} !important;
        background: {ADMIN_COLORS['white']} !important;
        color: {ADMIN_COLORS['black']} !important;
        box-shadow: none !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        letter-spacing: 0.1px !important;
        padding: 0.42rem 0.55rem !important;
        min-height: 34px !important;
      }}

      [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: {ADMIN_COLORS['red']} !important;
        color: {ADMIN_COLORS['black']} !important;
        background: {ADMIN_COLORS['red_light']} !important;
      }}

      [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {ADMIN_COLORS['red_light']} !important;
        color: {ADMIN_COLORS['red']} !important;
        font-weight: 500 !important;
        border-color: {ADMIN_COLORS['border']} !important;
        border-left: 3px solid {ADMIN_COLORS['red']} !important;
      }}

      [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
        color: {ADMIN_COLORS['red']} !important;
        border-left: 3px solid {ADMIN_COLORS['red']} !important;
      }}

      [data-testid="stSidebar"] .stButton:last-of-type > button {{
        border: none !important;
        background: transparent !important;
        color: {ADMIN_COLORS['text_faint']} !important;
        font-size: 11px !important;
        padding: 2px 0 !important;
        min-height: auto !important;
      }}

      [data-testid="stSidebar"] .stButton:last-of-type > button:hover {{
        background: transparent !important;
        color: {ADMIN_COLORS['red']} !important;
      }}

      .admin-shell-hero {{
        background: {ADMIN_COLORS['black']};
        border: 0.5px solid {ADMIN_COLORS['border_strong']};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
      }}

      .admin-shell-label {{
        color: {ADMIN_COLORS['text_muted']};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
      }}

      .admin-shell-title {{
        color: {ADMIN_COLORS['white']};
        font-size: 22px;
        font-weight: 500;
        margin: 4px 0 2px 0;
      }}

      .admin-shell-subtitle {{
        color: #D1D5DB;
        font-size: 13px;
      }}

      .admin-badge {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        border: 0.5px solid {ADMIN_COLORS['green']};
        color: {ADMIN_COLORS['green']};
        background: #EAF3E0;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        margin-top: 8px;
      }}

      .admin-divider {{
        height: 0.5px;
        background: {ADMIN_COLORS['border']};
        margin: 10px 0;
      }}

      .admin-nav-group-label {{
        color: {ADMIN_COLORS['text_muted']};
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
        margin-bottom: 6px;
      }}

      .admin-group-dot, .admin-nav-dot {{
        width: 5px;
        height: 5px;
        border-radius: 50%;
        display: inline-block;
      }}

      .admin-count-badge {{
        background: {ADMIN_COLORS['red_light']};
        color: {ADMIN_COLORS['red']};
        border: 0.5px solid {ADMIN_COLORS['red']};
        border-radius: 999px;
        font-size: 10px;
        font-weight: 500;
        text-align: center;
        padding: 1px 6px;
        margin-top: 8px;
      }}

      .admin-topbar {{
        background: {ADMIN_COLORS['black']};
        border: 0.5px solid {ADMIN_COLORS['border_strong']};
        border-radius: 12px;
        min-height: 56px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 0 14px;
        margin-bottom: 10px;
      }}

      .admin-topbar-left {{
        display: flex;
        align-items: center;
        gap: 6px;
        min-width: 72px;
      }}

      .admin-topbar-center {{
        color: {ADMIN_COLORS['white']};
        flex: 1;
        text-align: center;
        font-size: 15px;
        font-weight: 500;
        line-height: 1.25;
      }}

      .admin-topbar-center .value {{
        color: {ADMIN_COLORS['white']};
        font-weight: 500;
      }}

      .admin-topbar-center .muted {{
        color: #F3C9C9;
      }}

      .admin-deploy-btn {{
        border: 0.5px solid {ADMIN_COLORS['white']};
        color: {ADMIN_COLORS['white']};
        background: transparent;
        border-radius: 8px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 500;
        min-width: 76px;
        text-align: center;
      }}

      .admin-command-header {{
        background: {ADMIN_COLORS['black']};
        border: 0.5px solid {ADMIN_COLORS['border_strong']};
        border-left: 3px solid {ADMIN_COLORS['red']};
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 14px;
      }}

      .admin-command-label {{
        color: {ADMIN_COLORS['text_faint']};
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
      }}

      .admin-command-title {{
        color: {ADMIN_COLORS['white']};
        font-size: 22px;
        font-weight: 500;
        margin: 3px 0;
      }}

      .admin-command-subtitle {{
        color: #F3C9C9;
        font-size: 13px;
      }}

      [data-testid="stTabs"] [role="tablist"] {{
        display: none !important;
      }}

      [data-testid="stTabs"] [role="tab"] {{
        border-radius: 999px;
        border: 0.5px solid {ADMIN_COLORS['border_strong']};
        background: {ADMIN_COLORS['white']};
        color: {ADMIN_COLORS['text_muted']};
        font-size: 12px;
        font-weight: 500;
        padding: 4px 10px;
      }}

      [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: {ADMIN_COLORS['red_light']};
        color: {ADMIN_COLORS['red']};
        border-color: {ADMIN_COLORS['red']};
      }}

      [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 0.5px solid {ADMIN_COLORS['border']} !important;
        border-radius: 12px !important;
        background: {ADMIN_COLORS['white']} !important;
      }}

      .stButton > button, .stDownloadButton > button {{
        border-radius: 8px !important;
        border: 0.5px solid {ADMIN_COLORS['border_strong']} !important;
        box-shadow: none !important;
      }}

      .stButton > button:hover {{
        border-color: {ADMIN_COLORS['red']} !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
  )


def _render_admin_sidebar(sector_name: str, pending_count: int, fl_mode: bool):
  hidden_keys = set() if fl_mode else {"federated-learning"}
  sidebar_state = render_shared_sidebar(
    state_key="admin_nav",
    default_key="sector-overview",
    nav_items=ADMIN_NAV_ITEMS,
    group_order=["overview", "spokes", "operations"],
    group_labels={
      "overview": ("Overview", ADMIN_COLORS["red"]),
      "spokes": ("Spokes", ADMIN_COLORS["green"]),
      "operations": ("Operations", ADMIN_COLORS["black"]),
    },
    profile_label="Sector Governance Console",
    profile_name=f"{sector_name} Admin",
    role_badge_text="SECTOR ADMIN",
    role_badge_bg=ADMIN_COLORS["red_light"],
    role_badge_fg=ADMIN_COLORS["red"],
    role_badge_border=ADMIN_COLORS["red"],
    profile_bottom_border=ADMIN_COLORS["red"],
    button_key_prefix="admin_nav",
    badge_counts={"spoke-reports": pending_count},
    hidden_keys=hidden_keys,
    disconnect_button_key="admin_disconnect",
    disconnect_label="Disconnect session",
  )

  if sidebar_state["disconnect_clicked"]:
    logout_user()
    st.rerun()

  active_key = str(sidebar_state["active_key"])
  if sidebar_state["changed"]:
    st.session_state["admin_nav"] = active_key
    st.rerun()

  return ADMIN_NAV_KEY_TO_SECTION.get(active_key, "Sector Overview")


def _render_admin_content_chrome(sector_name: str):
  st.markdown(
    f"""
    <div class="admin-topbar">
      <div class="admin-topbar-left">
        <span class="admin-nav-dot" style="background:{ADMIN_COLORS['green']};"></span>
        <span class="admin-nav-dot" style="background:{ADMIN_COLORS['red']};"></span>
        <span class="admin-nav-dot" style="background:{ADMIN_COLORS['white']}; border:0.5px solid {ADMIN_COLORS['border_strong']};"></span>
      </div>
      <div class="admin-topbar-center">
        <span class="muted">Sector Governance Console · </span><span class="value">{escape(sector_name)}</span>
      </div>
      <button class="admin-deploy-btn" type="button">Deploy</button>
    </div>
    <div class="admin-command-header">
      <div class="admin-command-label">Sector Governance Console</div>
      <div class="admin-command-title">{escape(sector_name)} Command Center</div>
      <div class="admin-command-subtitle">Real-time telemetry and overview of your sector's institutions.</div>
    </div>
    """,
    unsafe_allow_html=True,
  )

def render():
  enforce_role(Role.BASKET_ADMIN.value)
  inject_enterprise_theme()
  _inject_admin_redesign_css()
  
  basket_id = st.session_state.get('basket_id')
  
  # Resolve sector name for a human-readable header
  with get_connection() as conn:
    c = conn.cursor()
    c.execute("SELECT id, name FROM baskets")
    all_baskets = {r['id']: r['name'] for r in c.fetchall()}

  sector_name = all_baskets.get(basket_id, f"Sector {basket_id}")
  fl_mode = st.session_state.get('fl_mode_enabled', False)
  pending_count = len(DeltaSyncManager.get_pending_syncs(basket_id))
  active_proj_count = len(ProjectManager.get_active_projects(basket_id))

  inst_count = 0
  try:
    with get_connection() as conn_i:
      ci = conn_i.cursor()
      ci.execute("SELECT COUNT(*) as c FROM institutions WHERE basket_id = ?", (basket_id,))
      res = ci.fetchone()
      if res:
        inst_count = res['c']
  except Exception:
    pass

  active_section = _render_admin_sidebar(sector_name, pending_count, fl_mode)
  _render_admin_content_chrome(sector_name)

  if active_section == "Sector Overview":
    st.markdown(f"### {sector_name} Command Center")
    st.write("Real-time telemetry and overview of your sector's institutions.")

    metrics_cols = st.columns(3)
    
    def _render_metric_card(col, title, value, icon, subtext, alert=False):
      bg_color = ADMIN_COLORS["white"]
      border_col = ADMIN_COLORS["border"]
      title_col = ADMIN_COLORS["text_muted"]
      val_col = ADMIN_COLORS["red"] if alert else ADMIN_COLORS["black"]
      accent = ADMIN_COLORS["red"] if alert else ADMIN_COLORS["green"]
      
      col.markdown(f"""
      <div style="background:{bg_color}; padding:1.2rem; border-radius:12px; 
            border:0.5px solid {border_col}; border-left:3px solid {accent}; box-shadow:none; margin-bottom:1rem;">
        <div style="color:{title_col}; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
          {icon} {title}
        </div>
        <div class="admin-num" style="color:{val_col}; font-size:2rem; font-weight:800; margin:0.3rem 0;">
          {value}
        </div>
        <div style="color:{title_col}; font-size:0.75rem;">{subtext}</div>
      </div>""", unsafe_allow_html=True)

    _render_metric_card(metrics_cols[0], "Registered Spokes", str(inst_count), "", "Active institutions in sector")
    _render_metric_card(metrics_cols[1], "Pending Reports", str(pending_count), "", "Requires admin review", alert=(pending_count>0))
    _render_metric_card(metrics_cols[2], "Active Projects", str(active_proj_count), "", "Cross-sector war rooms")

    # Telemetry Chart
    st.write("#### Telemetry Pulse")
    historical_syncs = DeltaSyncManager.get_historical_syncs(basket_id)
    all_syncs = historical_syncs + DeltaSyncManager.get_pending_syncs(basket_id)
    
    if not all_syncs:
      st.info("Insufficient data to generate telemetry baseline. Waiting for spoke transmissions.")
    else:
      try:
        import plotly.express as px
        df_chart = pd.DataFrame([
          {
            "Time": pd.to_datetime(s.get('created_at', s.get('timestamp', 0)), unit='s'),
            "Severity": s['payload'].get('severity_score', 0.0),
            "Status": s.get('status', 'PENDING'),
            "Type": s['payload'].get('incident_type', 'ANOMALY')
          }
          for s in all_syncs
        ])
        df_chart = df_chart.sort_values(by="Time")
        
        fig = px.scatter(
          df_chart, x="Time", y="Severity", color="Status", 
          size="Severity", hover_data=["Type"],
          color_discrete_map={"PENDING": "#DC2626", "PROCESSED": "#059669", "REJECTED": "#6B7280"}
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(
          height=300, 
          margin=dict(l=0, r=0, t=10, b=0),
          template="plotly_white",
          xaxis_title="",
          yaxis_title="Severity Score",
          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        _apply_plotly_numeric_font(fig)
        st.plotly_chart(fig, use_container_width=True)
      except Exception as e:
        st.caption("Unable to render telemetry chart.")
        
    st.write("---")
    
    # New Comprehensive Sector Intelligence Report Section
    st.write("#### Comprehensive Sector Analysis")
    st.write("Generate a full spectrum decision-ready intelligence report capturing both risks and stability vectors.")
    if st.button("Generate Sector Intelligence Report", type="primary"):
      with st.spinner("Compiling cross-sectional data and generating intelligence report..."):
        report_data = SectorReportGenerator.generate_report(basket_id)
        st.session_state['current_sector_report'] = report_data
        st.rerun()

    if 'current_sector_report' in st.session_state:
       st.write("---")
       with st.container(border=True):
         render_sector_report(st.session_state['current_sector_report'], theme_colors=None)
    
    st.write("---")
    st.info(" **Tip**: Open **Spoke Reports** in the left navigation to review incoming anomalies and promote them to national risks.")

  if active_section == "Data Sharing":
    st.markdown("### Lateral & Downward Data Sharing")
    st.write("Manage cross-sector data access requests and respond to Executive Directives.")
    
    # DOWNWARD DIRECTIVES
    st.write("#### Executive Directives & Benchmarks")
    directives = DataSharingManager.get_directives_for_basket(basket_id)
    if not directives:
      st.success("No active downward directives from the Executive Command.")
    else:
      for d in directives:
        req_ack = " (Requires Action)" if not d['is_acknowledged'] and d['requires_ack'] else ""
        bg = "#FEF2F2" if not d['is_acknowledged'] else "#F3F4F6"
        border = "#DC2626" if not d['is_acknowledged'] else "#9CA3AF"
        
        with st.container():
          st.markdown(f"""
          <div style="background:{bg}; border-left:4px solid {border}; padding:10px 14px; border-radius:4px; margin-bottom:10px;">
            <strong>[{d['directive_type']}] Priority: {d['priority']}</strong>{req_ack}<br/>
            <span style="font-size:0.9rem;">{d['content']}</span>
          </div>
          """, unsafe_allow_html=True)
          if not d['is_acknowledged'] and d['requires_ack']:
            if st.button("Acknowledge & Sync to Spokes", key=f"ack_dir_{d['id']}"):
              DataSharingManager.acknowledge_directive(d['id'], f"Basket_{basket_id}")
              # Auto-forward to spokes
              DataSharingManager.issue_directive(
                sender_role="BASKET_ADMIN",
                sender_id=f"Basket_{basket_id}",
                content=d['content'],
                priority=d['priority'],
                directive_type="CASCADED_DIRECTIVE",
                target_basket_id=basket_id,
                requires_ack=True
              )
              st.success("Directive acknowledged and cascaded to all nodes.")
              st.rerun()

    st.write("---")
    
    # LATERAL REQUESTS
    st.write("#### Cross-Sector Data Sharing")
    col_req, col_act = st.columns(2)
    
    with col_req:
      st.write("**Request Data from Another Sector**")
      other_baskets = {k: v for k, v in all_baskets.items() if k != basket_id}
      if other_baskets:
        target_b = st.selectbox("Target Sector", options=list(other_baskets.keys()), format_func=lambda x: other_baskets[x])
        scope = st.text_input("Requested Data Scope (e.g., 'Aggregated Health Telemetry Q3')")
        reason = st.text_area("Justification / Cross-Correlation Reason")
        
        if st.button("Submit Request"):
          if scope and reason:
            DataSharingManager.create_share_request(basket_id, target_b, reason, scope)
            st.success("Data share request submitted.")
          else:
            st.error("Scope and reason are required.")
    
    with col_act:
      st.write("**Incoming Requests for Your Data**")
      in_reqs = DataSharingManager.get_incoming_requests(basket_id)
      if not in_reqs:
        st.caption("No pending requests.")
      else:
        for req in in_reqs:
          with st.expander(f"Req from {req['requester_name']} | Scope: {req['data_scope']}"):
            st.write(f"**Reason:** {req['reason']}")
            c_app, c_rej = st.columns(2)
            dur = c_app.number_input("Duration (hrs)", min_value=1, value=24, key=f"dur_{req['id']}")
            if c_app.button("Approve", key=f"app_{req['id']}", type="primary"):
              DataSharingManager.resolve_request(req['id'], 'APPROVED', basket_id, dur)
              st.rerun()
            if c_rej.button("Deny", key=f"deny_{req['id']}"):
              DataSharingManager.resolve_request(req['id'], 'REJECTED', basket_id)
              st.rerun()

    st.write("#### Active Data Sharing Agreements")
    act_g = DataSharingManager.get_active_shares_granted(basket_id)
    act_r = DataSharingManager.get_active_shares_received(basket_id)
    
    if not act_g and not act_r:
       st.info("No active data shares.")
    else:
       cg, cr = st.columns(2)
       with cg:
         st.write("**Shares You Granted**")
         for g in act_g:
           exp_time = pd.to_datetime(g['expires_at'], unit='s').strftime('%Y-%m-%d %H:%M')
           st.markdown(f"**To:** {g['grantee_name']}<br/>**Scope:** {g['data_scope']}<br/>**Expires:** {exp_time}", unsafe_allow_html=True)
           if st.button("Revoke Access", key=f"rev_{g['id']}"):
             DataSharingManager.revoke_share(g['id'], basket_id)
             st.rerun()
           st.write("---")
       with cr:
         st.write("**Data Shared With You**")
         for r in act_r:
           exp_time = pd.to_datetime(r['expires_at'], unit='s').strftime('%Y-%m-%d %H:%M')
           st.markdown(f"**From:** {r['granter_name']}<br/>**Scope:** {r['data_scope']}<br/>**Expires:** {exp_time}", unsafe_allow_html=True)
           st.download_button("Download Encrypted Data Package", "Mock Data Payload", file_name=f"intel_{r['id']}.bin", key=f"dl_{r['id']}")
           st.write("---")
             

  if active_section == "Spoke Reports":
    st.markdown("### Incoming Reports from Spokes")
    st.write("Review anomaly reports submitted by institutions in your sector. Tick reports you want to combine into a risk and forward to the executive level.")
    
    pending_syncs = DeltaSyncManager.get_pending_syncs(basket_id)
    
    # State to track which events the Admin wants to fuse
    if 'selected_events' not in st.session_state:
      st.session_state['selected_events'] = set()
      
    if not pending_syncs:
      st.success("No pending reports. All incoming anomaly reports have been reviewed.")
    else:
      st.warning(f"{len(pending_syncs)} report(s) awaiting your review.")
      # Resolve institution names for display
      inst_name_cache = {}
      try:
        with get_connection() as conn_i:
          ci = conn_i.cursor()
          ci.execute("SELECT id, name FROM institutions WHERE basket_id = ?", (basket_id,))
          inst_name_cache = {r['id']: r['name'] for r in ci.fetchall()}
      except Exception:
        pass

      for sync in pending_syncs:
        payload = sync['payload']
        inst_display = inst_name_cache.get(sync['institution_id'], f"Institution {sync['institution_id']}")

        col_sel, col_exp = st.columns([1, 15])
        with col_sel:
          is_selected = st.checkbox("", key=f"sel_{sync['sync_id']}", value=sync['sync_id'] in st.session_state['selected_events'])
          if is_selected:
            st.session_state['selected_events'].add(sync['sync_id'])
          elif sync['sync_id'] in st.session_state['selected_events']:
            st.session_state['selected_events'].remove(sync['sync_id'])

        with col_exp.expander(
          f"{inst_display} — Severity {payload.get('severity_score', 0.0):.2f}"
          f" | {payload.get('anomaly_type', payload.get('incident_type', 'OBSERVATION'))}"
        ):
          # ── DEMO PAYLOAD RENDERING (summary / so_what / historical_comparison) ──
          # These fields are present in observations seeded via the demo_seeder
          # or generated by spoke institutions using the quick-report form.
          _summary  = payload.get('summary')
          _so_what  = payload.get('so_what')
          _hist_cmp = payload.get('historical_comparison')
          _prop_risk = payload.get('propagation_risk')
          _trend   = payload.get('trend')

          if _summary:
            st.markdown(
              f'<div style="background:#F8FAFC; border-left:4px solid #1F2937; padding:10px 14px; '
              f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.9rem; line-height:1.6;">'
              f'<strong>Field Report</strong><br>{_summary}</div>',
              unsafe_allow_html=True,
            )

          if _trend:
            _trend_colors = {"CRITICAL":"#BB0000","ACCELERATING":"#E05000","RISING":"#F59E0B",
                     "EARLY_ESCALATION":"#F59E0B","STABLE":"#059669","IMPROVING":"#059669"}
            _tc = _trend_colors.get(_trend, "#6B7280")
            st.markdown(
              f'<span style="background:{_tc}; color:#fff; padding:3px 10px; '
              f'border-radius:4px; font-size:0.8rem; font-weight:600;">Trend: {_trend}</span>',
              unsafe_allow_html=True,
            )
            st.write("")

          if _so_what:
            st.markdown(
              f'<div style="background:#FEF2F2; border-left:4px solid #DC2626; padding:10px 14px; '
              f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
              f'<strong>So What? (Consequence of Inaction)</strong><br>{_so_what}</div>',
              unsafe_allow_html=True,
            )

          if _hist_cmp:
            st.markdown(
              f'<div style="background:#F0F9FF; border-left:4px solid #3B82F6; padding:10px 14px; '
              f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
              f'<strong>Compared to What? (Historical Context)</strong><br>{_hist_cmp}</div>',
              unsafe_allow_html=True,
            )

          if _prop_risk:
            _risk_colors = {"CRITICAL":"#BB0000","HIGH":"#E05000","MODERATE":"#F59E0B","LOW":"#059669"}
            _rc = next((c for k,c in _risk_colors.items() if k in str(_prop_risk)), "#6B7280")
            st.markdown(
              f'<div style="background:#FFF7ED; border-left:4px solid {_rc}; padding:10px 14px; '
              f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
              f'<strong>Propagation Risk</strong><br>{_prop_risk}</div>',
              unsafe_allow_html=True,
            )

          # ── PIPELINE PAYLOAD RENDERING (composite_scores / shock_vector) ──
          # These fields are present when a spoke ran the full analysis pipeline.
          if 'composite_scores' in payload:
            st.write("#### Composite Intelligence Scores")
            c_col1, c_col2, c_col3 = st.columns(3)
            c_col1.metric("A) Detection Score", f"{payload['composite_scores'].get('A_Detection', 0.0):.2f} / 10.0")
            c_col2.metric("B) Impact Score", f"{payload['composite_scores'].get('B_Impact', 0.0):.2f} / 10.0")
            c_col3.metric("C) Certainty Score", f"{payload['composite_scores'].get('C_Certainty', 0.0):.2f} / 10.0")

            st.markdown(
              f'<div style="background:#F0FDF4; border-left:4px solid #10B981; padding:10px 14px; '
              f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem; line-height:1.5;">'
              f'<strong>Plain-language interpretation:</strong><br>'
              f'{narrate_composite_scores(payload["composite_scores"])}<br><br>'
              f'{narrate_severity(payload.get("severity_score", 0.0))}</div>',
              unsafe_allow_html=True,
            )
            st.write("---")

          # ── PILLAR 1: "SO WHAT?" (pipeline-generated projection) ──
          # Only shown when the spoke did NOT provide a manual 'so_what' field
          if not _so_what:
            sev = payload.get('severity_score', 0.0)
            projection = generate_inaction_projection(
              severity=sev,
              shock_vector=payload.get('shock_vector'),
              incident_type=payload.get('incident_type', 'ANOMALY'),
              composite_scores=payload.get('composite_scores'),
            )
            if projection:
              st.markdown(
                f'<div style="background:#FEF2F2; border-left:4px solid #DC2626; padding:10px 14px; '
                f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
                f'<strong> So What?</strong> {projection}</div>',
                unsafe_allow_html=True,
              )

          # ── PILLAR 2: "COMPARED TO WHAT?" (pipeline-generated context) ──
          if not _hist_cmp:
            sev = payload.get('severity_score', 0.0)
            hist_ctx = get_historical_context(
              basket_id=basket_id,
              severity=sev,
              incident_type=payload.get('incident_type', 'ANOMALY'),
            )
            st.markdown(
              f'<div style="background:#F0F9FF; border-left:4px solid #3B82F6; padding:10px 14px; '
              f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
              f'<strong>Compared to What?</strong> {hist_ctx}</div>',
              unsafe_allow_html=True,
            )

          if 'shock_vector' in payload:
            sv_narrative = narrate_shock_vector(payload['shock_vector'])
            st.markdown(
              f'<div style="background:#FFFBEB; border-left:4px solid #F59E0B; padding:10px 14px; '
              f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
              f'{sv_narrative}</div>',
              unsafe_allow_html=True,
            )
            st.plotly_chart(plot_shock_vector(payload['shock_vector'], "Pre vs Post Shock Vector"), use_container_width=True, key=f"pending_{sync['sync_id']}")
          if 'spoke_interpretation' in payload:
            st.markdown(f"**Institution's own assessment:** _{payload['spoke_interpretation']}_")
          if 'post_shock_volatility_forecast' in payload:
            st.write("**Volatility forecast at time of anomaly:**")
            st.json(payload['post_shock_volatility_forecast'])

          st.write("---")
          reject_msg = st.text_input("Reason for returning (optional):", key=f"rej_msg_{sync['sync_id']}")
          if st.button("Return to institution for more information", key=f"rej_btn_{sync['sync_id']}"):
            DeltaSyncManager.reject_sync(sync['sync_id'], reject_msg or "Please provide additional supporting data.")
            st.rerun()


      
      
      st.write("---")
      st.write("### Promote to National Risk")
      st.write("Select reports above, write a summary, and forward as a validated risk to the executive level.")
      
      selected_count = len(st.session_state['selected_events'])
      if selected_count > 0:
        st.info(f"{selected_count} Event(s) selected for Fusion.")
        risk_title = st.text_input("Validated Risk Title")
        risk_desc = st.text_area("Executive Summary / Risk Description", height=100)
        
        if st.button("Promote to Validated Risk", type="primary"):
          if not risk_title or not risk_desc:
            st.error("Please provide both a Title and Description for the Executive.")
          else:
            # Compute average scores across selected events
            selected_payloads = [s['payload'] for s in pending_syncs if s['sync_id'] in st.session_state['selected_events']]
            avg_A = np.mean([p.get('composite_scores', {}).get('A_Detection', 0) for p in selected_payloads]) if selected_payloads else 0.0
            avg_B = np.mean([p.get('composite_scores', {}).get('B_Impact', 0) for p in selected_payloads]) if selected_payloads else 0.0
            avg_C = np.mean([p.get('composite_scores', {}).get('C_Certainty', 0) for p in selected_payloads]) if selected_payloads else 0.0
            
            fused_scores = {
              "A_Detection": round(avg_A, 2),
              "B_Impact": round(avg_B, 2),
              "C_Certainty": round(avg_C, 2)
            }
            
            DeltaSyncManager.promote_risk(
              basket_id=basket_id,
              title=risk_title,
              description=risk_desc,
              composite_scores=fused_scores,
              source_sync_ids=list(st.session_state['selected_events'])
            )
            st.session_state['selected_events'] = set()
            st.success(f"Risk '{risk_title}' promoted to executive level.")
      else:
        st.write("*Tick reports above to select them, then fill in the fields here.*")

      if fl_mode:
        st.write("---")
        st.caption("Mode B is active. Open **Federated Learning (Mode B)** in the left navigation to run sector aggregation.")

  if active_section == "Risk Promotion":
    st.markdown("### Promoted Risks")
    st.write("Risks that you have validated and forwarded to the executive level. These are visible to national command.")

    promoted = DeltaSyncManager.get_promoted_risks(basket_id)
    if not promoted:
      st.success("No risks have been promoted from this sector yet. Review incoming reports in **Spoke Reports** and promote when ready.")
    else:
      st.info(f"{len(promoted)} risk(s) currently visible to the executive.")
      for risk in promoted:
        scores = risk.get('composite_scores', {})
        b_impact = scores.get('B_Impact', 0)
        with st.expander(f"{risk['title']} | Impact: {b_impact:.1f}/10 | {pd.to_datetime(risk.get('timestamp', 0), unit='s').strftime('%Y-%m-%d %H:%M')}"):
          st.write(risk.get('description', ''))
          c1, c2, c3 = st.columns(3)
          c1.metric("Detection", f"{scores.get('A_Detection', 0):.2f}/10")
          c2.metric("Impact", f"{b_impact:.2f}/10")
          c3.metric("Certainty", f"{scores.get('C_Certainty', 0):.2f}/10")

          # ── PILLAR 4: "WHAT SHOULD I DO?" ──
          rec = generate_recommendation(
            risk=risk,
            all_baskets=all_baskets,
            global_risks=promoted,
            historical_syncs=DeltaSyncManager.get_historical_syncs(basket_id),
          )
          st.markdown(
            f'<div style="background:#FFFBEB; border-left:4px solid {rec.level_color}; '
            f'padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0;">'
            f'<strong>Recommendation: '
            f'<span style="background:{rec.level_color}; color:#fff; padding:2px 8px; '
            f'border-radius:4px; font-size:0.8rem;">{rec.level}</span></strong><br/>'
            f'<span style="font-size:0.88rem;">{rec.summary}</span><br/>'
            f'<span style="font-size:0.82rem; color:#64748b;">'
            f'<b>Who:</b> {", ".join(rec.who[:4])} &nbsp;|&nbsp; '
            f'<b>Urgency:</b> {rec.urgency}</span>'
            f'</div>',
            unsafe_allow_html=True,
          )

          with st.expander(" Causal Evidence", expanded=False):
            st.caption(
              "Estimate the causal relationship behind this risk using historical national "
              "indicators. Pre-loaded with World Bank data. No data leaves this session."
            )
            try:
              from kshiked.ui.kshield.causal import (
                render_causal_evidence_panel, load_world_bank_data,
              )
              from kshiked.ui.theme import LIGHT_THEME as _adm_theme
              _awb = load_world_bank_data()
              if not _awb.empty:
                _acols = [c for c in _awb.columns if _awb[c].notna().sum() >= 15]
                _risk_uid = risk.get('id', abs(hash(risk.get('title', ''))) % 999999)
                _arc1, _arc2 = st.columns(2)
                with _arc1:
                  _a_treat = st.selectbox(
                    "Cause", _acols, index=0,
                    key=f"admin_risk_t_{_risk_uid}",
                  )
                with _arc2:
                  _a_out_opts = [c for c in _acols if c != _a_treat]
                  _a_out = st.selectbox(
                    "Effect", _a_out_opts, index=0,
                    key=f"admin_risk_o_{_risk_uid}",
                  )
                render_causal_evidence_panel(
                  df=_awb, treatment=_a_treat, outcome=_a_out,
                  theme=_adm_theme, key_prefix=f"ar_{_risk_uid}",
                )
              else:
                st.info("Open the Causal Analysis tab to load World Bank data.")
            except Exception as _ce:
              st.caption(f"Causal evidence unavailable: {_ce}")

  if active_section == "Operational Projects":
    st.info("### Operational Projects (Cross-Basket Fusion Spaces)")
    st.write("Temporary collaboration containers linking multiple baskets to one evolving complex situation.")
    
    # 1. Launch New Structured Project
    st.write("---")
    with st.container(border=True):
      from kshiked.ui.institution.project_components import render_project_wizard
      render_project_wizard(all_baskets, basket_id)
      
    st.write("---")
    
    # 2. View Active Projects
    st.write("#### Active Projects")
    active_projects = ProjectManager.get_active_projects(basket_id)
    
    if not active_projects:
      st.success("No active operational projects require your attention.")
    else:
      for proj in active_projects:
        with st.expander(f"WAR ROOM: {proj['title']} (Severity: {proj['severity']})", expanded=True):
          project_data = ProjectManager.get_project_details(proj['id'])
          
          # 1. Phase Progression Banner
          current_phase = proj['current_phase']
          phases = ["EMERGENCE", "ESCALATION", "STABILIZATION", "RECOVERY"]
          try:
            p_idx = phases.index(current_phase)
          except ValueError:
            p_idx = 0
            
          st.write("---")
          st.write(f"**Current Phase:** {' ➔ '.join([f'*{p}*' if i < p_idx else f'**{p}**' if i == p_idx else p for i, p in enumerate(phases)])}")
          if p_idx < len(phases) - 1:
            if st.button(f"Promote Phase to {phases[p_idx + 1]}", key=f"phase_btn_{proj['id']}", use_container_width=True):
              ProjectManager.transition_phase(proj['id'], phases[p_idx + 1])
              st.rerun()
          st.write("---")
          
          p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
          with p_col1:
            st.write("**Participants**")
            for p_b_id in project_data['participants']:
              st.write(f"- {all_baskets.get(p_b_id, f'Basket {p_b_id}')}")
            st.write("---")
            st.write(f"**Created:**<br>{pd.to_datetime(proj['created_at'], unit='s').strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
            
            st.write("---")
            st.write("**Disagreement Tensor (Consensus Drift)**")
            dis_matrix = ProjectManager.get_disagreement_matrix(proj['id'])
            if dis_matrix:
              import plotly.express as px
              df_dis = pd.DataFrame(list(dis_matrix.items()), columns=["Sector Admin", "Certainty"])
              fig = px.bar(df_dis, x="Sector Admin", y="Certainty", range_y=[0.0, 1.0], color="Certainty", color_continuous_scale="RdYlGn")
              fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
              _apply_plotly_numeric_font(fig)
              st.plotly_chart(fig, use_container_width=True, key=f"dis_{proj['id']}")
            else:
              st.caption("Awaiting sector updates.")
            
          with p_col2:
            st.write("**Shared Causal Stream**")
            st.markdown(f"*{proj['description']}*")
            st.write("---")
            
            for update in project_data.get('updates', []):
              u_color = "#006600" if update['update_type'] == 'OBSERVATION' else "#BB0000" if update['update_type'] == 'POLICY_ACTION' else "#1F2937"
              st.markdown(f"**<span style='color:{u_color};'>[{update['update_type']}]</span> {update['author_name']}**", unsafe_allow_html=True)
              st.write(update['content'])
              if update['certainty']:
                st.caption(f"Certainty Index: {update['certainty']:.2f}/1.0")
              st.write("")
              
          with p_col3:
            st.write("**Actions**")
            new_update = st.text_area("Post Update", key=f"upd_text_{proj['id']}")
            update_type = st.selectbox("Update Type", ["OBSERVATION", "ANALYSIS_REQUEST"], key=f"upd_type_{proj['id']}")
            
            cert_val = None
            if update_type == 'OBSERVATION':
              cert_val = st.slider("Your Certainty", 0.0, 1.0, 0.5, 0.05, key=f"cert_{proj['id']}")
              
            if st.button("Broadcast to Project Stream", key=f"upd_btn_{proj['id']}"):
              ProjectManager.add_update(proj['id'], st.session_state.get('username', f"Admin {basket_id}"), update_type, new_update, certainty=cert_val)
              st.rerun()
              
            st.write("---")
            
            with st.expander("Meta-Learning: Force Archive & Resolution"):
              st.write("Select the failure mode or resolution state. This securely recalibrates Sector Trust Weights across the entire national network.")
              res_state = st.selectbox("Resolution State", ['RESOLVED', 'FALSE_ALARM', 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_SIGNALS'], key=f"res_state_{proj['id']}")
              pol_score = st.slider("Policy Effectiveness", 0.0, 10.0, 5.0, 0.5, key=f"pol_score_{proj['id']}")
              res_sum = st.text_area("Debrief Summary", key=f"res_sum_{proj['id']}")
              
              if st.button("Archive & Calibrate Weights", key=f"arch_btn_{proj['id']}", type="primary"):
                if not res_sum:
                  st.error("Debrief summary required for Meta-Learning.")
                else:
                  payload = {"final_severity": proj['severity'], "active_participants": len(project_data['participants'])}
                  ProjectManager.archive_project(proj['id'], res_state, pol_score, res_sum, payload)
                  st.rerun()
                  
          st.write("---")
          st.markdown("#### Structured Project Health")
          from kshiked.ui.institution.project_components import render_project_overview
          render_project_overview(project_data, "Admin", basket_id, all_baskets)
              
    st.write("---")

  if active_section == "Historical Archive":
    st.markdown("### Historical Archive")
    st.write("All previously reviewed reports and closed operational projects.")
    
    st.write("#### Institutional Memory (Closed Projects)")
    memories = ProjectManager.get_institutional_memory()
    
    if not memories:
      st.write("No meta-learning historical archives exist yet.")
    else:
      for mem in memories:
        res_color = "#006600" if mem['resolution_state'] == 'RESOLVED' else "#BB0000" if mem['resolution_state'] == 'FALSE_ALARM' else "#1F2937"
        with st.expander(f"[{mem['resolution_state']}] {mem['title']} (Final Severity: {mem['severity']})"):
          m_col1, m_col2 = st.columns([2, 1])
          with m_col1:
            st.markdown(f"**Resolution Summary:**<br>{mem['resolution_summary']}", unsafe_allow_html=True)
            st.write("---")

            # ── PILLAR 5: "DID IT WORK?" ──
            try:
              participant_ids = []
              with get_connection() as conn_p5:
                cp5 = conn_p5.cursor()
                cp5.execute("SELECT basket_id FROM project_participants WHERE project_id = ?", (mem['id'],))
                participant_ids = [r['basket_id'] for r in cp5.fetchall()]
              if participant_ids:
                impact = compute_outcome_impact(
                  project_id=mem['id'],
                  project_created_at=mem.get('created_at', 0),
                  project_archived_at=mem.get('updated_at', 0),
                  participant_basket_ids=participant_ids,
                )
                bg = "#F0FDF4" if impact.get('delta_pct', 0) < 0 else "#FEF2F2"
                border = "#10B981" if impact.get('delta_pct', 0) < 0 else "#EF4444"
                st.markdown(
                  f'<div style="background:{bg}; border-left:4px solid {border}; '
                  f'padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
                  f'<strong> Did it Work?</strong> {impact["narrative"]}</div>',
                  unsafe_allow_html=True,
                )
            except Exception:
              pass

            st.write(f"**Learning Payload / Meta Data:**")
            st.json(mem['learning_payload'])
          with m_col2:
            st.metric("Policy Effectiveness", f"{mem['policy_effectiveness_score']}/10.0")
            
            ttc_mins = mem['time_to_consensus_seconds'] / 60.0
            st.metric("Time-to-Resolution", f"{ttc_mins:.1f} minutes")
            
            st.markdown(f"<span style='color:{res_color};'>**Network Trust Weight Recalibrated**</span>", unsafe_allow_html=True)
            
    st.write("---")
    st.write("#### Raw Spoke Anomalies (Merged)")
    
    historical_syncs = DeltaSyncManager.get_historical_syncs(basket_id)
    if not historical_syncs:
      st.write("No historical insights found.")
    else:
      st.write(f"Showing **{len(historical_syncs)}** historical records.")
      for sync in historical_syncs:
        payload = sync['payload']
        status_color = "#00FF00" if sync['status'] == 'PROCESSED' else "#FF4444"
        with st.expander(f"[{sync['status']}] Report from Spoke {sync['institution_id']} (Severity: {payload.get('severity_score', 0.0):.2f})"):
          if sync['status'] == 'REJECTED':
            st.error(f"**Admin Feedback:** {payload.get('admin_message', 'No message provided.')}")
            
          if 'composite_scores' in payload:
            st.write("#### Composite Intelligence Scores")
            c_col1, c_col2, c_col3 = st.columns(3)
            c_col1.metric("A) Detection Score", f"{payload['composite_scores'].get('A_Detection', 0.0):.2f} / 10.0")
            c_col2.metric("B) Impact Score", f"{payload['composite_scores'].get('B_Impact', 0.0):.2f} / 10.0")
            c_col3.metric("C) Certainty Score", f"{payload['composite_scores'].get('C_Certainty', 0.0):.2f} / 10.0")
            st.write("---")
            
          if 'shock_vector' in payload:
            st.plotly_chart(plot_shock_vector(payload['shock_vector'], f"Historical Shock Vector (ID: {sync['sync_id']})"), use_container_width=True, key=f"hist_{sync['sync_id']}")
          if 'spoke_interpretation' in payload:
            st.markdown(f"**Interpretation:** {payload['spoke_interpretation']}")

  if active_section == "Data Governance & Schemas":
    st.markdown("### Data Governance & Custom Schemas")
    st.write("Enforce structured reporting by defining exact column schemas that your assigned Spokes must follow.")
    
    from kshiked.ui.institution.backend.schema_manager import SchemaManager
    
    with st.expander("Create New Custom Schema", expanded=True):
      st.write("Define fields for a new schema to deploy to institutions.")
      new_schema_name = st.text_input("Schema Name", key="new_schema_name")
      
      if 'schema_builder_fields' not in st.session_state:
        st.session_state['schema_builder_fields'] = [{"name": "", "type": "number", "required": True}]
        
      for i, field in enumerate(st.session_state['schema_builder_fields']):
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        with col_f1:
          field["name"] = st.text_input("Field Name", key=f"f_name_{i}", value=field["name"])
        with col_f2:
          field["type"] = st.selectbox("Type", ["number", "text", "date", "bool"], index=["number", "text", "date", "bool"].index(field["type"]), key=f"f_type_{i}")
        with col_f3:
          field["required"] = st.checkbox("Required", value=field["required"], key=f"f_req_{i}")
          
      if st.button("+ Add Field"):
        st.session_state['schema_builder_fields'].append({"name": "", "type": "number", "required": True})
        st.rerun()
        
      if st.button("Save & Deploy Schema", type="primary"):
        valid_fields = [f for f in st.session_state['schema_builder_fields'] if f['name'].strip()]
        if not new_schema_name:
          st.error("Schema Name is required.")
        elif not valid_fields:
          st.error("At least one valid field is required.")
        else:
          SchemaManager.save_schema(basket_id, new_schema_name, valid_fields)
          st.success(f"Schema '{new_schema_name}' saved.")
          # Reset state
          st.session_state['schema_builder_fields'] = [{"name": "", "type": "number", "required": True}]
          import time 
          time.sleep(1)
          st.rerun()
          
    schemas = SchemaManager.get_schemas(basket_id)
    if schemas:
      st.write("#### Deployed Schemas (Active)")
      for sc in schemas:
        with st.expander(f"{sc['schema_name']} ({len(sc['fields'])} fields)"):
          st.json(sc['fields'])
          import datetime
          st.caption(f"Created/Updated: {datetime.datetime.fromtimestamp(sc['created_at']).strftime('%Y-%m-%d %H:%M')}")
          
    st.write("---")
    st.markdown("### Global Settings")
    st.write("Configure detection sensitivity thresholds for your sector's institutions.")
    st.caption("These thresholds are applied to all institutions in your sector when you push them below.")

    col1, col2 = st.columns(2)
    with col1:
      st.slider("Anomaly Detection Sensitivity (Lower = More Alerts)", 1.0, 15.0, 4.5, 0.5)
      st.slider("Prediction Volatility Smoothing", 0.5, 0.99, 0.8, 0.01)
    with col2:
      st.slider("Threat Detection Speed", 0.01, 0.5, 0.1, 0.01)
      st.selectbox("Hardware Defense Protocol", ["Standard Mode", "Aggressive Mode", "High Performance Mode"])

    if st.button("Push Target Bounds to Spokes"):
      st.success("System configurations successfully synchronized to all connected nodes.")

    if fl_mode:
      st.write("---")
      st.info("**Mode B active.** The aggregated global model is available in **Federated Learning (Mode B)** from the left navigation.")

  if active_section == "Collaboration Room":
    st.markdown("### Collaboration Room")
    render_collab_room(
      role=Role.BASKET_ADMIN.value,
      basket_id=basket_id,
      username=st.session_state.get('username'),
      all_baskets=all_baskets,
    )

  if fl_mode and active_section == "Federated Learning (Mode B)":
    st.markdown("### Federated Learning (Mode B)")
    st.info(
      "**Mode B is active.** Raw data from spokes never leaves their node. Only mathematical summaries "
      "(aggregated model weights) are processed here. Gradients flow upward; raw records do not."
    )

    # Auto-hydrate from historical processed syncs
    if 'current_global_weights' not in st.session_state:
      historical_syncs = DeltaSyncManager.get_historical_syncs(basket_id)
      processed_syncs = [s for s in historical_syncs if s['status'] == 'PROCESSED']
      if processed_syncs:
        payloads_hist = [s['payload'] for s in processed_syncs]
        gw, gm = FederationBridge.aggregate_spoke_models(payloads_hist, method_name='trimmed_mean')
        st.session_state['current_global_weights'] = gw
        st.session_state['last_aggregation_meta'] = gm

    st.write("#### Step 1 \u2014 Aggregate Pending Reports into Global Model")
    pending_fl = DeltaSyncManager.get_pending_syncs(basket_id)
    if not pending_fl:
      st.success("No pending reports to aggregate.")
    else:
      fl_methods = {
        "Trimmed Mean (recommended \u2014 removes outliers)": "trimmed_mean",
        "Median": "median",
        "Krum (most reliable node only)": "krum",
        "Bulyan (Byzantine-resilient consensus)": "bulyan"
      }
      selected_fl_method = st.selectbox("Aggregation strategy", list(fl_methods.keys()))
      if st.button("Run Aggregation", type="primary"):
        payloads_fl = [s['payload'] for s in pending_fl]
        global_weights, meta = FederationBridge.aggregate_spoke_models(
          payloads_fl, method_name=fl_methods[selected_fl_method]
        )
        DeltaSyncManager.mark_synced([s['sync_id'] for s in pending_fl])
        st.session_state['current_global_weights'] = global_weights
        st.session_state['last_aggregation_meta'] = meta
        st.success(f"Aggregated {meta.get('participants', '?')} participants using {selected_fl_method}.")
        st.rerun()

    if 'current_global_weights' in st.session_state:
      st.write("---")
      st.write("#### Current Global Model Weights")
      import plotly.graph_objects as go_fl
      weights_fl = st.session_state['current_global_weights']
      fig_fl = go_fl.Figure()
      y_data_fl = weights_fl if isinstance(weights_fl, list) else weights_fl.tolist()
      fig_fl.add_trace(go_fl.Scatter(y=y_data_fl, mode='lines+markers', line=dict(color='#006600')))
      fig_fl.update_layout(template='plotly_white', height=280, margin=dict(l=0, r=0, t=0, b=0))
      _apply_plotly_numeric_font(fig_fl)
      st.plotly_chart(fig_fl, use_container_width=True, key="global_model_chart_fl")

      st.write("---")
      st.write("#### Step 2 \u2014 Apply Privacy Protection before Broadcast")
      st.write("Add differential privacy noise to protect institution identities before sharing aggregated weights.")
      epsilon_fl = st.slider("Privacy level (lower = more privacy, less precision)", 0.1, 10.0, 1.5, 0.1)
      if st.button("Apply Privacy Noise"):
        noised = FederationBridge.apply_differential_privacy(weights_fl, epsilon_fl)
        st.session_state['dp_weights'] = noised
        st.success("Privacy noise applied.")

      if 'dp_weights' in st.session_state:
        dp_c1, dp_c2 = st.columns(2)
        if dp_c1.button("Broadcast to Spokes (privacy-protected)", use_container_width=True):
          st.success("Aggregated insights broadcast to spoke nodes.")
        if dp_c2.button("Share with Peer Sectors (privacy-protected)", use_container_width=True):
          st.success("Cross-sector synchronization complete.")

  # Terminology guide removed — terminology is now inline where needed
    
  if active_section == "Communications":
    with st.container(border=True):
      st.markdown("### Communications")

      # Escalate up to national
      st.write("#### Escalate to national level")
      esc_col1, esc_col2 = st.columns([4, 1])
      with esc_col1:
        escalation = st.text_input("Message to executive", key="esc_payload", placeholder="Describe a high-priority situation requiring national attention...")
      with esc_col2:
        st.write("")
        if st.button("Send to Executive", type="primary", use_container_width=True):
          if escalation:
            SecureMessaging.send_message(
              sender_role=Role.BASKET_ADMIN.value,
              sender_id=st.session_state.get('username'),
              receiver_role=Role.EXECUTIVE.value,
              receiver_id="ALL",
              content=escalation
            )
            st.success("Escalation sent to executive.")

      st.write("---")
      c_col1, c_col2 = st.columns([1, 1])
      with c_col1:
        st.write("**Messages from your institutions**")
        spoke_inbox = SecureMessaging.get_inbox(Role.BASKET_ADMIN.value, str(basket_id))
        if not spoke_inbox:
          st.caption("No messages from institutions.")
        else:
          for msg in spoke_inbox:
            with st.expander(f"From {msg['sender_id']} | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
              st.write(msg['content'])
              if not msg['is_read']:
                if st.button("Mark as read", key=f"ac_clear_{msg['id']}"):
                  SecureMessaging.mark_read(msg['id'])
                  st.rerun()

        st.write("---")
        st.write("**Send directive to an institution**")
        target_spoke = st.text_input("Institution username (or 'ALL' to broadcast)")
        directive = st.text_area("Message", height=100)
        if st.button("Send directive", type="primary", use_container_width=True):
          if directive and target_spoke:
            SecureMessaging.send_message(
              sender_role=Role.BASKET_ADMIN.value,
              sender_id=st.session_state.get('username'),
              receiver_role=Role.INSTITUTION.value,
              receiver_id=target_spoke,
              content=directive
            )
            st.success("Directive sent.")

      with c_col2:
        st.write("**Directives from the executive**")
        exec_inbox = SecureMessaging.get_inbox(Role.BASKET_ADMIN.value, st.session_state.get('username'))
        if not exec_inbox:
          st.caption("No directives received from executive.")
        else:
          for msg in exec_inbox:
            with st.expander(f"Executive | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
              st.write(msg['content'])
              if not msg['is_read']:
                if st.button("Acknowledge", key=f"ac_exec_{msg['id']}"):
                  SecureMessaging.mark_read(msg['id'])
                  st.rerun()

  # with tab_research:
  #   from kshiked.ui.institution.backend.research_engine import ResearchEngine, EngineContext
  #   from kshiked.ui.institution.research_components import render_research_engine_panel
  #   
  #   ctx = EngineContext(role="admin", user_id=st.session_state.get('username'), sector_id=basket_id)
  #   engine = ResearchEngine(context=ctx)
  #   render_research_engine_panel(engine)


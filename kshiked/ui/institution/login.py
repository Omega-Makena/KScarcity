import streamlit as st
import sys
import os
import qrcode
import secrets
from io import BytesIO
from pathlib import Path
from html import escape

# Ensure backend package is accessible and Scarcity engine handles imports correctly.
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import (
  login_user, logout_user, generate_totp_secret, 
  verify_totp_token, enable_user_2fa, is_developer_session, hash_password
)
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.style import inject_enterprise_theme, get_base64_of_bin_file
from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.institution.shared_sidebar import render_shared_sidebar


KENYAN_COLORS = {
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


INSTITUTION_NAV_ITEMS = [
  ("analysis", "Data Intake", "data-intake", KENYAN_COLORS["red"], None),
  ("analysis", "Signal Analysis", "signal-analysis", KENYAN_COLORS["red"], None),
  ("analysis", "Granger Causality", "granger-causality", KENYAN_COLORS["red"], None),
  ("analysis", "Causal Network", "causal-network", KENYAN_COLORS["red"], None),
  ("analysis", "Cross-Correlations", "cross-correlations", KENYAN_COLORS["red"], None),
  ("analysis", "Effect Estimation", "effect-estimation", KENYAN_COLORS["red"], None),
  ("operations", "Active Projects", "active-projects", KENYAN_COLORS["green"], None),
  ("operations", "Inbox", "inbox", KENYAN_COLORS["green"], None),
  ("operations", "Collaboration Room", "collaboration-room", KENYAN_COLORS["green"], None),
  ("models", "Model Configuration", "model-configuration", KENYAN_COLORS["black"], None),
  ("models", "Federated Learning", "federated-learning", KENYAN_COLORS["black"], "Mode B"),
]


INSTITUTION_NAV_KEY_TO_SECTION = {
  "data-intake": "Data Intake",
  "signal-analysis": "Signal Analysis",
  "granger-causality": "Granger Causality",
  "causal-network": "Causal Network",
  "cross-correlations": "Cross-Correlations",
  "effect-estimation": "Effect Estimation",
  "active-projects": "Active Projects",
  "inbox": "Inbox",
  "collaboration-room": "Collaboration Room",
  "model-configuration": "Model Configuration",
  "federated-learning": "FL Training Log",
}


@st.cache_data(show_spinner=False)
def _get_gok_logo_b64() -> str:
  logo_path = Path(project_root) / "GOK.png"
  if not logo_path.exists():
    return ""
  return get_base64_of_bin_file(str(logo_path))


def _query_value(query_params, key, default=""):
  value = query_params.get(key, default)
  if isinstance(value, list):
    return str(value[0]) if value else str(default)
  return str(value)


def _resolve_basket_name(basket_id):
  try:
    if not basket_id:
      return "Unknown Sector"
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("SELECT name FROM baskets WHERE id = ?", (basket_id,))
      row = c.fetchone()
      if row and row["name"]:
        return str(row["name"])
  except Exception:
    pass
  return f"Sector {basket_id}" if basket_id else "Unknown Sector"


def _resolve_institution_name(institution_id, fallback_access_id):
  try:
    if institution_id:
      with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM institutions WHERE id = ?", (institution_id,))
        row = c.fetchone()
        if row and row["name"]:
          return str(row["name"])
  except Exception:
    pass
  return fallback_access_id or "Institution Node"


def _inject_kenyan_shell_css():
  logo_b64 = _get_gok_logo_b64()

  if logo_b64:
    background_css = f"""
      .stApp {{
        background-image: url("data:image/png;base64,{logo_b64}") !important;
        background-size: 360px auto !important;
        background-repeat: no-repeat !important;
        background-position: center center !important;
        background-color: {KENYAN_COLORS['surface']} !important;
      }}

      [data-testid="stAppViewContainer"] {{
        background-color: rgba(248, 247, 245, 0.93) !important;
      }}
    """
  else:
    background_css = f"""
      .stApp {{
        background: {KENYAN_COLORS['surface']} !important;
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
      .kpi-value,
      .mono,
      .stNumberInput input,
      input[type="number"],
      .js-plotly-plot .xtick text,
      .js-plotly-plot .ytick text,
      .js-plotly-plot .hovertext {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-variant-numeric: tabular-nums;
      }}

      {background_css}

      [data-testid="stHeader"] {{
        background: transparent !important;
        border-bottom: none !important;
      }}

      [data-testid="stSidebarCollapsedControl"] {{
        display: block !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 10010 !important;
      }}

      [data-testid="stSidebarCollapsedControl"] button {{
        background: {KENYAN_COLORS['white']} !important;
        color: {KENYAN_COLORS['black']} !important;
        border: 0.5px solid {KENYAN_COLORS['border_strong']} !important;
        border-radius: 8px !important;
        box-shadow: none !important;
      }}

      [data-testid="stSidebarCollapsedControl"] button:hover {{
        border-color: {KENYAN_COLORS['red']} !important;
        color: {KENYAN_COLORS['red']} !important;
      }}

      [data-testid="stSidebar"] {{
        background: {KENYAN_COLORS['white']} !important;
        width: 210px !important;
        min-width: 210px !important;
        max-width: 210px !important;
        border-right: 0.5px solid {KENYAN_COLORS['border']} !important;
      }}

      [data-testid="stSidebar"] > div:first-child {{
        background: {KENYAN_COLORS['white']} !important;
      }}

      [data-testid="stSidebar"] .block-container {{
        padding-top: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-bottom: 0.75rem !important;
      }}

      [data-testid="stSidebar"] [role="radiogroup"] {{
        display: flex !important;
        gap: 8px !important;
      }}

      [data-testid="stSidebar"] [role="radiogroup"] > label {{
        flex: 1 1 0 !important;
        border-radius: 999px !important;
        border: 0.5px solid {KENYAN_COLORS['border_strong']} !important;
        background: {KENYAN_COLORS['white']} !important;
        color: {KENYAN_COLORS['text_muted']} !important;
        padding: 4px 8px !important;
        margin: 0 !important;
      }}

      [data-testid="stSidebar"] [role="radiogroup"] > label p {{
        font-size: 12px !important;
        font-weight: 500 !important;
        text-align: center !important;
      }}

      [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
        background: {KENYAN_COLORS['green']} !important;
        border-color: {KENYAN_COLORS['green']} !important;
      }}

      [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{
        color: {KENYAN_COLORS['white']} !important;
      }}

      [data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        text-align: left !important;
        border-radius: 8px !important;
        border: 0.5px solid {KENYAN_COLORS['border_strong']} !important;
        background: {KENYAN_COLORS['white']} !important;
        color: {KENYAN_COLORS['black']} !important;
        box-shadow: none !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        letter-spacing: 0.1px !important;
        padding: 0.42rem 0.55rem !important;
        min-height: 34px !important;
      }}

      [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: {KENYAN_COLORS['red']} !important;
        color: {KENYAN_COLORS['black']} !important;
        background: {KENYAN_COLORS['red_light']} !important;
      }}

      [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {KENYAN_COLORS['red_light']} !important;
        color: {KENYAN_COLORS['red']} !important;
        font-weight: 500 !important;
        border-color: {KENYAN_COLORS['border']} !important;
        border-left: 3px solid {KENYAN_COLORS['red']} !important;
      }}

      [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
        background: {KENYAN_COLORS['red_light']} !important;
        color: {KENYAN_COLORS['red']} !important;
        border-left: 3px solid {KENYAN_COLORS['red']} !important;
      }}

      [data-testid="stSidebar"] .stButton:last-of-type > button {{
        border: none !important;
        background: transparent !important;
        color: {KENYAN_COLORS['text_faint']} !important;
        font-size: 11px !important;
        padding: 2px 0 !important;
        min-height: auto !important;
      }}

      [data-testid="stSidebar"] .stButton:last-of-type > button:hover {{
        background: transparent !important;
        color: {KENYAN_COLORS['red']} !important;
      }}

      .main .block-container {{
        padding-top: 1rem !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
        padding-bottom: 24px !important;
        max-width: 100% !important;
      }}

      .k-topbar {{
        background: {KENYAN_COLORS['black']};
        border: 0.5px solid {KENYAN_COLORS['border_strong']};
        border-left: 3px solid {KENYAN_COLORS['red']};
        border-radius: 12px;
        min-height: 56px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 0 14px;
        margin-bottom: 10px;
      }}

      .k-topbar-left {{
        display: flex;
        align-items: center;
        gap: 6px;
        min-width: 72px;
      }}

      .k-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
      }}

      .k-topbar-center {{
        color: {KENYAN_COLORS['white']};
        flex: 1;
        text-align: center;
        font-size: 15px;
        font-weight: 500;
        line-height: 1.25;
      }}

      .k-topbar-center .muted {{
        color: #F3C9C9;
      }}

      .k-topbar-center .value {{
        color: {KENYAN_COLORS['white']};
        font-weight: 500;
      }}

      .k-deploy-btn {{
        border: 0.5px solid {KENYAN_COLORS['white']};
        color: {KENYAN_COLORS['white']};
        background: transparent;
        border-radius: 8px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 500;
        min-width: 76px;
        text-align: center;
      }}

      .k-profile {{
        background: {KENYAN_COLORS['white']};
        border: 0.5px solid {KENYAN_COLORS['border']};
        border-bottom: 3px solid {KENYAN_COLORS['red']};
        border-radius: 12px;
        padding: 12px;
      }}

      .k-access-id {{
        color: {KENYAN_COLORS['black']};
        font-size: 14px;
        font-weight: 500;
        line-height: 1.2;
      }}

      .k-clearance-pill {{
        display: inline-flex;
        align-items: center;
        background: {KENYAN_COLORS['green_light']};
        color: {KENYAN_COLORS['green']};
        border: 0.5px solid {KENYAN_COLORS['green']};
        border-radius: 999px;
        padding: 2px 8px;
        margin-top: 8px;
        font-size: 11px;
        font-weight: 500;
      }}

      .k-sector-label {{
        color: {KENYAN_COLORS['text_faint']};
        font-size: 11px;
        margin-top: 6px;
      }}

      .k-block-label {{
        color: {KENYAN_COLORS['text_muted']};
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
        margin-bottom: 8px;
      }}

      .k-mode-row {{
        display: flex;
        gap: 8px;
      }}

      .k-mode-pill {{
        display: inline-flex;
        justify-content: center;
        align-items: center;
        flex: 1;
        border-radius: 999px;
        border: 0.5px solid {KENYAN_COLORS['border_strong']};
        background: {KENYAN_COLORS['white']};
        color: {KENYAN_COLORS['text_muted']};
        font-size: 12px;
        font-weight: 500;
        padding: 5px 8px;
        text-decoration: none;
      }}

      .k-mode-pill.active {{
        background: {KENYAN_COLORS['green']};
        color: {KENYAN_COLORS['white']};
        border-color: {KENYAN_COLORS['green']};
      }}

      .k-divider {{
        height: 0.5px;
        background: {KENYAN_COLORS['border']};
        margin: 10px 0;
      }}

      .k-nav-group-label {{
        color: {KENYAN_COLORS['text_muted']};
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
        margin-bottom: 6px;
      }}

      .k-group-dot {{
        width: 5px;
        height: 5px;
        border-radius: 50%;
        display: inline-block;
      }}

      .k-nav-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        border: 0.5px solid transparent;
        border-left: 3px solid transparent;
        border-radius: 8px;
        background: {KENYAN_COLORS['white']};
        color: {KENYAN_COLORS['text_muted']};
        font-size: 13px;
        font-weight: 400;
        text-decoration: none;
        padding: 7px 8px;
        margin-bottom: 4px;
      }}

      .k-nav-item:hover {{
        background: {KENYAN_COLORS['red_light']};
        color: {KENYAN_COLORS['black']};
      }}

      .k-nav-item.active {{
        background: {KENYAN_COLORS['red_light']};
        color: {KENYAN_COLORS['red']};
        border-left-color: {KENYAN_COLORS['red']};
        font-weight: 500;
      }}

      .k-nav-dot {{
        width: 5px;
        height: 5px;
        border-radius: 50%;
        flex: 0 0 5px;
      }}

      .k-nav-text {{
        flex: 1;
      }}

      .k-nav-tag {{
        background: {KENYAN_COLORS['black']};
        color: {KENYAN_COLORS['white']};
        border-radius: 999px;
        padding: 1px 6px;
        font-size: 10px;
        font-weight: 500;
        margin-top: 8px;
        text-align: center;
      }}

      .k-disconnect {{
        color: {KENYAN_COLORS['text_faint']};
        font-size: 11px;
        text-decoration: none;
      }}

      .k-disconnect:hover {{
        color: {KENYAN_COLORS['red']};
      }}

      .k-sidebar-footer {{
        margin-top: 14px;
        padding-top: 10px;
        border-top: 0.5px solid {KENYAN_COLORS['border']};
      }}

      .k-page-title {{
        color: {KENYAN_COLORS['black']};
        font-size: 18px;
        font-weight: 500;
        margin: 0;
      }}

      .k-page-desc {{
        color: {KENYAN_COLORS['text_muted']};
        font-size: 13px;
        font-weight: 400;
        max-width: 560px;
        margin-top: 4px;
        margin-bottom: 16px;
      }}

      [data-testid="stExpander"] {{
        background: {KENYAN_COLORS['white']} !important;
        border: 0.5px solid {KENYAN_COLORS['border']} !important;
        border-radius: 8px !important;
        box-shadow: none !important;
      }}

      [data-testid="stExpander"]:hover {{
        border-color: {KENYAN_COLORS['red']} !important;
      }}

      [data-testid="stExpander"] summary {{
        color: {KENYAN_COLORS['black']} !important;
        font-size: 13px !important;
        font-weight: 500 !important;
      }}

      [data-testid="stFileUploaderDropzone"] {{
        background: {KENYAN_COLORS['white']} !important;
        border: 1.5px dashed {KENYAN_COLORS['border_strong']} !important;
        border-radius: 12px !important;
        padding: 16px !important;
      }}

      [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {KENYAN_COLORS['red']} !important;
      }}

      [data-testid="stFileUploaderDropzone"] small {{
        color: {KENYAN_COLORS['text_faint']} !important;
      }}

      [data-testid="stFileUploaderDropzone"] button {{
        border: 0.5px solid {KENYAN_COLORS['red']} !important;
        color: {KENYAN_COLORS['red']} !important;
        background: {KENYAN_COLORS['white']} !important;
        border-radius: 8px !important;
      }}

      [data-testid="stFileUploaderDropzone"] button:hover {{
        background: {KENYAN_COLORS['red']} !important;
        color: {KENYAN_COLORS['white']} !important;
      }}

      .stButton > button, .stDownloadButton > button {{
        border-radius: 8px !important;
        border: 0.5px solid {KENYAN_COLORS['border_strong']} !important;
        box-shadow: none !important;
      }}

      .stTabs {{
        display: none !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
  )


def _render_institution_sidebar(institution_name, parent_authority, sector_name):
  nav_keys = {item[2] for item in INSTITUTION_NAV_ITEMS}

  mode_q = str(st.session_state.get("portal_mode", "live")).lower()
  if mode_q not in {"live", "historical"}:
    mode_q = "live"
  st.session_state["portal_mode"] = mode_q

  nav_q = str(st.session_state.get("institution_nav", "data-intake")).lower()
  if nav_q not in nav_keys:
    nav_q = "data-intake"
  st.session_state["institution_nav"] = nav_q

  access_id = st.session_state.get("username", "Unknown")
  clearance_level = st.session_state.get("role", "UNKNOWN")
  mode_changed = False

  def _render_mode_controls():
    nonlocal mode_q, mode_changed
    st.markdown('<div class="k-block-label">View mode</div><div class="k-divider"></div>', unsafe_allow_html=True)
    mode_label = "Live" if mode_q == "live" else "Historical"
    if st.session_state.get("institution_mode_selector") != mode_label:
      st.session_state["institution_mode_selector"] = mode_label

    selected_mode = st.radio(
      "View mode",
      ["Live", "Historical"],
      key="institution_mode_selector",
      horizontal=True,
      label_visibility="collapsed",
    )
    if selected_mode.lower() != mode_q:
      mode_q = selected_mode.lower()
      mode_changed = True
    st.markdown('<div class="k-divider"></div>', unsafe_allow_html=True)

  fl_enabled = bool(st.session_state.get("fl_mode_enabled", False))
  hidden_keys = set() if fl_enabled else {"federated-learning"}

  sidebar_state = render_shared_sidebar(
    state_key="institution_nav",
    default_key="data-intake",
    nav_items=INSTITUTION_NAV_ITEMS,
    group_order=["analysis", "operations", "models"],
    group_labels={
      "analysis": ("Analysis", KENYAN_COLORS["red"]),
      "operations": ("Operations", KENYAN_COLORS["green"]),
      "models": ("Models & Privacy", KENYAN_COLORS["black"]),
    },
    profile_label=sector_name,
    profile_name=access_id,
    role_badge_text=clearance_level,
    role_badge_bg=KENYAN_COLORS["green_light"],
    role_badge_fg=KENYAN_COLORS["green"],
    role_badge_border=KENYAN_COLORS["green"],
    profile_bottom_border=KENYAN_COLORS["green"],
    button_key_prefix="institution_nav",
    hidden_keys=hidden_keys,
    pre_nav_renderer=_render_mode_controls,
    disconnect_button_key="institution_disconnect",
    disconnect_label="Disconnect session",
  )

  if sidebar_state["disconnect_clicked"]:
    logout_user()
    st.rerun()

  nav_q = str(sidebar_state["active_key"])
  if sidebar_state["changed"] or mode_changed:
    st.session_state["portal_mode"] = mode_q
    st.session_state["institution_nav"] = nav_q
    st.rerun()

  st.markdown(
    f"""
    <div class="k-topbar">
      <div class="k-topbar-left">
        <span class="k-dot" style="background:{KENYAN_COLORS['green']};"></span>
        <span class="k-dot" style="background:{KENYAN_COLORS['red']};"></span>
        <span class="k-dot" style="background:{KENYAN_COLORS['white']}; border:0.5px solid {KENYAN_COLORS['border_strong']};"></span>
      </div>
      <div class="k-topbar-center">
        <span class="value">{escape(institution_name)}</span>
        <span class="muted"> reporting to </span><span class="value">{escape(parent_authority)}</span>
      </div>
      <button class="k-deploy-btn" type="button">Deploy</button>
    </div>
    """,
    unsafe_allow_html=True,
  )


def _is_2fa_bypass_enabled() -> bool:
  """Temporary switch to bypass 2FA after phase-1 credential check.

  Default is OFF. Set SCACE4_SKIP_2FA=1/true/on only for explicit test sessions.
  """
  raw = os.getenv("SCACE4_SKIP_2FA")
  if raw is None:
    return False
  return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_signup_baskets() -> list[tuple[int, str]]:
  try:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("SELECT id, name FROM baskets ORDER BY name ASC")
      return [(int(r["id"]), str(r["name"])) for r in c.fetchall()]
  except Exception:
    return []


def _register_institution_signup(username: str, password: str, institution_name: str, basket_id: int) -> tuple[bool, str]:
  uname = str(username or "").strip()
  inst_name = str(institution_name or "").strip()
  if not uname:
    return False, "Username is required."
  if len(uname) < 4:
    return False, "Username must be at least 4 characters."
  if not password or len(str(password)) < 8:
    return False, "Password must be at least 8 characters."
  if not inst_name:
    return False, "Institution name is required."
  if not basket_id:
    return False, "Sector selection is required."

  try:
    with get_connection() as conn:
      c = conn.cursor()

      c.execute("SELECT id FROM users WHERE username = ?", (uname,))
      if c.fetchone() is not None:
        return False, "That username already exists."

      c.execute("SELECT id, basket_id FROM institutions WHERE name = ?", (inst_name,))
      existing_inst = c.fetchone()
      if existing_inst is None:
        api_key = f"signup_{secrets.token_hex(12)}"
        c.execute(
          "INSERT INTO institutions (name, basket_id, api_key) VALUES (?, ?, ?)",
          (inst_name, int(basket_id), api_key),
        )
        institution_id = int(c.lastrowid)
      else:
        institution_id = int(existing_inst["id"])
        existing_basket_id = int(existing_inst["basket_id"])
        if existing_basket_id != int(basket_id):
          return False, "Institution already exists under a different sector."

      c.execute(
        "INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?, ?, ?, ?, ?)",
        (uname, hash_password(password), Role.INSTITUTION.value, int(basket_id), institution_id),
      )
      conn.commit()
      return True, "Signup complete. You can now sign in."
  except Exception as exc:
    return False, f"Signup failed: {exc}"

def render_landing_page():
  inject_enterprise_theme(include_watermark=True)
  st.markdown("<h1 style='text-align: center; color: #1F2937; padding-top: 2rem;'>K-Scarcity</h1>", unsafe_allow_html=True)
  st.markdown("<h3 style='text-align: center; color: #475569; padding-bottom: 2rem;'>National Intelligence & Systemic Risk Gateway</h3>", unsafe_allow_html=True)
  _, center_col, _ = st.columns([1, 10, 1])
  
  with center_col:
    # 5 Ws + 1 H bundled into a single html block for faster rendering using CSS Grid
    st.markdown(
      """
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
      """,
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
      if 'phase1_passed' in st.session_state:
        del st.session_state['phase1_passed']
      st.rerun()
      
    st.write("")

    if not st.session_state.get('phase1_passed', False):
      auth_mode = st.radio(
        "Access",
        options=["Sign In", "Sign Up"],
        horizontal=True,
      )
    else:
      auth_mode = "Sign In"

    if auth_mode == "Sign Up" and not st.session_state.get('phase1_passed', False):
      baskets = _load_signup_baskets()
      if not baskets:
        st.error("No sectors available for signup. Please contact an administrator.")
        return

      basket_names = [name for _, name in baskets]
      basket_lookup = {name: bid for bid, name in baskets}

      with st.form("signup_form"):
        st.markdown("### Create Institution Account")
        institution_name = st.text_input("Institution Name")
        selected_sector = st.selectbox("Sector", basket_names)
        signup_username = st.text_input("Create Username")
        signup_password = st.text_input("Create Password", type="password")
        signup_password_confirm = st.text_input("Confirm Password", type="password")
        signup_submit = st.form_submit_button("Create Account", use_container_width=True)

        if signup_submit:
          if signup_password != signup_password_confirm:
            st.error("Passwords do not match.")
          else:
            ok, msg = _register_institution_signup(
              username=signup_username,
              password=signup_password,
              institution_name=institution_name,
              basket_id=int(basket_lookup[selected_sector]),
            )
            if ok:
              st.session_state["signup_prefill_username"] = str(signup_username or "").strip()
              st.success(msg)
              st.rerun()
            else:
              st.error(msg)
      return
    
    # ─── PHASE 1: Username & Password ──────────────────────
    if not st.session_state.get('phase1_passed', False):
      with st.form("login_form"):
        username = st.text_input("Username / Spoke ID", value=str(st.session_state.get("signup_prefill_username", "")))
        password = st.text_input("Passkey", type="password") # Re-enabled password
        submitted = st.form_submit_button("Authenticate", use_container_width=True)
        
        if submitted:
          # Actually check if user exists. (For demo still accepts empty password if backend allows)
          if login_user(username, password): 
            st.session_state.pop("signup_prefill_username", None)
            st.session_state['phase1_passed'] = True
            st.session_state["authenticated"] = True
            st.rerun()
          else:
            st.error("Invalid Credentials.")

    # ─── PHASE 2: Two-Factor Authentication (disabled) ─────
    else:
      st.info(f"Identity Verified: **{st.session_state.get('username')}**")
      st.caption("Two-factor authentication is currently disabled for this deployment.")

def route_authenticated_user():
  role = st.session_state.get('role')

  if role == Role.INSTITUTION.value:
    institution_id = st.session_state.get("institution_id")
    basket_id = st.session_state.get("basket_id")
    access_id = st.session_state.get("username", "Institution Node")

    institution_name = _resolve_institution_name(institution_id, access_id)
    sector_name = _resolve_basket_name(basket_id)
    parent_authority = sector_name

    _inject_kenyan_shell_css()
    _render_institution_sidebar(
      institution_name=institution_name,
      parent_authority=parent_authority,
      sector_name=sector_name,
    )

    if st.session_state.get("portal_mode", "live") == "historical":
      from kshiked.ui.institution.history_tab import render_history_tab
      from kshiked.ui.theme import LIGHT_THEME
      render_history_tab(LIGHT_THEME)
      return

    active_key = st.session_state.get("institution_nav", "data-intake")
    active_section = INSTITUTION_NAV_KEY_TO_SECTION.get(active_key, "Data Intake")
    from kshiked.ui.institution.local_dashboard import render as render_spoke
    render_spoke(active_section=active_section, use_enterprise_theme=False)
    return

  # Route based on explicit Role Based Access Controls
  if role == Role.EXECUTIVE.value:
    if is_developer_session():
      from kshiked.ui.institution.developer_dashboard import render as render_developer
      render_developer()
    else:
      from kshiked.ui.institution.executive_dashboard import render as render_executive
      render_executive()
  elif role == Role.BASKET_ADMIN.value:
    from kshiked.ui.institution.admin_governance import render as render_basket_admin
    render_basket_admin()
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

"""
Main Layout and Chat Interface for Policy Intelligence.

Styled to match the SENTINEL glassmorphism dark theme.
"""
from __future__ import annotations
import streamlit as st
import urllib.request

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .context import init_session_state, handle_user_input, process_bill_input
from .metrics import render_sidebar
from .live_policy import render_live_policy_impact


# ── Quick-suggestion chips shown under the welcome card ──────────────
_QUICK_SUGGESTIONS = [
    "Finance Bill 2026",
    "SHIF Phase 2",
    "Housing Levy Increase",
    "Digital Services Tax",
    "Fuel Subsidy Removal",
]


def _inject_chat_css(theme):
    """Inject CSS that skins Streamlit's chat widgets to match the dashboard."""
    st.markdown(f"""<style>
    /* ── Chat container ──────────────────────────────── */
    [data-testid="stChatMessageContent"] {{
        font-family: 'Space Mono', 'Courier New', monospace !important;
        font-size: 0.88rem !important;
        line-height: 1.7 !important;
        color: {theme.text_primary} !important;
    }}

    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
        background: {theme.bg_card} !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid {theme.border_default} !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.65rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
    }}

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: linear-gradient(135deg,
            rgba(0,255,136,0.06) 0%,
            rgba(31,51,29,0.45) 100%) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(0,255,136,0.18) !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.65rem !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
    }}

    /* Avatar icons */
    [data-testid="chatAvatarIcon-assistant"] {{
        background: linear-gradient(135deg, {theme.accent_primary}, {theme.accent_info}) !important;
        color: {theme.bg_primary} !important;
        border-radius: 10px !important;
    }}
    [data-testid="chatAvatarIcon-user"] {{
        background: linear-gradient(135deg, {theme.bg_secondary}, {theme.bg_tertiary}) !important;
        border: 1px solid rgba(0,255,136,0.3) !important;
        color: {theme.accent_primary} !important;
        border-radius: 10px !important;
    }}

    /* ── Chat input bar ──────────────────────────────── */
    [data-testid="stChatInput"] {{
        background: {theme.bg_card} !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid {theme.border_default} !important;
        border-radius: 14px !important;
        padding: 0.15rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {theme.accent_primary} !important;
        box-shadow: 0 0 0 2px rgba(0,255,136,0.15), 0 4px 20px rgba(0,0,0,0.3) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        font-family: 'Space Mono', monospace !important;
        color: {theme.text_primary} !important;
        font-size: 0.88rem !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {theme.text_muted} !important;
        opacity: 0.7 !important;
    }}
    /* Send button */
    [data-testid="stChatInput"] button {{
        color: {theme.accent_primary} !important;
        background: transparent !important;
        border: none !important;
        transition: transform 0.2s ease !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        transform: scale(1.15) !important;
    }}

    /* ── Welcome card ────────────────────────────────── */
    .policy-welcome {{
        background: {theme.bg_card};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {theme.border_default};
        border-radius: 16px;
        padding: 2rem 2.25rem;
        margin: 0.75rem 0 1.5rem;
        box-shadow: 0 6px 28px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }}
    .policy-welcome::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg,
            rgba(0,255,136,0.04) 0%,
            transparent 60%);
        pointer-events: none;
    }}
    .policy-welcome-title {{
        font-family: 'Space Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: {theme.accent_primary};
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }}
    .policy-welcome-desc {{
        font-family: 'Space Mono', monospace;
        color: {theme.text_secondary};
        font-size: 0.85rem;
        line-height: 1.8;
        margin-bottom: 0.25rem;
    }}
    .policy-welcome-hint {{
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: {theme.text_muted};
        margin-top: 1rem;
    }}

    /* ── Quick suggestion chips ──────────────────────── */
    .chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }}
    /* Override the home-page 320px card style for chip buttons */
    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] {{
        height: auto !important;
        min-height: 0 !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.76rem !important;
        letter-spacing: 0.3px !important;
        border-radius: 20px !important;
        background: {theme.bg_card} !important;
        border: 1px solid {theme.border_default} !important;
        color: {theme.text_secondary} !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:hover {{
        border-color: {theme.accent_primary} !important;
        color: {theme.accent_primary} !important;
        background: rgba(0,255,136,0.06) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 10px rgba(0,255,136,0.12) !important;
    }}
    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] p {{
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }}

    /* ── Status pill ─────────────────────────────────── */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: {theme.text_muted};
        letter-spacing: 0.5px;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        background: {theme.bg_card};
        border: 1px solid {theme.border_subtle};
        margin-bottom: 0.75rem;
    }}
    .status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }}
    .status-dot.connected {{
        background: {theme.accent_success};
        box-shadow: 0 0 6px {theme.accent_success};
        animation: pulse-status 2s ease infinite;
    }}
    .status-dot.offline {{
        background: {theme.accent_danger};
        box-shadow: 0 0 6px {theme.accent_danger};
    }}
    @keyframes pulse-status {{
        0%,100% {{ opacity: 1; }}
        50%     {{ opacity: 0.5; }}
    }}

    /* ── Section divider ─────────────────────────────── */
    .chat-divider {{
        border: none;
        border-top: 1px solid {theme.border_subtle};
        margin: 1rem 0;
    }}
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════

def render_policy_chat_interface(theme, sidebar_enabled=True, show_title=True, data=None):
    """Reusable policy chat interface component."""
    init_session_state()
    _inject_chat_css(theme)

    # Sidebar controls & Metrics
    if sidebar_enabled:
        render_sidebar(theme, data)

    # Main area header
    if show_title:
        st.markdown(
            '<div class="section-header">POLICY INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

    # Live policy trajectories
    render_live_policy_impact(theme)

    # Status indicator
    _render_status_pill()

    # Chat history (including styled welcome)
    _render_chat_history(theme)

    # Chat input
    user_input = st.chat_input("Ask about any Kenyan policy or bill …")
    if user_input:
        handle_user_input(user_input, theme)


# ═══════════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════════

def _render_status_pill():
    """Minimal connection pill that matches the glass theme."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        st.markdown(
            '<div class="status-pill">'
            '<span class="status-dot connected"></span>'
            'OLLAMA CONNECTED &mdash; qwen2.5:3b + nomic-embed-text'
            '</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<div class="status-pill">'
            '<span class="status-dot offline"></span>'
            'OLLAMA OFFLINE &mdash; run <code>ollama serve</code>'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Chat display
# ═══════════════════════════════════════════════════════════════════════════

def _render_chat_history(theme):
    """Render all chat messages with glassmorphism styling."""
    msgs = st.session_state.policy_chat_messages

    # First message is the welcome — render it as a styled card
    if msgs and msgs[0].get("role") == "assistant" and len(msgs) == 1:
        _render_welcome_card(theme)
        return

    # Render all messages
    for idx, msg in enumerate(msgs):
        role = msg["role"]
        content = msg["content"]

        # Skip the original welcome text (index 0) — we rendered it above
        if idx == 0 and role == "assistant" and not msg.get("metadata"):
            _render_welcome_card(theme)
            continue

        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)

            metadata = msg.get("metadata", {})
            if metadata.get("type") == "bill_analysis":
                _render_inline_visuals(metadata, theme)


def _render_welcome_card(theme):
    """Display the themed welcome card with quick-action chips."""
    st.markdown(f"""
    <div class="policy-welcome">
        <div class="policy-welcome-title">Policy Intelligence</div>
        <div class="policy-welcome-desc">
            Analyze any Kenyan policy &mdash; bills, gazette notices, levies, regulations,
            executive orders. Paste text, upload a PDF, drop a URL, or simply
            type a policy name below.
        </div>
        <div class="policy-welcome-hint">
            Powered by Ollama (qwen2.5:3b) + KShield synthetic data
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick suggestion chips rendered as small Streamlit buttons
    cols = st.columns(len(_QUICK_SUGGESTIONS))
    for col, label in zip(cols, _QUICK_SUGGESTIONS):
        with col:
            if st.button(label, key=f"impact_chip_{label}", use_container_width=True):
                process_bill_input(title=label, theme=theme)


# ═══════════════════════════════════════════════════════════════════════════
# Inline analysis visuals
# ═══════════════════════════════════════════════════════════════════════════

def _render_inline_visuals(metadata: dict, theme):
    """Render county risk chart and provision severity chart."""
    county_risks = metadata.get("county_risks", {})
    provisions_data = metadata.get("provisions_data", [])

    if county_risks and HAS_PLOTLY:
        sorted_counties = sorted(
            county_risks.items(), key=lambda x: x[1], reverse=True
        )[:15]
        if sorted_counties:
            counties, risks = zip(*sorted_counties)
            colors = [
                theme.accent_danger if r >= 0.7 else
                theme.accent_warning if r >= 0.4 else
                theme.accent_success
                for r in risks
            ]
            fig = go.Figure(go.Bar(
                x=list(risks),
                y=list(counties),
                orientation="h",
                marker_color=colors,
                text=[f"{r:.0%}" for r in risks],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(text="County Risk Assessment",
                           font=dict(size=13, color=theme.text_secondary)),
                xaxis_title="Risk Score",
                yaxis=dict(autorange="reversed"),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Mono, monospace",
                          color=theme.text_primary, size=11),
                xaxis=dict(range=[0, 1.1],
                           gridcolor=theme.border_subtle),
                yaxis_gridcolor=theme.border_subtle,
            )
            st.plotly_chart(fig, use_container_width=True)

    if provisions_data and HAS_PLOTLY:
        labels = [p["clause_id"] for p in provisions_data[:8]]
        severities = [p["severity"] for p in provisions_data[:8]]
        colors = [
            theme.accent_danger if s >= 0.8 else
            theme.accent_warning if s >= 0.5 else
            theme.accent_success
            for s in severities
        ]
        fig = go.Figure(go.Bar(
            x=labels,
            y=severities,
            marker_color=colors,
            text=[f"{s:.0%}" for s in severities],
            textposition="outside",
        ))
        fig.update_layout(
            title=dict(text="Provision Severity",
                       font=dict(size=13, color=theme.text_secondary)),
            yaxis_title="Severity",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono, monospace",
                      color=theme.text_primary, size=11),
            yaxis=dict(range=[0, 1.1],
                       gridcolor=theme.border_subtle),
            xaxis_gridcolor=theme.border_subtle,
        )
        st.plotly_chart(fig, use_container_width=True)

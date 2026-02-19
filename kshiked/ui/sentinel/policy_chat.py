"""
Policy Intelligence — SENTINEL Dashboard Chat Panel

Streamlit-based conversational interface for policy impact analysis.
Embedded as a view in the SENTINEL Command Center.

Styled to match the SENTINEL glassmorphism dark theme.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Optional

from ._shared import st, logger, HAS_PLOTLY, HAS_PANDAS

if HAS_PLOTLY:
    import plotly.graph_objects as go
    import plotly.express as px
if HAS_PANDAS:
    import pandas as pd


# ── Quick-suggestion chips ───────────────────────────────────────────────
_QUICK_SUGGESTIONS = [
    "Finance Bill 2026",
    "SHIF Phase 2",
    "Housing Levy Increase",
    "Digital Services Tax",
    "Fuel Subsidy Removal",
]


# ═══════════════════════════════════════════════════════════════════════════
# Async helper
# ═══════════════════════════════════════════════════════════════════════════

def _run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════
# Chat CSS — matches the dashboard glassmorphism theme
# ═══════════════════════════════════════════════════════════════════════════

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

    /* ── Quick suggestion chips ────────────────────── */
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
    .status-dot-sm {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }}
    .status-dot-sm.connected {{
        background: {theme.accent_success};
        box-shadow: 0 0 6px {theme.accent_success};
        animation: pulse-status 2s ease infinite;
    }}
    .status-dot-sm.offline {{
        background: {theme.accent_danger};
        box-shadow: 0 0 6px {theme.accent_danger};
    }}
    @keyframes pulse-status {{
        0%,100% {{ opacity: 1; }}
        50%     {{ opacity: 0.5; }}
    }}
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main Renderer
# ═══════════════════════════════════════════════════════════════════════════

def render_policy_chat(theme):
    """Render the standalone Policy Intelligence chat view."""
    _ensure_ollama_warm()
    render_policy_chat_interface(theme, sidebar_enabled=True, show_title=True)


def _ensure_ollama_warm():
    """Fire a background warm-up request once per session so the LLM is loaded."""
    if st.session_state.get("_ollama_warmed"):
        return
    st.session_state["_ollama_warmed"] = True
    try:
        import threading
        import urllib.request
        import json

        def _warm():
            try:
                payload = json.dumps({
                    "model": "qwen2.5:3b",
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"num_predict": 1},
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=60)
            except Exception:
                pass

        threading.Thread(target=_warm, daemon=True).start()
    except Exception:
        pass


def render_policy_chat_interface(theme, sidebar_enabled=True, show_title=True, data=None):
    """Reusable policy chat interface component."""
    _init_session_state()
    _inject_chat_css(theme)

    # Sidebar controls & Metrics
    if sidebar_enabled:
        _render_sidebar(theme, data)

    # Main area header
    if show_title:
        st.markdown(
            '<div class="section-header">POLICY INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

    # Status indicator
    _render_status_pill()

    # Chat history (including styled welcome)
    _render_chat_history(theme)

    # Chat input
    user_input = st.chat_input("Ask about any Kenyan policy or bill …")
    if user_input:
        _handle_user_input(user_input, theme)


# ═══════════════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════════════

def _init_session_state():
    """Initialize Streamlit session state for the policy chatbot."""
    if "policy_chat_messages" not in st.session_state:
        st.session_state.policy_chat_messages = [
            {
                "role": "assistant",
                "content": "_welcome_",  # placeholder — rendered as styled card
            }
        ]
    if "policy_session" not in st.session_state:
        st.session_state.policy_session = None
    if "policy_bill_text" not in st.session_state:
        st.session_state.policy_bill_text = ""
    if "policy_bill_title" not in st.session_state:
        st.session_state.policy_bill_title = ""


def _get_or_create_session():
    """Get or create the ChatSession (pure data — no async objects)."""
    if st.session_state.policy_session is None:
        try:
            from kshiked.pulse.llm.policy_chatbot import ChatSession
            st.session_state.policy_session = ChatSession()
        except Exception as e:
            logger.error(f"Failed to create ChatSession: {e}")
            st.error(f"Session initialization failed: {e}")
            return None
    return st.session_state.policy_session


# ═══════════════════════════════════════════════════════════════════════════
# Status pill
# ═══════════════════════════════════════════════════════════════════════════

def _render_status_pill():
    """Minimal connection pill that matches the glass theme."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        st.markdown(
            '<div class="status-pill">'
            '<span class="status-dot-sm connected"></span>'
            'OLLAMA CONNECTED &mdash; qwen2.5:3b + nomic-embed-text'
            '</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<div class="status-pill">'
            '<span class="status-dot-sm offline"></span>'
            'OLLAMA OFFLINE &mdash; run <code>ollama serve</code>'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar(theme, data=None):
    """Render the sidebar with metrics and context controls."""
    
    # 1. Impact Metrics (if data is provided) - Moved from old tabs
    if data:
        st.sidebar.markdown("### Live Impact Metrics")
        # st.sidebar.info(f"DEBUG: Data found. ESI: {len(data.esi_indicators) if data.esi_indicators else 0}, Prims: {len(data.primitives) if data.primitives else 0}")
        
        # Mini ESI Chart
        esi = data.esi_indicators
        if esi and HAS_PLOTLY:
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
                    _process_bill_input(text=bill_text, title=bill_title, theme=theme)

        elif input_mode == "Upload PDF":
            uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="policy_pdf_upload")
            bill_title = st.text_input("Title (opt)", key="policy_pdf_title")
            if st.button("Analyze PDF", key="policy_analyze_pdf"):
                if uploaded:
                    _process_bill_input(
                        pdf_bytes=uploaded.getvalue(),
                        title=bill_title or uploaded.name,
                        theme=theme,
                    )

        elif input_mode == "Enter URL":
            url = st.text_input("Bill URL", key="policy_url_input")
            bill_title = st.text_input("Title", key="policy_url_title")
            if st.button("Analyze URL", key="policy_analyze_url"):
                if url.strip():
                    _process_bill_input(url=url, title=bill_title, theme=theme)

    # Active bill info
    session = st.session_state.get("policy_session")
    if session and session.has_bill:
        bill = session.bill
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Active Analysis**: {bill.title}")
        if st.sidebar.button("Clear Context", key="policy_clear"):
            _clear_session()
            st.rerun()

    # Quick policies
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Show Quick Examples"):
        quick_policies = ["Finance Bill 2026", "SHIF Phase 2", "Housing Levy Increase"]
        for policy in quick_policies:
            if st.sidebar.button(f"Analyze {policy}", key=f"quick_{policy}"):
                _process_bill_input(title=policy, theme=theme)


# ═══════════════════════════════════════════════════════════════════════════
# Chat Display
# ═══════════════════════════════════════════════════════════════════════════

def _render_chat_history(theme):
    """Render all chat messages with glassmorphism styling."""
    msgs = st.session_state.policy_chat_messages

    # If only the initial welcome exists, show the styled card
    if msgs and msgs[0].get("role") == "assistant" and len(msgs) == 1:
        _render_welcome_card(theme)
        return

    for idx, msg in enumerate(msgs):
        role = msg["role"]
        content = msg["content"]

        # Replace the placeholder welcome with the styled card
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

    # Quick suggestion chips
    cols = st.columns(len(_QUICK_SUGGESTIONS))
    for col, label in zip(cols, _QUICK_SUGGESTIONS):
        with col:
            if st.button(label, key=f"sentinel_chip_{label}", use_container_width=True):
                _process_bill_input(title=label, theme=theme)


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


# ═══════════════════════════════════════════════════════════════════════════
# Input Processing
# ═══════════════════════════════════════════════════════════════════════════

async def _async_process_bill(session, text="", title="", pdf_bytes=None, url=""):
    """Run the full bill analysis pipeline inside a fresh async context."""
    from kshiked.pulse.llm.policy_chatbot import PolicyChatbot
    async with PolicyChatbot() as chatbot:
        return await chatbot.process_bill(
            session, text=text, title=title, pdf_bytes=pdf_bytes, url=url,
        )


async def _async_ask(session, question: str):
    """Handle a follow-up question inside a fresh async context."""
    from kshiked.pulse.llm.policy_chatbot import PolicyChatbot
    async with PolicyChatbot() as chatbot:
        return await chatbot.ask(session, question)


def _process_bill_input(
    text: str = "",
    title: str = "",
    pdf_bytes: Optional[bytes] = None,
    url: str = "",
    theme=None,
):
    """Process a bill input through the chatbot pipeline."""
    session = _get_or_create_session()
    if session is None:
        return

    # Add user message to chat
    if title and not text and not pdf_bytes and not url:
        user_msg = f"Analyze: **{title}**"
    elif url:
        user_msg = f"Analyze this policy: {url}"
    elif pdf_bytes:
        user_msg = f"Uploaded PDF: **{title or 'document'}**"
    elif text:
        preview = text[:150] + "..." if len(text) > 150 else text
        user_msg = f"Analyze this policy:\n\n> {preview}"
    else:
        user_msg = "Analyze policy"

    st.session_state.policy_chat_messages.append({
        "role": "user",
        "content": user_msg,
    })

    # Run analysis with progress
    progress = st.empty()
    with st.spinner("Analyzing policy..."):
        try:
            progress.info("Initializing LLM and extracting provisions...")
            response = _run_async(
                _async_process_bill(session, text=text, title=title, pdf_bytes=pdf_bytes, url=url)
            )
            progress.empty()
        except Exception as e:
            progress.empty()
            logger.error(f"Bill processing failed: {e}")
            traceback.print_exc()
            response = (
                f"**Analysis failed:** {e}\n\n"
                "Make sure Ollama is running (`ollama serve`) with `qwen2.5:3b` "
                "and `nomic-embed-text` models."
            )

    # Build serializable metadata for chart rendering
    metadata = {"type": "bill_analysis"}
    if session.has_bill:
        metadata["bill_title"] = session.bill.title
        metadata["provision_count"] = session.bill.provision_count
        # Store chart data as plain dicts (safe for session_state)
        metadata["provisions_data"] = [
            {"clause_id": p.clause_id, "severity": p.severity}
            for p in session.bill.top_provisions[:8]
        ]
    if session.has_prediction:
        metadata["has_prediction"] = True
        metadata["county_risks"] = dict(session.prediction.county_risks)

    st.session_state.policy_chat_messages.append({
        "role": "assistant",
        "content": response,
        "metadata": metadata,
    })

    st.rerun()


def _handle_user_input(user_input: str, theme):
    """Handle user chat input — route to bill analysis or follow-up."""
    session = _get_or_create_session()
    if session is None:
        return

    # Determine if this is a new bill query or a follow-up
    policy_keywords = [
        "bill", "levy", "tax", "act", "policy", "amendment", "shif", "nhif",
        "housing", "finance", "education", "fuel", "digital", "security",
        "gazette", "health", "sha", "analyze", "agriculture", "devolution",
        "university", "funding", "transport", "employment",
    ]
    is_policy_query = any(kw in user_input.lower() for kw in policy_keywords)

    # If no bill loaded, treat as bill title query
    if not session.has_bill:
        _process_bill_input(title=user_input, theme=theme)
        return

    # If bill loaded and it looks like a new policy, analyze it
    if is_policy_query and any(kw in user_input.lower() for kw in [
        "analyze", "bill", "act", "new", "what about",
    ]):
        _process_bill_input(title=user_input, theme=theme)
        return

    # Follow-up question about the current bill
    st.session_state.policy_chat_messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.spinner("Thinking..."):
        try:
            response = _run_async(_async_ask(session, user_input))
        except Exception as e:
            logger.error(f"Chat question failed: {e}")
            traceback.print_exc()
            response = f"**Error:** {e}"

    st.session_state.policy_chat_messages.append({
        "role": "assistant",
        "content": response,
    })

    st.rerun()


def _clear_session():
    """Clear the current policy chat session."""
    st.session_state.policy_chat_messages = []
    st.session_state.policy_session = None
    st.session_state.policy_bill_text = ""
    st.session_state.policy_bill_title = ""
    _init_session_state()

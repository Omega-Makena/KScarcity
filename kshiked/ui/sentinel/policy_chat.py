"""
Policy Intelligence — SENTINEL Dashboard Chat Panel

Streamlit-based conversational interface for policy impact analysis.
Embedded as a view in the SENTINEL Command Center.

Features:
- Chat interface with st.chat_input / st.chat_message
- Bill input via paste, PDF upload, URL, or title
- Real-time analysis with progress indicators
- Inline risk cards and county risk display
- Conversation memory across interactions
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ._shared import st, logger, HAS_PLOTLY, HAS_PANDAS

# Lazy-load plotly/pandas only when available
if HAS_PLOTLY:
    import plotly.graph_objects as go
    import plotly.express as px
if HAS_PANDAS:
    import pandas as pd


def render_policy_chat(theme):
    """Render the Policy Intelligence chat view."""
    _init_session_state()

    # Sidebar controls
    _render_sidebar(theme)

    # Main area
    st.markdown(
        '<div class="section-header">POLICY INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    # Chat history
    _render_chat_history(theme)

    # Chat input
    user_input = st.chat_input("Ask about any Kenyan policy or bill...")
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
                "content": (
                    "Welcome to **Policy Intelligence**. I can analyze any Kenyan "
                    "policy — bills, gazette notices, levies, regulations, executive orders.\n\n"
                    "**To get started:**\n"
                    "- Type a bill name (e.g., 'Finance Bill 2026', 'SHIF Amendment')\n"
                    "- Paste bill text in the sidebar\n"
                    "- Upload a PDF in the sidebar\n"
                    "- Enter a URL to a bill\n\n"
                    "I'll extract provisions, search historical data, and predict social impact."
                ),
            }
        ]
    if "policy_chatbot" not in st.session_state:
        st.session_state.policy_chatbot = None
    if "policy_session" not in st.session_state:
        st.session_state.policy_session = None
    if "policy_processing" not in st.session_state:
        st.session_state.policy_processing = False
    if "policy_bill_text" not in st.session_state:
        st.session_state.policy_bill_text = ""
    if "policy_bill_title" not in st.session_state:
        st.session_state.policy_bill_title = ""


def _get_or_create_chatbot():
    """Get or lazily create the PolicyChatbot instance."""
    if st.session_state.policy_chatbot is None:
        try:
            from kshiked.pulse.llm.policy_chatbot import PolicyChatbot, ChatSession
            st.session_state.policy_chatbot = PolicyChatbot()
            st.session_state.policy_session = ChatSession()
        except Exception as e:
            logger.error(f"Failed to create PolicyChatbot: {e}")
            st.error(f"Policy chatbot initialization failed: {e}")
            return None, None
    return st.session_state.policy_chatbot, st.session_state.policy_session


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar(theme):
    """Render the sidebar with bill input controls."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Bill Input")

    input_mode = st.sidebar.radio(
        "Input method",
        ["Type/Ask", "Paste Text", "Upload PDF", "Enter URL"],
        key="policy_input_mode",
        horizontal=True,
    )

    if input_mode == "Paste Text":
        bill_title = st.sidebar.text_input(
            "Bill title (optional)",
            key="policy_sidebar_title",
            placeholder="e.g., Finance Bill 2026",
        )
        bill_text = st.sidebar.text_area(
            "Paste bill text",
            key="policy_sidebar_text",
            height=200,
            placeholder="Paste the full bill text, gazette notice, or policy document here...",
        )
        if st.sidebar.button("🔍 Analyze Bill", key="policy_analyze_paste"):
            if bill_text.strip():
                st.session_state.policy_bill_text = bill_text
                st.session_state.policy_bill_title = bill_title
                _process_bill_input(
                    text=bill_text,
                    title=bill_title,
                    theme=theme,
                )

    elif input_mode == "Upload PDF":
        uploaded = st.sidebar.file_uploader(
            "Upload bill PDF",
            type=["pdf"],
            key="policy_pdf_upload",
        )
        bill_title = st.sidebar.text_input(
            "Bill title (optional)",
            key="policy_pdf_title",
            placeholder="e.g., Housing Levy Amendment",
        )
        if st.sidebar.button("🔍 Analyze PDF", key="policy_analyze_pdf"):
            if uploaded:
                _process_bill_input(
                    pdf_bytes=uploaded.getvalue(),
                    title=bill_title or uploaded.name,
                    theme=theme,
                )
            else:
                st.sidebar.warning("Please upload a PDF first.")

    elif input_mode == "Enter URL":
        url = st.sidebar.text_input(
            "Bill URL",
            key="policy_url_input",
            placeholder="https://parliament.go.ke/bills/...",
        )
        bill_title = st.sidebar.text_input(
            "Bill title (optional)",
            key="policy_url_title",
        )
        if st.sidebar.button("🔍 Analyze URL", key="policy_analyze_url"):
            if url.strip():
                _process_bill_input(url=url, title=bill_title, theme=theme)
            else:
                st.sidebar.warning("Please enter a URL.")

    # else: "Type/Ask" — just use the chat input at the bottom

    # Active bill info
    if st.session_state.policy_session and st.session_state.policy_session.has_bill:
        bill = st.session_state.policy_session.bill
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Active Bill")
        st.sidebar.markdown(f"**{bill.title}**")
        st.sidebar.markdown(f"Provisions: {bill.provision_count}")
        st.sidebar.markdown(f"Severity: {bill.total_severity:.0%}")
        st.sidebar.markdown(f"Sectors: {', '.join(bill.sectors)}")

        if st.sidebar.button("🗑️ Clear & Start New", key="policy_clear"):
            _clear_session()
            st.rerun()

    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Policies")
    quick_policies = [
        "Finance Bill 2026",
        "SHIF Phase 2",
        "Housing Levy Increase",
        "Fuel Levy Hike",
        "Digital Services Tax",
        "University Funding Model",
    ]
    for policy in quick_policies:
        if st.sidebar.button(f"📄 {policy}", key=f"quick_{policy}"):
            _process_bill_input(title=policy, theme=theme)


# ═══════════════════════════════════════════════════════════════════════════
# Chat Display
# ═══════════════════════════════════════════════════════════════════════════

def _render_chat_history(theme):
    """Render all chat messages."""
    for msg in st.session_state.policy_chat_messages:
        role = msg["role"]
        content = msg["content"]

        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)

            # Render inline visualizations if present
            metadata = msg.get("metadata", {})
            if metadata.get("type") == "bill_analysis" and metadata.get("has_prediction"):
                _render_inline_visuals(theme)


def _render_inline_visuals(theme):
    """Render county risk map and provision severity chart if data is available."""
    session = st.session_state.policy_session
    if not session or not session.prediction:
        return

    pred = session.prediction
    bill = session.bill

    # County risk chart
    if pred.county_risks and HAS_PLOTLY:
        sorted_counties = sorted(
            pred.county_risks.items(), key=lambda x: x[1], reverse=True
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
                title="County Risk Assessment",
                xaxis_title="Risk Score",
                yaxis=dict(autorange="reversed"),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme.text_primary, size=11),
                xaxis=dict(range=[0, 1.1]),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Provision severity chart
    if bill and bill.provisions and HAS_PLOTLY:
        provisions = bill.top_provisions[:8]
        labels = [f"{p.clause_id}" for p in provisions]
        severities = [p.severity for p in provisions]
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
            title="Provision Severity",
            yaxis_title="Severity",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_primary, size=11),
            yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Input Processing
# ═══════════════════════════════════════════════════════════════════════════

def _process_bill_input(
    text: str = "",
    title: str = "",
    pdf_bytes: Optional[bytes] = None,
    url: str = "",
    theme=None,
):
    """Process a bill input through the chatbot pipeline."""
    chatbot, session = _get_or_create_chatbot()
    if chatbot is None:
        return

    # Add user message to chat
    if title and not text and not pdf_bytes and not url:
        st.session_state.policy_chat_messages.append({
            "role": "user",
            "content": f"Analyze: **{title}**",
        })
    elif url:
        st.session_state.policy_chat_messages.append({
            "role": "user",
            "content": f"Analyze this policy: {url}",
        })
    elif pdf_bytes:
        st.session_state.policy_chat_messages.append({
            "role": "user",
            "content": f"📄 Uploaded PDF: **{title or 'document'}**",
        })
    elif text:
        preview = text[:150] + "..." if len(text) > 150 else text
        st.session_state.policy_chat_messages.append({
            "role": "user",
            "content": f"Analyze this policy:\n\n> {preview}",
        })

    # Run analysis
    with st.spinner("🔍 Analyzing policy... (extracting provisions, searching evidence, predicting impact)"):
        try:
            loop = asyncio.new_event_loop()
            response = loop.run_until_complete(
                chatbot.process_bill(
                    session,
                    text=text,
                    title=title,
                    pdf_bytes=pdf_bytes,
                    url=url,
                )
            )
            loop.close()
        except Exception as e:
            logger.error(f"Bill processing failed: {e}")
            import traceback
            traceback.print_exc()
            response = f"Analysis failed: {e}\n\nPlease check that Ollama is running (`ollama serve`)."

    # Add response
    metadata = {}
    if session.has_prediction:
        metadata = {
            "type": "bill_analysis",
            "bill_title": session.bill.title if session.bill else "",
            "provision_count": session.bill.provision_count if session.bill else 0,
            "has_prediction": True,
        }

    st.session_state.policy_chat_messages.append({
        "role": "assistant",
        "content": response,
        "metadata": metadata,
    })

    st.rerun()


def _handle_user_input(user_input: str, theme):
    """Handle user chat input."""
    chatbot, session = _get_or_create_chatbot()
    if chatbot is None:
        return

    # Check if it looks like a bill reference (no bill loaded yet)
    if not session.has_bill and any(kw in user_input.lower() for kw in [
        "bill", "levy", "tax", "act", "policy", "amendment", "shif", "nhif",
        "housing", "finance", "education", "fuel", "digital", "security",
        "gazette", "health", "sha", "analyze", "agriculture", "devolution",
    ]):
        _process_bill_input(title=user_input, theme=theme)
        return

    # If bill loaded, it's a follow-up question
    if not session.has_bill:
        # No bill, not a known keyword — try as title anyway
        _process_bill_input(title=user_input, theme=theme)
        return

    # Follow-up question
    st.session_state.policy_chat_messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.spinner("Thinking..."):
        try:
            loop = asyncio.new_event_loop()
            response = loop.run_until_complete(
                chatbot.ask(session, user_input)
            )
            loop.close()
        except Exception as e:
            logger.error(f"Chat question failed: {e}")
            response = f"Error: {e}"

    st.session_state.policy_chat_messages.append({
        "role": "assistant",
        "content": response,
    })

    st.rerun()


def _clear_session():
    """Clear the current policy chat session."""
    st.session_state.policy_chat_messages = []
    st.session_state.policy_chatbot = None
    st.session_state.policy_session = None
    st.session_state.policy_bill_text = ""
    st.session_state.policy_bill_title = ""
    _init_session_state()

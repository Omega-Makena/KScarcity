"""
Session Context and Input Handling for Policy Chat.
"""
from __future__ import annotations
import traceback
import logging
import streamlit as st
from .llm import run_async, async_process_bill, async_ask

logger = logging.getLogger(__name__)

def init_session_state():
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

def get_or_create_session():
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

def clear_session():
    """Clear the current policy chat session."""
    st.session_state.policy_chat_messages = []
    st.session_state.policy_session = None
    st.session_state.policy_bill_text = ""
    st.session_state.policy_bill_title = ""
    init_session_state()

def process_bill_input(
    text: str = "",
    title: str = "",
    pdf_bytes: bytes | None = None,
    url: str = "",
    theme=None,
):
    """Process a bill input through the chatbot pipeline."""
    session = get_or_create_session()
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
            if url:
                progress.info("Fetching content from URL...")
            elif pdf_bytes:
                progress.info("Extracting text from PDF...")
            else:
                progress.info("Extracting provisions via LLM...")
            response = run_async(
                async_process_bill(session, text=text, title=title, pdf_bytes=pdf_bytes, url=url)
            )
            progress.empty()
        except Exception as e:
            progress.empty()
            logger.error(f"Bill processing failed: {e}")
            traceback.print_exc()
            error_hint = ""
            if "timeout" in str(e).lower() or "connect" in str(e).lower():
                error_hint = (
                    "\n\nOllama may be loading the model for the first time "
                    "(this can take 30-60s). Please try again."
                )
            response = (
                f"**Analysis failed:** {e}{error_hint}\n\n"
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

def handle_user_input(user_input: str, theme):
    """Handle user chat input — route to bill analysis or follow-up."""
    session = get_or_create_session()
    if session is None:
        return

    # Check if the user pasted a URL
    stripped = user_input.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        process_bill_input(url=stripped, theme=theme)
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
        process_bill_input(title=user_input, theme=theme)
        return

    # If bill loaded and it looks like a new policy, analyze it
    if is_policy_query and any(kw in user_input.lower() for kw in [
        "analyze", "bill", "act", "new", "what about",
    ]):
        process_bill_input(title=user_input, theme=theme)
        return

    # Follow-up question about the current bill
    st.session_state.policy_chat_messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.spinner("Thinking..."):
        try:
            response = run_async(async_ask(session, user_input))
        except Exception as e:
            logger.error(f"Chat question failed: {e}")
            traceback.print_exc()
            response = f"**Error:** {e}"

    st.session_state.policy_chat_messages.append({
        "role": "assistant",
        "content": response,
    })

    st.rerun()

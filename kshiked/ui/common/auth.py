"""
Access Code Authentication Gate

Each module (K-SHIELD, K-PULSE, etc.) requires a unique access code.
Codes are stored in a config file and validated per-session.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

# Config file path — stores hashed access codes per module
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
ACCESS_CONFIG_FILE = CONFIG_DIR / "access_codes.json"


def _load_access_config() -> Dict:
    """Load access code configuration from file."""
    if ACCESS_CONFIG_FILE.exists():
        try:
            return json.loads(ACCESS_CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_access_config(config: Dict) -> None:
    """Save access code configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ACCESS_CONFIG_FILE.write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def _hash_code(code: str) -> str:
    """Hash an access code for secure storage."""
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def set_access_code(module_name: str, code: str) -> None:
    """Set the access code for a module (admin function)."""
    config = _load_access_config()
    config[module_name] = {
        "hash": _hash_code(code),
        "enabled": True,
    }
    _save_access_config(config)


def check_access(module_name: str, theme) -> bool:
    """
    Render the access code gate for a module.
    
    Returns True if user has valid access, False if blocked at gate.
    The gate is rendered as a styled card matching the dashboard aesthetic.
    """
    # Session state key for this module
    session_key = f"access_granted_{module_name}"
    
    # Already authenticated this session
    if st.session_state.get(session_key, False):
        return True
    
    # Load config
    config = _load_access_config()
    module_config = config.get(module_name, {})
    
    # If no code configured for this module, grant access (dev mode)
    if not module_config.get("hash"):
        st.session_state[session_key] = True
        return True
    
    # If module access is disabled
    if not module_config.get("enabled", True):
        st.error(f"Access to {module_name} is currently disabled.")
        _render_back_button()
        return False
    
    # Render the access gate
    _render_access_gate(module_name, module_config, theme, session_key)
    return False


def _render_back_button():
    """Render a back-to-home button."""
    if st.button("← Back to Home", key="auth_back_home"):
        st.session_state.current_view = "HOME"
        st.rerun()


def _render_access_gate(module_name: str, module_config: Dict, theme, session_key: str):
    """Render the access code input form."""
    
    st.markdown(f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
    ">
        <div style="
            background: linear-gradient(160deg, rgba(31, 51, 29, 0.6), rgba(20, 38, 22, 0.4));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 255, 136, 0.15);
            border-radius: 16px;
            padding: 3rem;
            max-width: 420px;
            width: 100%;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        ">
            <div style="
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(90deg, {theme.text_primary}, {theme.accent_primary});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: 3px;
                margin-bottom: 0.5rem;
            ">{module_name}</div>
            <div style="
                color: {theme.text_muted};
                font-size: 0.9rem;
                margin-bottom: 2rem;
                letter-spacing: 1px;
            ">RESTRICTED ACCESS — ENTER YOUR CODE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the input
    _, col_input, _ = st.columns([1, 2, 1])
    with col_input:
        code_input = st.text_input(
            "Access Code",
            type="password",
            key=f"access_code_input_{module_name}",
            label_visibility="collapsed",
            placeholder="Enter access code...",
        )
        
        col_submit, col_back = st.columns(2)
        with col_submit:
            if st.button("Authenticate", key=f"auth_submit_{module_name}", use_container_width=True):
                if code_input and _hash_code(code_input) == module_config["hash"]:
                    st.session_state[session_key] = True
                    st.rerun()
                else:
                    st.error("Invalid access code.")
        
        with col_back:
            if st.button("← Back", key=f"auth_back_{module_name}", use_container_width=True):
                st.session_state.current_view = "HOME"
                st.rerun()

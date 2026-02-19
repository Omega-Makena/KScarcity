"""
Navigation components — back buttons, breadcrumbs.
"""

from __future__ import annotations
import streamlit as st


def render_back_button(label: str = "← Back", target: str = "HOME", key_suffix: str = ""):
    """Render a styled back button that navigates to a target view."""
    key = f"nav_back_{target}_{key_suffix}" if key_suffix else f"nav_back_{target}"
    if st.button(label, key=key):
        st.session_state.current_view = target
        st.rerun()


def render_breadcrumb(path: list[tuple[str, str]], theme):
    """
    Render a breadcrumb navigation bar.
    
    Args:
        path: List of (label, target_view) tuples. Last item is current (no link).
        theme: ThemeColors instance
    """
    crumbs_html = []
    for i, (label, target) in enumerate(path):
        if i < len(path) - 1:
            crumbs_html.append(
                f'<span style="color: {theme.accent_primary}; cursor: pointer; '
                f'opacity: 0.7;">{label}</span>'
            )
        else:
            crumbs_html.append(
                f'<span style="color: {theme.text_primary}; font-weight: 600;">{label}</span>'
            )
    
    separator = f' <span style="color: {theme.text_muted}; margin: 0 0.5rem;">›</span> '
    
    st.markdown(f"""
    <div style="
        padding: 0.8rem 1.5rem;
        font-size: 0.85rem;
        letter-spacing: 1px;
    ">{separator.join(crumbs_html)}</div>
    """, unsafe_allow_html=True)
    
    # Render clickable buttons for non-current breadcrumb items
    # (hidden, triggered by the HTML spans above is not possible in Streamlit,
    #  so we use small cols with buttons)
    if len(path) > 1:
        cols = st.columns(len(path) + 2)
        for i, (label, target) in enumerate(path[:-1]):
            with cols[i]:
                if st.button(f"↩ {label}", key=f"breadcrumb_{target}_{i}"):
                    st.session_state.current_view = target
                    st.rerun()

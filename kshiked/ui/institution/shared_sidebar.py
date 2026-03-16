from __future__ import annotations

from html import escape
from typing import Callable, Iterable

import streamlit as st


SHARED_COLORS = {
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


def render_shared_sidebar(
  *,
  state_key: str,
  default_key: str,
  nav_items: Iterable[tuple[str, str, str, str, str | None]],
  group_order: Iterable[str],
  group_labels: dict[str, tuple[str, str]],
  profile_label: str,
  profile_name: str,
  role_badge_text: str,
  role_badge_bg: str,
  role_badge_fg: str,
  role_badge_border: str,
  profile_bottom_border: str,
  button_key_prefix: str,
  badge_counts: dict[str, int] | None = None,
  hidden_keys: set[str] | None = None,
  pre_nav_renderer: Callable[[], None] | None = None,
  disconnect_button_key: str = "shared_disconnect",
  disconnect_label: str = "Disconnect session",
) -> dict[str, object]:
  """Render the common sidebar shell and return routing state.

  Returns:
    {"active_key": str, "changed": bool, "disconnect_clicked": bool}
  """
  nav_keys = {item[2] for item in nav_items}
  active_key = str(st.session_state.get(state_key, default_key)).lower()
  if active_key not in nav_keys:
    active_key = default_key
  st.session_state[state_key] = active_key

  changed = False
  disconnect_clicked = False
  hidden_keys = hidden_keys or set()
  badge_counts = badge_counts or {}
  group_order = list(group_order)

  with st.sidebar:
    st.markdown(
      f"""
      <div style="background:{SHARED_COLORS['white']}; border:0.5px solid {SHARED_COLORS['border']}; border-bottom:3px solid {profile_bottom_border}; border-radius:12px; padding:12px;">
        <div style="color:{SHARED_COLORS['text_muted']}; font-size:10px; text-transform:uppercase; letter-spacing:0.8px; font-weight:500;">{escape(profile_label)}</div>
        <div style="color:{SHARED_COLORS['black']}; font-size:14px; font-weight:500; line-height:1.2; margin-top:4px; letter-spacing:0.1px;">{escape(profile_name)}</div>
        <div style="display:inline-flex; align-items:center; background:{role_badge_bg}; color:{role_badge_fg}; border:0.5px solid {role_badge_border}; border-radius:999px; padding:2px 8px; margin-top:8px; font-size:11px; font-weight:500;">{escape(role_badge_text)}</div>
      </div>
      <div style="height:0.5px; background:{SHARED_COLORS['border']}; margin:10px 0;"></div>
      """,
      unsafe_allow_html=True,
    )

    if pre_nav_renderer is not None:
      pre_nav_renderer()

    for group_key in group_order:
      group_label, group_color = group_labels[group_key]
      st.markdown(
        f'<div style="color:{SHARED_COLORS["text_muted"]}; font-size:10px; text-transform:uppercase; letter-spacing:0.9px; display:flex; align-items:center; gap:6px; font-weight:500; margin-bottom:6px;"><span style="width:5px; height:5px; border-radius:50%; background:{group_color}; display:inline-block;"></span>{escape(group_label)}</div>',
        unsafe_allow_html=True,
      )

      for item_group, item_label, item_key, item_color, item_tag in nav_items:
        if item_group != group_key:
          continue
        if item_key in hidden_keys:
          continue

        button_type = "primary" if item_key == active_key else "secondary"
        badge_count = int(badge_counts.get(item_key, 0) or 0)

        if badge_count > 0:
          dot_col, button_col, badge_col = st.columns([0.12, 0.70, 0.18], gap="small")
          with dot_col:
            st.markdown(
              f'<div style="width:5px; height:5px; border-radius:50%; background:{item_color}; margin-top:11px;"></div>',
              unsafe_allow_html=True,
            )
          with button_col:
            if st.button(item_label, key=f"{button_key_prefix}_{item_key}", use_container_width=True, type=button_type):
              if active_key != item_key:
                active_key = item_key
                changed = True
          with badge_col:
            st.markdown(
              f'<div style="background:{SHARED_COLORS["red_light"]}; color:{SHARED_COLORS["red"]}; border:0.5px solid {SHARED_COLORS["red"]}; border-radius:999px; font-size:10px; font-weight:500; text-align:center; padding:1px 6px; margin-top:8px;">{badge_count}</div>',
              unsafe_allow_html=True,
            )
        elif item_tag:
          dot_col, button_col, tag_col = st.columns([0.12, 0.70, 0.18], gap="small")
          with dot_col:
            st.markdown(
              f'<div style="width:5px; height:5px; border-radius:50%; background:{item_color}; margin-top:11px;"></div>',
              unsafe_allow_html=True,
            )
          with button_col:
            if st.button(item_label, key=f"{button_key_prefix}_{item_key}", use_container_width=True, type=button_type):
              if active_key != item_key:
                active_key = item_key
                changed = True
          with tag_col:
            st.markdown(
              f'<div style="background:{SHARED_COLORS["green_light"]}; color:{SHARED_COLORS["green"]}; border:0.5px solid {SHARED_COLORS["green"]}; border-radius:999px; font-size:10px; font-weight:500; text-align:center; padding:1px 6px; margin-top:8px;">{escape(str(item_tag))}</div>',
              unsafe_allow_html=True,
            )
        else:
          dot_col, button_col = st.columns([0.12, 0.88], gap="small")
          with dot_col:
            st.markdown(
              f'<div style="width:5px; height:5px; border-radius:50%; background:{item_color}; margin-top:11px;"></div>',
              unsafe_allow_html=True,
            )
          with button_col:
            if st.button(item_label, key=f"{button_key_prefix}_{item_key}", use_container_width=True, type=button_type):
              if active_key != item_key:
                active_key = item_key
                changed = True

      if group_key != group_order[-1]:
        st.markdown(
          f'<div style="height:0.5px; background:{SHARED_COLORS["border"]}; margin:10px 0;"></div>',
          unsafe_allow_html=True,
        )

    st.markdown(
      f'<div style="height:0.5px; background:{SHARED_COLORS["border"]}; margin:10px 0;"></div>',
      unsafe_allow_html=True,
    )
    if st.button(disconnect_label, key=disconnect_button_key, use_container_width=False):
      disconnect_clicked = True

  return {
    "active_key": active_key,
    "changed": changed,
    "disconnect_clicked": disconnect_clicked,
  }

"""
Reusable Landing Page Template

Renders a hero section with animated SVG background paths
and a 2×2 card grid. Used by HOME and each module's landing page.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import streamlit as st


def render_landing(
    theme,
    title: str,
    subtitle: str,
    tagline: str,
    cards: List[Tuple[str, str, str]],
    view_prefix: str = "",
    back_target: Optional[str] = None,
):
    """
    Render a landing page with hero + animated bg paths + 2×2 card grid.
    
    Args:
        theme: ThemeColors instance
        title: Hero title text
        subtitle: Hero description text
        tagline: Small tagline below subtitle
        cards: List of (title, description, target_view) tuples
        view_prefix: Prefix for session state view keys (e.g. "kshield_")
        back_target: If set, render a back button to this view
    """
    # CSS for hero + animation
    hero_css = f"""
    <style>
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes flowPath {{
        0% {{ stroke-dashoffset: 2400; opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ stroke-dashoffset: 0; opacity: 0; }}
    }}
    
    .landing-wrapper {{
        position: relative;
        width: 100%;
        min-height: 250px;
        overflow: hidden;
    }}
    
    .bg-paths-svg {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        opacity: 0.4;
        pointer-events: none;
        z-index: 0;
    }}
    
    .bg-path {{
        fill: none;
        stroke-dasharray: 2400;
        stroke-dashoffset: 2400;
        animation: flowPath 10s ease-in-out infinite;
    }}
    
    .hero-container {{
        position: relative;
        z-index: 1;
        text-align: center;
        padding: 1rem 2rem 0.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    
    .hero-title-wrapper {{
        overflow: hidden;
        display: inline-block;
    }}
    
    .hero-title {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, {theme.text_primary}, {theme.accent_primary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 6px;
        text-transform: uppercase;
        animation: fadeIn 1.5s ease-out forwards;
    }}
    
    .hero-desc {{
        font-size: 1.5rem;
        color: #E6EAF0;
        font-weight: 300;
        letter-spacing: 2px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0;
        animation: fadeIn 1.5s ease-out forwards;
        animation-delay: 1.2s;
    }}
    
    .hero-subtitle {{
        opacity: 0;
        font-size: 1.0rem;
        color: {theme.text_muted};
        font-weight: 400;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-top: 1rem;
        animation: fadeIn 1s ease-out forwards;
        animation-delay: 2.5s; 
    }}
    </style>
    """
    st.markdown(hero_css, unsafe_allow_html=True)
    
    # Hero with Background Paths
    st.markdown(f"""
    <div class="landing-wrapper">
        <svg class="bg-paths-svg" viewBox="0 0 1200 600" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="pg1" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="{theme.accent_primary}" stop-opacity="0"/>
                    <stop offset="50%" stop-color="{theme.accent_primary}" stop-opacity="0.6"/>
                    <stop offset="100%" stop-color="{theme.accent_primary}" stop-opacity="0"/>
                </linearGradient>
                <linearGradient id="pg2" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="{theme.accent_info}" stop-opacity="0"/>
                    <stop offset="50%" stop-color="{theme.accent_info}" stop-opacity="0.4"/>
                    <stop offset="100%" stop-color="{theme.accent_info}" stop-opacity="0"/>
                </linearGradient>
            </defs>
            <path class="bg-path" d="M-100 300Q200 100,500 250T900 200T1300 350" stroke="url(#pg1)" stroke-width="2.5" style="animation-delay:0s"/>
            <path class="bg-path" d="M-100 400Q300 200,600 350T1000 250T1400 400" stroke="url(#pg1)" stroke-width="2" style="animation-delay:1.5s"/>
            <path class="bg-path" d="M-100 150Q250 350,550 180T950 320T1400 150" stroke="url(#pg2)" stroke-width="2" style="animation-delay:3s"/>
            <path class="bg-path" d="M-100 500Q350 280,650 420T1050 300T1400 500" stroke="url(#pg1)" stroke-width="1.5" style="animation-delay:4.5s"/>
            <path class="bg-path" d="M-100 50Q200 250,500 100T900 280T1300 80" stroke="url(#pg2)" stroke-width="2" style="animation-delay:6s"/>
            <path class="bg-path" d="M-100 250Q400 50,700 280T1100 100T1400 300" stroke="url(#pg1)" stroke-width="1.5" style="animation-delay:2s"/>
        </svg>
        <div class="hero-container">
            <div class="hero-title-wrapper">
                <div class="hero-title">{title}</div>
            </div>
            <div class="hero-desc">{subtitle}</div>
            <div class="hero-subtitle">{tagline}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Back button (if applicable)
    if back_target:
        if st.button("← Back", key=f"landing_back_{view_prefix}"):
            st.session_state.current_view = back_target
            st.rerun()
    
    # Card Grid — 2×2
    def render_card(col, title, desc, target):
        with col:
            label = f"{title}\n\n{desc}"
            full_target = f"{view_prefix}{target}" if view_prefix else target
            if st.button(label, key=f"card_btn_{full_target}", use_container_width=True):
                st.session_state.current_view = full_target
                st.rerun()
    
    # Render in rows of 2
    for i in range(0, len(cards), 2):
        row_cards = cards[i:i+2]
        if len(row_cards) == 2:
            _, c1, c2, _ = st.columns([1, 3, 3, 1])
            render_card(c1, *row_cards[0])
            render_card(c2, *row_cards[1])
        else:
            _, c1, _ = st.columns([1, 3, 1])
            render_card(c1, *row_cards[0])
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {theme.text_muted}; font-size: 0.8rem;">
        SENTINEL v2.0 &bull; STRATEGIC COMMAND &amp; CONTROL
    </div>
    """, unsafe_allow_html=True)

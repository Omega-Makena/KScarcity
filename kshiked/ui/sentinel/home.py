"""Home / Landing page for SENTINEL."""

from ._shared import st


def render_home(theme):
    """Render the landing page with hero section and navigation cards."""

    # CSS for Hero and Card-Buttons
    hero_css = f"""
    <style>
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes gradient-shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes glow {{
        0% {{ text-shadow: 0 0 10px {theme.accent_primary}55; }}
        50% {{ text-shadow: 0 0 25px {theme.accent_primary}, 0 0 10px {theme.text_primary}; }}
        100% {{ text-shadow: 0 0 10px {theme.accent_primary}55; }}
    }}
    @keyframes typing {{
        from {{ width: 0 }}
        to {{ width: 100% }}
    }}
    @keyframes flowPath {{
        0% {{ stroke-dashoffset: 2400; opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ stroke-dashoffset: 0; opacity: 0; }}
    }}

    .home-wrapper {{
        position: relative;
        width: 100%;
        min-height: 300px;
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
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        margin: 0 auto;
        border-right: .15em solid {theme.accent_primary};
        animation:
            typing 2.5s steps(30, end),
            blink-caret .75s step-end infinite;
        width: 100%;
        max-width: 800px;
    }}

    @keyframes blink-caret {{
        from, to {{ border-color: transparent }}
        50% {{ border-color: {theme.accent_primary} }}
    }}

    .hero-title {{
        font-family: 'Courier New', monospace;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, {theme.text_primary}, {theme.accent_primary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 4px;
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
    # Shimmer + card polish CSS
    card_css = f"""
    <style>
    @keyframes shimmer-slide {{
        0% {{ background-position: -200% center; }}
        100% {{ background-position: 200% center; }}
    }}

    .shimmer-text {{
        background: linear-gradient(
            90deg,
            {theme.text_muted} 0%,
            {theme.accent_primary} 40%,
            {theme.text_primary} 50%,
            {theme.accent_primary} 60%,
            {theme.text_muted} 100%
        );
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer-slide 4s linear infinite;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
    }}

    .card-icon {{
        font-size: 2rem;
        margin-bottom: 0.25rem;
        opacity: 0.7;
    }}
    </style>
    """

    st.markdown(hero_css + card_css, unsafe_allow_html=True)

    # Hero with Background Paths
    st.markdown(f"""
    <div class="home-wrapper">
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
                <div class="hero-title">K-SCARCITY</div>
            </div>
            <div class="hero-desc">The Autonomous Economic Defense &amp; Simulation Platform</div>
            <div class="hero-subtitle">-- Powered by Scarcity --</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation Cards -- 2x2 Grid (now 3x2 or just appended)
    cards = [
        ("K-SHIELD", "Run large-scale economic simulations across sectors. Test policy scenarios, model shocks, and evaluate cascading risk using adaptive ABM agents.", "KSHIELD"),
        ("K-PULSE", "Continuously ingest and analyze live signals to detect anomalies. Monitor behavioral shifts and generate early warning intelligence in real-time.", "SIGNALS"),
        ("K-COLLAB", "Enable organizations to collaboratively train models and generate insights using federated learning and secure aggregation.", "FEDERATION"),
        ("K-EDUCATION", "Translate complex security intelligence into clear public knowledge through explainable analytics and accessible awareness dashboards.", "DOCS"),
        ("Institution Portal", "Securely upload weekly organizational data to participate in federated analysis and automatically trigger local online learning rounds.", "INSTITUTION"),
    ]

    def render_card(col, title, desc, target):
        with col:
            # Button with title + description inside the card
            if st.button(f"{title}\n\n{desc}", key=f"card_btn_{target}", use_container_width=True):
                st.session_state.current_view = target
                st.rerun()

    # Top Row  (spacer | card | card | spacer)
    _, c1_top, c2_top, _ = st.columns([1, 3, 3, 1])
    render_card(c1_top, *cards[0])
    render_card(c2_top, *cards[1])

    # Middle Row
    _, c1_bot, c2_bot, _ = st.columns([1, 3, 3, 1])
    render_card(c1_bot, *cards[2])
    render_card(c2_bot, *cards[3])

    # Bottom Row
    _, c1_inst, _, _ = st.columns([1, 3, 3, 1])
    render_card(c1_inst, *cards[4])

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {theme.text_muted}; font-size: 0.8rem;">
        SENTINEL v2.0 &bull; STRATEGIC COMMAND &amp; CONTROL
    </div>
    """, unsafe_allow_html=True)

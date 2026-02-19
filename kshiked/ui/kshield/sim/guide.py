"""Simulation Guide and Data Profile tab."""

from ._shared import st, pd, HAS_DATA_STACK


def render_simulation_guide(theme):
    st.markdown(
        '<div class="section-header">SIMULATION GUIDE &mdash; FULL TUTORIAL</div>',
        unsafe_allow_html=True,
    )
    nav_col, content_col = st.columns([1, 3])
    sections = [
        "1) What this card does",
        "2) Scenarios explained",
        "3) Policy templates",
        "4) How the SFC engine works",
        "5) Reading the results",
        "6) Sensitivity & heatmaps",
        "7) Comparing trajectories",
        "8) Connecting to Causal & Terrain",
        "9) Tips & common mistakes",
    ]
    with nav_col:
        st.markdown("**Guide Navigation**")
        section = st.radio("Jump to", sections, key="sim_guide_section",
                           label_visibility="collapsed")
    with content_col:
        if section == sections[0]:
            st.markdown("""
            This card runs **forward-looking economic simulations** using Kenya's
            calibrated Stock-Flow Consistent (SFC) model.

            You pick a shock scenario (e.g. oil crisis, drought), choose a policy
            response (CBK tightening, fiscal stimulus, etc.), select which outcome
            dimensions to watch, and run the simulation over 20-100 quarters.

            The engine is calibrated from **available economic data** or
            generic middle-income defaults if no external data is loaded.
            This ensures the simulation always runs, regardless of data availability.
            """)
        elif section == sections[1]:
            st.markdown("""
            **9 pre-built scenarios** cover Kenya's real risk landscape:

            | Category | Scenario | Key Shocks |
            |----------|----------|------------|
            | Supply | Oil Price Spike (+30%) | supply + FX |
            | Supply | Severe Drought (-20% Agri) | supply + demand |
            | Supply | Food Price Surge (+25%) | supply (ramped) |
            | External | Shilling Depreciation (-15%) | FX |
            | External | Global Recession | demand + FX |
            | External | Foreign Aid Cut (-30%) | fiscal |
            | Fiscal | Sovereign Debt Crisis | fiscal + FX |
            | Combined | Perfect Storm | supply + demand + FX |
            | Fiscal | Government Stimulus Boom | fiscal |

            Each includes a **context narrative** explaining real-world precedents.
            You can also build **custom scenarios** with your own shock magnitudes.
            """)
        elif section == sections[2]:
            st.markdown("""
            **8 policy templates** model real Government / CBK responses:

            - **Do Nothing** — let markets adjust
            - **CBK Tightening** — raise rates +2pp
            - **Aggressive Tightening** — major rate hike + CRR increase
            - **Fiscal Stimulus** — more spending + subsidies
            - **Austerity / IMF Package** — spending cuts + tax hikes
            - **Kenya 2016 Rate Cap** — interest rate cap at 11%
            - **Expansionary Mix** — lower rates + targeted subsidies
            - **Price Controls** — cap fuel + food prices

            Each template pre-fills the monetary and fiscal instrument sliders.
            You can override any slider after selecting a template.
            """)
        elif section == sections[3]:
            st.markdown("""
            The **SFC (Stock-Flow Consistent)** engine models 4 sectors:
            Households, Firms, Government, and Banking.

            Key equations:
            - **Phillips Curve** (New Keynesian): inflation responds to output gap
              with anchoring to prevent runaway spirals
            - **Taylor Rule**: interest rate responds to inflation and output gaps
            - **Fiscal block**: taxes, spending, subsidies, deficit, debt accumulation
            - **Household block**: consumption, savings, welfare
            - **Financial stability**: credit growth, leverage, banking health score

            All parameters are **calibrated from available data** (or generic
            middle-income defaults) using `kenya_calibration.py`.
            """)
        elif section == sections[4]:
            st.markdown("""
            After running, you'll see:

            1. **Impact delta cards** — final value + change for each watched dimension
            2. **Trajectory chart** — time-series of all selected dimensions
            3. **Shock onset marker** — vertical line showing when the shock hits

            Interpretation tips:
            - Green arrows (up for growth, down for inflation) = good outcomes
            - Red arrows = concerning movements
            - The shock onset marker helps you see lag effects
            """)
        elif section == sections[5]:
            st.markdown("""
            The **Sensitivity tab** shows a policy-outcome correlation heatmap:
            - Blue = policy instrument correlated with positive outcome
            - Red = correlated with negative outcome
            - Near zero = low sensitivity

            This helps identify which policy levers have the strongest effect
            on which outcomes, based on the simulation's trajectory data.
            """)
        elif section == sections[6]:
            st.markdown("""
            The **Compare tab** lets you run multiple scenarios back-to-back
            and overlay their trajectories on a single chart. This is useful for:

            - Comparing "do nothing" vs active policy response
            - Testing mild vs aggressive policy actions
            - Checking whether combined shocks are worse than sum of parts
            """)
        elif section == sections[7]:
            st.markdown("""
            All three K-SHIELD cards share the same data infrastructure:

            - **Causal** discovers relationships between indicators
            - **Terrain** maps the policy landscape and stability regions
            - **Simulation** runs forward scenarios with calibrated models

            When you load data in any card, it's shared via the
            "Shared K-SHIELD Dataset" option. If no external data is loaded,
            the simulation uses generic middle-income calibration defaults
            so you can always run scenarios.
            """)
        elif section == sections[8]:
            st.markdown("""
            **Tips:**
            - Start with a named scenario before building custom ones
            - Use "Do Nothing" policy first to see the raw shock effect
            - Then compare with an active policy to measure the difference
            - Watch at least 5 dimensions for a holistic view
            - 50 quarters (12.5 years) is usually enough to see full dynamics

            **Common mistakes:**
            - Running too few quarters (< 20) — dynamics haven't played out
            - Ignoring the shock onset — effects lag by 2-4 quarters
            - Comparing scenarios with different step counts
            - Not checking calibration confidence (shown in results header)
            """)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA PROFILE (matches terrain pattern)
# ═════════════════════════════════════════════════════════════════════════════

def render_data_profile(df: "pd.DataFrame", theme):
    n_rows, n_cols = df.shape
    coverage = f"{df.index.min()}" + " - " + f"{df.index.max()}" if len(df) > 0 else "N/A"
    completeness = f"{df.notna().mean().mean():.0%}"

    st.markdown(f"""
    <div style="display: flex; gap: 2rem; padding: 0.6rem 0; font-size: 0.78rem;
                color: {theme.text_muted}; flex-wrap: wrap;">
        <span>Rows: <b style="color:{theme.text_primary}">{n_rows}</b></span>
        <span>Columns: <b style="color:{theme.text_primary}">{n_cols}</b></span>
        <span>Coverage: <b style="color:{theme.text_primary}">{coverage}</b></span>
        <span>Completeness: <b style="color:{theme.text_primary}">{completeness}</b></span>
    </div>
    """, unsafe_allow_html=True)

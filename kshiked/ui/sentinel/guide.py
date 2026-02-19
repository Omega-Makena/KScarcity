"""System Guide tab - comprehensive in-app walkthrough."""

from ._shared import st


def render_system_guide_tab(theme):
    """Comprehensive in-app guide for all dashboard modules."""
    st.markdown('<div class="section-header">SYSTEM GUIDE &mdash; COMPLETE WALKTHROUGH</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.85rem;">'
        "This guide explains what each part of SENTINEL does, why it exists, "
        "what data it uses, and how to interpret outputs safely.</p>",
        unsafe_allow_html=True,
    )

    sections = [
        "1) System Purpose",
        "2) Data Pipeline",
        "3) Live Threat Map",
        "4) Executive Overview",
        "5) Signal Intelligence",
        "6) Causal Analysis",
        "7) Simulation",
        "8) Escalation Pathways",
        "9) Federation",
        "10) Operations",
        "11) Document Intelligence",
        "12) Reliability & Failure Modes",
        "13) Quick Glossary",
    ]
    selected = st.radio("Guide Sections", sections, key="system_guide_section")

    if selected == "1) System Purpose":
        st.markdown(
            """
            SENTINEL is a decision-support platform for economic risk and social instability monitoring.

            Core job:
            1. Observe incoming signals.
            2. Estimate risk and likely causal paths.
            3. Simulate policy scenarios.
            4. Surface operational alerts and decision windows.

            It is not a single prediction model. It is a multi-module system combining
            monitoring, causal inference, simulation, and coordination views.
            """
        )
    elif selected == "2) Data Pipeline":
        st.markdown(
            """
            Data flow is connector-based:

            1. `PulseConnector`: live signal ingestion and threat metrics.
            2. `ScarcityConnector`: hypothesis discovery and Granger-style causal candidates.
            3. `SimulationConnector`: scenario execution and trajectory outputs.
            4. `FederationConnector`: multi-agency status and contribution snapshots.
            5. `Document Intelligence`: news and local dossier content retrieval.

            Then `get_dashboard_data(...)` aggregates these into one `DashboardData` object,
            which all tabs consume.
            """
        )
    elif selected == "3) Live Threat Map":
        st.markdown(
            """
            What you see:
            - County-level risk markers.
            - Active signal counters.
            - Top-risk counties summary.

            Why it exists:
            - Gives immediate geographic situational awareness.
            - Answers: where is risk concentrating now?

            How to read:
            - Higher county risk score means greater current stress concentration.
            - Counter trends matter more than a single snapshot.
            """
        )
    elif selected == "4) Executive Overview":
        st.markdown(
            """
            What you see:
            - National threat status indicator.
            - Time-to-escalation gauge.
            - Top threat cards.
            - Competing hypothesis panel.

            Why it exists:
            - Fast decision brief for leadership.

            How to read:
            - Treat this as summary, then drill into Signals/Causal/Simulation tabs.
            - High-level status should be validated by underlying evidence panels.
            """
        )
    elif selected == "5) Signal Intelligence":
        st.markdown(
            """
            What you see:
            - Multi-signal intensity gauges.
            - Signal cascade map.
            - Co-occurrence heatmap.
            - Silence detector and risk timeline.

            Why it exists:
            - Detect early shifts in narrative and stress patterns.

            Key interpretation:
            - Co-occurrence shows which signals rise together.
            - Silence/going-dark may indicate migration to less visible channels.
            - Timeline helps separate temporary spikes from persistent trend changes.
            """
        )
    elif selected == "6) Causal Analysis":
        st.markdown(
            """
            Two layers are used:

            1. Granger layer:
            - Tests whether past X helps predict future Y.
            - Useful for directional early-warning relationships.
            - Not absolute proof of real-world cause.

            2. Structural estimation layer (Scarcity):
            - Runs effect types (estimands) like ATE/ATT/ATC/CATE/LATE/mediation.
            - Produces effect direction, magnitude, confidence intervals, and agreement diagnostics.

            Why estimands can fail:
            - Missing required assumptions (instrument for LATE, mediator for mediation).
            - Missing dependencies (for example econml for CATE/ITE).
            - Runtime worker issues in parallel mode.

            Fallback policy:
            - User policy: continue or fail-fast per estimand errors.
            - Runtime fallback: automatic retry in sequential mode if parallel workers fail.
            """
        )
    elif selected == "7) Simulation":
        st.markdown(
            """
            What you see:
            - Scenario builder (shocks and policy constraints).
            - Run output trajectory.
            - 3D/4D path visualization.
            - Policy sensitivity view.

            Why it exists:
            - Answer "what if we apply this shock/policy?" before real-world action.

            How to read:
            - Compare start vs end outcome deltas.
            - Look at trajectory shape, not just final point.
            - Use sensitivity matrix to see which policy levers move which outcomes most.
            """
        )
    elif selected == "8) Escalation Pathways":
        st.markdown(
            """
            What you see:
            - Decision-latency countdown.
            - Fragility/escalation placeholders or computed metrics when available.

            Why it exists:
            - Prioritize response timing under uncertainty.

            How to read:
            - Shorter time-to-decision means less room for delayed intervention.
            - Escalation logic is strongest when corroborated by Signals + Causal + Simulation.
            """
        )
    elif selected == "9) Federation":
        st.markdown(
            """
            What you see:
            - Agency participation and contribution snapshots.
            - Convergence/timeline panels when available.

            Why it exists:
            - Shows multi-agency coordination health.

            How to read:
            - Contribution imbalance can indicate coordination risk.
            - Active status is not enough; contribution quality and timeliness matter.
            """
        )
    elif selected == "10) Operations":
        st.markdown(
            """
            What you see:
            - County risk table.
            - Alerts feed.
            - Network analysis and economic satisfaction panels.

            Why it exists:
            - Converts analytics into actionable operational queue.

            How to read:
            - Prioritize counties by risk and trend.
            - Use alert severity + recency together.
            - Cross-check network and satisfaction metrics for intervention design.
            """
        )
    elif selected == "11) Document Intelligence":
        st.markdown(
            """
            What you see:
            - Live news stream by category.
            - Local dossier browser with extracted documents.

            Why it exists:
            - Provides evidence context and narrative validation for quantitative signals.

            How to read:
            - Use source credibility and recency.
            - Link narrative shifts to signal and causal changes.
            """
        )
    elif selected == "12) Reliability & Failure Modes":
        st.markdown(
            """
            Common failure patterns:
            - Missing optional libraries (causal/plot dependencies).
            - Sparse or misaligned time-series causing weak causal validity.
            - Parallel worker failures in constrained runtimes.
            - Demo fallback data when upstream connectors are unavailable.

            What the system does:
            - Surfaces explicit warnings and stage-level errors.
            - Skips invalid effect types with reasons.
            - Falls back to safer execution mode when possible.

            Analyst rule:
            - Do not trust one chart in isolation.
            - Require consistency across at least Signals + Causal + Simulation.
            """
        )
    elif selected == "13) Quick Glossary":
        st.markdown(
            """
            - `Signal`: measurable indicator extracted from incoming data.
            - `Hypothesis`: candidate relationship discovered by the engine.
            - `Granger causality`: predictive direction test using lagged values.
            - `Estimand`: the exact effect question being estimated.
            - `Heatmap`: color grid where color intensity encodes value strength.
            - `Confounder`: variable that influences both cause and outcome.
            - `Instrument`: proxy variable used for LATE identification.
            - `Mediator`: variable on the path between cause and outcome.
            - `CI (confidence interval)`: plausible effect range estimate.
            - `Fallback`: automatic safer mode used when preferred execution fails.
            """
        )

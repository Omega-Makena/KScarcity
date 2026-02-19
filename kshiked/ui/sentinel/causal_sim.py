"""Causal Analysis + Legacy Simulation tab (Scenario Platform v2.0)."""

from ._shared import (
    st, go, np, components, HAS_PLOTLY, DashboardData, THREAT_LEVELS,
)


def render_causal_tab(data: DashboardData, theme):
    """Causal analysis tab — uses World Bank Kenya data directly."""
    from kshield.causal import render_causal
    render_causal(theme)


def render_simulation_tab(data: DashboardData, theme):
    """
    Professional Scenario Platform Workspace.
    Layout: [Library (1)] [Builder (1)] [Run & Compare (2)]
    """
    st.markdown('<div class="section-header">SCENARIO PLATFORM (v2.0)</div>', unsafe_allow_html=True)

    if "active_scenario" not in st.session_state:
        st.session_state.active_scenario = {
            "name": "New Scenario",
            "shocks": [],
            "policies": [],
            "base_settings": {"steps": 50, "dt": 1.0},
        }

    from kshiked.ui.connector import SimulationConnector
    connector = SimulationConnector()
    connector.connect()

    col_lib, col_build, col_run = st.columns([1, 1, 2])

    # --- Pane 1: Scenario Library ---
    with col_lib:
        st.markdown("**Library**")
        st.markdown(f"""
        <div style="background-color: {theme.bg_secondary}; padding: 1rem; border-radius: 8px; border: 1px solid {theme.border_default}; height: 100%;">
        """, unsafe_allow_html=True)
        if st.button("+ New Scenario", use_container_width=True):
            st.session_state.active_scenario = {"name": "New Scenario", "shocks": [], "policies": []}
            st.session_state.active_scenario_id = None
            st.rerun()
        st.markdown("---")
        scenarios = connector.list_scenarios()
        for s in scenarios:
            if st.button(f"{s.get('name', 'Untitled')}", key=s.get('id'), use_container_width=True):
                loaded = connector.load_scenario(s.get('id'))
                if loaded:
                    st.session_state.active_scenario = loaded.to_dict()
                    st.session_state.active_scenario_id = loaded.id
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Pane 2: Scenario Builder ---
    with col_build:
        st.markdown("**Builder**")
        st.markdown(f"""
        <div style="background-color: {theme.bg_secondary}; padding: 1rem; border-radius: 8px; border: 1px solid {theme.border_default}; height: 100%;">
        """, unsafe_allow_html=True)
        scen = st.session_state.active_scenario
        new_name = st.text_input("Name", scen.get("name", "New Scenario"))
        scen["name"] = new_name

        st.markdown("##### Shocks")
        if st.button("Add Shock", key="add_shock"):
            scen.setdefault("shocks", []).append({
                "target": "demand_shock", "magnitude": 0.1,
                "start_time": 5, "duration": 5, "shape": "ramp",
            })
        shocks = scen.get("shocks", [])
        for i, shock in enumerate(shocks):
            with st.expander(f"Shock {i+1}: {shock.get('target')}", expanded=True):
                shock["target"] = st.selectbox("Target", ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"], key=f"s_t_{i}", index=0)
                shock["magnitude"] = st.number_input("Mag", value=shock.get("magnitude", 0.0), key=f"s_m_{i}")
                shock["start_time"] = st.number_input("Start (t)", value=shock.get("start_time", 5), key=f"s_st_{i}")
                shock["duration"] = st.number_input("Duration", value=shock.get("duration", 5), key=f"s_d_{i}")
                shock["shape"] = st.selectbox("Shape", ["step", "ramp", "pulse", "decay"], key=f"s_sh_{i}", index=1)
                if st.button("Remove", key=f"rem_s_{i}"):
                    shocks.pop(i)
                    st.rerun()

        st.markdown("##### Policies (Constraints)")
        if st.button("Add Constraint", key="add_pol"):
            scen.setdefault("policies", []).append({"name": "Rate Cap", "key": "interest_rate", "max_value": 0.10})
        policies = scen.get("policies", [])
        for i, pol in enumerate(policies):
            with st.expander(f"Pol {i+1}: {pol.get('name')}"):
                pol["name"] = st.text_input("Name", pol.get("name"), key=f"p_n_{i}")
                pol["key"] = st.selectbox("Metric", ["interest_rate", "gdp_growth", "inflation", "unemployment"], key=f"p_k_{i}")
                pol["max_value"] = st.number_input("Max Cap", value=pol.get("max_value", 0.1), key=f"p_mx_{i}")

        if st.button("Save Scenario", use_container_width=True):
            saved_id = connector.save_scenario(scen)
            if saved_id:
                st.session_state.active_scenario_id = saved_id
                st.success("Saved!")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Pane 3: Run & Result ---
    with col_run:
        st.markdown("**Simulation & Analysis**")
        if st.button("RUN SCENARIO", type="primary", use_container_width=True):
            with st.spinner("Compiling and Running..."):
                try:
                    from scarcity.simulation.scenario import Scenario
                    scen_obj = Scenario.from_dict(st.session_state.active_scenario)
                    result = connector.run_scenario_object(scen_obj)
                    st.session_state.sim_state = result
                except Exception as e:
                    st.error(f"Run Error: {e}")

        sim_state = st.session_state.get("sim_state")
        if sim_state:
            traj = sim_state.trajectory
            if traj:
                start = traj[0]["outcomes"]
                end = traj[-1]["outcomes"]
                c1, c2, c3 = st.columns(3)
                c1.metric("GDP Impact", f"{(end['gdp_growth']-start['gdp_growth'])*100:.2f}pp")
                c2.metric("Inflation Impact", f"{(end['inflation']-start['inflation'])*100:.2f}pp")
                c3.metric("Unemployment Impact", f"{(end['unemployment']-start['unemployment'])*100:.2f}pp")

            col_flux, col_cube = st.columns(2)
            with col_flux:
                st.markdown("##### 3D Economic Flux Engine")
                try:
                    from flux_viz import get_flux_graph_html as _flux_fn
                except ImportError:
                    _flux_fn = None
                if _flux_fn:
                    html = _flux_fn(traj, height=400)
                    components.html(html, height=400)
                else:
                    st.info("Flux Viz module not loaded")
            with col_cube:
                st.markdown("##### 4D State Cube")
                _render_4d_simulation(sim_state, theme, selected_shock_key="demand_shock")

        st.markdown("---")
        st.markdown('<div class="section-header">POLICY IMPACT SENSITIVITY</div>', unsafe_allow_html=True)
        _render_policy_sensitivity(data, theme)
        st.markdown("---")
        _render_economic_terrain(data, theme)


def _render_4d_simulation(simulation_state, theme, selected_shock_key="demand_shock"):
    if not HAS_PLOTLY or not simulation_state or not simulation_state.trajectory:
        st.info("No simulation data available. Run a scenario to visualize.")
        return
    trajectory = simulation_state.trajectory
    outcome_key = st.selectbox("Z-Axis Outcome", ["GDP Growth", "Inflation", "Unemployment"], index=0, key="viz_outcome_select")
    outcome_map = {"GDP Growth": "gdp_growth", "Inflation": "inflation", "Unemployment": "unemployment"}
    z_key = outcome_map[outcome_key]
    try:
        t_vals = [f["t"] for f in trajectory]
        x_vals = [f["shock_vector"].get(selected_shock_key, 0.0) for f in trajectory]
        y_vals = [f["policy_vector"].get("policy_rate", 0.0) * 100 for f in trajectory]
        z_vals = [f["outcomes"].get(z_key, 0.0) * 100 for f in trajectory]
    except KeyError as e:
        st.error(f"Data schema mismatch: Missing key {e}")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals, mode='lines+markers',
        marker=dict(size=6, color=t_vals, colorscale='Viridis',
                    colorbar=dict(title="Time (t)", thickness=10, x=0.9), symbol='circle'),
        line=dict(color=theme.accent_primary, width=5), name='Scenario Path',
        text=[f"t={t}<br>Shock={x:.2f}<br>Rate={y:.2f}%<br>{outcome_key}={z:.2f}%"
              for t, x, y, z in zip(t_vals, x_vals, y_vals, z_vals)], hoverinfo='text',
    ))
    fig.add_trace(go.Scatter3d(x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]], mode='markers',
                               marker=dict(size=10, color='white', symbol='diamond'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[x_vals[-1]], y=[y_vals[-1]], z=[z_vals[-1]], mode='markers',
                               marker=dict(size=10, color=theme.accent_danger, symbol='x'), name='End'))
    fig.update_layout(
        height=500, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"State Cube: {selected_shock_key} -> Policy -> {outcome_key}", font=dict(color=theme.text_muted)),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title=f'{selected_shock_key} (Force)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            yaxis=dict(title='Policy Rate (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            zaxis=dict(title=f'{outcome_key} (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
        ),
        legend=dict(x=0, y=1, font=dict(color=theme.text_muted), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_policy_sensitivity(data: DashboardData, theme):
    if not HAS_PLOTLY:
        return
    sim_state = st.session_state.get("sim_state") or data.simulation
    trajectory = getattr(sim_state, "trajectory", None)
    if not trajectory or len(trajectory) < 6:
        st.info("Policy sensitivity unavailable: run a scenario to generate trajectory data.")
        return
    policy_keys = sorted({k for frame in trajectory for k in frame.get("policy_vector", {}).keys()})
    outcome_keys = sorted({k for frame in trajectory for k in frame.get("outcomes", {}).keys()})
    if not policy_keys or not outcome_keys:
        st.info("Policy/outcome vectors missing from simulation frames.")
        return
    impacts = []
    for pol in policy_keys:
        p = np.array([float(f.get("policy_vector", {}).get(pol, 0.0)) for f in trajectory], dtype=float)
        row = []
        for out in outcome_keys:
            y = np.array([float(f.get("outcomes", {}).get(out, 0.0)) for f in trajectory], dtype=float)
            if len(p) < 3 or np.allclose(p.std(), 0.0) or np.allclose(y.std(), 0.0):
                row.append(0.0)
            else:
                row.append(float(np.corrcoef(p, y)[0, 1]))
        impacts.append(row)
    fig = go.Figure(go.Heatmap(
        z=impacts, x=[o.replace("_", " ").title() for o in outcome_keys],
        y=[p.replace("_", " ").title() for p in policy_keys],
        colorscale=[[0, theme.accent_danger], [0.5, '#ffffff'], [1, theme.accent_success]],
        zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in impacts], texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': theme.text_muted},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_economic_terrain(data: DashboardData, theme):
    if not HAS_PLOTLY or not np:
        return
    x = np.linspace(0, 20, 50)
    y = np.linspace(0, 20, 50)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * ((X - 5)**2 + (Y - 5)**2)
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis_r', opacity=0.8, showscale=False, name='Stability Landscape'))
    if data.simulation:
        curr_infl = getattr(data.simulation, 'inflation', 0.0)
        curr_unemp = getattr(data.simulation, 'unemployment', 0.0)
    else:
        return
    curr_z = 0.5 * ((curr_infl - 5)**2 + (curr_unemp - 5)**2)
    fig.add_trace(go.Scatter3d(
        x=[curr_infl], y=[curr_unemp], z=[curr_z + 20], mode='markers+text',
        marker=dict(size=8, color=theme.accent_danger, line=dict(width=2, color='white')),
        text=[f"CURRENT<br>Infl: {curr_infl:.1f}%<br>Unemp: {curr_unemp:.1f}%"],
        textposition="top center", name='Current State',
    ))
    fig.add_trace(go.Scatter3d(
        x=[curr_infl, curr_infl], y=[curr_unemp, curr_unemp], z=[curr_z, curr_z + 20],
        mode='lines', line=dict(width=2, color=theme.accent_danger, dash='dash'), showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(x=[5], y=[5], z=[0], mode='markers',
                               marker=dict(size=6, color=theme.accent_success, opacity=0.8), name='Equilibrium Target'))
    fig.update_layout(
        title=dict(text="Stability Phase Space (Valley = Optimal)", font=dict(color=theme.text_muted, size=12)),
        scene=dict(
            xaxis=dict(title='Inflation (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            yaxis=dict(title='Unemployment (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            zaxis=dict(title='Instability Potential', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=30, b=0), height=400, paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

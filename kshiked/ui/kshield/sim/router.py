"""K-SHIELD Simulation — main router and entry point."""

from ._shared import (
    st, pd, HAS_DATA_STACK,
    get_shared_dataset, set_shared_dataset,
    load_world_bank_data, validate_and_load_upload,
)
from .guide import render_simulation_guide
from .scenario_config import render_scenario_config
from .run import run_simulation, render_scenario_runner_tab
from .core_analysis import (
    render_sensitivity_tab, render_state_cube_tab,
    render_compare_tab, render_diagnostics_tab,
)
from .advanced import (
    render_phase_explorer_tab, render_irf_tab,
    render_flow_sankey_tab, render_monte_carlo_tab,
    render_stress_matrix_tab,
)
from .param_surface import render_parameter_surface_tab
from .research import (
    render_io_sectors_tab, render_inequality_tab,
    render_financial_tab, render_open_economy_tab,
    render_research_engine_tab,
)


def render_simulation(theme, data=None):
    """
    Render the full simulation card with grouped category navigation.
    Called from kshield/page.py router.
    """
    if not HAS_DATA_STACK:
        st.error("Required packages missing: pandas / numpy. Install them for simulation analytics.")
        return

    # Compact header row
    hdr_l, hdr_r = st.columns([3, 2])
    with hdr_l:
        st.markdown(
            '<div class="section-header" style="margin-bottom:0;">'
            'ECONOMIC SCENARIO ENGINE</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        source = st.selectbox(
            "Data",
            ["Generic (No Data Required)", "World Bank (Kenya)", "Upload CSV", "Shared K-SHIELD"],
            key="sim_data_source",
            label_visibility="collapsed",
        )

    # Load data
    df = pd.DataFrame()
    shared_df, shared_meta = get_shared_dataset()
    using_generic = False

    if source == "Generic (No Data Required)":
        using_generic = True
    elif source == "World Bank (Kenya)":
        df = load_world_bank_data()
        if df.empty:
            st.warning("World Bank CSV not found -- falling back to generic middle-income calibration.")
            using_generic = True
        set_shared_dataset(df, "World Bank (Kenya)", "SIMULATION")
    elif source == "Shared K-SHIELD":
        if shared_df is None or shared_df.empty:
            st.warning("No shared dataset. Load data in another K-SHIELD card first.")
            return
        df = shared_df
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="sim_upload")
        if uploaded:
            loaded_df, err = validate_and_load_upload(uploaded)
            if err:
                st.error(err)
                return
            df = loaded_df
            st.session_state["sim_uploaded_df"] = df
            set_shared_dataset(df, "Uploaded CSV", "SIMULATION")
        elif "sim_uploaded_df" in st.session_state:
            df = st.session_state["sim_uploaded_df"]
        if df.empty:
            return

    # Compact data status
    has_run = bool(st.session_state.get("sim_trajectory"))
    run_badge = (f"<span style='color:{theme.accent_success};'>SIM READY</span>"
                 if has_run else
                 f"<span style='color:{theme.text_muted}; opacity:0.5;'>NOT RUN</span>")
    if using_generic or df.empty:
        data_info = "Generic calibration (middle-income defaults)"
    else:
        n_rows, n_cols = df.shape
        coverage = f"{df.index.min()}-{df.index.max()}" if len(df) > 0 else "N/A"
        completeness = f"{df.notna().mean().mean():.0%}"
        data_info = f"{n_rows} rows | {n_cols} vars | {coverage} | {completeness}"
    st.markdown(
        f"<div style='font-size:0.72rem; color:{theme.text_muted}; "
        f"padding:0.2rem 0 0.5rem; display:flex; gap:1.5rem; flex-wrap:wrap;'>"
        f"<span>{data_info}</span>"
        f"<span>{run_badge}</span></div>",
        unsafe_allow_html=True,
    )

    # Lazy import simulation modules
    try:
        from kshiked.simulation.kenya_calibration import (
            calibrate_from_data, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS
        )
        from kshiked.simulation.scenario_templates import (
            SCENARIO_LIBRARY, POLICY_TEMPLATES, get_scenario_by_id,
            build_custom_scenario, SHOCK_REGISTRY, POLICY_INSTRUMENT_REGISTRY,
            SHOCK_SHAPES, merge_shock_vectors, merge_policy_instruments,
        )
        from scarcity.simulation.sfc import SFCEconomy, SFCConfig
    except ImportError as e:
        st.error(f"Simulation engine modules not available: {e}")
        return

    # Category navigation -- 4 groups
    category = st.radio(
        "nav",
        ["Setup & Run", "Core Analysis", "Advanced", "Research Modules"],
        horizontal=True,
        key="sim_category",
        label_visibility="collapsed",
    )

    st.markdown(f"<div style='border-top:1px solid {theme.border_default}; "
                f"margin:-0.5rem 0 0.8rem;'></div>", unsafe_allow_html=True)

    # CATEGORY 1: SETUP & RUN
    if category == "Setup & Run":
        scenario_cfg = render_scenario_config(
            theme, SCENARIO_LIBRARY, POLICY_TEMPLATES,
            get_scenario_by_id, build_custom_scenario,
            OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS,
            SHOCK_REGISTRY, POLICY_INSTRUMENT_REGISTRY, SHOCK_SHAPES,
        )
        run_clicked = run_simulation(
            theme, SFCEconomy, SFCConfig,
            calibrate_from_data, scenario_cfg,
            merge_shock_vectors, merge_policy_instruments,
        )

        setup_tabs = st.tabs(["Results", "Guide & Tutorial"])
        with setup_tabs[0]:
            render_scenario_runner_tab(theme, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS, run_clicked)
        with setup_tabs[1]:
            render_simulation_guide(theme)

    # CATEGORY 2: CORE ANALYSIS
    elif category == "Core Analysis":
        if not st.session_state.get("sim_trajectory"):
            st.info("Run a simulation in **Setup & Run** first.")
            return
        core_tabs = st.tabs([
            "Sensitivity", "3D Cube", "Compare",
            "Phase", "Impulse Response", "Flow Dynamics",
        ])
        with core_tabs[0]:
            render_sensitivity_tab(theme, OUTCOME_DIMENSIONS)
        with core_tabs[1]:
            render_state_cube_tab(theme, OUTCOME_DIMENSIONS)
        with core_tabs[2]:
            render_compare_tab(theme, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS)
        with core_tabs[3]:
            render_phase_explorer_tab(theme)
        with core_tabs[4]:
            render_irf_tab(theme)
        with core_tabs[5]:
            render_flow_sankey_tab(theme)

    # CATEGORY 3: ADVANCED
    elif category == "Advanced":
        if not st.session_state.get("sim_trajectory"):
            st.info("Run a simulation in **Setup & Run** first.")
            return
        adv_tabs = st.tabs([
            "Monte Carlo", "Stress Matrix",
            "Parameter Surface", "Diagnostics",
        ])
        with adv_tabs[0]:
            render_monte_carlo_tab(theme, SFCEconomy, SFCConfig,
                                    calibrate_from_data, build_custom_scenario)
        with adv_tabs[1]:
            render_stress_matrix_tab(theme, SCENARIO_LIBRARY,
                                      SFCEconomy, SFCConfig, calibrate_from_data)
        with adv_tabs[2]:
            render_parameter_surface_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data)
        with adv_tabs[3]:
            render_diagnostics_tab(theme)

    # CATEGORY 4: RESEARCH MODULES
    elif category == "Research Modules":
        res_tabs = st.tabs([
            "Research Engine", "IO Sectors",
            "Inequality", "Financial", "Open Economy",
        ])
        with res_tabs[0]:
            render_research_engine_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data)
        with res_tabs[1]:
            render_io_sectors_tab(theme)
        with res_tabs[2]:
            render_inequality_tab(theme)
        with res_tabs[3]:
            render_financial_tab(theme)
        with res_tabs[4]:
            render_open_economy_tab(theme)

"""Federation / Federated Databases tab."""

from __future__ import annotations

from ._shared import DashboardData, HAS_PANDAS, HAS_PLOTLY, logger, pd, px, st


@st.cache_resource
def _get_federation_manager():
    from federated_databases import get_scarcity_federation

    return get_scarcity_federation()


def _build_route_url() -> tuple[str, str]:
    route_path = "/?view=FEDERATION"
    host = st.get_option("server.address") or "localhost"
    port = st.get_option("server.port") or 8501
    full_url = f"http://{host}:{port}{route_path}"
    return route_path, full_url


def render_federation_tab(data: DashboardData, theme):
    manager = _get_federation_manager()

    route_path, full_url = _build_route_url()
    if not st.session_state.get("_fed_route_logged"):
        logger.info("Federation / Federated Databases route: %s", full_url)
        st.session_state["_fed_route_logged"] = True

    st.markdown('<div class="section-header">FEDERATION / FEDERATED DATABASES</div>', unsafe_allow_html=True)

    st.caption("Stable route path")
    st.code(route_path, language="text")
    st.caption("Direct URL")
    st.code(full_url, language="text")

    left, right = st.columns([2, 2])

    with left:
        st.markdown("#### Node Registration")
        node_id = st.text_input("Node ID", value="org_a", key="fed_node_id")
        county_filter = st.text_input("County Filter (optional)", value="", key="fed_county_filter")
        backend = st.selectbox("Backend", ["sqlite"], key="fed_backend")

        if st.button("Register Node", key="fed_register_node", type="primary"):
            try:
                node = manager.register_node(node_id=node_id, backend=backend, county_filter=county_filter or None)
                st.success(f"Node registered: {node.node_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Registration failed: {exc}")

    with right:
        st.markdown("#### Sync Controls")
        lr = st.slider("Learning Rate", min_value=0.01, max_value=0.50, value=0.12, step=0.01, key="fed_lr")
        lookback = st.selectbox("Live Window", [24, 48, 72, 168], index=0, key="fed_lookback")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Trigger Sync Round", key="fed_sync_round"):
                try:
                    result = manager.run_sync_round(learning_rate=float(lr), lookback_hours=int(lookback))
                    st.success(
                        f"Round {result.round_number} complete: participants={result.participants}, "
                        f"samples={result.total_samples}, loss={result.global_loss:.4f}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Sync failed: {exc}")
        with c2:
            if st.button("Run Smoke Round", key="fed_smoke"):
                try:
                    smoke = manager.run_smoke_round()
                    st.success(
                        "Smoke run passed: "
                        f"single_loss={smoke['single_node']['loss']:.4f}, "
                        f"sync_round={smoke['sync_round']['round_number']}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Smoke run failed: {exc}")

    status = manager.get_status()
    latest_round = status.get("latest_round") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes", f"{status.get('node_count', 0)}")
    m2.metric("Rounds", f"{status.get('round_count', 0)}")
    m3.metric("Last Round", str(latest_round.get("round_number", "-")))
    loss_val = latest_round.get("global_loss")
    m4.metric("Global Loss", f"{float(loss_val):.4f}" if loss_val is not None else "-")

    st.markdown("---")

    nodes = status.get("nodes", [])
    if nodes:
        st.markdown("#### Node Storage + ML Status")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
        else:
            st.write(nodes)
    else:
        st.info("No nodes registered yet.")

    round_history = manager.get_round_history(limit=30)
    if round_history:
        st.markdown("#### Sync Metrics")
        if HAS_PLOTLY and HAS_PANDAS:
            hist_df = pd.DataFrame(round_history).sort_values("round_number")
            fig = px.line(
                hist_df,
                x="round_number",
                y=["global_loss", "global_gradient_norm"],
                markers=True,
                color_discrete_sequence=[theme.accent_warning, theme.accent_primary],
            )
            fig.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme.text_primary),
                legend_title_text="Metric",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif HAS_PANDAS:
            hist_df = pd.DataFrame(round_history).sort_values("round_number")
            st.line_chart(hist_df.set_index("round_number")[["global_loss", "global_gradient_norm"]])

    st.markdown("#### Exchange / Audit Log")
    audit_rows = manager.get_exchange_log(limit=80)
    if audit_rows:
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        else:
            st.write(audit_rows)
    else:
        st.info("No audit events yet. Trigger a sync round to generate exchange records.")

    # Legacy agency cards still shown from existing connector data.
    agencies = getattr(data, "agencies", [])
    if agencies:
        st.markdown("---")
        st.markdown("#### Legacy Agency Feed")
        cols = st.columns(min(5, len(agencies)))
        for idx, agency in enumerate(agencies[:5]):
            with cols[idx]:
                status_color = theme.accent_success if agency.status == "active" else theme.accent_warning
                st.markdown(
                    f"""
                    <div class="agency-card {'active' if agency.status == 'active' else ''}">
                        <div class="name">{agency.name}</div>
                        <div class="status" style="color: {status_color};">{agency.status.upper()}</div>
                        <div style="margin-top:0.5rem; font-size:0.8rem;">Contribution: {agency.contribution_score:.0%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.caption(f"Audit path: {status.get('audit_path')}")
    st.caption(f"Control DB: {status.get('control_db')}")

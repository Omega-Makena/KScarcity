"""Federation / Federated Data Access tab with K-Collab nested federation controls."""

from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from ._shared import DashboardData, HAS_PANDAS, HAS_PLOTLY, go, logger, np, pd, px, st

from federated_databases.policy.engine import AccessContext
from k_collab.topology.schema import topology_preview, validate_topology, TopologyValidationError
from k_collab.ui.services import get_kcollab_services
from k_collab.projects.registry import CollaborationProject


@st.cache_resource
def _get_federation_manager():
    from federated_databases import get_scarcity_federation

    return get_scarcity_federation()


@st.cache_resource
def _get_kcollab() -> Dict[str, Any]:
    return get_kcollab_services()


def _build_route_url() -> tuple[str, str]:
    route_path = "/?view=FEDERATION"
    host = st.get_option("server.address") or "localhost"
    port = st.get_option("server.port") or 8501
    full_url = f"http://{host}:{port}{route_path}"
    return route_path, full_url


def _default_topology_payload() -> Dict[str, Any]:
    return {
        "name": "security_nested_federation",
        "nodes": [
            {
                "node_id": "agency_a",
                "level": 1,
                "node_type": "agency",
                "parent_id": None,
                "agency_id": "agency_a",
                "domains": ["intel", "cyber"],
                "clearance": "SECRET",
            },
            {
                "node_id": "agency_b",
                "level": 1,
                "node_type": "agency",
                "parent_id": None,
                "agency_id": "agency_b",
                "domains": ["finance", "immigration"],
                "clearance": "SECRET",
            },
            {
                "node_id": "a_ops",
                "level": 2,
                "node_type": "department",
                "parent_id": "agency_a",
                "agency_id": "agency_a",
                "domains": ["intel"],
                "clearance": "RESTRICTED",
            },
            {
                "node_id": "b_ops",
                "level": 2,
                "node_type": "department",
                "parent_id": "agency_b",
                "agency_id": "agency_b",
                "domains": ["finance"],
                "clearance": "RESTRICTED",
            },
            {
                "node_id": "a_ops_nairobi",
                "level": 3,
                "node_type": "site",
                "parent_id": "a_ops",
                "agency_id": "agency_a",
                "domains": ["intel"],
                "clearance": "RESTRICTED",
            },
        ],
        "trust_edges": [
            {"source": "agency_a", "target": "agency_b", "channel": "cross_agency_aggregate"},
            {"source": "a_ops", "target": "b_ops", "channel": "need_to_know"},
        ],
    }


def _ensure_topology_seed(services: Dict[str, Any]) -> None:
    if services["topology"].latest() is None:
        services["topology"].save(_default_topology_payload(), actor="system", message="seed_default_topology")


def _render_topology_builder(services: Dict[str, Any]) -> None:
    topology_store = services["topology"]
    _ensure_topology_seed(services)

    latest = topology_store.get_payload()
    default_text = json.dumps(latest, indent=2)

    st.markdown("#### Pane 1: Federation Topology Builder")
    st.caption("Versioned topology config with schema validation and trust-boundary diffing.")

    top_col, right_col = st.columns([3, 2])
    with top_col:
        raw = st.text_area(
            "Topology JSON",
            value=default_text,
            height=360,
            key="kcollab_topology_json",
        )
        c1, c2 = st.columns(2)
        with c1:
            actor = st.text_input("Actor", value="analyst_a", key="kcollab_topology_actor")
        with c2:
            message = st.text_input("Version Message", value="topology update", key="kcollab_topology_msg")

        if st.button("Validate + Save Topology", key="kcollab_save_topology", type="primary"):
            try:
                payload = json.loads(raw)
                rec = topology_store.save(payload, actor=actor, message=message)
                st.success(f"Saved topology version: {rec['version_id']}")
            except Exception as exc:
                st.error(f"Topology save failed: {exc}")

    with right_col:
        st.markdown("##### Graph Preview")
        try:
            preview_payload = json.loads(raw)
            st.code(topology_preview(preview_payload), language="text")
        except Exception:
            st.info("Preview unavailable until valid JSON is provided.")

    versions = topology_store.list_versions(limit=20)
    if versions:
        st.markdown("##### Version History")
        if HAS_PANDAS:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "version_id": v["version_id"],
                            "created_at": v["created_at"],
                            "actor": v["actor"],
                            "message": v.get("message", ""),
                        }
                        for v in versions
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        if len(versions) >= 2:
            ids = [v["version_id"] for v in versions]
            d1, d2 = st.columns(2)
            with d1:
                old_id = st.selectbox("Old Version", ids, index=min(1, len(ids) - 1), key="kcollab_topology_old")
            with d2:
                new_id = st.selectbox("New Version", ids, index=0, key="kcollab_topology_new")
            if st.button("Diff Versions", key="kcollab_topology_diff"):
                try:
                    diff = topology_store.diff(old_id, new_id)
                    st.json(diff)
                except Exception as exc:
                    st.error(f"Diff failed: {exc}")


def _render_federated_databases_pane(services: Dict[str, Any]) -> None:
    fed_db = services["fed_db"]
    projects = services["projects"]

    st.markdown("#### Pane 2: Federated Data Access (Virtualization Layer)")
    st.caption("Federated query execution: planner -> policy -> contract -> local pushdown -> suppression -> audit")
    st.info("K-Collab orchestrates access and computation above existing systems; it does not replace source storage.")

    st.markdown("##### Collaboration Project")
    project_rows = projects.all()
    project_ids = sorted(project_rows.keys()) if project_rows else []
    if not project_ids:
        st.info("No project found. Sync connectors to seed default project.")
    p1, p2 = st.columns([2, 3])
    with p1:
        selected_project = st.selectbox(
            "Project",
            options=project_ids or ["default_monitoring"],
            key="kcollab_project_select",
        )
    with p2:
        if st.button("Create/Update Minimal Project", key="kcollab_project_upsert"):
            try:
                connectors = fed_db.catalog.connectors()
                participants = sorted({c.get("node_id", "") for c in connectors if c.get("node_id")})
                fed_db.upsert_project(
                    CollaborationProject(
                        project_id=selected_project if project_ids else "default_monitoring",
                        name="Monitoring Collaboration",
                        objective="Cross-org aggregate intelligence analytics",
                        participants=participants,
                        allowed_datasets=["local_samples"],
                        allowed_domains=["intel", "finance", "security"],
                        allowed_computations=["analytics", "federated_ml"],
                        governance={"k_threshold_min": 3, "purpose_allowlist": ["monitoring", "research", "casework"]},
                    ),
                    actor="dashboard",
                )
                st.success("Project saved")
            except Exception as exc:
                st.error(f"Project save failed: {exc}")

    if st.button("Sync Node Connectors from Federation Manager", key="kcollab_sync_connectors"):
        try:
            synced = fed_db.register_default_from_manager(actor="dashboard")
            st.success(f"Synced connectors for {synced['nodes']} nodes")
        except Exception as exc:
            st.error(f"Connector sync failed: {exc}")

    connector_rows = fed_db.catalog.connectors()
    if connector_rows:
        st.markdown("##### Connectors Registry")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(connector_rows), use_container_width=True, hide_index=True)
        else:
            st.write(connector_rows)

    st.markdown("##### Data Contracts")
    contracts = fed_db.contracts.all()
    if contracts:
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(list(contracts.values())), use_container_width=True, hide_index=True)
        else:
            st.json(contracts)
    else:
        st.info("No contracts registered. Sync connectors to seed defaults.")

    st.markdown("##### Policy Composer")
    policy_payload = fed_db.policy.latest()
    with st.expander("Current Policy JSON", expanded=False):
        st.json(policy_payload)

    st.markdown("##### Compatibility Analysis + Basket Formation")
    cc1, cc2, cc3 = st.columns([2, 2, 2])
    with cc1:
        dataset_options = sorted(list(fed_db.contracts.all().keys())) or ["local_samples"]
        selected_dataset = st.selectbox("Dataset", options=dataset_options, key="kcollab_compat_dataset")
    with cc2:
        selected_operation = st.selectbox("Operation", options=["aggregate", "time_bucket"], key="kcollab_compat_operation")
    with cc3:
        if st.button("Run Compatibility", key="kcollab_run_compatibility"):
            try:
                report = fed_db.run_compatibility_analysis(
                    dataset_id=selected_dataset,
                    operation=selected_operation,
                    project_id=selected_project,
                    actor="dashboard",
                )
                st.session_state["kcollab_compatibility_result"] = report
            except Exception as exc:
                st.error(f"Compatibility analysis failed: {exc}")

    compatibility_result = st.session_state.get("kcollab_compatibility_result")
    if compatibility_result:
        st.markdown("###### Compatibility Map")
        node_scores = compatibility_result.get("node_scores", [])
        if HAS_PANDAS and node_scores:
            st.dataframe(pd.DataFrame(node_scores), use_container_width=True, hide_index=True)
        else:
            st.json(compatibility_result)

        st.markdown("###### Baskets")
        baskets = compatibility_result.get("baskets", [])
        if HAS_PANDAS and baskets:
            st.dataframe(pd.DataFrame(baskets), use_container_width=True, hide_index=True)
        else:
            st.json(baskets)

    st.markdown("##### Query Studio")
    q_col, c_col = st.columns([3, 2])
    with q_col:
        query_text = st.text_area(
            "Safe Query (JSON DSL or SQL subset)",
            value="SELECT county, COUNT(*) FROM local_samples GROUP BY county",
            height=140,
            key="kcollab_query_text",
        )
    with c_col:
        user_id = st.text_input("User", value="analyst_1", key="kcollab_ctx_user")
        role = st.selectbox("Role", ["analyst", "supervisor", "auditor"], key="kcollab_ctx_role")
        clearance = st.selectbox("Clearance", ["PUBLIC", "INTERNAL", "RESTRICTED", "SECRET"], index=2, key="kcollab_ctx_clear")
        purpose = st.selectbox("Purpose", ["monitoring", "research", "casework", "oversight", "audit"], index=0, key="kcollab_ctx_purpose")
        k_threshold = st.slider("k-threshold", min_value=2, max_value=15, value=3, key="kcollab_k_threshold")

    if st.button("Run Federated Query", key="kcollab_run_query", type="primary"):
        try:
            context = AccessContext(user_id=user_id, role=role, clearance=clearance, purpose=purpose)
            result = fed_db.run_query(
                query_text=query_text,
                context=context,
                actor=user_id,
                k_threshold=int(k_threshold),
                project_id=selected_project,
            )
            st.session_state["kcollab_query_result"] = result
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    query_result = st.session_state.get("kcollab_query_result")
    if query_result:
        st.markdown("###### Validation + Plan")
        st.json({"allowed": query_result.get("allowed"), "plan": query_result.get("plan"), "reasons": query_result.get("reasons", [])})

        if query_result.get("allowed"):
            rows = query_result.get("rows", [])
            st.markdown("###### Results")
            if HAS_PANDAS:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.write(rows)
            if query_result.get("suppressed"):
                st.warning(f"Suppressed groups: {len(query_result['suppressed'])}")
            if query_result.get("execution_trace"):
                st.markdown("###### Plan Viewer (Pushdown Trace)")
                st.json(query_result.get("execution_trace"))
            if query_result.get("baskets"):
                st.markdown("###### Basket Execution")
                st.json(query_result.get("baskets"))
            if query_result.get("non_iid_diagnostics"):
                st.markdown("###### Non-IID Diagnostics")
                st.json(query_result.get("non_iid_diagnostics"))
            if query_result.get("data_quality"):
                st.markdown("###### Data Quality")
                st.json(query_result.get("data_quality"))
            if query_result.get("provenance"):
                st.markdown("###### Provenance + Coverage")
                st.json(query_result.get("provenance"))
            if query_result.get("compatibility"):
                st.markdown("###### Compatibility Report Used")
                st.json(
                    {
                        "version_id": query_result["compatibility"].get("version_id"),
                        "baskets": query_result["compatibility"].get("baskets", []),
                        "excluded_nodes": query_result["compatibility"].get("excluded_nodes", []),
                    }
                )

    st.markdown("##### Audit / Chain-of-Custody")
    audit_rows = fed_db.audit_rows(limit=80)
    if audit_rows:
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        else:
            st.write(audit_rows)


def _render_federated_ml_pane(services: Dict[str, Any]) -> None:
    fed_ml = services["fed_ml"]
    topology = services["topology"].get_payload()
    projects = services["projects"]

    st.markdown("#### Pane 3: Federated ML (Nested Orchestration)")

    participants: List[str] = [n["node_id"] for n in topology.get("nodes", []) if int(n.get("level", 0) or 0) >= 2]

    j1, j2, j3 = st.columns([2, 2, 2])
    with j1:
        task_name = st.text_input("Task/Model", value="threat_criticality", key="kcollab_ml_task")
    with j2:
        vector_dim = st.slider("Vector Dim", min_value=4, max_value=128, value=8, key="kcollab_ml_dim")
    with j3:
        actor = st.text_input("Actor", value="ml_supervisor", key="kcollab_ml_actor")

    selected_nodes = st.multiselect(
        "Participating nodes (departments/sites)",
        options=participants,
        default=participants[: min(4, len(participants))],
        key="kcollab_ml_nodes",
    )
    project_ids = sorted(projects.all().keys()) or ["default_monitoring"]
    selected_project = st.selectbox("Project", options=project_ids, key="kcollab_ml_project")
    min_eps = st.slider("Min Remaining Epsilon", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="kcollab_ml_min_eps")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start Job", key="kcollab_ml_start", type="primary"):
            try:
                result = fed_ml.start_job(
                    actor=actor,
                    task_name=task_name,
                    selected_nodes=selected_nodes,
                    vector_dim=int(vector_dim),
                    project_id=selected_project,
                    min_remaining_epsilon=float(min_eps),
                )
                st.success(f"ML job started: {result['job_id']}")
            except Exception as exc:
                st.error(f"Start job failed: {exc}")

    with c2:
        if st.button("Run Round", key="kcollab_ml_round"):
            try:
                status = fed_ml.status()
                active = status.get("active_job")
                if not active:
                    raise RuntimeError("No active job")
                if np is None:
                    raise RuntimeError("NumPy is required for synthetic round updates")
                rng = np.random.default_rng()
                updates = {p["client_id"]: rng.normal(size=int(vector_dim)).tolist() for p in active["participants"]}
                output = fed_ml.run_round(actor=actor, updates=updates)
                st.session_state["kcollab_ml_round"] = output
                st.success("Round completed")
            except Exception as exc:
                st.error(f"Run round failed: {exc}")

    with c3:
        if st.button("Complete Job", key="kcollab_ml_complete"):
            try:
                done = fed_ml.complete_job(actor=actor)
                st.success(f"Model registered: {done['model_id']}")
            except Exception as exc:
                st.error(f"Complete job failed: {exc}")

    round_output = st.session_state.get("kcollab_ml_round")
    if round_output:
        st.markdown("##### Run Console")
        st.json(round_output)

    status = fed_ml.status()
    st.markdown("##### Model Registry")
    models = status.get("models", [])
    if models:
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(models), use_container_width=True, hide_index=True)
        else:
            st.write(models)


def _render_walkthrough_pane(services: Dict[str, Any]) -> None:
    fed_db = services["fed_db"]
    fed_ml = services["fed_ml"]

    st.markdown("#### Guided Coordination Walkthrough")
    st.caption("Runnable flow: registration -> publishing -> project -> compatibility -> analytics -> FL readiness")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        actor = st.text_input("Actor", value="coordinator_1", key="kcollab_walk_actor")
    with c2:
        user_id = st.text_input("User", value="analyst_1", key="kcollab_walk_user")
    with c3:
        project_id = st.text_input("Project ID", value="default_monitoring", key="kcollab_walk_project")

    d1, d2, d3 = st.columns([2, 2, 2])
    with d1:
        role = st.selectbox("Role", ["analyst", "supervisor", "auditor"], key="kcollab_walk_role")
    with d2:
        clearance = st.selectbox("Clearance", ["PUBLIC", "INTERNAL", "RESTRICTED", "SECRET"], index=2, key="kcollab_walk_clearance")
    with d3:
        purpose = st.selectbox("Purpose", ["monitoring", "research", "casework", "audit"], key="kcollab_walk_purpose")

    query_text = st.text_area(
        "Walkthrough Query",
        value="SELECT county, COUNT(*) FROM local_samples GROUP BY county",
        height=100,
        key="kcollab_walk_query",
    )
    k_threshold = st.slider("k-threshold", min_value=2, max_value=15, value=3, key="kcollab_walk_k")
    min_eps = st.slider("Min Remaining Epsilon (FL readiness)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="kcollab_walk_eps")

    if st.button("Run 6-Step Walkthrough", type="primary", key="kcollab_walk_run"):
        try:
            walkthrough = fed_db.run_guided_walkthrough(
                actor=actor,
                context=AccessContext(user_id=user_id, role=role, clearance=clearance, purpose=purpose),
                project_id=project_id,
                query_text=query_text,
                k_threshold=int(k_threshold),
            )
            fl_ready = fed_ml.readiness_from_baskets(
                baskets=walkthrough["steps"]["6_fl_ready_input"]["baskets"],
                project_id=project_id,
                min_remaining_epsilon=float(min_eps),
            )
            fl_hard = fed_ml.assess_hard_problems(
                readiness=fl_ready,
                round_output=st.session_state.get("kcollab_ml_round"),
            )
            gates = dict(walkthrough.get("quality_gates", {}))
            gates["fl_ready_state_verified"] = bool(fl_ready.get("ready"))
            gates["db_hard_problems_split_visible"] = bool(walkthrough["steps"].get("7_db_hard_problems"))
            gates["fl_hard_problems_split_visible"] = bool(fl_hard)
            gates["walkthrough_complete_without_manual_fixes"] = bool(gates.get("walkthrough_complete_without_manual_fixes")) and bool(
                fl_ready.get("ready")
            )
            walkthrough["steps"]["6_fl_ready_state"] = fl_ready
            walkthrough["steps"]["8_fl_hard_problems"] = fl_hard
            walkthrough["quality_gates"] = gates
            st.session_state["kcollab_walkthrough_result"] = walkthrough
        except Exception as exc:
            st.error(f"Walkthrough failed: {exc}")

    result = st.session_state.get("kcollab_walkthrough_result")
    if result:
        st.markdown("##### Quality Gates")
        st.json(result.get("quality_gates", {}))
        if result.get("steps", {}).get("7_db_hard_problems"):
            st.markdown("##### DB-Side Hard Problems")
            st.json(result["steps"]["7_db_hard_problems"])
        if result.get("steps", {}).get("8_fl_hard_problems"):
            st.markdown("##### FL-Side Hard Problems")
            st.json(result["steps"]["8_fl_hard_problems"])
        st.markdown("##### Step Outputs")
        st.json(result.get("steps", {}))


def _render_legacy_federation_controls(data: DashboardData, theme) -> None:
    manager = _get_federation_manager()

    st.markdown("#### Legacy Federation Controls")

    left, right = st.columns([2, 2])

    with left:
        st.markdown("##### Node Registration")
        node_id = st.text_input("Node ID", value="org_a", key="fed_node_id")
        county_filter = st.text_input("County Filter (optional)", value="", key="fed_county_filter")
        backend = st.selectbox("Backend", ["sqlite"], key="fed_backend")

        if st.button("Register Node", key="fed_register_node"):
            try:
                node = manager.register_node(node_id=node_id, backend=backend, county_filter=county_filter or None)
                st.success(f"Node registered: {node.node_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Registration failed: {exc}")

    with right:
        st.markdown("##### Sync Controls")
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

    nodes = status.get("nodes", [])
    if nodes:
        st.markdown("##### Node Storage + ML Status")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
        else:
            st.write(nodes)

    round_history = manager.get_round_history(limit=30)
    if round_history:
        st.markdown("##### Sync Metrics")
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

    st.markdown("##### Exchange / Audit Log")
    audit_rows = manager.get_exchange_log(limit=80)
    if audit_rows:
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        else:
            st.write(audit_rows)

    agencies = getattr(data, "agencies", [])
    if agencies:
        st.markdown("##### Legacy Agency Feed")
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


def _init_wizard_state(services: Dict[str, Any]) -> None:
    if "kcollab_wizard_step" not in st.session_state:
        st.session_state.kcollab_wizard_step = 0
    if "kcollab_topology_draft" not in st.session_state:
        _ensure_topology_seed(services)
        st.session_state.kcollab_topology_draft = services["topology"].get_payload()
    st.session_state.setdefault("kcollab_last_health", [])
    st.session_state.setdefault("kcollab_last_compat", None)
    st.session_state.setdefault("kcollab_last_query", None)
    st.session_state.setdefault("kcollab_last_fl_ready", None)


def _wizard_nav(steps: List[str]) -> int:
    current = int(st.session_state.get("kcollab_wizard_step", 0))
    st.progress((current + 1) / len(steps), text=f"Step {current + 1} of {len(steps)} — {steps[current]}")
    selected = st.selectbox("Workflow Step", options=steps, index=current, key="kcollab_wizard_step_select")
    st.session_state.kcollab_wizard_step = steps.index(selected)

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("Back", disabled=st.session_state.kcollab_wizard_step == 0, key="kcollab_wizard_back"):
            st.session_state.kcollab_wizard_step = max(0, st.session_state.kcollab_wizard_step - 1)
            st.rerun()
    with c2:
        if st.button("Next", disabled=st.session_state.kcollab_wizard_step >= len(steps) - 1, key="kcollab_wizard_next"):
            st.session_state.kcollab_wizard_step = min(len(steps) - 1, st.session_state.kcollab_wizard_step + 1)
            st.rerun()

    return int(st.session_state.kcollab_wizard_step)


def _render_step_topology(services: Dict[str, Any]) -> None:
    st.markdown("### Step 1 — Topology (no raw JSON required)")
    st.caption("Define agencies, departments, and sites. The graph preview updates as you build.")

    draft = dict(st.session_state.get("kcollab_topology_draft") or _default_topology_payload())
    draft.setdefault("nodes", [])
    draft.setdefault("trust_edges", [])

    nodes = list(draft.get("nodes", []))
    node_ids = [n.get("node_id") for n in nodes if n.get("node_id")]

    left, right = st.columns([3, 2])
    with left:
        with st.form("kcollab_add_node_form"):
            node_id = st.text_input("Node ID", value="")
            level = st.selectbox("Level", [1, 2, 3], index=0)
            node_type = st.selectbox("Node Type", ["agency", "department", "site"], index=0)
            parent_options = ["(none)"] + node_ids
            parent_id = st.selectbox("Parent", parent_options, index=0)
            agency_id = st.text_input("Agency ID (optional)", value="")
            domains = st.text_input("Domains (comma-separated)", value="intel")
            clearance = st.selectbox("Clearance", ["PUBLIC", "INTERNAL", "RESTRICTED", "SECRET"], index=2)
            add_node = st.form_submit_button("Add Node")

        if add_node:
            node_id = node_id.strip()
            if not node_id:
                st.error("Node ID is required.")
            elif node_id in node_ids:
                st.error("Node ID already exists.")
            else:
                parent_val = None if parent_id == "(none)" else parent_id
                if level > 1 and not parent_val:
                    st.error("Level 2/3 nodes require a parent.")
                else:
                    if not agency_id:
                        if level == 1:
                            agency_val = node_id
                        else:
                            parent = next((n for n in nodes if n.get("node_id") == parent_val), {})
                            agency_val = parent.get("agency_id") or parent_val or node_id
                    else:
                        agency_val = agency_id.strip()
                    nodes.append(
                        {
                            "node_id": node_id,
                            "level": int(level),
                            "node_type": node_type,
                            "parent_id": parent_val,
                            "agency_id": agency_val,
                            "domains": [d.strip() for d in domains.split(",") if d.strip()],
                            "clearance": clearance,
                        }
                    )
                    draft["nodes"] = nodes
                    st.session_state.kcollab_topology_draft = draft
                    st.success(f"Node added: {node_id}")

        if nodes:
            st.markdown("#### Nodes")
            if HAS_PANDAS:
                st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
            else:
                st.json(nodes)

            remove_nodes = st.multiselect("Remove nodes", options=node_ids, key="kcollab_remove_nodes")
            if st.button("Remove Selected Nodes", key="kcollab_remove_nodes_btn"):
                if remove_nodes:
                    nodes = [n for n in nodes if n.get("node_id") not in set(remove_nodes)]
                    edges = [e for e in draft.get("trust_edges", []) if e.get("source") not in set(remove_nodes) and e.get("target") not in set(remove_nodes)]
                    draft["nodes"] = nodes
                    draft["trust_edges"] = edges
                    st.session_state.kcollab_topology_draft = draft
                    st.success("Selected nodes removed.")

        st.markdown("#### Trust Boundaries")
        if node_ids:
            with st.form("kcollab_add_edge_form"):
                source = st.selectbox("Source", node_ids, key="kcollab_edge_source")
                target = st.selectbox("Target", node_ids, key="kcollab_edge_target")
                channel = st.selectbox("Channel", ["need_to_know", "cross_agency_aggregate", "standard"], index=0)
                add_edge = st.form_submit_button("Add Trust Edge")
            if add_edge:
                if source == target:
                    st.error("Source and target cannot be the same.")
                else:
                    edges = list(draft.get("trust_edges", []))
                    edges.append({"source": source, "target": target, "channel": channel})
                    draft["trust_edges"] = edges
                    st.session_state.kcollab_topology_draft = draft
                    st.success("Trust edge added.")
        else:
            st.info("Add at least one node before defining trust boundaries.")

        edges = list(draft.get("trust_edges", []))
        if edges:
            if HAS_PANDAS:
                st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)
            else:
                st.json(edges)
            edge_labels = [f"{e.get('source')} -> {e.get('target')} ({e.get('channel')})" for e in edges]
            remove_edges = st.multiselect("Remove edges", options=edge_labels, key="kcollab_remove_edges")
            if st.button("Remove Selected Edges", key="kcollab_remove_edges_btn"):
                remove_set = set(remove_edges)
                edges = [e for e, label in zip(edges, edge_labels) if label not in remove_set]
                draft["trust_edges"] = edges
                st.session_state.kcollab_topology_draft = draft
                st.success("Selected edges removed.")

        with st.expander("Advanced JSON (optional)"):
            raw = st.text_area("Topology JSON", value=json.dumps(draft, indent=2), height=220, key="kcollab_topology_raw")
            if st.button("Load JSON into Draft", key="kcollab_load_topology_json"):
                try:
                    st.session_state.kcollab_topology_draft = json.loads(raw)
                    st.success("Draft updated from JSON.")
                except Exception as exc:
                    st.error(f"JSON load failed: {exc}")

    with right:
        st.markdown("#### Graph Preview")
        st.code(topology_preview(draft), language="text")
        st.markdown("#### Validation")
        actor = st.text_input("Actor", value="analyst_a", key="kcollab_topology_actor_wiz")
        message = st.text_input("Version Message", value="topology update", key="kcollab_topology_msg_wiz")
        if st.button("Validate Topology", key="kcollab_validate_topology"):
            try:
                validate_topology(draft)
                st.success("Topology is valid.")
            except TopologyValidationError as exc:
                st.error(f"Validation error: {exc}")
            except Exception as exc:
                st.error(f"Validation failed: {exc}")
        if st.button("Save Topology Version", key="kcollab_save_topology_wiz", type="primary"):
            try:
                rec = services["topology"].save(draft, actor=actor, message=message)
                st.success(f"Saved topology version: {rec['version_id']}")
            except Exception as exc:
                st.error(f"Save failed: {exc}")


def _render_step_connectors(services: Dict[str, Any]) -> None:
    st.markdown("### Step 2 — Connectors & Dataset Publishing")
    st.caption("Connect to institutional systems and verify data contracts/mappings.")
    fed_db = services["fed_db"]

    if st.button("Sync Node Connectors", key="kcollab_sync_connectors_wiz"):
        try:
            synced = fed_db.register_default_from_manager(actor="wizard")
            st.success(f"Synced connectors for {synced['nodes']} nodes")
        except Exception as exc:
            st.error(f"Connector sync failed: {exc}")

    health = fed_db.connector_health()
    st.session_state.kcollab_last_health = health
    if health:
        st.markdown("#### Connector Health")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(health), use_container_width=True, hide_index=True)
        else:
            st.json(health)
    else:
        st.info("No connectors registered yet.")

    connectors = fed_db.catalog.connectors()
    if connectors:
        st.markdown("#### Connectors Registry")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(connectors), use_container_width=True, hide_index=True)
        else:
            st.json(connectors)

    contracts = fed_db.contracts.all()
    if contracts:
        st.markdown("#### Data Contracts")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(list(contracts.values())), use_container_width=True, hide_index=True)
        else:
            st.json(contracts)
    else:
        st.info("No contracts registered yet.")

    mappings = fed_db.canonical.all()
    if mappings:
        st.markdown("#### Canonical Mappings")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(list(mappings.values())), use_container_width=True, hide_index=True)
        else:
            st.json(mappings)


def _render_step_project(services: Dict[str, Any]) -> None:
    st.markdown("### Step 3 — Project Setup")
    fed_db = services["fed_db"]
    topology = services["topology"].get_payload()
    nodes = [n["node_id"] for n in topology.get("nodes", []) if int(n.get("level", 0) or 0) >= 2]
    datasets = sorted(fed_db.contracts.all().keys()) or ["local_samples"]

    with st.form("kcollab_project_form"):
        project_id = st.text_input("Project ID", value="default_monitoring")
        name = st.text_input("Project Name", value="Monitoring Collaboration")
        objective = st.text_area("Objective", value="Cross-org aggregate analytics and shared intelligence.", height=80)
        participants = st.multiselect("Participants", options=nodes, default=nodes[: min(4, len(nodes))])
        allowed_datasets = st.multiselect("Allowed Datasets", options=datasets, default=datasets[:1])
        allowed_domains = st.text_input("Allowed Domains (comma-separated)", value="intel,finance,security")
        c1, c2, c3 = st.columns(3)
        with c1:
            allow_analytics = st.checkbox("Allow Analytics", value=True)
        with c2:
            allow_federated_ml = st.checkbox("Allow Federated ML", value=True)
        with c3:
            k_threshold = st.slider("k-threshold min", min_value=2, max_value=10, value=3)
        purposes = st.multiselect("Purpose Allowlist", options=["monitoring", "research", "casework", "oversight", "audit"], default=["monitoring", "research"])
        required_clearance = st.selectbox("Required Clearance", ["PUBLIC", "INTERNAL", "RESTRICTED", "SECRET"], index=2)
        save_project = st.form_submit_button("Save Project", type="primary")

    if save_project:
        if not project_id.strip():
            st.error("Project ID is required.")
        elif not participants:
            st.error("Select at least one participant.")
        elif not allowed_datasets:
            st.error("Select at least one dataset.")
        else:
            try:
                fed_db.upsert_project(
                    CollaborationProject(
                        project_id=project_id.strip(),
                        name=name.strip() or project_id.strip(),
                        objective=objective.strip(),
                        participants=participants,
                        allowed_datasets=allowed_datasets,
                        allowed_domains=[d.strip() for d in allowed_domains.split(",") if d.strip()],
                        allowed_computations=[c for c, flag in {"analytics": allow_analytics, "federated_ml": allow_federated_ml}.items() if flag],
                        governance={
                            "k_threshold_min": int(k_threshold),
                            "purpose_allowlist": purposes,
                            "required_clearance": required_clearance,
                        },
                    ),
                    actor="wizard",
                )
                st.success("Project saved.")
            except Exception as exc:
                st.error(f"Project save failed: {exc}")

    projects = fed_db.projects.all()
    if projects:
        st.markdown("#### Current Projects")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(list(projects.values())), use_container_width=True, hide_index=True)
        else:
            st.json(projects)


def _render_step_compatibility(services: Dict[str, Any]) -> None:
    st.markdown("### Step 4 — Compatibility & Baskets")
    fed_db = services["fed_db"]
    projects = sorted(fed_db.projects.all().keys()) or ["default_monitoring"]
    datasets = sorted(fed_db.contracts.all().keys()) or ["local_samples"]

    c1, c2, c3 = st.columns(3)
    with c1:
        project_id = st.selectbox("Project", options=projects, key="kcollab_wiz_compat_project")
    with c2:
        dataset_id = st.selectbox("Dataset", options=datasets, key="kcollab_wiz_compat_dataset")
    with c3:
        operation = st.selectbox("Operation", ["aggregate", "time_bucket"], key="kcollab_wiz_compat_op")

    if st.button("Run Compatibility Analysis", type="primary", key="kcollab_wiz_run_compat"):
        try:
            report = fed_db.run_compatibility_analysis(dataset_id=dataset_id, operation=operation, project_id=project_id, actor="wizard")
            st.session_state.kcollab_last_compat = report
        except Exception as exc:
            st.error(f"Compatibility analysis failed: {exc}")

    report = st.session_state.get("kcollab_last_compat")
    if report:
        st.markdown("#### Node Scores")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(report.get("node_scores", [])), use_container_width=True, hide_index=True)
        else:
            st.json(report.get("node_scores", []))
        st.markdown("#### Baskets")
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(report.get("baskets", [])), use_container_width=True, hide_index=True)
        else:
            st.json(report.get("baskets", []))
        if report.get("excluded_nodes"):
            st.markdown("#### Excluded Nodes")
            st.json(report.get("excluded_nodes", []))


def _render_step_analytics(services: Dict[str, Any]) -> None:
    st.markdown("### Step 5 — Federated Analytics")
    fed_db = services["fed_db"]
    projects = sorted(fed_db.projects.all().keys()) or ["default_monitoring"]

    c1, c2, c3 = st.columns(3)
    with c1:
        project_id = st.selectbox("Project", options=projects, key="kcollab_wiz_query_project")
    with c2:
        role = st.selectbox("Role", ["analyst", "supervisor", "auditor"], key="kcollab_wiz_role")
    with c3:
        clearance = st.selectbox("Clearance", ["PUBLIC", "INTERNAL", "RESTRICTED", "SECRET"], index=2, key="kcollab_wiz_clearance")

    q1, q2, q3 = st.columns(3)
    with q1:
        user_id = st.text_input("User", value="analyst_1", key="kcollab_wiz_user")
    with q2:
        purpose = st.selectbox("Purpose", ["monitoring", "research", "casework", "oversight", "audit"], key="kcollab_wiz_purpose")
    with q3:
        k_threshold = st.slider("k-threshold", min_value=2, max_value=15, value=3, key="kcollab_wiz_k")

    query_text = st.text_area(
        "Query (SQL subset or JSON DSL)",
        value="SELECT county, COUNT(*) FROM local_samples GROUP BY county",
        height=120,
        key="kcollab_wiz_query_text",
    )

    if st.button("Run Federated Query", type="primary", key="kcollab_wiz_run_query"):
        try:
            result = fed_db.run_query(
                query_text=query_text,
                context=AccessContext(user_id=user_id, role=role, clearance=clearance, purpose=purpose),
                actor=user_id,
                k_threshold=int(k_threshold),
                project_id=project_id,
            )
            st.session_state.kcollab_last_query = result
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    result = st.session_state.get("kcollab_last_query")
    if result:
        st.markdown("#### Validation + Plan")
        st.json({"allowed": result.get("allowed"), "reasons": result.get("reasons", []), "plan": result.get("plan")})
        if result.get("allowed"):
            st.markdown("#### Results")
            rows = result.get("rows", [])
            if HAS_PANDAS:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.json(rows)
            if result.get("provenance"):
                st.markdown("#### Provenance")
                st.json(result.get("provenance"))
            if result.get("baskets"):
                st.markdown("#### Basket Execution")
                st.json(result.get("baskets"))
        else:
            st.warning("Query denied. Review policy/project reasons above.")


def _render_step_fl_readiness(services: Dict[str, Any]) -> None:
    st.markdown("### Step 6 — FL Readiness")
    fed_ml = services["fed_ml"]
    fed_db = services["fed_db"]

    report = st.session_state.get("kcollab_last_compat")
    if not report:
        st.info("Run compatibility analysis first to generate baskets.")
        return

    projects = sorted(fed_db.projects.all().keys()) or ["default_monitoring"]
    c1, c2, c3 = st.columns(3)
    with c1:
        project_id = st.selectbox("Project", options=projects, key="kcollab_wiz_fl_project")
    with c2:
        min_participants = st.slider("Min participants per basket", min_value=1, max_value=5, value=2, key="kcollab_wiz_fl_min_participants")
    with c3:
        min_eps = st.slider("Min remaining epsilon", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="kcollab_wiz_fl_eps")

    if st.button("Assess FL Readiness", type="primary", key="kcollab_wiz_fl_ready"):
        try:
            readiness = fed_ml.readiness_from_baskets(
                baskets=report.get("baskets", []),
                project_id=project_id,
                min_participants_per_basket=int(min_participants),
                min_remaining_epsilon=float(min_eps),
            )
            st.session_state.kcollab_last_fl_ready = readiness
        except Exception as exc:
            st.error(f"Readiness check failed: {exc}")

    readiness = st.session_state.get("kcollab_last_fl_ready")
    if readiness:
        st.markdown("#### Readiness Summary")
        st.json(readiness)


def _render_step_hard_problems(services: Dict[str, Any]) -> None:
    st.markdown("### Step 7 — Hard Problems (Split)")
    fed_db = services["fed_db"]
    fed_ml = services["fed_ml"]

    health = st.session_state.get("kcollab_last_health", [])
    compatibility = st.session_state.get("kcollab_last_compat") or {}
    query_result = st.session_state.get("kcollab_last_query") or {}
    readiness = st.session_state.get("kcollab_last_fl_ready") or {}

    if not compatibility or not query_result:
        st.info("Run compatibility and analytics steps first for full reports.")

    db_report = fed_db.hard_problems.assess(
        health=health,
        compatibility=compatibility,
        query_result=query_result,
    )
    st.markdown("#### DB-Side Hard Problems")
    st.json(db_report)

    fl_report = fed_ml.assess_hard_problems(
        readiness=readiness,
        round_output=st.session_state.get("kcollab_ml_round"),
    )
    st.markdown("#### FL-Side Hard Problems")
    st.json(fl_report)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower())
    return cleaned.strip("_") or "collaboration"


def _render_topology_graph(payload: Dict[str, Any], theme) -> None:
    if not HAS_PLOTLY:
        st.code(topology_preview(payload), language="text")
        return

    nodes = list(payload.get("nodes", []))
    if not nodes:
        st.info("Add organizations and departments to see the collaboration graph.")
        return

    levels = {1: [], 2: [], 3: []}
    for node in nodes:
        lvl = int(node.get("level", 1) or 1)
        levels.setdefault(lvl, []).append(node)

    positions: Dict[str, tuple[float, float]] = {}
    for lvl in [1, 2, 3]:
        group = levels.get(lvl, [])
        width = max(1, len(group))
        for idx, node in enumerate(sorted(group, key=lambda n: n.get("node_id", ""))):
            x = (idx + 1) / (width + 1)
            y = 1.0 - (lvl - 1) * 0.42
            positions[str(node.get("node_id"))] = (x, y)

    line_x: List[float] = []
    line_y: List[float] = []
    for node in nodes:
        parent = node.get("parent_id")
        nid = str(node.get("node_id", ""))
        if parent and parent in positions and nid in positions:
            x0, y0 = positions[parent]
            x1, y1 = positions[nid]
            line_x += [x0, x1, None]
            line_y += [y0, y1, None]

    trust_x: List[float] = []
    trust_y: List[float] = []
    for edge in payload.get("trust_edges", []):
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if src in positions and dst in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            trust_x += [x0, x1, None]
            trust_y += [y0, y1, None]

    color_map = {"agency": "#2cc8a7", "department": "#4db3ff", "site": "#f8b84e"}
    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_color: List[str] = []
    for node in nodes:
        nid = str(node.get("node_id", ""))
        if nid not in positions:
            continue
        x, y = positions[nid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{nid} ({node.get('node_type', 'node')})")
        node_color.append(color_map.get(str(node.get("node_type", "")).lower(), "#c7ccd5"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="rgba(150,160,180,0.55)", width=2),
            hoverinfo="skip",
            name="Hierarchy",
        )
    )
    if trust_x:
        fig.add_trace(
            go.Scatter(
                x=trust_x,
                y=trust_y,
                mode="lines",
                line=dict(color="rgba(44,200,167,0.9)", width=2, dash="dot"),
                hoverinfo="skip",
                name="Trust Links",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=20, color=node_color, line=dict(width=1, color="rgba(255,255,255,0.8)")),
            name="Entities",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[-0.2, 1.1]),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.text_primary),
    )
    st.plotly_chart(fig, use_container_width=True)


def _init_collab_experience_state(services: Dict[str, Any]) -> None:
    _ensure_topology_seed(services)
    st.session_state.setdefault("kcollab_flow_step", 0)
    st.session_state.setdefault("kcollab_active_project", "default_monitoring")
    st.session_state.setdefault("kcollab_topology_draft", services["topology"].get_payload())
    st.session_state.setdefault("kcollab_last_health", [])
    st.session_state.setdefault("kcollab_last_compat", None)
    st.session_state.setdefault("kcollab_last_query", None)
    st.session_state.setdefault("kcollab_last_fl_ready", None)


def _render_flow_nav() -> int:
    steps = [
        "1. Create Collaboration",
        "2. Add Organizations",
        "3. Connect Organizations",
        "4. Generate Shared Insights",
        "5. Prepare Model Training",
        "6. Trust & Risk Report",
    ]
    current = int(st.session_state.get("kcollab_flow_step", 0))
    current = max(0, min(current, len(steps) - 1))

    selected = st.select_slider("Collaboration Flow", options=steps, value=steps[current], key="kcollab_flow_slider")
    st.session_state.kcollab_flow_step = steps.index(selected)

    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if st.button("Back", key="kcollab_flow_back", disabled=st.session_state.kcollab_flow_step == 0):
            st.session_state.kcollab_flow_step = max(0, st.session_state.kcollab_flow_step - 1)
            st.rerun()
    with c2:
        if st.button("Next", key="kcollab_flow_next", disabled=st.session_state.kcollab_flow_step == len(steps) - 1):
            st.session_state.kcollab_flow_step = min(len(steps) - 1, st.session_state.kcollab_flow_step + 1)
            st.rerun()

    st.progress((st.session_state.kcollab_flow_step + 1) / len(steps))
    return int(st.session_state.kcollab_flow_step)


def _render_collab_step_create(services: Dict[str, Any]) -> None:
    st.markdown("### Create a Collaboration")
    st.caption("Define the mission and participants. K-Collab handles technical data integration behind the scenes.")

    fed_db = services["fed_db"]
    topology = st.session_state.get("kcollab_topology_draft", {})
    participant_candidates = [n["node_id"] for n in topology.get("nodes", []) if int(n.get("level", 0) or 0) >= 2]
    dataset_options = sorted(fed_db.contracts.all().keys()) or ["local_samples"]

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Collaboration name", value="Regional Threat Intelligence Collaboration")
        objective = st.text_area("Goal", value="Share insights across organizations while preserving each participant's sensitive data.", height=110)
        dataset_id = st.selectbox("Primary insight source", options=dataset_options)
    with col2:
        participants = st.multiselect(
            "Participating departments",
            options=participant_candidates,
            default=participant_candidates[: min(4, len(participant_candidates))],
        )
        allow_training = st.checkbox("Enable joint model training", value=True)
        privacy_level = st.selectbox("Privacy level", ["Standard", "High", "Strict"], index=1)

    if st.button("Create Collaboration", type="primary", key="kcollab_create_collaboration"):
        if not participants:
            st.error("Select at least one participating department.")
            return
        project_id = _slugify(name)
        privacy_k = {"Standard": 3, "High": 4, "Strict": 5}[privacy_level]
        try:
            fed_db.upsert_project(
                CollaborationProject(
                    project_id=project_id,
                    name=name.strip() or project_id,
                    objective=objective.strip(),
                    participants=participants,
                    allowed_datasets=[dataset_id],
                    allowed_domains=["intel", "finance", "security"],
                    allowed_computations=["analytics", "federated_ml"] if allow_training else ["analytics"],
                    governance={"k_threshold_min": privacy_k, "purpose_allowlist": ["monitoring", "research", "casework"]},
                ),
                actor="collab_owner",
            )
            st.session_state.kcollab_active_project = project_id
            st.success(f"Collaboration created: {name}")
        except Exception as exc:
            st.error(f"Could not create collaboration: {exc}")

    project_rows = fed_db.projects.all()
    if project_rows:
        labels = {pid: row.get("name", pid) for pid, row in project_rows.items()}
        selected = st.selectbox(
            "Active collaboration",
            options=list(labels.keys()),
            index=list(labels.keys()).index(st.session_state.get("kcollab_active_project"))
            if st.session_state.get("kcollab_active_project") in labels
            else 0,
            format_func=lambda x: labels.get(x, x),
            key="kcollab_active_project_select",
        )
        st.session_state.kcollab_active_project = selected


def _render_collab_step_network(services: Dict[str, Any], theme) -> None:
    st.markdown("### Add Organizations and Departments")
    st.caption("Model who collaborates with whom. No technical setup required in this step.")

    draft = dict(st.session_state.get("kcollab_topology_draft", services["topology"].get_payload()))
    draft.setdefault("nodes", [])
    draft.setdefault("trust_edges", [])
    nodes = list(draft.get("nodes", []))
    node_ids = [n.get("node_id") for n in nodes if n.get("node_id")]

    c_left, c_right = st.columns([3, 2])
    with c_left:
        with st.form("kcollab_add_entity_form"):
            display_name = st.text_input("Entity name", value="")
            entity_type = st.selectbox("Entity type", ["Organization", "Department", "Site"], index=0)
            parent = st.selectbox("Belongs to", options=["(none)"] + node_ids, index=0)
            focus = st.text_input("Focus areas (comma-separated)", value="intel")
            clearance = st.selectbox("Sensitivity tier", ["PUBLIC", "INTERNAL", "RESTRICTED", "SECRET"], index=2)
            add_entity = st.form_submit_button("Add entity")

        if add_entity:
            nid = _slugify(display_name)
            if not display_name.strip():
                st.error("Entity name is required.")
            elif nid in node_ids:
                st.error("That entity already exists.")
            else:
                level = {"Organization": 1, "Department": 2, "Site": 3}[entity_type]
                node_type = {"Organization": "agency", "Department": "department", "Site": "site"}[entity_type]
                parent_id = None if parent == "(none)" else parent
                if level > 1 and not parent_id:
                    st.error("Departments and sites must belong to an organization or department.")
                else:
                    if level == 1:
                        agency_id = nid
                    else:
                        parent_node = next((n for n in nodes if n.get("node_id") == parent_id), {})
                        agency_id = parent_node.get("agency_id") or parent_id or nid
                    nodes.append(
                        {
                            "node_id": nid,
                            "level": level,
                            "node_type": node_type,
                            "parent_id": parent_id,
                            "agency_id": agency_id,
                            "domains": [x.strip() for x in focus.split(",") if x.strip()],
                            "clearance": clearance,
                        }
                    )
                    draft["nodes"] = nodes
                    st.session_state.kcollab_topology_draft = draft
                    st.success(f"Added {entity_type.lower()}: {display_name}")

        if nodes:
            view_rows = [{"entity_id": n.get("node_id"), "type": n.get("node_type"), "parent": n.get("parent_id"), "focus_areas": ", ".join(n.get("domains", []))} for n in nodes]
            if HAS_PANDAS:
                st.dataframe(pd.DataFrame(view_rows), use_container_width=True, hide_index=True)
            else:
                st.write(view_rows)

            remove_ids = st.multiselect("Remove entities", options=node_ids, key="kcollab_remove_entities")
            if st.button("Remove selected entities", key="kcollab_remove_entities_btn"):
                nodes = [n for n in nodes if n.get("node_id") not in set(remove_ids)]
                edges = [e for e in draft.get("trust_edges", []) if e.get("source") not in set(remove_ids) and e.get("target") not in set(remove_ids)]
                draft["nodes"] = nodes
                draft["trust_edges"] = edges
                st.session_state.kcollab_topology_draft = draft
                st.success("Entities removed.")

        st.markdown("#### Collaboration links")
        if node_ids:
            edge_c1, edge_c2, edge_c3 = st.columns(3)
            with edge_c1:
                src = st.selectbox("From", node_ids, key="kcollab_edge_src_simple")
            with edge_c2:
                dst = st.selectbox("To", node_ids, key="kcollab_edge_dst_simple")
            with edge_c3:
                ch = st.selectbox("Link type", ["need_to_know", "cross_agency_aggregate", "standard"], key="kcollab_edge_ch_simple")
            if st.button("Add link", key="kcollab_add_link_btn"):
                if src == dst:
                    st.error("Select two different entities.")
                else:
                    edges = list(draft.get("trust_edges", []))
                    edges.append({"source": src, "target": dst, "channel": ch})
                    draft["trust_edges"] = edges
                    st.session_state.kcollab_topology_draft = draft
                    st.success("Collaboration link added.")

        if st.button("Save collaboration map", type="primary", key="kcollab_save_map_simple"):
            try:
                saved = services["topology"].save(draft, actor="collab_owner", message="collaboration_map_update")
                st.success(f"Collaboration map saved (version {saved['version_id']})")
            except Exception as exc:
                st.error(f"Could not save collaboration map: {exc}")

    with c_right:
        st.markdown("#### Collaboration graph")
        _render_topology_graph(draft, theme)
        try:
            validate_topology(draft)
            st.success("Collaboration map is valid.")
        except Exception as exc:
            st.warning(f"Map needs attention: {exc}")


def _render_collab_step_connect(services: Dict[str, Any]) -> None:
    st.markdown("### Connect Organizations")
    st.caption("K-Collab auto-connects sources and harmonizes formats in the background.")

    fed_db = services["fed_db"]
    if st.button("Connect all participating organizations", type="primary", key="kcollab_connect_orgs"):
        try:
            synced = fed_db.register_default_from_manager(actor="collab_owner")
            st.success(f"Connected {synced['nodes']} organizations.")
        except Exception as exc:
            st.error(f"Connection failed: {exc}")

    health = fed_db.connector_health()
    st.session_state.kcollab_last_health = health
    if health:
        rows = [
            {
                "Organization Node": h.get("node_id"),
                "Connection Status": "Ready" if h.get("healthy") else "Needs attention",
                "Note": h.get("reason", ""),
            }
            for h in health
        ]
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.write(rows)
    else:
        st.info("No connected organizations yet.")


def _render_collab_step_insights(services: Dict[str, Any]) -> None:
    st.markdown("### Generate Shared Insights")
    st.caption("Run safe cross-organization analytics while preserving each participant's trade secrets.")

    fed_db = services["fed_db"]
    project_id = st.session_state.get("kcollab_active_project", "default_monitoring")
    dataset_options = sorted(fed_db.contracts.all().keys()) or ["local_samples"]

    c1, c2, c3 = st.columns(3)
    with c1:
        dataset_id = st.selectbox("Insight source", dataset_options, key="kcollab_insight_dataset")
    with c2:
        insight_type = st.selectbox("Insight type", ["Activity by county", "Activity by sector", "Average criticality by county"], key="kcollab_insight_type")
    with c3:
        privacy_k = st.slider("Privacy protection level", min_value=2, max_value=10, value=3, key="kcollab_insight_k")

    query_map = {
        "Activity by county": f"SELECT county, COUNT(*) FROM {dataset_id} GROUP BY county",
        "Activity by sector": f"SELECT sector, COUNT(*) FROM {dataset_id} GROUP BY sector",
        "Average criticality by county": f"SELECT county, AVG(criticality) FROM {dataset_id} GROUP BY county",
    }
    query_text = query_map[insight_type]
    st.text_input("Query generated automatically", value=query_text, disabled=True)

    if st.button("Generate insights", type="primary", key="kcollab_generate_insights_btn"):
        try:
            compat = fed_db.run_compatibility_analysis(
                dataset_id=dataset_id,
                operation="aggregate",
                project_id=project_id,
                actor="collab_owner",
            )
            st.session_state.kcollab_last_compat = compat
            result = fed_db.run_query(
                query_text=query_text,
                context=AccessContext(user_id="collab_owner", role="supervisor", clearance="SECRET", purpose="monitoring"),
                actor="collab_owner",
                k_threshold=int(privacy_k),
                project_id=project_id,
            )
            st.session_state.kcollab_last_query = result
        except Exception as exc:
            st.error(f"Could not generate insights: {exc}")

    result = st.session_state.get("kcollab_last_query")
    if result:
        if not result.get("allowed"):
            st.warning("Insights could not be generated with current policy/project settings.")
            st.write(result.get("reasons", []))
            return
        prov = result.get("provenance", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Coverage", f"{float(prov.get('coverage_score', 0.0))*100:.1f}%")
        m2.metric("Quality", f"{float(prov.get('quality_score', 0.0))*100:.1f}%")
        m3.metric("Contributing organizations", f"{len(prov.get('contributing_institutions', []))}")
        rows = result.get("rows", [])
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.write(rows)
        excluded = prov.get("excluded_institutions", [])
        if excluded:
            st.info("Some organizations were safely excluded due to policy/compatibility constraints.")
            if HAS_PANDAS:
                st.dataframe(pd.DataFrame(excluded), use_container_width=True, hide_index=True)
            else:
                st.write(excluded)


def _render_collab_step_training(services: Dict[str, Any]) -> None:
    st.markdown("### Prepare Model Training")
    st.caption("Check if the collaboration is ready for joint model training.")

    fed_db = services["fed_db"]
    fed_ml = services["fed_ml"]
    project_id = st.session_state.get("kcollab_active_project", "default_monitoring")
    compat = st.session_state.get("kcollab_last_compat")
    if not compat:
        st.info("Generate shared insights first to build compatibility baskets.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        min_participants = st.slider("Minimum teams per basket", min_value=1, max_value=5, value=2, key="kcollab_train_min_p")
    with c2:
        min_eps = st.slider("Minimum privacy budget", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="kcollab_train_eps")
    with c3:
        task_name = st.text_input("Training objective", value="shared_threat_signal_model", key="kcollab_train_task")

    if st.button("Check training readiness", type="primary", key="kcollab_check_training_ready"):
        try:
            ready = fed_ml.readiness_from_baskets(
                baskets=compat.get("baskets", []),
                project_id=project_id,
                min_participants_per_basket=int(min_participants),
                min_remaining_epsilon=float(min_eps),
            )
            st.session_state.kcollab_last_fl_ready = ready
        except Exception as exc:
            st.error(f"Readiness check failed: {exc}")

    ready = st.session_state.get("kcollab_last_fl_ready")
    if ready:
        if ready.get("ready"):
            st.success("Ready for joint training.")
        else:
            st.warning("Not ready for joint training yet.")
            st.write(ready.get("reasons", []))
        m1, m2, m3 = st.columns(3)
        m1.metric("Usable baskets", len(ready.get("usable_baskets", [])))
        m2.metric("Excluded baskets", len(ready.get("excluded_baskets", [])))
        m3.metric("Participant pool", ready.get("participant_pool_size", 0))

        if ready.get("ready") and st.button("Start training session", key="kcollab_start_training_session"):
            try:
                selected_nodes = sorted({m for b in ready.get("usable_baskets", []) for m in b.get("members", [])})
                started = fed_ml.start_job(
                    actor="collab_owner",
                    task_name=task_name,
                    selected_nodes=selected_nodes,
                    project_id=project_id,
                    vector_dim=16,
                    min_remaining_epsilon=float(min_eps),
                )
                st.success(f"Training session started: {started['job_id']}")
            except Exception as exc:
                st.error(f"Could not start training: {exc}")


def _render_collab_step_trust(services: Dict[str, Any]) -> None:
    st.markdown("### Trust and Risk Report")
    st.caption("Plain-language status of collaboration safety, quality, and readiness.")

    fed_db = services["fed_db"]
    fed_ml = services["fed_ml"]
    db_report = fed_db.hard_problems.assess(
        health=st.session_state.get("kcollab_last_health", []),
        compatibility=st.session_state.get("kcollab_last_compat") or {},
        query_result=st.session_state.get("kcollab_last_query") or {},
    )
    fl_report = fed_ml.assess_hard_problems(
        readiness=st.session_state.get("kcollab_last_fl_ready") or {},
        round_output=st.session_state.get("kcollab_ml_round"),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Data collaboration")
        st.metric("Overall status", db_report.get("overall_status", "unknown").upper())
        if HAS_PANDAS:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Check": p.get("name"), "Status": p.get("status"), "What to do": p.get("action")}
                        for p in db_report.get("problems", [])
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write(db_report)
    with c2:
        st.markdown("#### Model collaboration")
        st.metric("Overall status", fl_report.get("overall_status", "unknown").upper())
        if HAS_PANDAS:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Check": p.get("name"), "Status": p.get("status"), "What to do": p.get("action")}
                        for p in fl_report.get("problems", [])
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write(fl_report)


def _render_kcollab_wizard(services: Dict[str, Any], data: DashboardData, theme) -> None:
    _init_collab_experience_state(services)
    st.markdown("#### K-Collab Collaboration Experience")
    st.caption("Collaborate across organizations, preserve trade secrets, and generate trusted shared intelligence.")

    step = _render_flow_nav()
    if step == 0:
        _render_collab_step_create(services)
    elif step == 1:
        _render_collab_step_network(services, theme)
    elif step == 2:
        _render_collab_step_connect(services)
    elif step == 3:
        _render_collab_step_insights(services)
    elif step == 4:
        _render_collab_step_training(services)
    else:
        _render_collab_step_trust(services)

    admin_mode = st.toggle("Show technical/admin panels", value=False, key="kcollab_admin_mode")
    if admin_mode:
        with st.expander("Technical panels", expanded=False):
            tabs = st.tabs(
                [
                    "Topology Builder (JSON)",
                    "Federated Data Access",
                    "Guided Walkthrough",
                    "Federated ML",
                    "Legacy Control",
                ]
            )
            with tabs[0]:
                _render_topology_builder(services)
            with tabs[1]:
                _render_federated_databases_pane(services)
            with tabs[2]:
                _render_walkthrough_pane(services)
            with tabs[3]:
                _render_federated_ml_pane(services)
            with tabs[4]:
                _render_legacy_federation_controls(data, theme)


def render_federation_tab(data: DashboardData, theme):
    st.markdown('<div class="section-header">K-COLLAB</div>', unsafe_allow_html=True)
    st.caption("Cross-organization collaboration for safe shared insights and joint model training.")

    services = _get_kcollab()
    _render_kcollab_wizard(services, data, theme)

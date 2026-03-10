"""
Federated Learning Dashboard — Streamlit UI for FL monitoring and data upload.

Upload data → triggers ``fl.data_ready`` → automatic training → aggregation.
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path

st.set_page_config(page_title="Federated Learning", page_icon="", layout="wide")

# ── Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .fl-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .fl-header h1 { color: #e94560; margin: 0; }
    .fl-header p { color: #a8a8b3; margin-top: 0.5rem; }
    .metric-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p { color: #a8a8b3; margin: 0.3rem 0 0 0; font-size: 0.85rem; }
    .status-running { color: #4ade80; font-weight: bold; }
    .status-stopped { color: #f87171; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def _get_federation_manager():
    """Lazy-load the federation manager."""
    try:
        from federated_databases.scarcity_federation import get_scarcity_federation
        return get_scarcity_federation()
    except Exception as e:
        st.error(f"Could not load federation manager: {e}")
        return None


def _get_model_registry():
    """Lazy-load available models."""
    try:
        from federated_databases.model_registry import FLModelRegistry
        return FLModelRegistry.list_models()
    except Exception:
        return ["logistic"]


# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="fl-header">
    <h1>Federated Learning</h1>
    <p>Event-driven distributed training across county nodes</p>
</div>
""", unsafe_allow_html=True)

fm = _get_federation_manager()
if fm is None:
    st.stop()

# ── Status Overview ──────────────────────────────────────────────────
status = fm.get_status()
nodes = status.get("nodes", [])
round_history = fm.get_round_history(limit=50)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🏢 Nodes", len(nodes))
with col2:
    st.metric("🔄 Rounds", status.get("round_count", 0))
with col3:
    latest = status.get("latest_round")
    loss_val = f"{latest['global_loss']:.4f}" if latest and latest.get("global_loss") else "—"
    st.metric("📉 Latest Loss", loss_val)
with col4:
    total_samples = sum(n.get("sample_count", 0) for n in nodes)
    st.metric("Total Samples", f"{total_samples:,}")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────
tab_upload, tab_training, tab_nodes, tab_history, tab_config = st.tabs([
    "📤 Data Upload", "🧠 Training", "🏢 Nodes", "📜 Round History", "⚙️ Config"
])

# ── Tab: Data Upload ─────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload Data to Trigger Training")

    col_up1, col_up2 = st.columns([2, 1])

    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload CSV data file",
            type=["csv"],
            help="Upload a CSV with columns: timestamp, location_county, threat_score, etc."
        )

    with col_up2:
        target_node = st.selectbox(
            "Target Node",
            options=[n["node_id"] for n in nodes] if nodes else ["org_a"],
            index=0,
        )
        model_name = st.selectbox(
            "Training Model",
            options=_get_model_registry(),
            index=0,
        )

    if uploaded_file is not None:
        if st.button("🚀 Upload & Train", type="primary", use_container_width=True):
            with st.spinner(f"Training on {target_node} with {model_name}..."):
                # Save uploaded file
                upload_dir = Path("data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_path = upload_dir / uploaded_file.name
                save_path.write_bytes(uploaded_file.getvalue())

                # Ingest and train
                try:
                    fm.ingest_live_batch(source_path=str(save_path))
                    result = fm.run_single_node_training(
                        node_id=target_node,
                        model_name=model_name,
                    )
                    st.success(
                        f"Training complete! "
                        f"Loss: {result['loss']:.4f}, "
                        f"Samples: {result['sample_count']}"
                    )
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.divider()
    st.subheader("Or Run a Full Sync Round")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        round_model = st.selectbox(
            "Model for Sync Round",
            options=_get_model_registry(),
            key="round_model",
        )
    with col_r2:
        round_lr = st.number_input("Learning Rate", value=0.12, step=0.01, key="round_lr")
    with col_r3:
        source_path = st.text_input(
            "Data Source",
            value="data/synthetic_kenya_policy/tweets.csv",
            key="round_source",
        )

    if st.button("🔄 Run Sync Round", use_container_width=True):
        with st.spinner("Running federated sync round..."):
            try:
                result = fm.run_sync_round(
                    learning_rate=round_lr,
                    model_name=round_model,
                    source_path=source_path,
                )
                st.success(
                    f"Round #{result.round_number} complete! "
                    f"Participants: {result.participants}, "
                    f"Loss: {result.global_loss:.4f}"
                )
            except Exception as e:
                st.error(f"Sync round failed: {e}")

# ── Tab: Training ────────────────────────────────────────────────────
with tab_training:
    st.subheader("Available Training Models")

    models = _get_model_registry()
    for m in models:
        with st.expander(f"🤖 {m}", expanded=False):
            if m == "logistic":
                st.write("Binary classifier using sigmoid + cross-entropy. Backward compatible with original federation manager.")
            elif m == "hypothesis_ensemble":
                st.write("Ensemble of 15+ hypothesis types: Causal (Granger), Correlational, Temporal (VAR), Functional, Equilibrium, plus RLS regression.")
            elif m == "rls_online":
                st.write("Online Recursive Least Squares estimator for streaming regression.")
            else:
                st.write(f"Custom registered model: {m}")

# ── Tab: Nodes ───────────────────────────────────────────────────────
with tab_nodes:
    st.subheader("Registered Federation Nodes")

    if nodes:
        df_nodes = pd.DataFrame(nodes)
        display_cols = [c for c in ["node_id", "backend", "county_filter", "sample_count", "model_updates"] if c in df_nodes.columns]
        st.dataframe(df_nodes[display_cols] if display_cols else df_nodes, use_container_width=True)
    else:
        st.info("No nodes registered yet.")

    st.divider()
    st.subheader("Register New Node")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        new_node_id = st.text_input("Node ID", placeholder="e.g., nairobi")
    with col_n2:
        county_filter = st.text_input("County Filter (optional)", placeholder="e.g., Nairobi")

    if st.button("➕ Register Node"):
        if new_node_id:
            try:
                fm.register_node(new_node_id, county_filter=county_filter or None)
                st.success(f"Node '{new_node_id}' registered!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

# ── Tab: History ─────────────────────────────────────────────────────
with tab_history:
    st.subheader("Sync Round History")

    if round_history:
        df_rounds = pd.DataFrame(round_history)
        display_cols = [c for c in [
            "round_number", "participants", "total_samples",
            "global_loss", "global_gradient_norm", "aggregation_method",
            "started_at", "completed_at"
        ] if c in df_rounds.columns]
        st.dataframe(
            df_rounds[display_cols] if display_cols else df_rounds,
            use_container_width=True,
        )

        # Loss chart
        if "global_loss" in df_rounds.columns and "round_number" in df_rounds.columns:
            st.line_chart(
                df_rounds.set_index("round_number")["global_loss"],
                use_container_width=True,
            )
    else:
        st.info("No rounds completed yet. Upload data or run a sync round.")

# ── Tab: Config ──────────────────────────────────────────────────────
with tab_config:
    st.subheader("Federation Configuration")

    st.json({
        "node_count": len(nodes),
        "control_db": status.get("control_db", "N/A"),
        "audit_path": status.get("audit_path", "N/A"),
        "available_models": _get_model_registry(),
    })

    st.subheader("CLI Commands")
    st.code("""
# Start coordinator
python scripts/run_fl.py coordinator --port 8765 --model hypothesis_ensemble

# Start client node
python scripts/run_fl.py client --coordinator ws://10.0.0.1:8765 --node-id nairobi

# Run single round
python scripts/run_fl.py round --model logistic

# Watch for new data
python scripts/run_fl.py watch --dir data/uploads/ --node-id nairobi

# Check status
python scripts/run_fl.py status

# List available models
python scripts/run_fl.py models
    """, language="bash")

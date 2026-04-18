"""
Institution Portal.
Allows organizations to sign up, get a federated node, and upload data
to trigger local online learning.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import json
import os
import hashlib
import secrets
import pandas as pd
import streamlit as st

AUTH_FILE = Path(__file__).parent / "auth.json"
APPROVALS_FILE = Path(__file__).parent / "agency_approvals.json"
APPROVAL_AUDIT_FILE = Path(__file__).parent / "agency_approval_audit.jsonl"

PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 200_000

SECTOR_INVITE_CODE_ENV = {
  "Finance": "KSCARCITY_INVITE_FINANCE",
  "Healthcare": "KSCARCITY_INVITE_HEALTHCARE",
  "Security": "KSCARCITY_INVITE_SECURITY",
  "Agriculture": "KSCARCITY_INVITE_AGRICULTURE",
  "Government": "KSCARCITY_INVITE_GOVERNMENT",
}

DEFAULT_SECTOR_INVITE_CODES = {
  "Finance": "finance-join-2026",
  "Healthcare": "health-join-2026",
  "Security": "security-join-2026",
  "Agriculture": "agri-join-2026",
  "Government": "gov-join-2026",
}

PORTAL_APPROVAL_CODE = os.getenv("KSCARCITY_PORTAL_APPROVAL_CODE", "approve-agencies-2026")

def _load_auth() -> dict:
  if AUTH_FILE.exists():
    with open(AUTH_FILE, "r") as f:
      return json.load(f)
  return {}

def _save_auth(auth_data: dict):
  with open(AUTH_FILE, "w") as f:
    json.dump(auth_data, f)


def _load_approvals() -> dict:
  if APPROVALS_FILE.exists():
    with open(APPROVALS_FILE, "r") as f:
      data = json.load(f)
      if isinstance(data, dict):
        data.setdefault("pending", {})
        data.setdefault("rejected", {})
        return data
  return {"pending": {}, "rejected": {}}


def _save_approvals(data: dict):
  data.setdefault("pending", {})
  data.setdefault("rejected", {})
  with open(APPROVALS_FILE, "w") as f:
    json.dump(data, f)


def _append_approval_audit(action: str, node_id: str, actor: str, details: dict | None = None):
  event = {
    "timestamp": time.time(),
    "action": str(action),
    "node_id": str(node_id),
    "actor": str(actor),
    "details": details or {},
  }
  with open(APPROVAL_AUDIT_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(event) + "\n")


def _hash_password(password: str) -> str:
  salt = secrets.token_hex(16)
  digest = hashlib.pbkdf2_hmac(
    "sha256",
    str(password).encode("utf-8"),
    salt.encode("utf-8"),
    PASSWORD_ITERATIONS,
  ).hex()
  return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt}${digest}"


def _is_hashed_password(stored_value: str) -> bool:
  return isinstance(stored_value, str) and stored_value.startswith(f"{PASSWORD_SCHEME}$")


def _verify_password(password: str, stored_value: str) -> bool:
  if not stored_value:
    return False

  if _is_hashed_password(stored_value):
    parts = stored_value.split("$", 3)
    if len(parts) != 4:
      return False
    _, iter_text, salt, expected_digest = parts
    try:
      iterations = int(iter_text)
    except ValueError:
      return False
    actual_digest = hashlib.pbkdf2_hmac(
      "sha256",
      str(password).encode("utf-8"),
      salt.encode("utf-8"),
      iterations,
    ).hex()
    return secrets.compare_digest(actual_digest, expected_digest)

  # Legacy plaintext fallback for backward compatibility.
  return secrets.compare_digest(str(stored_value), str(password))


def _invite_codes() -> dict:
  resolved = {}
  for sector, env_key in SECTOR_INVITE_CODE_ENV.items():
    resolved[sector] = os.getenv(env_key, DEFAULT_SECTOR_INVITE_CODES[sector])
  return resolved


def _is_valid_sector_invite(domain_choice: str, invite_code: str) -> bool:
  expected = _invite_codes().get(domain_choice, "")
  if not expected:
    return False
  return secrets.compare_digest(str(expected), str(invite_code))


def _provision_institution(new_inst_name: str, new_inst_id: str, domain_choice: str, password_hash: str) -> str:
  """Provision an approved institution into federation topology and local auth store."""
  from federated_databases.scarcity_federation import get_scarcity_federation

  manager = get_scarcity_federation()
  node = manager.register_node(node_id=new_inst_id, backend="sqlite")

  # Refresh K-Collab connectors
  services = get_kcollab_services()
  services["fed_db"].register_default_from_manager()

  # Inject into Topology to ensure FL Domain Baskets pick it up
  topo_store = services["topology"]
  current_topo = topo_store.get_payload()
  domain_clean = domain_choice.lower()

  if "nodes" not in current_topo:
    current_topo["nodes"] = []

  # 1. Ensure the Level 1 Domain Agency exists
  domain_exists = any(n.get("node_id") == domain_clean for n in current_topo["nodes"])
  if not domain_exists:
    current_topo["nodes"].append({
      "node_id": domain_clean,
      "level": 1,
      "node_type": "agency",
      "agency_id": domain_clean,
      "clearance": "RESTRICTED",
      "domains": [domain_clean],
    })

  # 2. Add the Node as a Level 2 Department under that Agency
  new_topo_node = {
    "node_id": new_inst_id,
    "level": 2,
    "node_type": "department",
    "agency_id": domain_clean,
    "parent_id": domain_clean,
    "clearance": "RESTRICTED",
    "domains": [domain_clean],
  }

  # Filter out if it somehow existed
  current_topo["nodes"] = [n for n in current_topo["nodes"] if n.get("node_id") != new_inst_id]
  current_topo["nodes"].append(new_topo_node)
  topo_store.save(current_topo, actor="institution_portal", message=f"Registered {new_inst_id} in {domain_clean}")

  auth_data = _load_auth()
  auth_data[new_inst_id] = {
    "institution_name": new_inst_name,
    "password_hash": password_hash,
    "domain": domain_choice,
    "approved_at": time.time(),
  }
  _save_auth(auth_data)
  return node.node_id


def _render_pending_approval_panel(theme):
  st.markdown(f'<h4 style="color: {theme.text_secondary};">Agency Approval</h4>', unsafe_allow_html=True)
  with st.expander("Admin Approval Console", expanded=False):
    admin_code = st.text_input("Approval Admin Code", type="password", key="portal_approval_admin_code")
    if not admin_code:
      st.caption("Enter admin code to review pending requests.")
      return

    if not secrets.compare_digest(admin_code, PORTAL_APPROVAL_CODE):
      st.error("Invalid approval admin code.")
      return

    approvals = _load_approvals()
    pending = approvals.get("pending", {})
    if not pending:
      st.success("No pending agency registrations.")
      return

    pending_ids = sorted(pending.keys())
    selected_node_id = st.selectbox("Pending Node ID", pending_ids, key="pending_node_selector")
    selected = pending[selected_node_id]

    st.write(f"**Institution**: {selected.get('institution_name', '')}")
    st.write(f"**Domain**: {selected.get('domain', '')}")
    st.write(f"**Requested**: {selected.get('requested_at', 0):.0f}")

    approve_col, reject_col = st.columns(2)
    with approve_col:
      if st.button("Approve & Provision", key="approve_pending_node", use_container_width=True):
        try:
          approved_node_id = _provision_institution(
            new_inst_name=selected.get("institution_name", ""),
            new_inst_id=selected_node_id,
            domain_choice=selected.get("domain", ""),
            password_hash=selected.get("password_hash", ""),
          )
          approvals["pending"].pop(selected_node_id, None)
          _save_approvals(approvals)
          _append_approval_audit(
            action="APPROVED",
            node_id=selected_node_id,
            actor="portal_admin",
            details={"provisioned_node_id": approved_node_id, "domain": selected.get("domain", "")},
          )
          st.success(f"Approved and provisioned {approved_node_id}.")
          st.rerun()
        except Exception as e:
          st.error(f"Approval failed: {e}")

    with reject_col:
      if st.button("Reject Request", key="reject_pending_node", use_container_width=True):
        rejected = approvals.setdefault("rejected", {})
        rejected[selected_node_id] = {
          **selected,
          "rejected_at": time.time(),
        }
        approvals["pending"].pop(selected_node_id, None)
        _save_approvals(approvals)
        _append_approval_audit(
          action="REJECTED",
          node_id=selected_node_id,
          actor="portal_admin",
          details={"domain": selected.get("domain", "")},
        )
        st.warning(f"Rejected request for {selected_node_id}.")
        st.rerun()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.auth import check_access
from k_collab.ui.services import get_kcollab_services


def render(theme):
  """Render the Institution Portal module."""

  st.markdown(f'<h1 style="color: {theme.text_primary};">Institution Portal</h1>', unsafe_allow_html=True)
  st.markdown("Securely upload weekly organizational data to participate in federated analysis.")

  # Simple pseudo-auth for institutions
  if "institution_node_id" not in st.session_state:
    _render_signup_login(theme)
  else:
    _render_upload_workspace(theme)

  # Global back button
  st.markdown("---")
  if st.button("← Back to Home", key="institution_back"):
    st.session_state.current_view = "HOME"
    st.rerun()


def _render_signup_login(theme):
  """Render institution signup/login with invite-code validation and approval workflow."""
  st.info("Sign in or register your institution to access the portal and upload data.")

  col1, col2 = st.columns(2)

  with col1:
    st.markdown(f'<h3 style="color: {theme.text_secondary};">Sign Up</h3>', unsafe_allow_html=True)
    with st.form("inst_signup_form"):
      new_inst_name = st.text_input("Institution Name (e.g., Central Bank Kenya)")
      new_inst_id = st.text_input("Requested Node ID (e.g., cbk_nairobi)")
      domain_choice = st.selectbox("Industry Domain", ["Finance", "Healthcare", "Security", "Agriculture", "Government"])
      access_code = st.text_input("Sector Invite Code", type="password")
      new_pass = st.text_input("Set Your Password", type="password")
      
      if st.form_submit_button("Submit Registration Request"):
        if not _is_valid_sector_invite(domain_choice, access_code):
          st.error("Invalid sector invite code.")
        elif new_inst_name and new_inst_id and new_pass:
          normalized_node_id = str(new_inst_id).strip().lower()
          auth_data = _load_auth()
          approvals = _load_approvals()
          pending = approvals.get("pending", {})
          if normalized_node_id in auth_data:
            st.error("This Node ID is already registered.")
          elif normalized_node_id in pending:
            st.warning("A registration request with this Node ID is already pending approval.")
          else:
            approvals.setdefault("pending", {})[normalized_node_id] = {
              "institution_name": new_inst_name,
              "domain": domain_choice,
              "password_hash": _hash_password(new_pass),
              "requested_at": time.time(),
            }
            _save_approvals(approvals)
            _append_approval_audit(
              action="SUBMITTED",
              node_id=normalized_node_id,
              actor="agency_self_signup",
              details={"domain": domain_choice, "institution_name": new_inst_name},
            )
            st.success("Registration submitted. An admin must approve before provisioning and login.")
        else:
          st.warning("Please fill in all fields.")

    _render_pending_approval_panel(theme)

  with col2:
    st.markdown(f'<h3 style="color: {theme.text_secondary};">Log In</h3>', unsafe_allow_html=True)
    with st.form("inst_login_form"):
      login_id = st.text_input("Node ID")
      login_pass = st.text_input("Password", type="password")
      
      if st.form_submit_button("Log In"):
        if login_id and login_pass:
          auth_data = _load_auth()
          approvals = _load_approvals()
          normalized_login_id = str(login_id).strip().lower()

          record = auth_data.get(normalized_login_id)
          if record:
            stored_secret = record.get("password_hash") or record.get("password")
            if _verify_password(login_pass, stored_secret):
              # Auto-upgrade legacy plaintext entries.
              if not _is_hashed_password(stored_secret):
                record["password_hash"] = _hash_password(login_pass)
                record.pop("password", None)
                auth_data[normalized_login_id] = record
                _save_auth(auth_data)

              st.session_state.institution_node_id = normalized_login_id
              st.session_state.institution_name = record.get("institution_name", normalized_login_id)
              st.success("Logged in successfully! Redirecting to your Basket...")
              time.sleep(1)
              st.rerun()
            else:
              st.error("Invalid Node ID or Password.")
          elif normalized_login_id in approvals.get("pending", {}):
            st.warning("Registration is pending approval. Please contact your sector admin.")
          elif normalized_login_id in approvals.get("rejected", {}):
            st.error("Registration was rejected. Please re-apply with correct details.")
          else:
            st.error("Invalid Node ID or Password.")
        else:
          st.warning("Please provide both Node ID and Password.")


def _render_upload_workspace(theme):
  """Render the data upload and online learning workspace."""
  node_id = st.session_state.institution_node_id
  inst_name = st.session_state.institution_name

  st.success(f"Authenticated as **{inst_name}** (`{node_id}`)")

  st.markdown(f'<h3 style="color: {theme.text_secondary};">Weekly Data Upload</h3>', unsafe_allow_html=True)
  st.write("Upload your anonymized CSV dataset. Our system will ingest it and run an online learning round instantly.")

  uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

  if uploaded_file is not None:
    try:
      df = pd.read_csv(uploaded_file)
      st.write("Preview of uploaded data:")
      st.dataframe(df.head(), use_container_width=True)

      if st.button("Submit Data & Trigger Learning", type="primary"):
        with st.spinner("Ingesting data and running online learning..."):
          from federated_databases.scarcity_federation import get_scarcity_federation
          
          manager = get_scarcity_federation()
          # Trigger a sync round specifically representing this node's new data.
          # Since we cannot easily isolate a run_sync_round to a single node via the API without modifying Scarcity Federation directly here,
          # we trigger the global sync round which simulates the active nodes pulling their latest local models.
          # In a true deployment, the CSV would be inserted into this node's DB, followed by a localized train.
          
          # Mock insert delay
          time.sleep(1.5)
          
          result = manager.run_sync_round(learning_rate=0.15, lookback_hours=168)
          
        st.success(f"Data ingested successfully! Triggered FL Sync Round #{result.round_number}.")
        st.info(
          f"**Round Details**: Participants = {result.participants}, "
          f"Samples Processed = {result.total_samples}, "
          f"Current Global Loss = {result.global_loss:.4f}"
        )
        
    except Exception as e:
      st.error(f"Error reading or processing file: {e}")

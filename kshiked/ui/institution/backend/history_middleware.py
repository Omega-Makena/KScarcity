import json
import os
import uuid
import hashlib
from datetime import datetime
from functools import wraps
import streamlit as st

from kshiked.ui.institution.backend.database import get_connection

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def save_analysis_history(analysis_type: str, sector: str, input_params: dict, result_payload: dict, summary: str):
    """Saves analysis history to DB and writes full payload to disk."""
    if not st.session_state.get("authenticated"):
        return
        
    # Deduplicate reactive Streamlit runs (prevent spamming history on slider moves/tab changes)
    param_str = json.dumps(input_params, cls=DatetimeEncoder, sort_keys=True)
    run_hash = hashlib.md5(f"{analysis_type}_{param_str}".encode()).hexdigest()
    
    if "saved_history_hashes" not in st.session_state:
        st.session_state["saved_history_hashes"] = set()
        
    if run_hash in st.session_state["saved_history_hashes"]:
        return  # Already saved this exact run configuration in this session
        
    st.session_state["saved_history_hashes"].add(run_hash)
    
    user_id = st.session_state.get("user_id", 0)
    username = st.session_state.get("username", "Unknown")
    role = st.session_state.get("role", "Unknown")
    basket_id = st.session_state.get("basket_id", 0)
    
    run_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Save full payload to disk
    file_path = os.path.join(HISTORY_DIR, f"{run_id}.json")
    try:
        with open(file_path, "w") as f:
            json.dump({
                "id": run_id,
                "timestamp": timestamp,
                "type": analysis_type,
                "inputs": input_params,
                "results": result_payload
            }, f, cls=DatetimeEncoder)
    except Exception as e:
        print(f"Failed to save history payload to disk: {e}")
        return
        
    # Save metadata to SQLite
    try:
        input_str = json.dumps(input_params, cls=DatetimeEncoder)
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_history (
                    id, timestamp, analysis_type, user_id, username, role, 
                    basket_id, sector, input_parameters, result_summary, full_result_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, timestamp, analysis_type, user_id, username, role, 
                  basket_id, sector, input_str, summary, file_path))
            conn.commit()
    except Exception as e:
        print(f"Failed to insert history to DB: {e}")

def get_analysis_history(role: str, user_id: int, basket_id: int):
    """Fetches analysis history based on RBAC rules."""
    query = "SELECT * FROM analysis_history"
    params = []
    
    if role == "Executive":
        # Can see everything
        pass
    elif role == "Admin":
        # Can see everything in their basket
        query += " WHERE basket_id = ?"
        params.append(basket_id)
    elif role == "Spoke" or role == "Institution":
        # Can only see their own runs
        query += " WHERE user_id = ?"
        params.append(user_id)
    else:
        # Failsafe
        query += " WHERE user_id = ?"
        params.append(user_id)
        
    query += " ORDER BY timestamp DESC"
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"Failed to fetch history: {e}")
        return []

def get_full_analysis_result(file_path: str):
    """Loads the full payload from disk."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load history payload: {e}")
        return None

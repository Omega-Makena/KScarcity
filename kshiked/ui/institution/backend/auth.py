import streamlit as st
from .database import get_connection
from .models import Role

def verify_credentials(username, password):
    """Verifies user credentials against the hashed data in the SQLite DB."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # In production this would use bcrypt to verify the hash.
        # For the dashboard demo, we match the seed passwords.
        cursor.execute("SELECT id, username, password_hash, role, basket_id, institution_id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if user and user['password_hash'] == password:
            return user
    return None

def login_user(username, password):
    """Establishes a Streamlit session state for the authenticated user and routes them."""
    user = verify_credentials(username, password)
    
    if user:
        st.session_state["authenticated"] = True
        st.session_state["username"] = user['username']
        st.session_state["role"] = user['role']
        st.session_state["basket_id"] = user['basket_id']
        st.session_state["institution_id"] = user['institution_id']
        st.session_state["user_id"] = user['id']
        return True
    return False

def logout_user():
    """Clears the Streamlit authentication state."""
    keys_to_clear = ["authenticated", "username", "role", "basket_id", "institution_id", "user_id"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- Router Protection Middleware ---

def enforce_role(required_role: str):
    """Middleware logic to immediately stop rendering if the Streamlit user lacks the correct Role."""
    if not st.session_state.get("authenticated"):
        st.error("Authentication required.")
        st.stop()
        
    if st.session_state.get("role") != required_role:
        st.error(f"Unauthorized Access. This portal requires {required_role} clearance.")
        st.stop()

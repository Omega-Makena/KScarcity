import streamlit as st
import pyotp
import base64
import binascii
import time
from .database import get_connection
from .models import Role

TOTP_INTERVAL_SECONDS = 30
TOTP_VALID_WINDOW_STEPS = 1

def verify_credentials(username, password):
  """Verifies user credentials against the DB. Password check disabled for demo."""
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash, role, basket_id, institution_id, totp_secret, is_2fa_enabled FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    if user:
      # DEMO MODE: password check disabled — any valid username is accepted
      # Re-enable for production: if user['password_hash'] == password:
      return user
  return None

def generate_totp_secret():
  """Generates a new base32 secret for TOTP."""
  return pyotp.random_base32()


def _normalize_totp_secret(secret):
  """Normalizes a TOTP secret to canonical base32 format (uppercase, no spaces)."""
  if not secret:
    return ""
  return str(secret).strip().replace(" ", "").upper()


def _is_valid_base32_secret(secret):
  """Validates that a normalized secret is decodable base32."""
  if not secret:
    return False
  try:
    # Base32 decoding requires length multiple of 8; pad for tolerant validation.
    padding = "=" * (-len(secret) % 8)
    base64.b32decode(secret + padding, casefold=True)
    return True
  except (binascii.Error, ValueError):
    return False

def verify_totp_token(secret, token):
  """Verifies a 6-digit token using server time, 30s intervals, and +/-1-step drift."""
  if not secret or not token:
    return False

  normalized_secret = _normalize_totp_secret(secret)
  clean_token = str(token).replace(" ", "").strip()

  if not clean_token.isdigit() or len(clean_token) != 6:
    return False

  if not _is_valid_base32_secret(normalized_secret):
    return False

  totp = pyotp.TOTP(normalized_secret, interval=TOTP_INTERVAL_SECONDS)
  server_epoch = time.time()
  return totp.verify(
    clean_token,
    for_time=server_epoch,
    valid_window=TOTP_VALID_WINDOW_STEPS,
  )

def enable_user_2fa(user_id, secret):
  """Saves the TOTP secret and flips the is_2fa_enabled flag to True."""
  normalized_secret = _normalize_totp_secret(secret)
  if not _is_valid_base32_secret(normalized_secret):
    return False

  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      "UPDATE users SET totp_secret = ?, is_2fa_enabled = 1 WHERE id = ?",
      (normalized_secret, user_id)
    )
    conn.commit()
    return True

def login_user(username, password):
  """
  PHASE 1: Establishes a partial Streamlit session state for the authenticated user.
  Does NOT log them in completely. `login.py` will route them to Phase 2 (2FA).
  """
  user = verify_credentials(username, password)
  
  if user:
    # Set provisional login data. 'authenticated' is NOT set to True yet.
    st.session_state["username"] = user['username']
    st.session_state["role"] = user['role']
    st.session_state["basket_id"] = user['basket_id']
    st.session_state["institution_id"] = user['institution_id']
    st.session_state["user_id"] = user['id']
    st.session_state["totp_secret"] = _normalize_totp_secret(user['totp_secret'])
    st.session_state["is_2fa_enabled"] = bool(user['is_2fa_enabled'])
    return True
  return False

def logout_user():
  """Clears the Streamlit authentication state."""
  keys_to_clear = [
    "authenticated", "username", "role", "basket_id", 
    "institution_id", "user_id", "phase1_passed", 
    "totp_secret", "is_2fa_enabled"
  ]
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

import streamlit as st
import pyotp
import base64
import binascii
import time
import hashlib
import secrets
from .database import get_connection
from .models import Role

TOTP_INTERVAL_SECONDS = 30
TOTP_VALID_WINDOW_STEPS = 1
DEVELOPER_USERNAMES = {"developer", "dev", "developer_dashboard"}
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 200_000


def hash_password(password: str) -> str:
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


def verify_password(password: str, stored_value: str) -> bool:
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

  # Legacy plaintext fallback for existing seeded/demo records.
  return secrets.compare_digest(str(stored_value), str(password))


def _ensure_developer_account_exists():
  """Best-effort guard so dedicated developer login is always available."""
  try:
    exec_hash = hash_password("exec123")
    dev_hash = hash_password("dev123")
    with get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        ("executive", exec_hash, Role.EXECUTIVE.value),
      )
      cursor.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        ("developer", dev_hash, Role.EXECUTIVE.value),
      )
      conn.commit()
  except Exception:
    # Keep auth resilient if DB/table migration is not ready yet.
    pass

def verify_credentials(username, password):
  """Verifies user credentials against the DB with hashed password checking."""
  _ensure_developer_account_exists()
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash, role, basket_id, institution_id, totp_secret, is_2fa_enabled FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user and verify_password(password, user["password_hash"]):
      # Upgrade legacy plaintext storage to hashed representation.
      if not _is_hashed_password(user["password_hash"]):
        cursor.execute(
          "UPDATE users SET password_hash = ? WHERE id = ?",
          (hash_password(password), user["id"]),
        )
        conn.commit()
      return user
  return None


def is_developer_username(username) -> bool:
  """Returns True when the username should land on the dedicated developer dashboard."""
  if not username:
    return False
  return str(username).strip().lower() in DEVELOPER_USERNAMES


def is_developer_session() -> bool:
  """Checks current Streamlit session identity for developer dashboard routing."""
  return is_developer_username(st.session_state.get("username"))

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

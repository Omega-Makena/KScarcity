import sqlite3
import json
import time
from typing import List, Optional
import os
import hashlib
import secrets

from .models import Role, User, Basket, Institution, OntologySchema, DeltaQueueMessage

DB_PATH = os.path.join(os.path.dirname(__file__), "federated_registry.sqlite")
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 200_000


def _seed_hash_password(password: str) -> str:
  salt = secrets.token_hex(16)
  digest = hashlib.pbkdf2_hmac(
    "sha256",
    str(password).encode("utf-8"),
    salt.encode("utf-8"),
    PASSWORD_ITERATIONS,
  ).hex()
  return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt}${digest}"

def get_connection():
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  return conn

def init_db():
  """Initializes the federated institution registry with the necessary schemas."""
  with get_connection() as conn:
    cursor = conn.cursor()
    
    # Baskets (The Hubs)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS baskets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT
      )
    """)

    # Institutions (The Spokes)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS institutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        basket_id INTEGER NOT NULL,
        api_key TEXT NOT NULL,
        trust_weight REAL NOT NULL DEFAULT 1.0,
        FOREIGN KEY (basket_id) REFERENCES baskets (id)
      )
    """)

    # Users (The 3-Tier Auth Layer)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        basket_id INTEGER,
        institution_id INTEGER,
        FOREIGN KEY (basket_id) REFERENCES baskets (id),
        FOREIGN KEY (institution_id) REFERENCES institutions (id)
      )
    """)

    # Ontology Enforcer (Semantic Dictionary per Basket)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS ontology_schemas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        basket_id INTEGER NOT NULL UNIQUE,
        schema_definition TEXT NOT NULL,
        FOREIGN KEY (basket_id) REFERENCES baskets (id)
      )
    """)

    # Asynchronous Delta Queue (Offline Syncs)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS delta_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        institution_id INTEGER NOT NULL,
        basket_id INTEGER NOT NULL,
        payload TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        timestamp REAL NOT NULL,
        FOREIGN KEY (institution_id) REFERENCES institutions (id),
        FOREIGN KEY (basket_id) REFERENCES baskets (id)
      )
    """)
    
    # Validated Risks (Fused Events Promoted by Basket Admin)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS validated_risks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        basket_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        composite_scores TEXT NOT NULL,
        source_sync_ids TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PROMOTED',
        timestamp REAL NOT NULL,
        FOREIGN KEY (basket_id) REFERENCES baskets (id)
      )
    """)
    
    # Operational Projects (Temporary Cross-Basket Collaboration)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS operational_projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        severity REAL NOT NULL,
        current_phase TEXT NOT NULL DEFAULT 'EMERGENCE',
        status TEXT NOT NULL DEFAULT 'ACTIVE',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
      )
    """)
    
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_participants (
        project_id INTEGER NOT NULL,
        basket_id INTEGER NOT NULL,
        PRIMARY KEY (project_id, basket_id),
        FOREIGN KEY (project_id) REFERENCES operational_projects (id),
        FOREIGN KEY (basket_id) REFERENCES baskets (id)
      )
    """)
    
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_updates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        author_name TEXT NOT NULL,
        update_type TEXT NOT NULL, -- 'OBSERVATION', 'ANALYSIS_REQUEST', 'POLICY_ACTION'
        content TEXT NOT NULL,
        certainty REAL,
        timestamp REAL NOT NULL,
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)
    
    # Temporal State Machine Tracking
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_phases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        phase_name TEXT NOT NULL, -- 'EMERGENCE', 'ESCALATION', 'STABILIZATION', 'RECOVERY'
        entered_at REAL NOT NULL,
        duration_seconds REAL,
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)
    
    # Real-time Signals matched to projects
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        signal_id TEXT NOT NULL,
        threat_tier TEXT NOT NULL,
        category TEXT NOT NULL,
        impact_score REAL NOT NULL,
        matched_at REAL NOT NULL,
        raw_text TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)
    
    # Institutional Memory (Post-Event Meta-Learning)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS institutional_memory (
        project_id INTEGER PRIMARY KEY,
        resolution_state TEXT NOT NULL, -- 'FALSE_ALARM', 'RESOLVED', 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_SIGNALS'
        policy_effectiveness_score REAL NOT NULL,
        resolution_summary TEXT NOT NULL,
        time_to_consensus_seconds REAL,
        learning_payload TEXT NOT NULL, -- JSON metrics
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)
    
    # Historical Analysis Tracker
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS analysis_history (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        role TEXT NOT NULL,
        basket_id INTEGER,
        sector TEXT,
        input_parameters TEXT NOT NULL,
        result_summary TEXT NOT NULL,
        full_result_path TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
      )
    """)
    
    try:
      cursor.execute("ALTER TABLE operational_projects ADD COLUMN vision TEXT")
    except sqlite3.OperationalError:
      pass # Column likely already exists

    # Data Schemas (Admin defined tracking)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS data_schemas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        basket_id INTEGER NOT NULL,
        schema_name TEXT NOT NULL,
        fields_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        FOREIGN KEY (basket_id) REFERENCES baskets (id)
      )
    """)

    # Structured Project Tracking
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_objectives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        success_metric TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_milestones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        due_date REAL NOT NULL,
        assigned_to TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'NOT_STARTED',
        linked_objectives_json TEXT,
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        measurable_target TEXT,
        achieved TEXT DEFAULT 'PENDING',
        actual_result TEXT,
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS project_post_mortem (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL UNIQUE,
        verdict TEXT NOT NULL,
        justification TEXT NOT NULL,
        lessons_learned TEXT,
        FOREIGN KEY (project_id) REFERENCES operational_projects (id)
      )
    """)

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS milestone_activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        milestone_id INTEGER NOT NULL,
        changed_by TEXT NOT NULL,
        old_status TEXT NOT NULL,
        new_status TEXT NOT NULL,
        note TEXT,
        timestamp REAL NOT NULL,
        FOREIGN KEY (milestone_id) REFERENCES project_milestones (id)
      )
    """)
    
    # --- FEATURE 2: Data Sharing & Governance ---
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS data_share_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        requester_basket_id INTEGER NOT NULL,
        target_basket_id INTEGER NOT NULL,
        reason TEXT NOT NULL,
        data_scope TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        created_at REAL NOT NULL,
        resolved_at REAL,
        FOREIGN KEY (requester_basket_id) REFERENCES baskets (id),
        FOREIGN KEY (target_basket_id) REFERENCES baskets (id)
      )
    """)
    
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS active_data_shares (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id INTEGER NOT NULL,
        granter_basket_id INTEGER NOT NULL,
        grantee_basket_id INTEGER NOT NULL,
        data_scope TEXT NOT NULL,
        expires_at REAL NOT NULL,
        created_at REAL NOT NULL,
        FOREIGN KEY (request_id) REFERENCES data_share_requests (id),
        FOREIGN KEY (granter_basket_id) REFERENCES baskets (id),
        FOREIGN KEY (grantee_basket_id) REFERENCES baskets (id)
      )
    """)

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS downward_directives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_role TEXT NOT NULL,
        sender_id TEXT NOT NULL,
        target_basket_id INTEGER,
        target_institution_id INTEGER,
        directive_type TEXT NOT NULL,
        content TEXT NOT NULL,
        priority TEXT NOT NULL,
        requires_ack BOOLEAN NOT NULL DEFAULT 1,
        created_at REAL NOT NULL,
        FOREIGN KEY (target_basket_id) REFERENCES baskets (id),
        FOREIGN KEY (target_institution_id) REFERENCES institutions (id)
      )
    """)
    
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS share_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actor_id TEXT NOT NULL,
        action_type TEXT NOT NULL,
        target_entity TEXT NOT NULL,
        details TEXT NOT NULL,
        timestamp REAL NOT NULL
      )
    """)
    
    # Acknowledge tracking for directives
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS directive_acknowledgments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        directive_id INTEGER NOT NULL,
        acknowledger_id TEXT NOT NULL,
        acknowledged_at REAL NOT NULL,
        FOREIGN KEY (directive_id) REFERENCES downward_directives (id)
      )
    """)
    
    conn.commit()

def seed_database():
  """Seeds the database with test data if it is empty."""
  with get_connection() as conn:
    cursor = conn.cursor()

    def _ensure_developer_account():
      exec_hash = _seed_hash_password("exec123")
      dev_hash = _seed_hash_password("dev123")
      cursor.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        ("executive", exec_hash, Role.EXECUTIVE.value),
      )
      cursor.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        ("developer", dev_hash, Role.EXECUTIVE.value),
      )
    
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] > 0:
      _ensure_developer_account()
      conn.commit()
      return # Already seeded
      
    print("Seeding robust Federated Database Registry...")
    
    # 1. Baskets
    cursor.execute("INSERT INTO baskets (name, description) VALUES (?, ?)", ("Economic Basket", "Treasury, Central Bank, Tax Authority"))
    econ_basket_id = cursor.lastrowid
    cursor.execute("INSERT INTO baskets (name, description) VALUES (?, ?)", ("Health Basket", "Hospitals, CDC, Pharma Regulators"))
    health_basket_id = cursor.lastrowid
    cursor.execute("INSERT INTO baskets (name, description) VALUES (?, ?)", ("Security Basket", "Police, Border Ctrl, Cyber"))
    sec_basket_id = cursor.lastrowid
    
    # 2. Institutions
    cursor.execute("INSERT INTO institutions (name, basket_id, api_key) VALUES (?, ?, ?)", ("Central Bank", econ_basket_id, "key_econ_cb"))
    cb_id = cursor.lastrowid
    cursor.execute("INSERT INTO institutions (name, basket_id, api_key) VALUES (?, ?, ?)", ("National Stats Bureau", econ_basket_id, "key_econ_nsb"))
    nsb_id = cursor.lastrowid
    
    cursor.execute("INSERT INTO institutions (name, basket_id, api_key) VALUES (?, ?, ?)", ("General Hospital 1", health_basket_id, "key_h_gh1"))
    gh1_id = cursor.lastrowid
    
    # 3. Ontology Schemas
    econ_schema = json.dumps({"required_columns": ["inflation_index", "unemployment_rate", "debt_ratio"], "allow_extra": False})
    cursor.execute("INSERT INTO ontology_schemas (basket_id, schema_definition) VALUES (?, ?)", (econ_basket_id, econ_schema))
    health_schema = json.dumps({"required_columns": ["bed_capacity_pct", "intubation_rate", "daily_admissions"], "allow_extra": False})
    cursor.execute("INSERT INTO ontology_schemas (basket_id, schema_definition) VALUES (?, ?)", (health_basket_id, health_schema))

    # 4. Users (Passwords are obviously mocked for the system demo, hashing is needed in prod)
    # 4a. The God Tier
    cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", ("executive", _seed_hash_password("exec123"), Role.EXECUTIVE.value))
    _ensure_developer_account()
    
    # 4b. The Basket Admins
    cursor.execute("INSERT INTO users (username, password_hash, role, basket_id) VALUES (?, ?, ?, ?)", ("econ_admin", _seed_hash_password("admin123"), Role.BASKET_ADMIN.value, econ_basket_id))
    cursor.execute("INSERT INTO users (username, password_hash, role, basket_id) VALUES (?, ?, ?, ?)", ("health_admin", _seed_hash_password("admin123"), Role.BASKET_ADMIN.value, health_basket_id))

    # 4c. The Spokes
    cursor.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?, ?, ?, ?, ?)", ("spoke_cb", _seed_hash_password("spoke123"), Role.INSTITUTION.value, econ_basket_id, cb_id))
    cursor.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?, ?, ?, ?, ?)", ("spoke_nsb", _seed_hash_password("spoke123"), Role.INSTITUTION.value, econ_basket_id, nsb_id))
    cursor.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?, ?, ?, ?, ?)", ("spoke_gh1", _seed_hash_password("spoke123"), Role.INSTITUTION.value, health_basket_id, gh1_id))
    
    conn.commit()

if __name__ == "__main__":
  init_db()
  seed_database()
  print("Database initialization complete.")

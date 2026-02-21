import sqlite3
import json
import time
from typing import List, Optional
import os

from .models import Role, User, Basket, Institution, OntologySchema, DeltaQueueMessage

DB_PATH = os.path.join(os.path.dirname(__file__), "federated_registry.sqlite")

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
                update_type TEXT NOT NULL,  -- 'OBSERVATION', 'ANALYSIS_REQUEST', 'POLICY_ACTION'
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
        
        conn.commit()

def seed_database():
    """Seeds the database with test data if it is empty."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] > 0:
            return  # Already seeded
            
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
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", ("executive", "exec123", Role.EXECUTIVE.value))
        
        # 4b. The Basket Admins
        cursor.execute("INSERT INTO users (username, password_hash, role, basket_id) VALUES (?, ?, ?, ?)", ("econ_admin", "admin123", Role.BASKET_ADMIN.value, econ_basket_id))
        cursor.execute("INSERT INTO users (username, password_hash, role, basket_id) VALUES (?, ?, ?, ?)", ("health_admin", "admin123", Role.BASKET_ADMIN.value, health_basket_id))

        # 4c. The Spokes
        cursor.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?, ?, ?, ?, ?)", ("spoke_cb", "spoke123", Role.INSTITUTION.value, econ_basket_id, cb_id))
        cursor.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?, ?, ?, ?, ?)", ("spoke_nsb", "spoke123", Role.INSTITUTION.value, econ_basket_id, nsb_id))
        cursor.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?, ?, ?, ?, ?)", ("spoke_gh1", "spoke123", Role.INSTITUTION.value, health_basket_id, gh1_id))
        
        conn.commit()

if __name__ == "__main__":
    init_db()
    seed_database()
    print("Database initialization complete.")

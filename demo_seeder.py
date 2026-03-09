"""
K-Scarcity Demo Scenario Seeder — Cholera Crisis
=================================================
Populates the federated_registry.sqlite database with a full, realistic
multi-sector demo scenario based on the Kenya Cholera Outbreak.

Creates:
  3 Baskets  : Public Health, Water & Sanitation, Security & Displacement
  9 Spokes   : 3 per basket (county-level health facilities, water boards, security units)
  3 Admins   : One per basket
  1 Executive: National Security Intelligence Executive
  Delta Syncs: Realistic spoke observations with cholera data
  Risks      : Promoted cross-spoke risks visible to admins
  Projects   : 2 active cross-basket operational projects
  Memory     : 1 archived learning record

Run this ONCE to reset and re-seed the demo database:
  python demo_seeder.py

THEN log in with:
  Spokes   : turkana_health / spoke123 | garissa_water / spoke123 | etc.
  Admins   : health_admin / admin123 | water_admin / admin123 | security_admin / admin123
  Executive: director_general / exec123
"""

import sqlite3
import json
import time
import os
import sys
from pathlib import Path

# Ensure backend imports work when run from the project root
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

DB_PATH = os.path.join(
    project_root, "kshiked", "ui", "institution", "backend", "federated_registry.sqlite"
)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# =============================================================================
# STEP 0: WIPE existing demo data while preserving schema
# =============================================================================
def reset_db():
    print("[RESET] Wiping existing data...")
    with get_conn() as conn:
        cur = conn.cursor()
        # Delete in reverse FK dependency order
        for table in [
            "institutional_memory", "project_updates", "project_phases",
            "project_participants", "operational_projects",
            "validated_risks", "delta_queue", "ontology_schemas",
            "users", "institutions", "baskets"
        ]:
            cur.execute(f"DELETE FROM {table}")
            cur.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
        conn.commit()
    print("[RESET] Done.\n")

# =============================================================================
# STEP 1: BASKETS (the 3 sector hubs)
# =============================================================================
def seed_baskets(cur):
    baskets = [
        ("Public Health Intelligence",
         "County-level health facilities, disease surveillance units, and the CDC Kenya. "
         "Responsible for tracking cholera incidence, health facility capacity, and patient flows."),
        ("Water & Sanitation Authority",
         "Regional water boards and WASH programme coordinators managing contamination "
         "indices, borehole quality, and sanitation coverage across high-risk counties."),
        ("Security & Displacement Command",
         "Kenya Police regional units, UNHCR displacement monitoring, and NIS border "
         "intelligence. Tracks security incidents, health worker obstruction, and IDP flows."),
    ]
    ids = []
    for name, desc in baskets:
        cur.execute("INSERT INTO baskets (name, description) VALUES (?,?)", (name, desc))
        ids.append(cur.lastrowid)
        print(f"  [BASKET] {name} → id={ids[-1]}")
    return ids  # health_bid, water_bid, security_bid


# =============================================================================
# STEP 2: INSTITUTIONS (9 spokes — real county-level entities)
# =============================================================================
def seed_institutions(cur, health_bid, water_bid, security_bid):
    # Spoke institutions mapped to actual cholera seed counties
    spokes = [
        # --- PUBLIC HEALTH BASKET ---
        ("Turkana County Health Department",  health_bid,   "KEY_HEALTH_TURKANA"),
        ("Garissa County Referral Hospital",  health_bid,   "KEY_HEALTH_GARISSA"),
        ("Marsabit Disease Surveillance Unit",health_bid,   "KEY_HEALTH_MARSABIT"),

        # --- WATER & SANITATION BASKET ---
        ("Northern Water Works Development Agency", water_bid, "KEY_WATER_NWWDA"),
        ("Tana & Athi Rivers Development Authority", water_bid,"KEY_WATER_TARDA"),
        ("Coast Water Services Board",         water_bid,   "KEY_WATER_COAST"),

        # --- SECURITY & DISPLACEMENT BASKET ---
        ("Turkana NPS Regional Command",       security_bid,"KEY_SEC_TURKANA"),
        ("NFD Security Intelligence Unit",     security_bid,"KEY_SEC_NFD"),
        ("UNHCR Northern Kenya Displacement",  security_bid,"KEY_SEC_UNHCR"),
    ]
    ids = []
    for name, bid, key in spokes:
        cur.execute(
            "INSERT INTO institutions (name, basket_id, api_key) VALUES (?,?,?)",
            (name, bid, key)
        )
        ids.append(cur.lastrowid)
        print(f"  [SPOKE]  {name} → id={ids[-1]}")
    return ids  # list of 9 institution ids


# =============================================================================
# STEP 3: ONTOLOGY SCHEMAS (field definitions per basket)
# =============================================================================
def seed_ontologies(cur, health_bid, water_bid, security_bid):
    schemas = [
        (health_bid, {
            "required_columns": [
                "cholera_cases", "health_facility_capacity",
                "daily_admissions", "ors_kits_available"
            ],
            "allow_extra": True,
            "description": "Daily epidemiological report from county health facility"
        }),
        (water_bid, {
            "required_columns": [
                "contamination_index", "borehole_quality_pct",
                "households_served", "rainfall_mm"
            ],
            "allow_extra": True,
            "description": "WASH field report — water quality and sanitation metrics"
        }),
        (security_bid, {
            "required_columns": [
                "security_incident_rate", "health_worker_obstruction",
                "displaced_population", "misinformation_signal"
            ],
            "allow_extra": True,
            "description": "Security and displacement intelligence brief"
        }),
    ]
    for bid, schema in schemas:
        cur.execute(
            "INSERT INTO ontology_schemas (basket_id, schema_definition) VALUES (?,?)",
            (bid, json.dumps(schema))
        )
    print("  [SCHEMA] Ontologies defined for all 3 baskets")


# =============================================================================
# STEP 4: USERS (3 admins, 9 spokes, 1 executive)
# =============================================================================
def seed_users(cur, health_bid, water_bid, security_bid, inst_ids):
    tk_h, ga_h, ma_h, nw_w, ta_w, co_w, tk_s, nf_s, un_s = inst_ids

    users = [
        # --- EXECUTIVE ---
        ("director_general", "exec123", "EXECUTIVE",      None,         None),

        # --- BASKET ADMINS ---
        ("health_admin",     "admin123","BASKET_ADMIN",   health_bid,   None),
        ("water_admin",      "admin123","BASKET_ADMIN",   water_bid,    None),
        ("security_admin",   "admin123","BASKET_ADMIN",   security_bid, None),

        # --- SPOKES: HEALTH ---
        ("turkana_health",   "spoke123","INSTITUTION",    health_bid,   tk_h),
        ("garissa_health",   "spoke123","INSTITUTION",    health_bid,   ga_h),
        ("marsabit_health",  "spoke123","INSTITUTION",    health_bid,   ma_h),

        # --- SPOKES: WATER ---
        ("nwwda_water",      "spoke123","INSTITUTION",    water_bid,    nw_w),
        ("tarda_water",      "spoke123","INSTITUTION",    water_bid,    ta_w),
        ("coast_water",      "spoke123","INSTITUTION",    water_bid,    co_w),

        # --- SPOKES: SECURITY ---
        ("turkana_security", "spoke123","INSTITUTION",    security_bid, tk_s),
        ("nfd_security",     "spoke123","INSTITUTION",    security_bid, nf_s),
        ("unhcr_northern",   "spoke123","INSTITUTION",    security_bid, un_s),
    ]

    uid_map = {}
    for username, pwd, role, bid, iid in users:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?,?,?,?,?)",
            (username, pwd, role, bid, iid)
        )
        uid_map[username] = cur.lastrowid
        print(f"  [USER]   {username:20s} | {role:15s} | basket={bid} inst={iid}")
    return uid_map


# =============================================================================
# STEP 5: DELTA QUEUE — spoke insights (the data flowing upward)
# =============================================================================
def seed_delta_queue(cur, health_bid, water_bid, security_bid, inst_ids):
    """
    Simulate 7 days of spoke observations being uploaded to the delta queue.
    Each observation mirrors the cholera synthetic data for realism.
    """
    tk_h, ga_h, ma_h, nw_w, ta_w, co_w, tk_s, nf_s, un_s = inst_ids
    base_ts = time.time() - 7 * 86400  # 7 days ago
    day = 86400
    sync_ids = []

    def push(iid, bid, payload, days_ago):
        ts = base_ts + (7 - days_ago) * day
        cur.execute(
            "INSERT INTO delta_queue (institution_id, basket_id, payload, status, timestamp) VALUES (?,?,?,'PENDING',?)",
            (iid, bid, json.dumps(payload), ts)
        )
        sid = cur.lastrowid
        sync_ids.append(sid)
        return sid

    # --- HEALTH SYNCS ---
    push(tk_h, health_bid, {
        "summary": "Rapid onset diarrhoeal disease, 147 cases in 72 hours. ORS stockouts in 3 facilities.",
        "county": "Turkana", "cholera_cases": 147, "daily_admissions": 62,
        "health_facility_capacity": 0.31, "ors_kits_available": 12,
        "severity_score": 0.78, "anomaly_detected": True,
        "anomaly_type": "CASE_SPIKE", "trend": "ACCELERATING",
        "propagation_risk": "HIGH — Lodwar General at 34% capacity",
        "historical_comparison": "Highest case count in 14 months for this county",
        "so_what": "If trend continues 4 more days, facility capacity breaches 10%, patient diversion begins",
    }, 6)

    push(ga_h, health_bid, {
        "summary": "Garissa County reports 89 confirmed cholera cases. 2 deaths recorded. Influx from Dadaab camp.",
        "county": "Garissa", "cholera_cases": 89, "daily_admissions": 34,
        "health_facility_capacity": 0.45, "ors_kits_available": 67,
        "severity_score": 0.62, "anomaly_detected": True,
        "anomaly_type": "DISPLACEMENT_INFLUX", "trend": "RISING",
        "propagation_risk": "MEDIUM — Dadaab IDP camp connection identified",
        "historical_comparison": "2nd highest since 2021 rainy season event",
        "so_what": "Displacement from Turkana county is seeding secondary outbreak here",
    }, 5)

    push(ma_h, health_bid, {
        "summary": "Marsabit District Hospital overwhelmed. Médecins Sans Frontières requested.",
        "county": "Marsabit", "cholera_cases": 203, "daily_admissions": 91,
        "health_facility_capacity": 0.18, "ors_kits_available": 4,
        "severity_score": 0.91, "anomaly_detected": True,
        "anomaly_type": "FACILITY_COLLAPSE", "trend": "CRITICAL",
        "propagation_risk": "CRITICAL — Below 20% capacity. MSF intervention requested.",
        "historical_comparison": "No comparable crisis in recorded history for this county",
        "so_what": "Without intervention in 48 hours, untreated cholera case fatality rate rises from 1% to 25%",
    }, 4)

    # --- WATER SYNCS ---
    push(nw_w, water_bid, {
        "summary": "Contamination index in Turkana boreholes at 0.82. Flash flood event on Day 3 washed faecal matter into Turkwell river.",
        "counties": ["Turkana", "Samburu"], "contamination_index": 0.82,
        "borehole_quality_pct": 0.14, "households_served": 12400,
        "rainfall_mm": 48.3, "severity_score": 0.84,
        "anomaly_detected": True, "anomaly_type": "CONTAMINATION_SPIKE",
        "so_what": "Turkwell river serves as drinking source for ~45,000 households downstream",
    }, 6)

    push(ta_w, water_bid, {
        "summary": "Tana River contamination at 0.67 — above safe threshold. Industrial waste suspected near Garissa town.",
        "counties": ["Garissa", "Tana River"], "contamination_index": 0.67,
        "borehole_quality_pct": 0.38, "households_served": 89000,
        "rainfall_mm": 12.1, "severity_score": 0.61,
        "anomaly_detected": True, "anomaly_type": "INDUSTRIAL_CONTAMINATION",
        "so_what": "Secondary water source for 89,000 households. If not treated, cholera transmission amplifies",
    }, 5)

    push(co_w, water_bid, {
        "summary": "Coast Water Services normal operations. Baseline report — no anomalies.",
        "counties": ["Mombasa", "Kilifi"], "contamination_index": 0.18,
        "borehole_quality_pct": 0.76, "households_served": 340000,
        "rainfall_mm": 6.5, "severity_score": 0.12,
        "anomaly_detected": False, "trend": "STABLE",
    }, 3)

    # --- SECURITY SYNCS ---
    push(tk_s, security_bid, {
        "summary": "Health worker convoy attacked in Lokichar sub-county. 2 nurses injured. Supply vehicle seized.",
        "county": "Turkana", "security_incident_rate": 9,
        "health_worker_obstruction": 0.71, "displaced_population": 8300,
        "misinformation_signal": 0.74, "severity_score": 0.82,
        "anomaly_detected": True, "anomaly_type": "HEALTH_WORKER_THREAT",
        "incident_type": "ARMED_OBSTRUCTION",
        "so_what": "Health worker safety risk will reduce intervention coverage. Estimated 40% reduction in treatment reach",
        "historical_comparison": "First such incident in Turkana in 18 months",
    }, 5)

    push(nf_s, security_bid, {
        "summary": "Misinformation spread via WhatsApp in Garissa — 'cholera vaccine causes infertility' narrative reaching rural communities.",
        "counties": ["Garissa", "Wajir", "Mandera"], "security_incident_rate": 3,
        "health_worker_obstruction": 0.41, "displaced_population": 14200,
        "misinformation_signal": 0.83, "severity_score": 0.69,
        "anomaly_detected": True, "anomaly_type": "MISINFORMATION_CLUSTER",
        "platform": "WhatsApp", "estimated_reach": 28000,
        "so_what": "Vaccine hesitancy is actively undermining ring vaccination programme in 3 counties",
    }, 4)

    push(un_s, security_bid, {
        "summary": "IDP count in Northern Kenya — 22,400 registered. Turkana → Marsabit movement corridor active.",
        "counties": ["Turkana", "Marsabit", "Isiolo"],
        "security_incident_rate": 2, "health_worker_obstruction": 0.10,
        "displaced_population": 22400, "misinformation_signal": 0.28,
        "severity_score": 0.54, "movement_corridor": "Turkana → Marsabit → Isiolo",
        "primary_driver": "Cholera outbreak + water scarcity",
        "so_what": "IDP movement is mechanically seeding the outbreak into previously unaffected counties",
    }, 3)

    print(f"  [DELTA]  {len(sync_ids)} spoke insights queued")
    return sync_ids


# =============================================================================
# STEP 6: VALIDATED RISKS (admin-promoted, visible to executive)  
# =============================================================================
def seed_validated_risks(cur, health_bid, water_bid, security_bid, sync_ids):
    ts_now = time.time()
    risks = []

    risks.append(cur.execute(
        "INSERT INTO validated_risks (basket_id, title, description, composite_scores, source_sync_ids, status, timestamp) VALUES (?,?,?,?,?,'PROMOTED',?)",
        (health_bid,
         "CRITICAL: Tri-County Cholera Outbreak — Collapse Risk in Marsabit",
         "Three northern counties (Turkana, Garissa, Marsabit) are simultaneously experiencing "
         "a cholera outbreak with exponential growth. Marsabit District Hospital is at 18% capacity "
         "with an imminent collapse threshold. Emergency medical supplies and clinical staff "
         "redeployment are urgently required within 48 hours to prevent uncontrolled CFR escalation.",
         json.dumps({
             "severity": 0.91, "trend": 0.88, "anomaly_confidence": 0.95,
             "threat_index": 0.89, "composite_risk": 0.91
         }),
         json.dumps(sync_ids[:3]),
         ts_now - 3600)
    ).lastrowid)

    risks.append(cur.execute(
        "INSERT INTO validated_risks (basket_id, title, description, composite_scores, source_sync_ids, status, timestamp) VALUES (?,?,?,?,?,'PROMOTED',?)",
        (water_bid,
         "HIGH: Turkwell River Contamination — 45,000 Households at Risk",
         "Flash flood event on Day 3 washed faecal matter into the Turkwell River, which serves "
         "as the primary water source for approximately 45,000 households across Turkana and Samburu. "
         "Borehole quality at 14%. If not chlorinated within 72 hours, downstream transmission "
         "will structurally amplify the current cholera outbreak by an estimated 3x.",
         json.dumps({
             "severity": 0.84, "trend": 0.77, "anomaly_confidence": 0.88,
             "threat_index": 0.81, "composite_risk": 0.84
         }),
         json.dumps(sync_ids[3:5]),
         ts_now - 7200)
    ).lastrowid)

    risks.append(cur.execute(
        "INSERT INTO validated_risks (basket_id, title, description, composite_scores, source_sync_ids, status, timestamp) VALUES (?,?,?,?,?,'PROMOTED',?)",
        (security_bid,
         "HIGH: Health Worker Safety Collapses in Turkana — Vaccine Programme Undermined",
         "Armed obstruction of health workers in Lokichar sub-county and simultaneous "
         "WhatsApp-based misinformation reaching 28,000 people are together causing the Ring "
         "Vaccination Programme to lose effectiveness. Without security escort for health teams, "
         "treatment coverage will drop 40% within 5 days. The misinformation cluster is independently "
         "suppressing uptake even in areas with secure access.",
         json.dumps({
             "severity": 0.82, "trend": 0.71, "anomaly_confidence": 0.86,
             "threat_index": 0.79, "composite_risk": 0.81
         }),
         json.dumps(sync_ids[6:8]),
         ts_now - 1800)
    ).lastrowid)

    print(f"  [RISK]   {len(risks)} risks promoted to executive level")
    return risks


# =============================================================================
# STEP 7: OPERATIONAL PROJECTS (cross-basket collaboration)
# =============================================================================
def seed_projects(cur, health_bid, water_bid, security_bid):
    ts_now = time.time()
    projects = []

    # Project 1: Active crisis response (ESCALATION phase)
    cur.execute(
        "INSERT INTO operational_projects (title, description, severity, current_phase, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("Operation CLEAR WATER — Emergency Turkwell River Chlorination",
         "Cross-agency emergency response to contamination of Turkwell River system. "
         "Coordinating NWWDA water treatment teams, health facility ORS pre-positioning, "
         "and security escort for response convoys in Turkana County.",
         0.87, "ESCALATION", "ACTIVE", ts_now - 5 * 86400, ts_now - 3600)
    )
    p1_id = cur.lastrowid
    cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at, duration_seconds) VALUES (?,?,?,?)",
                (p1_id, "EMERGENCE", ts_now - 5 * 86400, 2 * 86400))
    cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at) VALUES (?,?,?)",
                (p1_id, "ESCALATION", ts_now - 3 * 86400))
    for bid in [health_bid, water_bid, security_bid]:
        cur.execute("INSERT INTO project_participants (project_id, basket_id) VALUES (?,?)", (p1_id, bid))

    # Add realistic project thread
    thread = [
        (ts_now - 4.8*86400, "NWWDA Operations", "OBSERVATION",
         "Turkwell contamination confirmed at 0.82 index. Four boreholes in Lodwar sub-county shut down. "
         "Estimated 45,000 households affected. Emergency chlorination trucks dispatched from Kitale depot.",
         0.88),
        (ts_now - 4.2*86400, "Turkana Health Dept", "OBSERVATION",
         "Correlation confirmed: cholera case spike (147 cases) temporally correlated with river contamination event. "
         "ORS stockout in Lodwar General Hospital. Emergency restocking required within 24 hours.",
         0.91),
        (ts_now - 3.5*86400, "water_admin", "ANALYSIS_REQUEST",
         "Request to Public Health basket: Can you confirm transmission pathway via water? "
         "Need epidemiological link for Cabinet Secretary briefing by 17:00 today.",
         None),
        (ts_now - 3.1*86400, "health_admin", "OBSERVATION",
         "Confirmed: Genomic sequencing of cholera strain from Lodwar General matches upstream sample from "
         "Turkwell river. Transmission pathway is definitively water-borne. Confidence: 94%.",
         0.94),
        (ts_now - 2.5*86400, "Turkana NPS Command", "OBSERVATION",
         "Security escort for chlorination convoy approved. Two APCs deployed. However: convoy attacked at "
         "Lokichar junction by unknown armed persons. 2 nurses injured, supply truck seized. Convoy redirected.",
         0.78),
        (ts_now - 1.8*86400, "security_admin", "POLICY_ACTION",
         "POLICY ACTION LOGGED: Requested National Police Service deploy additional rapid response unit to "
         "Lokichar before operational resumption. CS Security briefed. KDF airlifting ORS kits to Marsabit. "
         "Misinformation counter-narrative launched via Citizen Radio and SMS broadcast.",
         0.82),
        (ts_now - 0.5*86400, "health_admin", "OBSERVATION",
         "Positive signal: 48-hour trend shows Turkana case count growth slowing from 22%/day to 9%/day. "
         "Chlorination of 2 boreholes completed. Marsabit still critical — MSF team on ground.",
         0.77),
    ]
    for ts, author, utype, content, cert in thread:
        cur.execute(
            "INSERT INTO project_updates (project_id, author_name, update_type, content, certainty, timestamp) VALUES (?,?,?,?,?,?)",
            (p1_id, author, utype, content, cert, ts)
        )
    projects.append(p1_id)
    print(f"  [PROJECT] 'Operation CLEAR WATER' → id={p1_id} (ESCALATION, 7 updates)")

    # Project 2: Misinformation containment (EMERGENCE phase)
    cur.execute(
        "INSERT INTO operational_projects (title, description, severity, current_phase, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("Operation SIGNAL CLEAR — Misinformation Containment across NFD",
         "Coordinated intelligence and communications effort to neutralise the vaccine misinformation "
         "campaign spreading through WhatsApp networks in the North-Eastern counties. Involves NIS "
         "digital intelligence, public health messaging, and community health worker activation.",
         0.69, "EMERGENCE", "ACTIVE", ts_now - 2 * 86400, ts_now - 3600)
    )
    p2_id = cur.lastrowid
    cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at) VALUES (?,?,?)",
                (p2_id, "EMERGENCE", ts_now - 2 * 86400))
    for bid in [health_bid, security_bid]:
        cur.execute("INSERT INTO project_participants (project_id, basket_id) VALUES (?,?)", (p2_id, bid))

    thread2 = [
        (ts_now - 1.9*86400, "NFD Security Intel", "OBSERVATION",
         "WhatsApp cluster identified: 'cholera vaccine causes infertility' narrative. "
         "Origin traced to 3 primary spreaders in Mandera, Wajir, and Garissa. Estimated reach: 28,000 recipients.",
         0.83),
        (ts_now - 1.5*86400, "security_admin", "ANALYSIS_REQUEST",
         "Requesting Public Health basket to confirm vaccine safety messaging and provide factual counter-narrative "
         "for NIS to distribute. Need content by 14:00.",
         None),
        (ts_now - 1.1*86400, "health_admin", "POLICY_ACTION",
         "POLICY ACTION: Kenya CDC issued public advisory. Counter-narrative translated to Somali, Borana, and "
         "Turkana. Citizen Radio broadcast scheduled for 18:00. Community Health Volunteers activated in 47 villages.",
         0.75),
    ]
    for ts, author, utype, content, cert in thread2:
        cur.execute(
            "INSERT INTO project_updates (project_id, author_name, update_type, content, certainty, timestamp) VALUES (?,?,?,?,?,?)",
            (p2_id, author, utype, content, cert, ts)
        )
    projects.append(p2_id)
    print(f"  [PROJECT] 'Operation SIGNAL CLEAR' → id={p2_id} (EMERGENCE, 3 updates)")
    return projects


# =============================================================================
# STEP 8: INSTITUTIONAL MEMORY (1 archived resolved project for learning section)
# =============================================================================
def seed_memory(cur, health_bid, water_bid):
    ts_old = time.time() - 45 * 86400  # 45 days ago

    cur.execute(
        "INSERT INTO operational_projects (title, description, severity, current_phase, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("Rift Valley Fever Surveillance — February 2024",
         "Cross-basket surveillance project following suspected Rift Valley Fever cluster in Laikipia. "
         "Resolved as FALSE ALARM after PCR confirmation ruled out RVF. Highlights need for faster "
         "laboratory turnaround time to reduce unnecessary alarm.",
         0.55, "RECOVERY", "ARCHIVED", ts_old, ts_old + 12 * 86400)
    )
    mem_pid = cur.lastrowid

    for phase, duration in [("EMERGENCE", 3*86400), ("ESCALATION", 5*86400),
                              ("STABILIZATION", 3*86400), ("RECOVERY", 86400)]:
        cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at, duration_seconds) VALUES (?,?,?,?)",
                    (mem_pid, phase, ts_old, duration))
        ts_old += duration

    for bid in [health_bid, water_bid]:
        cur.execute("INSERT INTO project_participants (project_id, basket_id) VALUES (?,?)", (mem_pid, bid))

    cur.execute(
        "INSERT INTO institutional_memory (project_id, resolution_state, policy_effectiveness_score, resolution_summary, time_to_consensus_seconds, learning_payload) VALUES (?,?,?,?,?,?)",
        (mem_pid, "FALSE_ALARM", 0.71,
         "PCR results after 9 days confirmed no RVF. Early symptoms matched RVF presentation but were "
         "caused by a livestock brucellosis cluster. Key learning: lab turnaround time of 9 days is "
         "too slow for outbreak confirmation. Recommend point-of-care RDT procurement for future cases.",
         12 * 86400,
         json.dumps({
             "days_to_resolve": 12,
             "false_alarm_cause": "Brucellosis misidentified as RVF",
             "recommendation": "Procure point-of-care RDT for Northern counties",
             "trust_weight_adjustments": {"health_basket": "+0.05", "water_basket": "0.00"}
         }))
    )
    print(f"  [MEMORY] Archived project 'Rift Valley Fever Surveillance' → id={mem_pid}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*65)
    print(" K-SCARCITY DEMO SEEDER — Cholera Crisis Scenario")
    print("="*65 + "\n")

    reset_db()

    with get_conn() as conn:
        cur = conn.cursor()

        print("[1/6] Creating Baskets...")
        health_bid, water_bid, security_bid = seed_baskets(cur)

        print("\n[2/6] Creating Spoke Institutions...")
        inst_ids = seed_institutions(cur, health_bid, water_bid, security_bid)

        print("\n[3/6] Creating Ontology Schemas...")
        seed_ontologies(cur, health_bid, water_bid, security_bid)

        print("\n[4/6] Creating Users (Spokes, Admins, Executive)...")
        seed_users(cur, health_bid, water_bid, security_bid, inst_ids)

        print("\n[5/6] Seeding Spoke Delta Queue Observations...")
        sync_ids = seed_delta_queue(cur, health_bid, water_bid, security_bid, inst_ids)

        print("\n   Promoting risks to Executive level...")
        seed_validated_risks(cur, health_bid, water_bid, security_bid, sync_ids)

        print("\n   Creating Operational Projects...")
        seed_projects(cur, health_bid, water_bid, security_bid)

        print("\n   Seeding Institutional Memory...")
        seed_memory(cur, health_bid, water_bid)

        conn.commit()

    print("\n" + "="*65)
    print(" DEMO SCENARIO SEEDED SUCCESSFULLY")
    print("="*65)
    print("""
  LOGIN CREDENTIALS
  ─────────────────────────────────────────────────────
  EXECUTIVE
    director_general  /  exec123

  BASKET ADMINS
    health_admin      /  admin123   (Public Health Intelligence)
    water_admin       /  admin123   (Water & Sanitation Authority)
    security_admin    /  admin123   (Security & Displacement Command)

  SPOKES — HEALTH BASKET
    turkana_health    /  spoke123   (Turkana County Health Dept)
    garissa_health    /  spoke123   (Garissa County Referral Hospital)
    marsabit_health   /  spoke123   (Marsabit Disease Surveillance Unit)

  SPOKES — WATER BASKET
    nwwda_water       /  spoke123   (Northern Water Works)
    tarda_water       /  spoke123   (Tana & Athi Rivers)
    coast_water       /  spoke123   (Coast Water Services Board)

  SPOKES — SECURITY BASKET
    turkana_security  /  spoke123   (Turkana NPS Regional Command)
    nfd_security      /  spoke123   (NFD Security Intelligence Unit)
    unhcr_northern    /  spoke123   (UNHCR Northern Kenya)

  ─────────────────────────────────────────────────────
  2 ACTIVE PROJECTS:
    - Operation CLEAR WATER  (ESCALATION)
    - Operation SIGNAL CLEAR (EMERGENCE)
  3 PROMOTED RISKS visible to Executive
  9 SPOKE OBSERVATIONS in delta queue
  1 ARCHIVED PROJECT in Institutional Memory
  ─────────────────────────────────────────────────────
  Refresh your Streamlit browser tab and log in!
""")

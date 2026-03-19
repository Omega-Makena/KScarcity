"""
K-Scarcity Demo Scenario Seeder — Full Cross-Sectoral Cholera Crisis
=====================================================================
Creates a complete end-to-end demo with 7 SECTORS (baskets), each with
its own sector admin and 3-4 spoke institutions. Every spoke has unique,
realistic data from the cholera outbreak scenario.

SECTOR STRUCTURE (Basket = Sector):
  1. Public Health         -> 4 spokes (county hospitals, surveillance units)
  2. Water & Sanitation    -> 3 spokes (water boards, WASH coordinators)
  3. Transport & Roads     -> 3 spokes (road authorities, logistics hubs)
  4. Security & Border     -> 3 spokes (police, NIS, border control)
  5. Displacement & IDP    -> 3 spokes (UNHCR, Red Cross, county coordinators)
  6. Food & Markets        -> 3 spokes (market data, food banks, agriculture)
  7. Communications        -> 3 spokes (NIS digital, telcos, community radio)

TOTAL:
  7 sector admins | 22 spoke users | 1 executive  = 30 users
  22 delta queue entries (unique data per spoke)
  7 sector-level promoted risks
  3 cross-sector operational projects
  1 archived institutional memory

Run:  python demo_seeder.py
"""

import sqlite3
import json
import time
import os
import sys
from pathlib import Path

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
# RESET
# =============================================================================
def reset_db():
    print("[RESET] Clearing existing data...")
    with get_conn() as conn:
        cur = conn.cursor()
        for table in [
            "institutional_memory", "project_updates", "project_phases",
            "project_participants", "operational_projects",
            "project_objectives", "project_milestones", "project_outcomes", 
            "project_post_mortem", "milestone_activity_log", "data_schemas", 
            "analysis_history",
            "validated_risks", "delta_queue", "ontology_schemas",
            "users", "institutions", "baskets"
        ]:
            try:
                cur.execute(f"DELETE FROM {table}")
                cur.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
            except Exception:
                pass # Safe skip if table doesn't exist yet
        conn.commit()
    print("[RESET] Done.\n")

# =============================================================================
# SECTOR / BASKET DEFINITIONS + ONTOLOGIES
# =============================================================================
SECTORS = [
    {
        "name": "Public Health",
        "description":
            "County health departments, disease surveillance units, and the Kenya CDC. "
            "Tracks cholera incidence, facility capacity, case fatality rates, and patient flows.",
        "ontology": {
            "required_columns": ["cholera_cases","health_facility_capacity",
                                  "daily_admissions","ors_kits_available","cfr_percent"],
            "description": "Daily epidemiological brief from county health facility"
        },
        "admin": ("health_admin", "admin123"),
        "spokes": [
            ("Turkana County Health Department",  "KEY_H_TURKANA", "turkana_health",
             "spoke123",
             {
                 "county": "Turkana", "cholera_cases": 312, "daily_admissions": 87,
                 "health_facility_capacity": 0.24, "ors_kits_available": 4,
                 "cfr_percent": 2.3, "severity_score": 0.91,
                 "anomaly_detected": True, "anomaly_type": "FACILITY_NEAR_COLLAPSE",
                 "trend": "CRITICAL",
                 "summary": "Lodwar General at 24% capacity. ORS nearly depleted. "
                            "2 deaths in last 24 hours. MSF access denied at Lokichar checkpoint.",
                 "so_what": "If capacity breaches 10% within 48h, untreated CFR rises from 2% to >20%.",
                 "historical_comparison": "Highest case count in Turkana in 18 months.",
                 "propagation_risk": "CRITICAL — 3 bordering counties receiving displaced persons from Turkana."
             }),
            ("Garissa County Referral Hospital", "KEY_H_GARISSA", "garissa_health",
             "spoke123",
             {
                 "county": "Garissa", "cholera_cases": 189, "daily_admissions": 54,
                 "health_facility_capacity": 0.41, "ors_kits_available": 34,
                 "cfr_percent": 0.8, "severity_score": 0.71,
                 "anomaly_detected": True, "anomaly_type": "DISPLACEMENT_INFLUX",
                 "trend": "RISING",
                 "summary": "IDP influx from Turkana driving secondary outbreak. "
                            "85 new cases from Dadaab camp sector in last 3 days.",
                 "so_what": "IDP movement is mechanically exporting the outbreak. "
                            "Without displacement control, Garissa will reach Turkana severity in ~9 days.",
                 "historical_comparison": "2nd highest Garissa case count on record (2021 was peak).",
                 "propagation_risk": "HIGH — Dadaab camp (428,000 residents) not yet breached."
             }),
            ("Marsabit Disease Surveillance Unit", "KEY_H_MARSABIT", "marsabit_health",
             "spoke123",
             {
                 "county": "Marsabit", "cholera_cases": 98, "daily_admissions": 31,
                 "health_facility_capacity": 0.57, "ors_kits_available": 82,
                 "cfr_percent": 0.4, "severity_score": 0.54,
                 "anomaly_detected": True, "anomaly_type": "CASE_SPIKE",
                 "trend": "EARLY_ESCALATION",
                 "summary": "New cluster detected in Moyale town. "
                            "Contaminated borehole confirmed as source. WASH team deployed.",
                 "so_what": "If borehole not isolated within 48h, projection shows 300+ cases by end of week.",
                 "historical_comparison": "Comparable to 2022 rainy season event — intervened early then.",
                 "propagation_risk": "MODERATE — Moyale is a border crossing into Ethiopia."
             }),
            ("Kenya National CDC Field Team", "KEY_H_CDC", "cdc_field",
             "spoke123",
             {
                 "team": "CDC Field Epidemiology", "active_counties": ["Turkana","Garissa","Marsabit","Isiolo"],
                 "genomic_confirmed": True, "strain": "O1 El Tor Ogawa",
                 "outbreak_declared": True, "ring_vaccination_coverage": 0.38,
                 "cholera_cases": 599, "health_facility_capacity": 0.0, "ors_kits_available": 0,
                 "severity_score": 0.89,
                 "anomaly_detected": True, "anomaly_type": "MULTI_COUNTY_OUTBREAK",
                 "summary": "National outbreak officially confirmed across 4 counties. "
                            "Ring vaccination at 38% coverage — below herd threshold. "
                            "Misinformation actively suppressing vaccination uptake.",
                 "so_what": "At current trajectory, 12 counties will be affected within 21 days.",
                 "propagation_risk": "CRITICAL — Cross-county transmission via shared water sources confirmed."
             }),
        ]
    },
    {
        "name": "Water & Sanitation",
        "description":
            "Regional water boards and WASH program coordinators. Manages contamination "
            "indices, borehole quality, treatment capacity, and sanitation infrastructure.",
        "ontology": {
            "required_columns": ["contamination_index","borehole_quality_pct",
                                  "households_served","rainfall_mm","treatment_capacity_pct"],
            "description": "WASH field report — water quality and treatment capacity"
        },
        "admin": ("water_admin", "admin123"),
        "spokes": [
            ("Northern Water Works Development Agency", "KEY_W_NWWDA", "nwwda_water",
             "spoke123",
             {
                 "region": "Turkana + Samburu", "contamination_index": 0.82,
                 "borehole_quality_pct": 0.14, "households_served": 45000,
                 "rainfall_mm": 48.3, "treatment_capacity_pct": 0.18,
                 "severity_score": 0.84, "anomaly_detected": True,
                 "anomaly_type": "FLASH_FLOOD_CONTAMINATION",
                 "event": "Turkwell River flash flood on Day 3. Faecal matter washed into system.",
                 "summary": "Turkwell River contamination at 0.82 — 8x safe threshold. "
                            "Flash flood on Day 3 caused faecal contamination of Turkwell riparian zone. "
                            "4 boreholes shut. 45,000 households without clean water.",
                 "so_what": "Without emergency chlorination within 72h, downstream transmission amplifies outbreak 3x.",
                 "emergency_action": "Chlorination trucks dispatched from Kitale. ETA: 18 hours."
             }),
            ("Tana & Athi Rivers Development Authority", "KEY_W_TARDA", "tarda_water",
             "spoke123",
             {
                 "region": "Garissa + Tana River", "contamination_index": 0.67,
                 "borehole_quality_pct": 0.38, "households_served": 89000,
                 "rainfall_mm": 12.1, "treatment_capacity_pct": 0.51,
                 "severity_score": 0.62, "anomaly_detected": True,
                 "anomaly_type": "INDUSTRIAL_CONTAMINATION",
                 "event": "Suspected industrial discharge near Garissa town upstream.",
                 "summary": "Tana River contamination at 0.67, above safe threshold of 0.3. "
                            "Industrial discharge suspected upstream — confirmed by smell and colour. "
                            "89,000 households using Tana as secondary water source.",
                 "so_what": "Tana River secondary contamination elevates Garissa outbreak risk independently of the primary cluster."
             }),
            ("Coast Water Services Board", "KEY_W_COAST", "coast_water",
             "spoke123",
             {
                 "region": "Mombasa + Kilifi + Kwale", "contamination_index": 0.17,
                 "borehole_quality_pct": 0.78, "households_served": 412000,
                 "rainfall_mm": 5.2, "treatment_capacity_pct": 0.84,
                 "severity_score": 0.09, "anomaly_detected": False, "trend": "STABLE",
                 "summary": "Normal operations across Coast region. No anomalies detected. "
                            "Monitoring Mombasa port area for potential introduction via seafaring networks.",
                 "so_what": "Coast region currently clean but high population density (412k) makes it a high-consequence zone if outbreak reaches here."
             }),
        ]
    },
    {
        "name": "Transport & Logistics",
        "description":
            "Kenya Roads Authority regional offices, KeNHA, and logistics hubs. Tracks "
            "road accessibility, convoy routing, and supply chain disruptions affecting aid delivery.",
        "ontology": {
            "required_columns": ["road_accessibility_score","blocked_routes",
                                  "aid_convoy_eta_hours","fuel_availability_index"],
            "description": "Transport intelligence report for aid and supply corridor management"
        },
        "admin": ("transport_admin", "admin123"),
        "spokes": [
            ("Kenya National Highways Authority — Northern Region", "KEY_T_KENHA", "kenha_northern",
             "spoke123",
             {
                 "region": "Turkana + Marsabit corridor", "road_accessibility_score": 0.31,
                 "blocked_routes": ["A2 Highway Lokichar junction — armed incident"],
                 "aid_convoy_eta_hours": 34, "fuel_availability_index": 0.44,
                 "severity_score": 0.77, "anomaly_detected": True,
                 "anomaly_type": "ROUTE_BLOCKAGE",
                 "summary": "A2 Highway blocked at Lokichar junction following armed attack on health convoy. "
                            "Alternative B8 route adds 14 hours to ETA. ORS resupply for Lodwar General delayed.",
                 "so_what": "Every 12-hour delay in ORS delivery at Lodwar General directly increases CFR by ~0.5%.",
                 "reroute_proposed": "South via Kitale-Kapenguria-Lokichar alternative (B8). Requires security escort."
             }),
            ("KeNHA Eastern Corridor", "KEY_T_KENHA_E", "kenha_eastern",
             "spoke123",
             {
                 "region": "Garissa + Wajir corridor", "road_accessibility_score": 0.58,
                 "blocked_routes": [],
                 "aid_convoy_eta_hours": 8, "fuel_availability_index": 0.72,
                 "severity_score": 0.28, "anomaly_detected": False, "trend": "STABLE",
                 "summary": "Eastern corridor (Garissa-Nairobi A3) fully operational. "
                            "Fuel supplies adequate at Garissa town depot. Convoy support available.",
                 "so_what": "Eastern corridor can serve as a logistics staging point for Northern Kenya resupply."
             }),
            ("Air Kenya Humanitarian Logistics Unit", "KEY_T_AIR", "airkenya_logistics",
             "spoke123",
             {
                 "region": "National airlifts", "road_accessibility_score": 0.91,
                 "blocked_routes": [],
                 "aid_convoy_eta_hours": 2, "fuel_availability_index": 0.88,
                 "severity_score": 0.21,
                 "active_airlifts": ["ORS to Marsabit airstrip — 3MT", "Medical staff to Lodwar — 8 personnel"],
                 "anomaly_detected": False,
                 "summary": "2 active humanitarian airlifts underway. ORS (3 metric tons) en route to Marsabit. "
                            "8 clinical staff airlifted to Lodwar. Marsabit airstrip confirmed operational.",
                 "so_what": "Airlift is currently the only reliable supply route to Turkana. Road closure makes this critical path."
             }),
        ]
    },
    {
        "name": "Security & Border",
        "description":
            "National Police Service regional commands, National Intelligence Service, and KRA border control. "
            "Tracks security incidents, health worker obstruction, and border crossing patterns.",
        "ontology": {
            "required_columns": ["security_incident_rate","health_worker_obstruction",
                                  "border_crossing_volume","active_threat_level"],
            "description": "Security intelligence brief — incidents and threat assessment"
        },
        "admin": ("security_admin", "admin123"),
        "spokes": [
            ("Turkana NPS Regional Command", "KEY_S_NPS_TK", "nps_turkana",
             "spoke123",
             {
                 "county": "Turkana", "security_incident_rate": 9,
                 "health_worker_obstruction": 0.71, "border_crossing_volume": 340,
                 "active_threat_level": "HIGH",
                 "severity_score": 0.82, "anomaly_detected": True,
                 "anomaly_type": "HEALTH_WORKER_ATTACK",
                 "incident_log": [
                     "Armed persons seized health supply vehicle at Lokichar junction",
                     "2 nurses injured, treated at Lodwar General",
                     "ORS stockpile valued at KES 1.2M seized"
                 ],
                 "summary": "Health worker convoy attacked at Lokichar, Day 5. 2 nurses injured. "
                            "Supply truck seized by unknown armed group. Convoy rerouted to B8 highway.",
                 "so_what": "Security incidents are reducing health worker coverage by ~40% in affected areas. "
                            "Health teams now refusing to operate without armed escort.",
                 "response_requested": "Request 2 APCs security escort for all future convoy operations in Turkana North."
             }),
            ("North Eastern Intelligence Unit (NIS)", "KEY_S_NIS", "nis_northeast",
             "spoke123",
             {
                 "counties": ["Garissa","Wajir","Mandera"], "security_incident_rate": 4,
                 "health_worker_obstruction": 0.42, "border_crossing_volume": 1240,
                 "active_threat_level": "MEDIUM",
                 "severity_score": 0.71, "anomaly_detected": True,
                 "anomaly_type": "MISINFORMATION_THREAT",
                 "misinfo_platforms": ["WhatsApp","TikTok"],
                 "misinfo_narrative": "Cholera vaccine causes infertility — circulating in Somali-language networks",
                 "estimated_reach": 28000,
                 "summary": "WhatsApp cluster spreading anti-vaccine narrative confirmed in NFD. "
                            "Origin traced to 3 primary spreaders in Mandera, Wajir, and Garissa. "
                            "TikTok amplification identified. Estimated 28,000 unique recipients.",
                 "so_what": "Vaccine hesitancy is actively undermining ring vaccination programme. "
                            "Without counter-narrative, coverage will stall below herd immunity threshold."
             }),
            ("KRA Border Control — Moyale Crossing", "KEY_S_KRA", "kra_moyale",
             "spoke123",
             {
                 "crossing": "Moyale (Kenya-Ethiopia border)", "security_incident_rate": 1,
                 "health_worker_obstruction": 0.05, "border_crossing_volume": 870,
                 "active_threat_level": "LOW",
                 "severity_score": 0.41, "anomaly_detected": True,
                 "anomaly_type": "DISEASE_EXPORT_RISK",
                 "cholera_positives_screened": 3,
                 "summary": "3 cholera-positive individuals intercepted at Moyale crossing attempting to enter Ethiopia. "
                            "Cross-border notification issued to Ethiopian health authorities. "
                            "Screening now active on all northbound lanes.",
                 "so_what": "Without border health screening, outbreak will cross into Ethiopia within 7-14 days. "
                            "This would trigger an international public health emergency."
             }),
        ]
    },
    {
        "name": "Displacement & IDP",
        "description":
            "UNHCR Kenya, Kenya Red Cross, and county displacement coordinators. Monitors IDP "
            "population movements, refugee camp health, and displacement corridors.",
        "ontology": {
            "required_columns": ["displaced_population","primary_displacement_cause",
                                  "destination_county","camp_overcrowding_index"],
            "description": "Displacement intelligence brief — population movement and camp status"
        },
        "admin": ("displacement_admin", "admin123"),
        "spokes": [
            ("UNHCR Kenya — Northern Operations", "KEY_D_UNHCR", "unhcr_northern",
             "spoke123",
             {
                 "region": "Turkana + Marsabit + Isiolo", "displaced_population": 22400,
                 "primary_displacement_cause": "Cholera outbreak + water scarcity",
                 "destination_county": "Marsabit", "camp_overcrowding_index": 0.78,
                 "severity_score": 0.68, "anomaly_detected": True,
                 "anomaly_type": "MASS_DISPLACEMENT",
                 "movement_corridor": "Turkana -> Marsabit -> Isiolo (southward flight)",
                 "summary": "22,400 registered IDPs in northern corridor. Turkana-Marsabit movement active. "
                            "Kakuma camp at 178% design capacity. Sanitation in overflow zones critical.",
                 "so_what": "IDP movement is mechanically exporting the cholera outbreak to previously unaffected counties. "
                            "Marsabit case spike directly correlates with IDP arrival dates."
             }),
            ("Kenya Red Cross — Northern Kenya", "KEY_D_REDCROSS", "redcross_north",
             "spoke123",
             {
                 "region": "Turkana + Garissa", "displaced_population": 8300,
                 "primary_displacement_cause": "Security threats + cholera fear",
                 "destination_county": "Garissa", "camp_overcrowding_index": 0.61,
                 "severity_score": 0.52, "anomaly_detected": True,
                 "anomaly_type": "URBAN_IDP_INFLUX",
                 "camps_active": ["Garissa Town transit camp", "Lodwar municipal shelter"],
                 "summary": "8,300 IDPs recorded in Red Cross transit camps. Garissa town receiving "
                            "urban IDP influx from rural Turkana. Shelter capacity at 161%. "
                            "Daily ration provision active but ORS supplies at 3-day stock.",
                 "so_what": "Urban IDPs are driving Garissa's secondary cholera cluster. Camp WASH conditions are inadequate."
             }),
            ("Turkana County Displacement Coordination", "KEY_D_COUNTY", "turkana_displacement",
             "spoke123",
             {
                 "county": "Turkana", "displaced_population": 5100,
                 "primary_displacement_cause": "Armed insecurity + borehole closure",
                 "destination_county": "Lodwar", "camp_overcrowding_index": 0.34,
                 "severity_score": 0.44, "anomaly_detected": False,
                 "voluntary_return_rate": 0.08,
                 "summary": "5,100 internally displaced within Turkana county itself (Lodwar urban). "
                            "Return rate to villages very low (8%) — safety concerns and no clean water. "
                            "County coordination centre operational with 14 NGO partners.",
                 "so_what": "Low return rate indicates the primary conditions (security + water) driving displacement are unresolved."
             }),
        ]
    },
    {
        "name": "Food & Markets",
        "description":
            "Kenya National Bureau of Statistics market monitors, WFP food security teams, "
            "and the Kenya Agricultural & Livestock Research Organisation. Tracks food price shocks, "
            "market disruption, and food security indices across affected counties.",
        "ontology": {
            "required_columns": ["market_activity_index","food_price_index",
                                  "staple_shortage_flag","food_insecure_population"],
            "description": "Food security and market intelligence report"
        },
        "admin": ("markets_admin", "admin123"),
        "spokes": [
            ("WFP Kenya — Northern Operations", "KEY_F_WFP", "wfp_northern",
             "spoke123",
             {
                 "region": "Turkana + Marsabit", "market_activity_index": 0.28,
                 "food_price_index": 2.1, "staple_shortage_flag": True,
                 "food_insecure_population": 340000,
                 "severity_score": 0.79, "anomaly_detected": True,
                 "anomaly_type": "MARKET_COLLAPSE",
                 "key_commodities_affected": ["Maize flour (+110%)", "Sugar (+80%)", "ORS (unavailable)"],
                 "summary": "Northern Kenya markets near collapse due to combined cholera + security crisis. "
                            "Maize flour prices have doubled. Supply convoys from Eldoret deterred by security. "
                            "340,000 people estimated food-insecure in Turkana alone.",
                 "so_what": "Market collapse during a disease outbreak creates a nutrition-infection spiral — "
                            "malnourished patients have 4x higher cholera CFR."
             }),
            ("KNBS Market Price Monitoring — NFD", "KEY_F_KNBS", "knbs_markets",
             "spoke123",
             {
                 "region": "Garissa + Wajir + Mandera", "market_activity_index": 0.54,
                 "food_price_index": 1.42, "staple_shortage_flag": False,
                 "food_insecure_population": 89000,
                 "severity_score": 0.47, "anomaly_detected": True,
                 "anomaly_type": "PRICE_SPIKE",
                 "maize_price_change_pct": 42, "sugar_price_change_pct": 35,
                 "summary": "NFD markets seeing 35-42% price increases for staples. "
                            "Price shock driven by supply chain disruption from security incidents on A2/B8 highways.",
                 "so_what": "At current trajectory, NFD markets will cross the food-insecurity threshold in 12-15 days."
             }),
            ("AgriMarket Kenya — Livestock Price Index", "KEY_F_AGRI", "agri_livestock",
             "spoke123",
             {
                 "region": "Turkana + Baringo + Samburu", "market_activity_index": 0.39,
                 "food_price_index": 0.66, "staple_shortage_flag": False,
                 "food_insecure_population": 45000,
                 "livestock_sale_rate_change_pct": -58,
                 "severity_score": 0.55, "anomaly_detected": True,
                 "anomaly_type": "LIVESTOCK_DISTRESS_SALE",
                 "summary": "Pastoralist communities offloading livestock at distress prices (-58% vs normal). "
                            "Forced fire-sales indicate acute income shock from disrupted trade routes and water scarcity.",
                 "so_what": "Livestock distress sales signal systemic livelihood collapse. "
                            "Pastoral communities losing asset base cannot recover without targeted intervention."
             }),
        ]
    },
    {
        "name": "Communications & Information",
        "description":
            "NIS Digital Intelligence, Safaricom emergency coordination, and Kenya community radio network. "
            "Tracks misinformation spread, platform activity, and counter-narrative effectiveness.",
        "ontology": {
            "required_columns": ["misinformation_signal","platform","estimated_reach",
                                  "counter_narrative_coverage","network_availability_index"],
            "description": "Information environment assessment — misinformation and counter-narrative tracking"
        },
        "admin": ("comms_admin", "admin123"),
        "spokes": [
            ("NIS Digital Threat Analysis Unit", "KEY_C_NIS", "nis_digital",
             "spoke123",
             {
                 "platforms": ["WhatsApp","TikTok","Facebook"], "misinformation_signal": 0.83,
                 "estimated_reach": 28000, "counter_narrative_coverage": 0.21,
                 "network_availability_index": 0.87,
                 "severity_score": 0.79, "anomaly_detected": True,
                 "anomaly_type": "COORDINATED_DISINFO_CAMPAIGN",
                 "primary_narrative": "Cholera vaccine causes infertility",
                 "secondary_narrative": "Government is poisoning boreholes",
                 "origin_tracing": "3 primary spreaders identified — Mandera, Wajir, Garissa",
                 "summary": "Coordinated misinformation campaign confirmed across WhatsApp, TikTok, and Facebook. "
                            "Two narratives running simultaneously. Origin traced to 3 primary accounts. "
                            "Counter-narrative covering only 21% of the affected network.",
                 "so_what": "Misinformation is suppressing vaccination uptake below herd immunity threshold. "
                            "Vaccine hesitancy adds an estimated 4-7 days to outbreak resolution timeline."
             }),
            ("Safaricom Emergency Network Operations", "KEY_C_SAFARICOM", "safaricom_ops",
             "spoke123",
             {
                 "region": "Northern Kenya", "misinformation_signal": 0.0,
                 "network_availability_index": 0.71, "counter_narrative_coverage": 0.0,
                 "estimated_reach": 0,
                 "towers_down": 3, "towers_affected_counties": ["Turkana North"],
                 "severity_score": 0.41, "anomaly_detected": True,
                 "anomaly_type": "NETWORK_DEGRADATION",
                 "summary": "3 cell towers in Turkana North are offline following the security incident at Lokichar. "
                            "Coverage gap is ~340 sq km. SMS health alerts cannot reach affected communities. "
                            "Repair team deployment requires security clearance.",
                 "so_what": "Network degradation in the outbreak epicenter is preventing SMS-based health advisories "
                            "from reaching 64,000 residents in the communication blackspot."
             }),
            ("Kenya Community Radio Network — Northern", "KEY_C_RADIO", "community_radio",
             "spoke123",
             {
                 "region": "Turkana + Samburu + Marsabit", "misinformation_signal": 0.0,
                 "counter_narrative_coverage": 0.67, "network_availability_index": 0.92,
                 "estimated_reach": 180000,
                 "severity_score": 0.0, "anomaly_detected": False,
                 "broadcasts_active": [
                     "Cholera prevention — Turkana FM (Turkana language)",
                     "Vaccine safety — Marsabit FM (Borana + Samburu)",
                     "Water safety — Isiolo FM (Amharic)"
                 ],
                 "summary": "Community radio counter-narrative broadcasts active in 3 counties covering 180,000 listeners. "
                            "Programming in local languages (Turkana, Borana, Samburu, Amharic). "
                            "Reaching communities that have no digital access.",
                 "so_what": "Radio is the most effective channel in areas with network downtime. "
                            "Scaling this is the fastest way to close the counter-narrative coverage gap."
             }),
        ]
    },
]

# =============================================================================
# BUILD EVERYTHING
# =============================================================================
def seed(conn):
    cur = conn.cursor()
    basket_ids = {}
    inst_ids   = {}
    all_sync_ids = []

    print("[1/7] Creating 7 Sector Baskets + Ontologies...")
    for sector in SECTORS:
        cur.execute("INSERT INTO baskets (name, description) VALUES (?,?)",
                    (sector["name"], sector["description"]))
        bid = cur.lastrowid
        basket_ids[sector["name"]] = bid

        cur.execute("INSERT INTO ontology_schemas (basket_id, schema_definition) VALUES (?,?)",
                    (bid, json.dumps(sector["ontology"])))
        print(f"    Sector: {sector['name']} -> basket_id={bid}")

    print("\n[2/7] Creating Spoke Institutions...")
    for sector in SECTORS:
        bid = basket_ids[sector["name"]]
        inst_ids[sector["name"]] = []
        for spoke_name, api_key, username, password, data in sector["spokes"]:
            cur.execute("INSERT INTO institutions (name, basket_id, api_key) VALUES (?,?,?)",
                        (spoke_name, bid, api_key))
            iid = cur.lastrowid
            inst_ids[sector["name"]].append((iid, username, data))
            print(f"    [{sector['name'][:6]}] {spoke_name} -> inst_id={iid}")

    print("\n[3/7] Creating Users (1 Executive + 7 Sector Admins + 22 Spokes)...")
    # Executive
    cur.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?,?,?,?,?)",
                ("director_general", "exec123", "EXECUTIVE", None, None))
    print(f"    EXECUTIVE: director_general / exec123")

    # Sector admins + spokes
    for sector in SECTORS:
        bid  = basket_ids[sector["name"]]
        aname, apwd = sector["admin"]
        cur.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?,?,?,?,?)",
                    (aname, apwd, "BASKET_ADMIN", bid, None))
        print(f"    ADMIN    : {aname} / {apwd}  [{sector['name']}]")

        for iid, username, data in inst_ids[sector["name"]]:
            cur.execute("INSERT INTO users (username, password_hash, role, basket_id, institution_id) VALUES (?,?,?,?,?)",
                        (username, "spoke123", "INSTITUTION", bid, iid))
            print(f"    SPOKE    : {username} / spoke123  [{sector['name'][:12]}]")

    print("\n[4/7] Seeding Delta Queue (unique observation per spoke)...")
    base_ts = time.time() - 5 * 86400
    for sector in SECTORS:
        bid = basket_ids[sector["name"]]
        for idx, (iid, username, data) in enumerate(inst_ids[sector["name"]]):
            ts = base_ts + (idx * 3600)
            payload = {"sector": sector["name"], "spoke": username, **data}
            cur.execute(
                "INSERT INTO delta_queue (institution_id, basket_id, payload, status, timestamp) VALUES (?,?,?,'PENDING',?)",
                (iid, bid, json.dumps(payload), ts)
            )
            all_sync_ids.append(cur.lastrowid)
    print(f"    {len(all_sync_ids)} unique spoke observations queued")

    print("\n[5/7] Promoting Sector Risks to Executive...")
    risk_ids = []
    promoted_risks = [
        ("Public Health",
         "CRITICAL: Tri-County Cholera Outbreak — Marsabit / Turkana / Garissa",
         "Kenya CDC has confirmed a nationally declared multi-county cholera outbreak. "
         "Turkana is at 24% facility capacity, Marsabit has a confirmed borehole-source cluster, "
         "and Garissa is receiving IDP-driven secondary seeding from Dadaab camp. Without emergency "
         "medical supply airlift and ring vaccination scale-up, the CDC projects outbreak expansion "
         "to 12 counties within 21 days.",
         {"severity": 0.91, "trend": 0.89, "confidence": 0.95, "composite_risk": 0.91}),
        ("Water & Sanitation",
         "HIGH: Turkwell River Flash Flood Contamination — 45,000 Households at Risk",
         "A flash flood on Day 3 washed faecal matter into the Turkwell River, the primary water source "
         "for 45,000 households in Turkana and Samburu counties. Contamination index at 0.82. "
         "Without emergency chlorination within 72 hours, river contamination will structurally amplify "
         "the cholera outbreak by a factor of approximately 3.",
         {"severity": 0.84, "trend": 0.77, "confidence": 0.88, "composite_risk": 0.84}),
        ("Transport & Logistics",
         "HIGH: A2 Highway Blockage — ORS Supply Chain to Lodwar Disrupted",
         "Armed attack on a health supply convoy at Lokichar junction has blocked the A2 Highway, "
         "the primary land supply route to Lodwar General Hospital. Each 12-hour delay in ORS delivery "
         "is projected to increase cholera CFR by approximately 0.5%. Air Kenya humanitarian airlift is "
         "the only active supply route — insufficient alone to meet demand.",
         {"severity": 0.77, "trend": 0.71, "confidence": 0.82, "composite_risk": 0.77}),
        ("Security & Border",
         "HIGH: Health Worker Safety Breakdown + Moyale Cross-Border Risk",
         "Armed groups are obstructing health worker convoys in Turkana North, reducing treatment coverage "
         "by 40%. Simultaneously, the Moyale border crossing has recorded 3 cholera-positive travellers "
         "intercepted en route to Ethiopia. Without border health screening and security escorts, the "
         "outbreak is projected to go international within 7-14 days.",
         {"severity": 0.81, "trend": 0.74, "confidence": 0.86, "composite_risk": 0.81}),
        ("Displacement & IDP",
         "HIGH: 22,400 IDPs in Northern Corridor Mechanically Spreading Outbreak",
         "Mass population displacement from outbreak counties is creating a biological vector effect. "
         "Turkana-origin IDPs arriving in Marsabit and Garissa are the confirmed source of secondary "
         "outbreak clusters. Kakuma camp is at 178% capacity with inadequate sanitation. "
         "Unless displacement conditions are addressed, the outbreak cannot be spatially contained.",
         {"severity": 0.72, "trend": 0.68, "confidence": 0.83, "composite_risk": 0.72}),
        ("Food & Markets",
         "MEDIUM: Market Collapse in Northern Kenya — Nutrition-Infection Spiral Risk",
         "Cholera outbreak combined with supply chain disruption has caused market activity indices to "
         "fall to 0.28 in Turkana/Marsabit. Maize flour prices are up 110%. 340,000 people are food-insecure. "
         "Malnourished patients have 4x higher cholera CFR — a nutrition-infection spiral is beginning "
         "to materialize and will compound outbreak severity if unaddressed.",
         {"severity": 0.71, "trend": 0.65, "confidence": 0.79, "composite_risk": 0.71}),
        ("Communications & Information",
         "HIGH: Coordinated Anti-Vaccine Disinfo Campaign Suppressing Outbreak Response",
         "A coordinated misinformation campaign ('vaccine causes infertility') has reached 28,000 people "
         "across WhatsApp, TikTok, and Facebook in NFD. Ring vaccination coverage has stalled at 38% — "
         "below the 60% herd immunity threshold needed to contain cholera. Without counter-narrative "
         "scale-up, vaccination programme effectiveness will continue to decline.",
         {"severity": 0.79, "trend": 0.72, "confidence": 0.84, "composite_risk": 0.79}),
    ]

    ts_now = time.time()
    for sector_name, title, desc, scores in promoted_risks:
        bid = basket_ids[sector_name]
        sector_sync_ids = [sid for sid, (iid, uname, _) in
                           zip(all_sync_ids, inst_ids.get(sector_name, []))
                           if True][:3]
        cur.execute(
            "INSERT INTO validated_risks (basket_id, title, description, composite_scores, source_sync_ids, status, timestamp) VALUES (?,?,?,?,?,'PROMOTED',?)",
            (bid, title, desc, json.dumps(scores), json.dumps(sector_sync_ids[:3]), ts_now - 1800)
        )
        risk_ids.append(cur.lastrowid)
        ts_now -= 600

    print(f"    {len(risk_ids)} risks promoted across all 7 sectors")

    print("\n[6/7] Creating 3 Cross-Sector Operational Projects...")
    ts = time.time()

    # PROJECT 1: Full cross-sector (ESCALATION)
    cur.execute(
        "INSERT INTO operational_projects (title, description, severity, current_phase, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("Operation CONTAIN — National Cholera Crisis Response",
         "Cross-sector emergency project coordinating Public Health, Water, Transport, Security, "
         "Displacement, and Food sectors to contain the northern Kenya cholera outbreak. "
         "Primary objectives: restore clean water access, secure ORS supply chain, "
         "scale ring vaccination, and prevent cross-border spread to Ethiopia.",
         0.91, "ESCALATION", "ACTIVE", ts - 5*86400, ts - 2*3600)
    )
    p1 = cur.lastrowid
    cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at, duration_seconds) VALUES (?,?,?,?)",
                (p1, "EMERGENCE", ts - 5*86400, 2*86400))
    cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at) VALUES (?,?,?)",
                (p1, "ESCALATION", ts - 3*86400))
    for sname in ["Public Health","Water & Sanitation","Transport & Logistics",
                  "Security & Border","Displacement & IDP"]:
        cur.execute("INSERT INTO project_participants (project_id, basket_id) VALUES (?,?)",
                    (p1, basket_ids[sname]))

    for entry in [
        (ts-4.8*86400, "Turkana Health Dept", "OBSERVATION",
         "CRITICAL: Lodwar General Hospital at 24% capacity. ORS nearly depleted. "
         "2 deaths in last 24 hours. Requesting immediate supply convoy and clinical staff reinforcement.", 0.91),
        (ts-4.3*86400, "NWWDA", "OBSERVATION",
         "Turkwell River contamination confirmed at 0.82 index — 8x safe threshold. "
         "Flash flood source verified. Emergency chlorination trucks dispatched from Kitale. ETA 18 hours.", 0.88),
        (ts-3.9*86400, "Turkana NPS", "OBSERVATION",
         "Security incident at Lokichar: health convoy attacked. 2 nurses injured. Supply truck seized. "
         "A2 Highway now effectively blocked for civilian convoys. Requesting APC escort authorisation.", 0.82),
        (ts-3.5*86400, "KeNHA Northern", "OBSERVATION",
         "A2 Highway blockage confirmed. B8 alternative adds 14 hours. Fuel at Lodwar depot at 44%. "
         "Air Kenya humanitarian airlift activated as primary supply route.", 0.77),
        (ts-3.0*86400, "UNHCR Kenya", "OBSERVATION",
         "22,400 IDPs registered in Turkana-Marsabit corridor. Movement confirmed as seeding outbreak "
         "into Marsabit. Kakuma camp at 178% capacity with critical WASH conditions.", 0.79),
        (ts-2.5*86400, "health_admin", "ANALYSIS_REQUEST",
         "CONVERGENCE ASSESSMENT: All 6 sectors are reporting concurrent crises. "
         "Requesting Cabinet Secretaries for Health, Water, and Interior convene emergency coordination meeting. "
         "What is the single most leveraged intervention to break the cascade?", None),
        (ts-2.0*86400, "water_admin", "OBSERVATION",
         "WATER SECTOR RESPONSE: First chlorination truck reached Lodwar boreholes. 2 of 4 boreholes restored. "
         "Contamination index for Lodwar town now 0.51 (down from 0.82). Trend improving.", 0.74),
        (ts-1.5*86400, "security_admin", "POLICY_ACTION",
         "POLICY ACTION: 2 APCs approved for health convoy escort. KDF airlift authorised for Marsabit. "
         "NIS counter-narrative operation launched — radio and SMS confirmed as primary channels. "
         "Moyale border health screening operational with WHO support.", 0.83),
        (ts-1.0*86400, "transport_admin", "OBSERVATION",
         "Air Kenya completed ORS airlift to Marsabit (3 metric tons). 8 clinical staff delivered to Lodwar. "
         "B8 route ORS convoy departed Kitale under escort. Estimated Lodwar arrival: 14:00 tomorrow.", 0.86),
        (ts-0.4*86400, "health_admin", "OBSERVATION",
         "POSITIVE SIGNAL: Turkana case growth rate slowing — 22%/day to 8%/day over 48 hours. "
         "Lodwar General now at 31% capacity (up from 24%). Garissa still rising. Marsabit stable. "
         "Ring vaccination coverage up to 44% in Turkana. Still below threshold — scaling required.", 0.79),
    ]:
        ts_e, auth_e, utype_e, content_e, cert_e = entry
        cur.execute(
            "INSERT INTO project_updates (project_id, author_name, update_type, content, certainty, timestamp) VALUES (?,?,?,?,?,?)",
            (p1, auth_e, utype_e, content_e, cert_e, ts_e)
        )
    print(f"    Project 1: 'Operation CONTAIN' (ESCALATION, 10 updates) -> id={p1}")

    # PROJECT 2: Misinformation (EMERGENCE)
    cur.execute(
        "INSERT INTO operational_projects (title, description, severity, current_phase, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("Operation SIGNAL CLEAR — Misinformation Containment NFD",
         "Coordinated digital intelligence, public health communications, and community radio effort "
         "to neutralise the anti-vaccine misinformation campaign in North-Eastern and Northern Kenya. "
         "NIS Digital, Kenya CDC, and community radio networks operating in parallel.",
         0.72, "EMERGENCE", "ACTIVE", ts - 2*86400, ts - 3600)
    )
    p2 = cur.lastrowid
    cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at) VALUES (?,?,?)",
                (p2, "EMERGENCE", ts - 2*86400))
    for sname in ["Security & Border","Communications & Information","Public Health"]:
        cur.execute("INSERT INTO project_participants (project_id, basket_id) VALUES (?,?)",
                    (p2, basket_ids[sname]))

    for entry in [
        (ts-1.9*86400, "NIS Digital Unit", "OBSERVATION",
         "Two concurrent disinfo narratives confirmed. 3 primary spreaders traced. 28,000 unique reach. "
         "Coordinated campaign indicators: posting synchrony, shared phrasing, rapid amplification.", 0.83),
        (ts-1.5*86400, "comms_admin", "ANALYSIS_REQUEST",
         "Requesting Public Health basket to provide verified vaccine safety content in Somali, "
         "Borana, and Turkana languages for NIS distribution by 14:00.", None),
        (ts-1.1*86400, "NIS Digital Unit", "POLICY_ACTION",
         "POLICY ACTION: Primary spreader accounts reported to platform providers. "
         "WhatsApp Business API activated for verified health messaging. "
         "TikTok Kenya coordination call scheduled for 15:00.", 0.74),
        (ts-0.8*86400, "Community Radio Network", "OBSERVATION",
         "Counter-narrative radio broadcasts active in Turkana FM, Marsabit FM, and Isiolo FM "
         "covering 180,000 listeners in local languages. Vaccine safety messaging confirmed accurate by Kenya CDC.", 0.81),
        (ts-0.3*86400, "Safaricom Ops", "OBSERVATION",
         "Turkana North tower repair team cleared by security. 2 of 3 towers restored. "
         "SMS health advisory broadcast reaching 64,000 previously unreachable residents.", 0.78),
    ]:
        ts_e, auth_e, utype_e, content_e, cert_e = entry
        cur.execute(
            "INSERT INTO project_updates (project_id, author_name, update_type, content, certainty, timestamp) VALUES (?,?,?,?,?,?)",
            (p2, auth_e, utype_e, content_e, cert_e, ts_e)
        )
    print(f"    Project 2: 'Operation SIGNAL CLEAR' (EMERGENCE, 5 updates) -> id={p2}")

    # PROJECT 3: Market & Food crisis (EMERGENCE)
    cur.execute(
        "INSERT INTO operational_projects (title, description, severity, current_phase, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("Operation FOOD BRIDGE — Emergency Market Stabilisation",
         "Cross-sector emergency market intervention to prevent a nutrition-infection spiral "
         "in Turkana and surrounding counties. WFP emergency food rations, strategic "
         "grain reserve release, and alternative supply corridor activation from Eldoret.",
         0.68, "EMERGENCE", "ACTIVE", ts - 86400, ts - 1800)
    )
    p3 = cur.lastrowid
    cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at) VALUES (?,?,?)",
                (p3, "EMERGENCE", ts - 86400))
    for sname in ["Food & Markets","Transport & Logistics","Displacement & IDP"]:
        cur.execute("INSERT INTO project_participants (project_id, basket_id) VALUES (?,?)",
                    (p3, basket_ids[sname]))

    for entry in [
        (ts-0.9*86400, "WFP Kenya", "OBSERVATION",
         "Northern Kenya market activity at 0.28 — effectively collapsed. Maize flour +110%, sugar +80%. "
         "340,000 food-insecure in Turkana. Malnourished cholera patients have 4x higher CFR.", 0.79),
        (ts-0.6*86400, "markets_admin", "ANALYSIS_REQUEST",
         "Requesting Transport sector to open Kitale-Kapenguria-Lodwar alternative corridor (B8) for "
         "commercial resupply vehicles, not just aid convoys. This would normalize supply chain.", None),
        (ts-0.3*86400, "transport_admin", "POLICY_ACTION",
         "POLICY ACTION: B8 corridor opened for commercial traffic under police escort. "
         "National Strategic Grain Reserve release of 2,000MT approved for Turkana. "
         "WFP emergency nutrition rations pre-positioned at Lodwar distribution point.", 0.81),
    ]:
        ts_e, auth_e, utype_e, content_e, cert_e = entry
        cur.execute(
            "INSERT INTO project_updates (project_id, author_name, update_type, content, certainty, timestamp) VALUES (?,?,?,?,?,?)",
            (p3, auth_e, utype_e, content_e, cert_e, ts_e)
        )
    print(f"    Project 3: 'Operation FOOD BRIDGE' (EMERGENCE, 3 updates) -> id={p3}")

    print("\n[7/7] Seeding Institutional Memory (1 archived project)...")
    ts_old = time.time() - 60*86400
    cur.execute(
        "INSERT INTO operational_projects (title, description, severity, current_phase, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("Rift Valley Fever Alert — February 2024",
         "Multi-sector surveillance project following suspected RVF cluster in Laikipia-Baringo. "
         "Resolved as FALSE ALARM after PCR confirmation ruled out Rift Valley Fever.",
         0.55, "RECOVERY", "ARCHIVED", ts_old, ts_old + 12*86400)
    )
    mem_pid = cur.lastrowid
    for phase, d in [("EMERGENCE",3*86400),("ESCALATION",5*86400),("STABILIZATION",3*86400),("RECOVERY",86400)]:
        cur.execute("INSERT INTO project_phases (project_id, phase_name, entered_at, duration_seconds) VALUES (?,?,?,?)",
                    (mem_pid, phase, ts_old, d))
        ts_old += d
    for sname in ["Public Health","Water & Sanitation"]:
        cur.execute("INSERT INTO project_participants (project_id, basket_id) VALUES (?,?)",
                    (mem_pid, basket_ids[sname]))
    cur.execute(
        "INSERT INTO institutional_memory (project_id, resolution_state, policy_effectiveness_score, resolution_summary, time_to_consensus_seconds, learning_payload) VALUES (?,?,?,?,?,?)",
        (mem_pid, "FALSE_ALARM", 0.71,
         "Brucellosis misidentified as RVF. Lab turnaround time of 9 days too slow for outbreak confirmation. "
         "Recommendation: procure point-of-care RDT kits for all Northern counties.",
         12*86400, json.dumps({
             "days_to_resolve": 12, "false_alarm_cause": "Brucellosis misidentified as RVF",
             "recommendation": "Procure POC RDT for Northern counties",
         }))
    )
    print(f"    Archived: 'Rift Valley Fever Alert' -> id={mem_pid}")
    conn.commit()

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + "="*70)
    print("  K-SCARCITY DEMO SEEDER v2 — Full Cross-Sectoral Cholera Crisis")
    print("="*70 + "\n")
    reset_db()
    with get_conn() as conn:
        seed(conn)

    print("\n" + "="*70)
    print("  DEMO SCENARIO SEEDED SUCCESSFULLY")
    print("="*70)
    print("""
  EXECUTIVE
    director_general  /  exec123

  SECTOR ADMINS (Basket Admins)
    health_admin       /  admin123  (Public Health)
    water_admin        /  admin123  (Water & Sanitation)
    transport_admin    /  admin123  (Transport & Logistics)
    security_admin     /  admin123  (Security & Border)
    displacement_admin /  admin123  (Displacement & IDP)
    markets_admin      /  admin123  (Food & Markets)
    comms_admin        /  admin123  (Communications & Information)

  SPOKES / Spoke123 for all
    HEALTH    : turkana_health | garissa_health | marsabit_health | cdc_field
    WATER     : nwwda_water | tarda_water | coast_water
    TRANSPORT : kenha_northern | kenha_eastern | airkenya_logistics
    SECURITY  : nps_turkana | nis_northeast | kra_moyale
    DISPLACE  : unhcr_northern | redcross_north | turkana_displacement
    MARKETS   : wfp_northern | knbs_markets | agri_livestock
    COMMS     : nis_digital | safaricom_ops | community_radio

  3 ACTIVE PROJECTS (visible to executive):
    - Operation CONTAIN        (ESCALATION — 10 updates, 5 sectors)
    - Operation SIGNAL CLEAR   (EMERGENCE  — 5 updates, 3 sectors)
    - Operation FOOD BRIDGE    (EMERGENCE  — 3 updates, 3 sectors)

  Refresh Streamlit and log in!
""")

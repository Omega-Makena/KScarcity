"""
Kenya Cholera Outbreak Synthetic Data Generator
================================================
Generates a realistic multi-sector synthetic dataset simulating a cholera
outbreak across all 47 Kenyan counties over a configurable number of days.

Cross-sector interactions are explicitly modelled (not independent random walks).
Output: cholera_outbreak_kenya.csv (full) and sample_output.csv (first 10 days)

Author: K-Scarcity Analytics Engine
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import date, timedelta

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
RANDOM_SEED          = 42           # For reproducibility
SIMULATION_DAYS      = 90           # Total simulation duration
START_DATE           = date(2024, 3, 1)  # Simulation start date

# Outbreak seed counties (county_id values, 0-indexed from COUNTIES list)
# Prioritising known water-stress counties: Turkana(28), Garissa(6), Marsabit(24), Kwale(15)
SEED_COUNTIES        = [28, 6, 24]  # 3 origin counties

# Noise level: fraction of signal added as Gaussian noise
NOISE_LEVEL          = 0.08

# Probability of an external shock occurring on any given day (per shock type)
SHOCK_PROBABILITY    = 0.012

# Cross-sector thresholds (used in interaction logic)
CAPACITY_CRISIS_THRESHOLD    = 0.30  # Below this → security risk
CAPACITY_RECOVERY_LAG        = 4     # Days before capacity starts recovering
MISINFO_AMPLIFICATION_THRESH = 0.60  # Above this → security incidents spike
DISPLACEMENT_THRESHOLD       = 5000  # Displaced persons above this → market drops
SPREAD_CONTAMINATION_THRESH  = 0.45  # County contamination must be above this to seed neighbours

# Urban county IDs with higher baseline health capacity and displacement absorption
# Nairobi(18), Mombasa(1), Kisumu(14), Nakuru(4), Eldoret/Uasin Gishu(22)
URBAN_COUNTIES = [18, 1, 14, 4, 22]


# =============================================================================
# KENYA COUNTY DATA
# (county_id is 0-indexed for internal use; matches adjacency list below)
# =============================================================================
COUNTIES = [
    # id, name,            population, baseline_water_risk, has_urban_center
    (0,  "Mombasa",         1208333,  0.25, True),
    (1,  "Kwale",            866820,  0.65, False),
    (2,  "Kilifi",          1453787,  0.55, False),
    (3,  "Tana River",       315943,  0.70, False),
    (4,  "Lamu",             143920,  0.50, False),
    (5,  "Taita-Taveta",     340671,  0.45, False),
    (6,  "Garissa",          841353,  0.80, False),   # High water stress
    (7,  "Wajir",            781263,  0.78, False),
    (8,  "Mandera",         1025756,  0.82, False),
    (9,  "Marsabit",         459785,  0.80, False),   # High water stress
    (10, "Isiolo",           268002,  0.68, False),
    (11, "Meru",            1545714,  0.35, False),
    (12, "Tharaka-Nithi",    393177,  0.40, False),
    (13, "Embu",             608599,  0.38, False),
    (14, "Kitui",           1136187,  0.62, False),
    (15, "Machakos",        1421932,  0.40, False),
    (16, "Makueni",          987653,  0.55, False),
    (17, "Nyandarua",        638289,  0.30, False),
    (18, "Nyeri",            759164,  0.28, False),
    (19, "Kirinyaga",        610411,  0.30, False),
    (20, "Murang'a",        1056640,  0.32, False),
    (21, "Kiambu",          2417735,  0.22, True),    # Nairobi metro
    (22, "Turkana",         1155257,  0.85, False),   # Highest water stress
    (23, "West Pokot",       621241,  0.62, False),
    (24, "Samburu",          310327,  0.72, False),
    (25, "Trans-Nzoia",      990341,  0.35, False),
    (26, "Uasin Gishu",      1163186, 0.30, True),   # Eldoret urban
    (27, "Elgeyo Marakwet",  454480,  0.42, False),
    (28, "Nandi",            885711,  0.38, False),
    (29, "Baringo",          666763,  0.58, False),
    (30, "Laikipia",         518560,  0.45, False),
    (31, "Nakuru",          2162202,  0.28, True),   # Nakuru urban
    (32, "Narok",            1157873, 0.50, False),
    (33, "Kajiado",         1117840,  0.45, False),
    (34, "Kericho",          902777,  0.28, False),
    (35, "Bomet",            875689,  0.35, False),
    (36, "Kakamega",        1867579,  0.38, False),
    (37, "Vihiga",           590013,  0.42, False),
    (38, "Bungoma",         1670570,  0.40, False),
    (39, "Busia",            893681,  0.55, False),
    (40, "Siaya",            993183,  0.48, False),
    (41, "Kisumu",          1155574,  0.35, True),   # Kisumu urban
    (42, "Homa Bay",        1131950,  0.52, False),
    (43, "Migori",           1116436, 0.50, False),
    (44, "Kisii",           1266860,  0.40, False),
    (45, "Nyamira",          598046,  0.42, False),
    (46, "Nairobi",         4397073,  0.15, True),   # Capital, best infrastructure
]

# Re-map so the actual county ID is consistent with list position
COUNTY_DF = pd.DataFrame(COUNTIES,
    columns=["county_id","county_name","population","baseline_water_risk","urban"])

# =============================================================================
# COUNTY ADJACENCY GRAPH
# (edges represent shared borders — based on Kenya's geographic adjacency)
# =============================================================================
ADJACENCY = [
    (0, 1), (0, 2),                          # Mombasa
    (1, 2), (1, 5),                          # Kwale
    (2, 3), (2, 4), (2, 14),                 # Kilifi
    (3, 4), (3, 6), (3, 14),                 # Tana River
    (4, 3),                                  # Lamu
    (5, 1), (5, 16), (5, 33),               # Taita-Taveta
    (6, 3), (6, 7), (6, 10), (6, 14),       # Garissa
    (7, 6), (7, 8), (7, 9), (7, 10),        # Wajir
    (8, 7), (8, 9),                          # Mandera
    (9, 7), (9, 8), (9, 10), (9, 22), (9, 24), # Marsabit
    (10, 6), (10, 7), (10, 9), (10, 11), (10, 24), (10, 29), (10, 30), # Isiolo
    (11, 10), (11, 12), (11, 13), (11, 24), (11, 30), # Meru
    (12, 11), (12, 13),                      # Tharaka-Nithi
    (13, 11), (13, 12), (13, 15), (13, 19),  # Embu
    (14, 2), (14, 3), (14, 6), (14, 15), (14, 16), # Kitui
    (15, 13), (15, 14), (15, 16), (15, 21), (15, 46), # Machakos
    (16, 14), (16, 15), (16, 33),            # Makueni
    (17, 18), (17, 20), (17, 21), (17, 30),  # Nyandarua
    (18, 17), (18, 19), (18, 20), (18, 30),  # Nyeri
    (19, 13), (19, 18), (19, 20),            # Kirinyaga
    (20, 17), (20, 18), (20, 19), (20, 21),  # Murang'a
    (21, 15), (21, 17), (21, 20), (21, 31), (21, 33), (21, 46), # Kiambu
    (22, 9), (22, 23), (22, 24), (22, 29),   # Turkana
    (23, 22), (23, 25), (23, 27), (23, 29),  # West Pokot
    (24, 9), (24, 10), (24, 11), (24, 22), (24, 30), # Samburu
    (25, 23), (25, 26), (25, 27), (25, 38),  # Trans-Nzoia
    (26, 25), (26, 27), (26, 28), (26, 36), (26, 38), # Uasin Gishu
    (27, 23), (27, 25), (27, 26), (27, 28), (27, 29), # Elgeyo Marakwet
    (28, 26), (28, 27), (28, 31), (28, 34), (28, 36), # Nandi
    (29, 10), (29, 22), (29, 23), (29, 24), (29, 27), (29, 30), # Baringo
    (30, 10), (30, 11), (30, 17), (30, 18), (30, 24), (30, 29), (30, 31), # Laikipia
    (31, 21), (31, 28), (31, 29), (31, 30), (31, 32), (31, 33), (31, 34), # Nakuru
    (32, 31), (32, 33), (32, 34), (32, 35), (32, 43),  # Narok
    (33, 5), (33, 16), (33, 21), (33, 31), (33, 32), (33, 46), # Kajiado
    (34, 28), (34, 31), (34, 32), (34, 35), (34, 44),  # Kericho
    (35, 32), (35, 34), (35, 42), (35, 43), (35, 44), (35, 45), # Bomet
    (36, 26), (36, 28), (36, 37), (36, 38), (36, 39), (36, 41), # Kakamega
    (37, 36), (37, 38), (37, 41),            # Vihiga
    (38, 25), (38, 26), (38, 36), (38, 39),  # Bungoma
    (39, 36), (39, 38), (39, 40), (39, 41),  # Busia
    (40, 39), (40, 41), (40, 43),            # Siaya
    (41, 36), (41, 37), (41, 39), (41, 40), (41, 42), (41, 44), # Kisumu
    (42, 35), (42, 40), (42, 41), (42, 43), (42, 44), (42, 45), # Homa Bay
    (43, 32), (43, 35), (43, 40), (43, 42), (43, 45), # Migori
    (44, 34), (44, 35), (44, 41), (44, 42), (44, 45), # Kisii
    (45, 35), (45, 42), (45, 43), (45, 44),  # Nyamira
    (46, 15), (46, 21), (46, 31), (46, 33),  # Nairobi
]

def build_adjacency_graph():
    """Build a NetworkX graph of Kenya county adjacency."""
    G = nx.Graph()
    G.add_nodes_from(range(47))
    G.add_edges_from(ADJACENCY)
    return G

def get_neighbours(G, county_id):
    """Return list of neighbouring county IDs."""
    return list(G.neighbors(county_id))


# =============================================================================
# SHOCK EVENT GENERATOR
# =============================================================================
def generate_shocks(rng, n_days):
    """
    Generate 2-3 random external shock events at random timesteps.
    Each shock is a dict: {day, type, target_counties, magnitude}
    Shock types:
      - 'road_closure'    : road_accessibility_score drops sharply
      - 'rainfall_event'  : contamination_index spikes in affected counties
      - 'misinfo_spike'   : misinformation_signal suddenly jumps
    """
    shock_types = ['road_closure', 'rainfall_event', 'misinfo_spike']
    n_shocks = rng.integers(2, 4)   # 2 or 3 shocks
    shocks = []
    used_days = set()
    for _ in range(n_shocks):
        # Ensure shocks don't cluster on day 0 or the last day
        day = int(rng.integers(5, n_days - 10))
        while day in used_days:
            day = int(rng.integers(5, n_days - 10))
        used_days.add(day)

        shock_type = rng.choice(shock_types)
        # Random target counties (1-4 affected)
        n_targets = int(rng.integers(1, 5))
        targets = list(rng.choice(47, size=n_targets, replace=False).astype(int))
        magnitude = float(rng.uniform(0.3, 0.7))
        shocks.append({
            "day":      day,
            "type":     shock_type,
            "targets":  targets,
            "magnitude": magnitude,
            "duration": int(rng.integers(2, 8))  # Lasts 2-7 days
        })
        print(f"  [SHOCK] Day {day:3d}: '{shock_type}' in counties "
              f"{[COUNTY_DF.loc[t,'county_name'] for t in targets]} "
              f"(magnitude={magnitude:.2f}, duration={shocks[-1]['duration']}d)")
    return shocks


def is_shock_active(shocks, day, shock_type, county_id):
    """Check if a particular shock is active on a given day for a county."""
    for s in shocks:
        if (s["type"] == shock_type and
                s["day"] <= day < s["day"] + s["duration"] and
                county_id in s["targets"]):
            return s["magnitude"]
    return 0.0


# =============================================================================
# MAIN SIMULATION
# =============================================================================
def simulate(seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    G   = build_adjacency_graph()
    n   = 47  # Number of counties
    T   = SIMULATION_DAYS

    print(f"\n{'='*60}")
    print(f"K-SCARCITY: Kenya Cholera Synthetic Outbreak Simulation")
    print(f"{'='*60}")
    print(f"  Seed counties  : "
          f"{[COUNTY_DF.loc[s,'county_name'] for s in SEED_COUNTIES]}")
    print(f"  Duration       : {T} days from {START_DATE}")
    print(f"  Random seed    : {seed}")
    print(f"\n  Generating external shocks:")
    shocks = generate_shocks(rng, T)

    # -------------------------------------------------------------------------
    # State arrays — shape (T, n)
    # -------------------------------------------------------------------------
    contamination      = np.zeros((T, n))   # 0-1
    cholera_cases      = np.zeros((T, n))   # int
    capacity           = np.zeros((T, n))   # 0-1 (health facility capacity)
    road_score         = np.zeros((T, n))   # 0-1
    security_rate      = np.zeros((T, n))   # int (incidents)
    worker_obstruction = np.zeros((T, n))   # 0-1
    displaced          = np.zeros((T, n))   # int
    displacement_dest  = np.zeros((T, n), dtype=int)  # county_id
    market_index       = np.zeros((T, n))   # 0-1
    food_price         = np.zeros((T, n))   # float (index, baseline 1.0)
    misinfo_signal     = np.zeros((T, n))   # 0-1

    # -------------------------------------------------------------------------
    # Baseline initialisation (Day 0)
    # -------------------------------------------------------------------------
    for cid in range(n):
        row = COUNTY_DF.loc[cid]
        pop  = row["population"]
        water_risk = row["baseline_water_risk"]
        is_urban   = row["urban"]

        # Water contamination: seeded counties start high, others use baseline
        if cid in SEED_COUNTIES:
            contamination[0, cid] = min(0.95, water_risk + rng.uniform(0.2, 0.4))
        else:
            contamination[0, cid] = water_risk * rng.uniform(0.5, 0.9)

        # Health capacity: urban counties start higher
        capacity[0, cid] = (rng.uniform(0.7, 0.9) if is_urban
                            else rng.uniform(0.4, 0.7))

        # Roads: drier/remote counties have poorer infrastructure
        road_score[0, cid] = (rng.uniform(0.65, 0.90) if is_urban
                              else max(0.1, 1.0 - water_risk + rng.uniform(-0.1, 0.1)))

        # Market: inversely correlated with remoteness
        market_index[0, cid] = (rng.uniform(0.7, 0.95) if is_urban
                                else rng.uniform(0.45, 0.75))
        food_price[0, cid]   = rng.uniform(0.95, 1.05)  # Near baseline

        # Seed cholera cases in origin counties
        if cid in SEED_COUNTIES:
            cholera_cases[0, cid] = int(pop * rng.uniform(0.0005, 0.002))
        else:
            cholera_cases[0, cid] = int(rng.integers(0, 5))

        # Security, displacement, misinfo: low at day 0
        security_rate[0, cid]      = int(rng.integers(0, 3))
        displaced[0, cid]          = 0
        misinfo_signal[0, cid]     = rng.uniform(0.02, 0.12)
        displacement_dest[0, cid]  = cid  # No displacement yet

    # -------------------------------------------------------------------------
    # SIMULATION LOOP
    # -------------------------------------------------------------------------
    cross_sector_events = []  # Track when interaction rules fire

    for t in range(1, T):
        for cid in range(n):
            row = COUNTY_DF.loc[cid]
            pop  = int(row["population"])
            is_urban = row["urban"]
            neighbours = get_neighbours(G, cid)

            # --- Shock overlays ---
            road_shock    = is_shock_active(shocks, t, 'road_closure', cid)
            rain_shock    = is_shock_active(shocks, t, 'rainfall_event', cid)
            misinfo_shock = is_shock_active(shocks, t, 'misinfo_spike', cid)

            # ===== 1. WATER & SANITATION: contamination_index =================
            # Contamination is driven by:
            #   - Yesterday's contamination (slow decay)
            #   - Rainfall shocks (spike contamination)
            #   - Inflowing displaced population from contaminated counties
            inflow_contamination = 0.0
            for nb in neighbours:
                if contamination[t-1, nb] > SPREAD_CONTAMINATION_THRESH:
                    # Neighbouring contamination bleeds over proportionally to
                    # road connectivity (poor roads slow spread)
                    inflow_contamination += (
                        contamination[t-1, nb] * 0.04 * road_score[t-1, cid])
            inflow_contamination = min(0.15, inflow_contamination)  # Cap bleed-over

            # Slow recovery without intervention
            decay = 0.98 if cid not in SEED_COUNTIES else 0.995
            contamination[t, cid] = np.clip(
                contamination[t-1, cid] * decay
                + inflow_contamination
                + rain_shock * 0.30          # Rainfall shock spikes contamination
                + rng.normal(0, NOISE_LEVEL * 0.1),
                0.0, 1.0)

            # ===== 2. CHOLERA CASES ===========================================
            # INTERACTION: cases grow as function of contamination AND displaced
            # population flowing in FROM neighbouring counties
            inflow_displaced = sum(
                displaced[t-1, nb] * 0.1 for nb in neighbours
            )
            transmission_rate = (
                0.15 * contamination[t-1, cid]
                + 0.00005 * inflow_displaced
                + rng.normal(0, NOISE_LEVEL * 0.05))

            new_cases = max(0, int(
                cholera_cases[t-1, cid] * transmission_rate
                + rng.poisson(contamination[t-1, cid] * 2)))

            # Recovery: ~3-5% cases resolve daily
            recoveries = int(cholera_cases[t-1, cid] * rng.uniform(0.03, 0.05))
            cholera_cases[t, cid] = max(0,
                int(cholera_cases[t-1, cid] + new_cases - recoveries))

            # ===== 3. HEALTH FACILITY CAPACITY ================================
            # INTERACTION: capacity degrades as cholera_cases rise (3-5 day lag)
            # INTERACTION: road_score controls recovery rate
            lag = min(t, rng.integers(3, 6))
            case_pressure = cholera_cases[t-lag, cid] / max(1, pop * 0.01)
            capacity_degradation = np.clip(case_pressure * 0.08, 0, 0.12)
            capacity_natural_recovery = (
                0.015 * road_score[t-1, cid])  # Better roads → faster restock

            capacity[t, cid] = np.clip(
                capacity[t-1, cid]
                - capacity_degradation
                + capacity_natural_recovery
                + rng.normal(0, NOISE_LEVEL * 0.03),
                0.05, 0.98)

            # ===== 4. TRANSPORT: road_accessibility_score =====================
            # Road shocks drop the score; slow recovery over days
            road_score[t, cid] = np.clip(
                road_score[t-1, cid]
                - road_shock * 0.4        # Shock drops road score
                + 0.01                    # Slow daily repair
                + rng.normal(0, NOISE_LEVEL * 0.05),
                0.05, 1.0)

            # ===== 5. SECURITY ================================================
            # INTERACTION: incidents rise when both capacity is critical AND
            # misinformation is high
            base_security_level = int(rng.poisson(1.5))
            fired_security_interaction = False
            if (capacity[t-1, cid] < CAPACITY_CRISIS_THRESHOLD and
                    misinfo_signal[t-1, cid] > MISINFO_AMPLIFICATION_THRESH):
                # Security crisis: incidents spike
                security_spike = int(rng.poisson(8))
                security_rate[t, cid] = base_security_level + security_spike
                fired_security_interaction = True
                cross_sector_events.append(
                    f"Day {t:3d} | {row['county_name']:20s} | "
                    f"SECURITY SPIKE: capacity={capacity[t-1,cid]:.2f} "
                    f"+ misinfo={misinfo_signal[t-1,cid]:.2f} → "
                    f"{security_rate[t,cid]} incidents")
            else:
                security_rate[t, cid] = base_security_level

            # Health worker obstruction increases with security incidents
            worker_obstruction[t, cid] = np.clip(
                0.1 * security_rate[t, cid] / 10.0
                + rng.normal(0, NOISE_LEVEL * 0.05),
                0.0, 1.0)

            # ===== 6. DISPLACEMENT ============================================
            # People flee toward counties with:
            #   - Higher road accessibility
            #   - Lower cholera burden
            # INTERACTION: market activity drops when displacement exceeds threshold
            if cholera_cases[t, cid] > 50 or security_rate[t, cid] > 5:
                # People flee proportional to crisis severity
                crisis_index = (
                    cholera_cases[t, cid] / max(1, pop * 0.005)
                    + security_rate[t, cid] / 10.0)
                new_displaced = int(min(
                    pop * 0.003 * crisis_index + rng.poisson(50),
                    pop * 0.05))  # Max 5% of population displaced at once

                # Choose destination: best-road + lowest-case neighbour
                if neighbours:
                    scores = [
                        road_score[t-1, nb] - 0.5 * (
                            cholera_cases[t-1, nb] / max(1, COUNTY_DF.loc[nb, "population"] * 0.01))
                        for nb in neighbours
                    ]
                    best_nb = neighbours[int(np.argmax(scores))]
                    displacement_dest[t, cid] = best_nb
                    displaced[t, cid] = max(0,
                        int(displaced[t-1, cid] + new_displaced
                            - displaced[t-1, cid] * 0.05))
                else:
                    displacement_dest[t, cid] = cid
                    displaced[t, cid] = int(displaced[t-1, cid])
            else:
                # Slow return to home if conditions improve
                displaced[t, cid] = max(0, int(displaced[t-1, cid] * 0.95))
                displacement_dest[t, cid] = cid

            # ===== 7. FOOD & MARKETS ==========================================
            # INTERACTION: market drops when displacement > threshold OR security spikes
            market_pressure = 0.0
            if displaced[t, cid] > DISPLACEMENT_THRESHOLD:
                market_pressure += 0.04
                cross_sector_events.append(
                    f"Day {t:3d} | {row['county_name']:20s} | "
                    f"MARKET STRESS: displaced={displaced[t,cid]:,} > threshold")
            if security_rate[t, cid] > 5:
                market_pressure += 0.06

            market_index[t, cid] = np.clip(
                market_index[t-1, cid]
                - market_pressure
                + 0.008                   # Slow market recovery
                + rng.normal(0, NOISE_LEVEL * 0.05),
                0.05, 1.0)

            # Food prices rise inversely with market activity and with displacement
            food_price[t, cid] = np.clip(
                food_price[t-1, cid]
                + (1.0 - market_index[t, cid]) * 0.02
                + displaced[t, cid] / max(1, pop) * 0.1
                + rng.normal(0, NOISE_LEVEL * 0.02),
                0.5, 3.5)

            # ===== 8. MISINFORMATION SIGNAL ===================================
            # INTERACTION: amplified by security incidents and market disruption
            misinfo_base = rng.uniform(-0.01, 0.03)   # Random drift
            misinfo_amplification = (
                0.05 * (security_rate[t, cid] > 3)
                + 0.04 * (market_index[t, cid] < 0.5)
                + misinfo_shock * 0.4)                # Direct spike from shock

            misinfo_signal[t, cid] = np.clip(
                misinfo_signal[t-1, cid] * 0.95       # Natural decay
                + misinfo_base
                + misinfo_amplification
                + rng.normal(0, NOISE_LEVEL * 0.04),
                0.0, 1.0)

    # -------------------------------------------------------------------------
    # ASSEMBLE OUTPUT DATAFRAME
    # -------------------------------------------------------------------------
    print(f"\n  Assembling output ({T * n} rows)...")
    records = []
    dates = [START_DATE + timedelta(days=t) for t in range(T)]

    for t in range(T):
        for cid in range(n):
            records.append({
                "date":                  dates[t].isoformat(),
                "county_id":             cid,
                "county_name":           COUNTY_DF.loc[cid, "county_name"],
                "contamination_index":   round(float(contamination[t, cid]), 4),
                "cholera_cases":         int(cholera_cases[t, cid]),
                "health_facility_capacity": round(float(capacity[t, cid]), 4),
                "road_accessibility_score": round(float(road_score[t, cid]), 4),
                "security_incident_rate":   int(security_rate[t, cid]),
                "health_worker_obstruction": round(float(worker_obstruction[t, cid]), 4),
                "displaced_population":  int(displaced[t, cid]),
                "displacement_direction_county_id": int(displacement_dest[t, cid]),
                "market_activity_index": round(float(market_index[t, cid]), 4),
                "food_price_index":      round(float(food_price[t, cid]), 4),
                "misinformation_signal": round(float(misinfo_signal[t, cid]), 4),
            })

    df = pd.DataFrame(records)
    return df, cross_sector_events, shocks


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    df, events, shocks = simulate(seed=RANDOM_SEED)

    # Save full dataset
    out_path = "cholera_outbreak_kenya.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  [OUTPUT] Full dataset saved → {out_path}")

    # Save 10-day sample
    sample_path = "sample_output.csv"
    sample = df[df["date"] <= (date(2024, 3, 1) + timedelta(days=9)).isoformat()]
    sample.to_csv(sample_path, index=False)
    print(f"  [OUTPUT] 10-day sample saved → {sample_path}")

    # -------------------------------------------------------------------------
    # SIMULATION SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Outbreak Origin Counties:")
    for s in SEED_COUNTIES:
        print(f"    → {COUNTY_DF.loc[s, 'county_name']} "
              f"(water risk: {COUNTY_DF.loc[s, 'baseline_water_risk']})")

    # Peak case county and day
    pivot = df.pivot_table(index="date", columns="county_name",
                           values="cholera_cases", aggfunc="sum")
    peak_total = pivot.max()
    peak_county = peak_total.idxmax()
    peak_day_date = pivot[peak_county].idxmax()
    peak_day_num  = (date.fromisoformat(peak_day_date) - date(2024, 3, 1)).days
    peak_value    = int(peak_total[peak_county])

    print(f"\n  Peak Case County    : {peak_county}")
    print(f"  Peak Value          : {peak_value:,} cases")
    print(f"  Day of Peak         : Day {peak_day_num} ({peak_day_date})")

    # Total cases on final day
    final = df[df["date"] == df["date"].max()]
    total_final_cases = final["cholera_cases"].sum()
    affected_counties = (final["cholera_cases"] > 20).sum()
    print(f"\n  End-State (Day {SIMULATION_DAYS}):")
    print(f"    Total active cases     : {total_final_cases:,}")
    print(f"    Counties with >20 cases: {affected_counties}/47")
    print(f"    Total displaced persons: {final['displaced_population'].sum():,}")

    # Cross-sector interaction summary
    unique_events = list(dict.fromkeys(events))  # Deduplicate while preserving order
    print(f"\n  Cross-Sector Interactions Fired: {len(unique_events)} events")
    security_events = [e for e in unique_events if "SECURITY" in e]
    market_events   = [e for e in unique_events if "MARKET" in e]
    print(f"    Security spike events : {len(security_events)}")
    print(f"    Market stress events  : {len(market_events)}")

    if security_events:
        print(f"\n  Sample Security Events (first 3):")
        for e in security_events[:3]:
            print(f"    {e}")
    if market_events:
        print(f"\n  Sample Market Events (first 3):")
        for e in market_events[:3]:
            print(f"    {e}")

    print(f"\n  External Shocks:")
    for s in shocks:
        county_names = [COUNTY_DF.loc[t, 'county_name'] for t in s['targets']]
        print(f"    Day {s['day']:3d}: {s['type']:20s} → {county_names}")

    print(f"\n{'='*60}")
    print(f"Done. Check '{out_path}' and '{sample_path}'.")
    print(f"{'='*60}\n")

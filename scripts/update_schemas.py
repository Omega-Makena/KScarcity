import sqlite3
import json
import os

db_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\ui\institution\backend\federated_registry.sqlite"

SECTOR_FEATURES = {
    "Public Health": ["patient_intake", "essential_drug_stock", "bed_occupancy", "staff_availability"],
    "Water & Sanitation": ["reservoir_level", "water_quality_index", "pipe_pressure", "pump_operational_hrs"],
    "Transport & Logistics": ["fleet_availability", "fuel_reserves", "route_disruption_index", "maintenance_backlog"],
    "Security & Border": ["patrol_frequency", "incident_reports", "border_wait_time_hrs", "equipment_readiness"]
}

DEFAULT_FEATURES = ["generic_metric_1", "generic_metric_2", "generic_metric_3", "generic_metric_4"]

def main():
    if not os.path.exists(db_path):
        print("DB not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all basket IDs and names
    cursor.execute("SELECT id, name FROM baskets")
    baskets = cursor.fetchall()

    for basket in baskets:
        b_id = basket[0]
        b_name = basket[1]
        
        # Look up features for this specific sector, or use defaults
        features = SECTOR_FEATURES.get(b_name, DEFAULT_FEATURES)
        
        new_schema = {
            "required_columns": ["timestamp", "location", "sector"] + features + ["threat_score"],
            "allow_extra": True
        }
        schema_json = json.dumps(new_schema)

        # Check if row exists
        cursor.execute("SELECT 1 FROM ontology_schemas WHERE basket_id = ?", (b_id,))
        if cursor.fetchone():
            cursor.execute("UPDATE ontology_schemas SET schema_definition = ? WHERE basket_id = ?", (schema_json, b_id))
        else:
            cursor.execute("INSERT INTO ontology_schemas (basket_id, schema_definition) VALUES (?, ?)", (b_id, schema_json))

    conn.commit()
    conn.close()
    print(f"Updated ontology schemas with sector-specific features for {len(baskets)} baskets in the database.")

if __name__ == '__main__':
    main()

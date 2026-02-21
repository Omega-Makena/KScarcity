import sys
import os
import traceback

print("--- Testing Imports ---")

# Setup path mapping exactly as Streamlit uses it
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

print(f"Base Path: {base_path}")

modules_to_test = [
    "scarcity.runtime.bus",
    "scarcity.engine.anomaly",
    "scarcity.engine.forecasting",
    "scarcity.engine.store",
    "scarcity.engine.resource_manager",
    "scarcity.federation.aggregator",
    "scarcity.federation.privacy_guard",
    "kshiked.ui.institution.backend.scarcity_bridge",
    "kshiked.ui.institution.backend.federation_bridge",
    "kshiked.ui.institution.backend.executive_bridge"
]

for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"[OK] {mod}")
    except Exception as e:
        print(f"[FAIL] {mod} -> {type(e).__name__}: {e}")
        traceback.print_exc()

print("--- Test Complete ---")

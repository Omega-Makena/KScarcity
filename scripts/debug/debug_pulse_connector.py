
import logging
import sys
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Mock Streamlit session state if needed, though data_connector mostly relies on class state
class MockSessionState(dict):
    pass

# Patch sys.modules to avoid Streamlit errors if imported
# sys.modules['streamlit'] = type('MockStreamlit', (), {'session_state': MockSessionState()})

try:
    from kshiked.ui.data_connector import PulseConnector
    print("Successfully imported PulseConnector")
except ImportError as e:
    print(f"Failed to import PulseConnector: {e}")
    sys.exit(1)

def test_connector():
    connector = PulseConnector()
    print(f"Initial connected state: {connector._connected}")

    print("Attempting to connect...")
    success = connector.connect()
    print(f"Connect result: {success}")
    
    if not success:
        print("Connection failed. Cannot test live data.")
        return

    print("Waiting for data injection (5s)...")
    time.sleep(5)

    print("\n--- Primitives ---")
    prims = connector.get_primitives()
    print(prims)

    print("\n--- ESI Indicators ---")
    esi = connector.get_esi_indicators()
    print(esi)

    # Check if we are getting demo data values
    # Demo ESI Food is 0.65
    is_demo_esi = esi.get("Food") == 0.65
    print(f"\nIs Demo ESI? {is_demo_esi}")

if __name__ == "__main__":
    test_connector()

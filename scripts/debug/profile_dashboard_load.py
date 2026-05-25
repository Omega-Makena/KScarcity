
import time
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

# Setup logging with timestamp
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("profiler")

def profile_connector(name, connector_class):
    logger.info(f"[{name}] Starting instantiation...")
    start = time.time()
    try:
        conn = connector_class()
        inst_time = time.time() - start
        logger.info(f"[{name}] Instantiated in {inst_time:.4f}s")
        
        logger.info(f"[{name}] Connection starting...")
        conn_start = time.time()
        success = conn.connect()
        conn_time = time.time() - conn_start
        logger.info(f"[{name}] Connected in {conn_time:.4f}s (Success: {success})")
        
        return inst_time, conn_time
    except Exception as e:
        logger.error(f"[{name}] Failed: {e}")
        return 0, 0

def main():
    logger.info("Starting Dashboard Load Profile...")
    total_start = time.time()

    # Import overhead
    logger.info("Importing data_connector...")
    import_start = time.time()
    try:
        from kshiked.ui.data_connector import (
            ScarcityConnector, 
            PulseConnector, 
            FederationConnector, 
            SimulationConnector,
            get_dashboard_data
        )
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return
    logger.info(f"Imports took {time.time() - import_start:.4f}s")

    # Connector Profiling (Serial)
    logger.info("--- Serial Component Profiling ---")
    profile_connector("Scarcity", ScarcityConnector)
    profile_connector("Pulse", PulseConnector)
    profile_connector("Federation", FederationConnector)
    profile_connector("Simulation", SimulationConnector)

    # Full Aggregation Profiling
    logger.info("--- Full get_dashboard_data() Profiling ---")
    agg_start = time.time()
    data = get_dashboard_data()
    agg_time = time.time() - agg_start
    logger.info(f"get_dashboard_data() took {agg_time:.4f}s")

    logger.info(f"Total Script Time: {time.time() - total_start:.4f}s")

if __name__ == "__main__":
    main()

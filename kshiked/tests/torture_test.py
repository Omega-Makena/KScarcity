"""
Torture Test for Online Discovery Engine.

Feeds the engine with:
- NaNs and Nones
- Infinities
- Strings and Garbled text
- Unexpected Types
- Missing Keys
- Numeric Overflows

Goal: Crash the engine to identify weak points.
"""

import numpy as np
import math
import logging
from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TortureTest")

def run_torture():
    engine = OnlineDiscoveryEngine()
    vars = ["A", "B", "C"]
    
    # Initialize with clean schema
    schema = {"fields": [{"name": v} for v in vars]}
    engine.initialize(schema)
    
    logger.info("Starting Torture Test...")
    
    # 1. Clean Data (Baseline)
    engine.process_row({"A": 1.0, "B": 2.0, "C": 3.0})
    
    # 2. Missing Value (None)
    try:
        logger.info("Sending None...")
        engine.process_row({"A": 1.0, "B": None, "C": 3.0})
    except Exception as e:
        logger.error(f"Crashed on None: {e}")

    # 3. NaN
    try:
        logger.info("Sending NaN...")
        engine.process_row({"A": np.nan, "B": 2.0, "C": 3.0})
    except Exception as e:
        logger.error(f"Crashed on NaN: {e}")

    # 4. Strings
    try:
        logger.info("Sending String...")
        engine.process_row({"A": "garbage", "B": 2.0, "C": 3.0})
    except Exception as e:
        logger.error(f"Crashed on String: {e}")

    # 5. Infinity
    try:
        logger.info("Sending Infinity...")
        engine.process_row({"A": float('inf'), "B": 2.0, "C": 3.0})
    except Exception as e:
        logger.error(f"Crashed on Infinity: {e}")

    # 6. Overflow Keys (Noise)
    try:
        logger.info("Sending Extra Keys...")
        engine.process_row({"A": 1.0, "B": 2.0, "Z_Unknown": 99.0})
    except Exception as e:
        logger.error(f"Crashed on Extra Keys: {e}")

    # 7. Type Mismatch (List instead of float)
    try:
        logger.info("Sending List as Value...")
        engine.process_row({"A": [1, 2], "B": 2.0})
    except Exception as e:
        logger.error(f"Crashed on List Type: {e}")

    logger.info("Torture Test Complete.")

if __name__ == "__main__":
    run_torture()


import time
import logging
import sys
import threading

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("profiler")

def profile_import(module_name):
    start = time.time()
    try:
        __import__(module_name)
        duration = time.time() - start
        logger.info(f"Import {module_name}: {duration:.4f}s")
        return duration
    except Exception as e:
        logger.error(f"Import {module_name} failed: {e}")
        return 0

def main():
    logger.info("Starting K-SHIELD Load Profile...")
    
    # 1. Profile Imports (Simulating what happens when accessing sub-pages or warming)
    logger.info("--- Profiling Sub-module Imports ---")
    
    # These are the modules warmed in page.py
    modules = [
        "kshiked.ui.kshield.causal", # Adjusted path based on structure
        "kshiked.ui.kshield.terrain",
        "kshiked.ui.kshield.simulation",
        "kshiked.ui.kshield.impact",
    ]
    
    for mod in modules:
        profile_import(mod)

    # 2. Check if page.py itself is heavy
    logger.info("--- Profiling page.py Import ---")
    start = time.time()
    try:
        from kshiked.ui.kshield import page
    except ImportError:
        # Try local path if running from root
        try:
            import kshiked.ui.kshield.page as page
        except ImportError:
             logger.error("Could not import kshield.page")
    
    duration = time.time() - start
    logger.info(f"Import kshield.page: {duration:.4f}s")

if __name__ == "__main__":
    # Add project root to path
    import os
    sys.path.append(os.getcwd())
    main()

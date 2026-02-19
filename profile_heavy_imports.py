
import time
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("profiler")

def main():
    logger.info("Profiling heavy imports...")
    
    start = time.time()
    try:
        import kshiked.pulse.llm.policy_chatbot
    except ImportError:
        logger.error("Failed to import policy_chatbot")
        
    logger.info(f"policy_chatbot import: {time.time() - start:.4f}s")

    start = time.time()
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        pass
    logger.info(f"plotly import: {time.time() - start:.4f}s")

    start = time.time()
    try:
        import dowhy
        import econml
    except ImportError:
        pass
    logger.info(f"causal libs import: {time.time() - start:.4f}s")

if __name__ == "__main__":
    import os
    sys.path.append(os.getcwd())
    main()

"""
Proof of Concept: Online Relationship Discovery on Kenya Data.

Demonstrates the new `OnlineDiscoveryEngine` finding relationships 
(Compositional, Correlational, Functional) in the Kenya dataset
without off-line training.

Usage: python -m kshiked.tests.prove_scarcity_v2
(requires: pip install -e .)
"""

import pandas as pd
import time
import logging
from pathlib import Path

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.discovery import RelationshipType

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoC")

# Data path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "API_KEN_DS2_en_csv_v2_14659.csv"

def run_proof():
    # 1. Load Data
    logger.info(f"Loading dataset from {DATA_PATH}...")
    
    # Skip header rows if necessary (World Bank format often has 4 header rows)
    try:
        df = pd.read_csv(DATA_PATH, skiprows=4)
    except Exception:
        df = pd.read_csv(DATA_PATH)
        
    logger.info(f"Loaded {len(df)} rows. Columns: {len(df.columns)}")
    
    # 2. Pivot/Prepare Data
    # The World Bank data is likely [Country, Code, Indicator Name, Indicator Code, 1960, 1961...].
    # We need Time Series format: Rows=Years, Cols=Indicators.
    
    # Check structure
    if "Indicator Name" in df.columns:
        logger.info("Pivoting World Bank format to Time Series...")
        # Melt year columns
        id_vars = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
        val_vars = [c for c in df.columns if c not in id_vars and c.isdigit()]
        
        melted = df.melt(id_vars=id_vars, value_vars=val_vars, var_name="Year", value_name="Value")
        
        # Pivot: Index=Year, Columns=Indicator Code, Values=Value
        pivoted = melted.pivot(index="Year", columns="Indicator Code", values="Value")
        
        # Clean: Drop columns with too many NaNs
        pivoted = pivoted.dropna(axis=1, thresh=len(pivoted)*0.8)
        pivoted = pivoted.fillna(method='ffill').fillna(0.0)
        
        data_stream = pivoted
    else:
        logger.info("Assuming standard tabular format...")
        data_stream = df.select_dtypes(include=['number']).fillna(0.0)

    logger.info(f"Stream ready: {data_stream.shape[0]} timesteps, {data_stream.shape[1]} variables.")

    # 3. Initialize Engine
    engine = OnlineDiscoveryEngine()
    
    # Pass schema
    schema = {"fields": [{"name": c} for c in data_stream.columns]}
    engine.initialize(schema)
    
    # 4. Stream Loop
    logger.info("Starting Online Discovery Loop...")
    start_time = time.time()
    
    for i in range(len(data_stream)):
        # Convert row to dict
        row = data_stream.iloc[i].to_dict()
        
        # Engine Step
        stats = engine.process_row(row)
        
        if i % 10 == 0:
            logger.info(f"Step {i}: Active Hypotheses={stats['active_hypotheses']}, Groups={stats['groups']}")

    duration = time.time() - start_time
    logger.info(f"Finished processing in {duration:.2f}s ({len(data_stream)/duration:.1f} rows/s)")

    # 5. Extract Knowledge
    knowledge = engine.get_knowledge_graph()
    
    logger.info("="*40)
    logger.info("DISCOVERED RELATIONSHIPS (Top 10)")
    logger.info("="*40)
    
    for h in knowledge[:10]:
        m = h['metrics']
        logger.info(f"[{h['type'].upper()}] {h['variables']} (Conf: {m['confidence']:.3f}, Samples: {m['evidence']})")

if __name__ == "__main__":
    run_proof()

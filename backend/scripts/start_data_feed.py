"""
Start a synthetic data feed to the scarcity system.

This script continuously generates and sends synthetic data to the MPIE ingestion endpoint.
"""

import asyncio
import httpx
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000/api/v2"
INGEST_ENDPOINT = f"{API_BASE}/mpie/ingest"

# Configuration
ROWS_PER_WINDOW = 100
FEATURES = 16
INTERVAL_SECONDS = 2.0  # Send data every 2 seconds


async def generate_synthetic_data(window_id: int) -> dict:
    """Generate synthetic data window."""
    # Generate random data
    data = np.random.randn(ROWS_PER_WINDOW, FEATURES).astype(np.float32)
    
    return {
        "data": data.tolist(),
        "schema": {
            "features": FEATURES,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "window_id": window_id
    }


async def send_data_window(client: httpx.AsyncClient, window_id: int) -> bool:
    """Send a data window to the ingestion endpoint."""
    try:
        payload = await generate_synthetic_data(window_id)
        
        response = await client.post(INGEST_ENDPOINT, json=payload, timeout=5.0)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✓ Window {window_id}: Ingested {result['rows']} rows x {result['features']} features")
            return True
        else:
            logger.error(f"✗ Window {window_id}: Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Window {window_id}: Error - {e}")
        return False


async def main():
    """Main data feed loop."""
    logger.info("Starting synthetic data feed...")
    logger.info(f"Target: {INGEST_ENDPOINT}")
    logger.info(f"Window size: {ROWS_PER_WINDOW} rows x {FEATURES} features")
    logger.info(f"Interval: {INTERVAL_SECONDS}s")
    logger.info("-" * 60)
    
    window_id = 0
    
    async with httpx.AsyncClient() as client:
        # Check if API is available
        try:
            health_response = await client.get(f"{API_BASE}/health", timeout=5.0)
            if health_response.status_code != 200:
                logger.error("API health check failed. Is the backend running?")
                return
            logger.info("✓ Backend API is healthy")
        except Exception as e:
            logger.error(f"Cannot connect to backend: {e}")
            logger.error("Make sure the backend is running on http://localhost:8000")
            return
        
        # Start feeding data
        while True:
            success = await send_data_window(client, window_id)
            
            if success:
                window_id += 1
            
            await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nData feed stopped by user")

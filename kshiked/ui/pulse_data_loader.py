import pandas as pd
import requests
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger("sentinel.pulse")

class PulseDataLoader:
    """
    Ingests social media/signal data from:
    1. Local Kaggle CSVs (data/pulse)
    2. Live Hugging Face API (Kenya Hate Speech Superset)
    """
    
    HF_API_URL = "https://datasets-server.huggingface.co/rows?dataset=manueltonneau%2Fkenya-hate-speech-superset&config=default&split=train"
    
    def __init__(self, data_dir="data/pulse"):
        self.data_dir = Path(data_dir)
        self.hf_token = os.environ.get("HF_TOKEN")
        
    def load_combined_data(self) -> pd.DataFrame:
        """Loads both local and live data, normalized."""
        local = self.load_local_files()
        live = self.fetch_live_api_data()
        
        combined = pd.concat([local, live], ignore_index=True)
        return combined

    def load_local_files(self) -> pd.DataFrame:
        """Reads CSVs from data/pulse."""
        all_data = []
        if not self.data_dir.exists():
            logger.warning(f"{self.data_dir} not found.")
            return pd.DataFrame()
            
        for f in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                # Normalize columns manually based on known datasets
                norm_df = self._normalize_dataframe(df, source=f.name)
                all_data.append(norm_df)
            except Exception as e:
                logger.error(f"Failed to read {f}: {e}")
                
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def fetch_live_api_data(self, limit=100) -> pd.DataFrame:
        """Fetches rows from Hugging Face Datasets Server."""
        try:
            # Note: The user provided URL has parameters encoded in it already. 
            # Requests params argument will double encode if we aren't careful.
            # Let's use the base URL and pass params cleanly.
            base_url = "https://datasets-server.huggingface.co/rows"
            params = {
                "dataset": "manueltonneau/kenya-hate-speech-superset",
                "config": "default",
                "split": "train",
                "offset": 0,
                "length": limit
            }
            
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
                
            logger.info(f"Fetching from {base_url} with params: {params}")
            r = requests.get(base_url, headers=headers, params=params, timeout=10)
            
            if r.status_code != 200:
                logger.error(f"HF API Error {r.status_code}: {r.text}")
                return pd.DataFrame()
                
            r.raise_for_status()
            
            data = r.json()
            rows = [row['row'] for row in data.get('rows', [])]
            
            df = pd.DataFrame(rows)
            return self._normalize_dataframe(df, source="hf_api_live")
            
        except Exception as e:
            logger.error(f"HF API Fetch failed: {e}")
            return pd.DataFrame()

    def _normalize_dataframe(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Standardizes Schema: [text, timestamp, signal_type, severity, source]"""
        out = pd.DataFrame()
        
        # 1. Text Content
        if 'text' in df.columns:
            out['text'] = df['text']
        elif 'tweet' in df.columns:
            out['text'] = df['tweet']
        elif 'content' in df.columns:
            out['text'] = df['content']
        else:
            return pd.DataFrame() # Skip if no text
            
        # 2. Signal Type / Label
        if 'label' in df.columns:
            out['signal_type'] = df['label'].astype(str)
        elif 'sentiment' in df.columns:
            out['signal_type'] = df['sentiment'].astype(str)
        else:
            out['signal_type'] = 'unknown'
            
        # 3. Metadata
        out['source'] = source
        out['timestamp'] = datetime.now() # unexpected for static data, but useful for simulation baseline
        
        # Simple severity heuristic
        out['severity'] = 0.5
        
        return out

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = PulseDataLoader()
    
    # Test API
    print("Fetching live data...")
    df_live = loader.fetch_live_api_data(limit=5)
    print(f"Live Data: {len(df_live)} rows")
    if not df_live.empty:
        print(df_live[['text', 'signal_type']].head())
    
    # Test Local
    # print("\nLoading local data...")
    # df_local = loader.load_local_files()
    # print(f"Local Data: {len(df_local)} rows")

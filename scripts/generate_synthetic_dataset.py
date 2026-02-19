
import sys
import os

# Ensure the project root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scarcity.synthetic.pipeline import SyntheticPipeline

def main():
    print("Starting Synthetic Data Generation...")
    
    # Configuration
    NUM_ACCOUNTS = 200
    DURATION_DAYS = 14
    OUTPUT_DIR = "data/synthetic_kenya"
    
    pipeline = SyntheticPipeline(output_dir=OUTPUT_DIR)
    pipeline.run(num_accounts=NUM_ACCOUNTS, duration_days=DURATION_DAYS)

if __name__ == "__main__":
    main()

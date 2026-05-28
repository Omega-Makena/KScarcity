import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from benchmark.synthetic.pipeline import SyntheticBenchmark
from benchmark.synthetic.reporting import generate_report

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    schema_path = os.path.join(base_dir, "benchmark_schema.json")
    
    print("Initializing benchmark...", flush=True)
    bench = SyntheticBenchmark(schema_path=schema_path, seed=42, B_perm=100)
    
    print("Running sweep [3000]...", flush=True)
    results = bench.run_sweep([3000])
    
    out_dir = os.path.join(base_dir, "benchmark_results")
    generate_report(results, out_dir=out_dir)
    print("Done!", flush=True)

import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

data = json.loads((Path('scarcity/synthetic/benchmark_results/benchmark_data.json')).read_text())
fs = data.get('federation_scarcity', {})

print('=== SINGLE-COUNTRY EDGES ===')
for e in fs.get('single_edges', []):
    sym = '<>' if e.get('symmetric') else '->'
    print(f"  {e['source']:<28} {sym} {e['target']:<28} {e['type']:<14} conf={e['confidence']:.3f}")

print()
print('=== FEDERATED EDGES ===')
for e in fs.get('fed_edges', []):
    sym = '<>' if e.get('symmetric') else '->'
    print(f"  {e['source']:<28} {sym} {e['target']:<28} {e['type']:<14} conf={e['confidence']:.3f}  {e.get('plausibility','')}")

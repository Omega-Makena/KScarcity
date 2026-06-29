# Getting Started with Scarcity

A practical install-and-use guide for people consuming the `scarcity` library.

Scarcity is a toolkit of independent engines:

- **Relationship discovery** — an online "automated statistician" that learns which of
  15 relationship types hold in a data stream (`scarcity.engine`).
- **Causal inference** — an offline DoWhy-based estimation pipeline (`scarcity.causal`).
- **Simulation** — stock-flow-consistent (SFC) economy models (`scarcity.simulation`).
- **Federation** — privacy-preserving, multi-institution knowledge aggregation
  (`scarcity.federation`, `scarcity.fmi`, `scarcity.meta`).
- Plus supporting layers: `stream`, `governor`, `runtime`, `synthetic`, `analytics`.

You only install what you need. The core install is tiny; heavy dependencies
(`torch`, `dowhy`, `fastapi`, ...) live behind **optional extras**.

---

## 1. Requirements

- **Python ≥ 3.9**
- A virtual environment is strongly recommended.

```bash
python -m venv .venv
# Windows:        .\.venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 2. Install

### Core (lightweight)

```bash
pip install scarcity
```

Pulls only `numpy` + `pandas`. This already gives you:

- the **simulation** engine (`scarcity.simulation`)
- the **relationship-discovery** engine in pure-Python mode (see the note in §4.1)
- `meta`, `governor`, `runtime`, `analytics`, and the non-secure parts of `federation`/`fmi`

### Pick the extras you need

Install extras with the `scarcity[extra]` syntax (quote it in zsh):

```bash
pip install "scarcity[causal]"            # one extra
pip install "scarcity[causal,stream]"     # several at once
```

| Extra | Install when you want to… | Pulls in |
|-------|---------------------------|----------|
| `causal` | run the offline causal-inference pipeline (`run_causal`) | `dowhy` |
| `gpu` | use the fast vectorized / CUDA discovery backend | `torch`, `pytorch3d`, `pynvml` |
| `accel` | use the JIT-accelerated `engine.anomaly` / `engine.forecasting` modules | `numba` |
| `stream` | ingest live streams, shard, replay, websockets | `aiofiles`, `websockets`, `scikit-learn` |
| `federation` | use **secure** aggregation (encrypted federation transport) | `cryptography` |
| `fmi` | load YAML configs / compressed packets for the federation–meta interface | `pyyaml`, `zstandard` |
| `governor` | enable real system-resource governance | `psutil` |
| `dashboard` | run the bundled FastAPI dashboard / API surface | `fastapi`, `uvicorn`, `pydantic`, ... |
| `dev` | run the test suite, linters, type-checker | `pytest`, `ruff`, `mypy`, `scipy` |

### Convenience: (almost) everything

```bash
pip install "scarcity[all]"
```

`all` bundles every runtime extra **except `gpu`** and **except `dev`**.
`gpu` is excluded on purpose because `pytorch3d` often needs a matching CUDA
toolchain and a from-source build — install it deliberately:

```bash
pip install "scarcity[all]"
pip install "scarcity[gpu]"   # only if you actually need the GPU/vectorized backend
```

### Development install (from a clone)

```bash
git clone https://github.com/Omega-Makena/KScarcity.git
cd KScarcity
pip install -e ".[dev]"        # editable + test/lint toolchain
pytest                         # 387 tests collect; run the suite
```

---

## 3. Verify the install

```bash
python -c "import scarcity; print(scarcity.__version__)"
```

`import scarcity` is intentionally lazy — it will succeed even if you have not
installed the optional extras. A missing dependency only surfaces when you
actually touch the feature that needs it (see §6).

---

## 4. Quickstarts

### 4.1 Relationship discovery (core install)

Feed rows one at a time; read out the surviving hypotheses.

```python
from scarcity import OnlineDiscoveryEngine

# vectorized=False -> pure-Python backend, no torch needed (core install).
# small_dataset_mode=True keeps sparse relationship types alive on short series.
engine = OnlineDiscoveryEngine(vectorized=False, small_dataset_mode=True)

stream = [
    {"price": 1.0, "demand": 9.8, "income": 4.1},
    {"price": 1.1, "demand": 9.2, "income": 4.0},
    # ... one dict[str, float] per observation
]
for row in stream:
    engine.process_row(row)

# Surviving relationships above a confidence threshold:
for h in engine.export_hypothesis_summary(min_conf=0.5):
    print(h)   # {'vars': [...], 'type': '...', 'confidence': ..., 'evidence': ...}
```

> **Important — the default needs `torch`.** `OnlineDiscoveryEngine()` defaults to
> `vectorized=True`, which lazily imports `torch` (the GPU/tensor backend). On a
> **core install you must pass `vectorized=False`**, or you will get an
> `ImportError` for torch. To use the faster backend instead:
> `pip install "scarcity[gpu]"`, then `OnlineDiscoveryEngine(device="cuda")`.

### 4.2 Causal inference (`pip install "scarcity[causal]"`)

```python
import pandas as pd
from scarcity.causal import run_causal, EstimandSpec

df = pd.read_csv("observations.csv")

spec = EstimandSpec(
    treatment="price",
    outcome="demand",
    confounders=["season", "income"],
)

result = run_causal(df, spec)   # -> CausalRunResult
print(result)
```

### 4.3 Simulation (core install)

```python
from scarcity.simulation import SFCEconomy, SFCConfig

economy = SFCEconomy(SFCConfig())
history = economy.run(steps=100)   # list of per-step state dicts
print(history[-1])
```

### 4.4 Secure federation (`pip install "scarcity[federation,fmi]"`)

The federation layer aggregates knowledge across nodes without sharing raw data.
It has many moving parts (secure aggregation, gossip, hierarchical aggregation);
start from the high-level orchestrator:

```python
from scarcity.federation import HierarchicalFederation, HierarchicalFederationConfig
# See docs/scarcity-docs/federation for the full protocol and node lifecycle.
```

---

## 5. Choosing extras by use case

| Your goal | Install |
|-----------|---------|
| Mine relationships from a CSV / stream (pure Python) | `scarcity` |
| Same, but fast / on GPU | `scarcity[gpu]` |
| Estimate a treatment effect from observational data | `scarcity[causal]` |
| Run economic / stock-flow simulations | `scarcity` |
| Live streaming ingestion + replay | `scarcity[stream]` |
| Multi-institution federated learning (encrypted) | `scarcity[federation,fmi]` |
| Stand up the dashboard/API | `scarcity[dashboard]` |
| Just want it all (no GPU) | `scarcity[all]` |

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` when creating `OnlineDiscoveryEngine()` | default `vectorized=True` needs torch | pass `vectorized=False`, or `pip install "scarcity[gpu]"` |
| `ModuleNotFoundError: No module named 'dowhy'` | causal pipeline not installed | `pip install "scarcity[causal]"` |
| `ModuleNotFoundError: No module named 'numba'` importing `scarcity.engine.anomaly` / `.forecasting` | JIT modules not installed | `pip install "scarcity[accel]"` |
| `cryptography` errors in secure aggregation | secure transport not installed | `pip install "scarcity[federation]"` |
| `pip install "scarcity[gpu]"` fails building `pytorch3d` | needs a CUDA build toolchain | follow the official PyTorch3D install for your platform/CUDA version |
| `RuntimeError: PyYAML is required to load FMI configuration files` | FMI config support not installed | `pip install "scarcity[fmi]"` |

---

## 7. Where to go next

- Architecture and the 15 relationship types: the **Scarcity Engine** section of the
  [README](../README.md).
- Per-subpackage reference docs: [`docs/scarcity-docs/`](scarcity-docs/).
- Public API: `import scarcity; dir(scarcity)` lists the headline entry points and
  subpackages.

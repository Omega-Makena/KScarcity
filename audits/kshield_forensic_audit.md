# KSHIELD_AUDIT.md

## 1. Executive Summary
*(To be completed at the end of the audit)*
- [Pending]
- [Pending]

## 2. Data & Information Integrity

### A. Data Sources
| Dataset / Source | Type | Location | Origin | Size/Status |
| :--- | :--- | :--- | :--- | :--- |
| **World Dev Indicators** | Static CSV | `API_KEN_DS2_en_csv_v2_14659.csv` | World Bank (WDI) | 974KB, ~1500 lines. <br>Last Upd: 2025-12-19 |
| **Social Scrapers** | Live Stream | `kshiked/pulse/scrapers/` | FB, X, Insta, Reddit | Implementation: Hybrid (Lib + Playwright) |
| **Domain Config** | Config JSON | `backend/data/domains.json` | Local | 2KB (Static list) |

### B. Integrity Findings
**1. Static Data (WDI CSV):**
- **Completeness**: Significant missingness. Many indicator rows have empty strings `""` for recent years (2023, 2024).
- **Scope**: Kenya-only (`"Kenya","KEN"`). Hardcoded to Kenya context.
- **Evidence**: `view_file` on CSV showed indicators like "Intentional homicides" with sparse data.

**2. Dynamic Data (Pulse Scrapers):**
- **Collection Method**: Automated scraping via `facebook-scraper` and `Playwright`.
- **Sampling Bias (High Risk)**:
  - `facebook_scraper.py` limits scraping to first **3 pages** to avoid detection: `for page in self.config.pages[:3]: # Limit pages to avoid detection`.
  - This effectively ignores other configured pages, creating a silent coverage gap.
- **Reliability**: Uses "search" queries or page lists. Relies on potentially fragile DOM selectors or 3rd party libs (`facebook_scraper`).
- **Standardization**: Good usage of `ScraperResult` dataclass (`base.py`) to unify fields (`geo_location`, `text`, `posted_at`) across platforms.

**3. Versioning & Lineage:**
- **Code**: Git used.
- **Data**: CSV filename implies versioning (`v2_14659`), but no DVC or formal data versioning found for the repo.


## 3. Technical / System Architecture

### A. Entry Points & Flow
- **CLI/Launcher**: `run_kshield.py` orchestrates startup.
  - **Risk**: Uses `subprocess.Popen` to launch Streamlit. No supervision or restart policy if dashboard crashes.
  - **Hardcoding**: Theme colors, port (8501 default), and paths (`scarcity/dashboard.py`) are hardcoded in the launcher.
- **Backend**: `backend/app/main.py` (FastAPI).
  - **Status**: Defines v1 (deprecated) and v2 (active) APIs.
  - **Mismatch**: `main.py` docstring says Federation/Simulation are "coming soon", while `README.md` claims "Status: Production Ready".

### B. Configuration & Drift
- **Assumption Ledger**:
  | Location | Assumption | Risk | Configurable? |
  | :--- | :--- | :--- | :--- |
  | `facebook_scraper.py:130` | User Agent `Chrome/120...` | Detection/Blocking | No (Hardcoded) |
  | `run_kshield.py:101` | Theme Colors | Branding rigidity | No |
  | `main.py:98` | Dataset Registry Path | Path breakage | No |
  | `backend/app/core/config.py` | *(Pending inspection)* | - | - |

- **Env Parity**: `run_kshield.py` checks for `.env` but proceeds without it (warns only). Scrapers rely on this for credentials.

### C. Single Points of Failure (SPOF)
- **Startup**: `run_kshield.py` is a synchronous script. If it dies, the whole system stops.
- **Data Dependency**: `API_KEN_DS2...csv`. If this file is missing or corrupted, `SimulationManager` likely fails.

### C. Readiness Ratings (Evidence-based)
| Subsystem | Rating | Why | Evidence |
|---|---:|---|---|
| **Scarcity Discovery** | PROTOTYPE | Logic sound, but hardcoded priors & unverified leakage | `discovery.py:147` (Valid update order) |
| **Logic/Ident** | RISK | Hardcoded `proceed_when_unidentifiable=True` | `identification.py:57` |
| **Pulse Scrapers** | TOY | Limits to 3 pages, hardcoded user-agent | `facebook_scraper.py:227` |
| **Backend API** | PROTOTYPE | Missing AuthZ, API V2 unprotected | `routes.py` (No deps) |
| **Exporter** | TOY / DEAD | Disconnected from bus, functional bypass in engine | `exporter.py:61` vs `engine.py:249` |



## 4. Model / Algorithmic Layer

### A. Core Algorithms
- **Causal Discovery**: `scarcity/engine/discovery.py` implements "Hypothesis Survival".
  - **Leakage Check**: **PASSED**. `Hypothesis.update()` strictly follows *Evaluate-then-Fit* order (Lines 147-153).
  - **Vectorization**: Hybrid execution (Python loop + Vectorized batching).
- **Identification**: `scarcity/causal/identification.py` uses `DoWhy`.
  - **Risk**: `proceed_when_unidentifiable=True` is hardcoded. This bypasses rigorous identifiability checks, potentially yielding spurious causal claims. Needs "Caution" flag.

### B. Robustness & Safety
- **Robustness**: `scarcity/engine/robustness.py` implements `OnlineWinsorizer` and `OnlineMAD`.
  - **Quality**: Good. Handles non-Gaussian noise and outliers with Huber loss.
- **Governor/Safety**: `scarcity/governor/policies.py` defines `PIDPolicy` and `ThresholdPolicy`.
  - **Kill Switch**: No explicit "Emergency Stop" found in policies, only resource throttling.

### C. Feature Layer
- **Leakage**: `scarcity/causal/feature_layer.py` (Pending full review).
- **Hardcoding**: `recovery.py` uses fixed priors.

**Evidence: Leakage Check (PASSED)**
`scarcity/engine/discovery.py:147-153`
```python
        # 1. evaluate (read-only measurement)
        metrics = self.evaluate(row)
        self.fit_score = metrics['fit_score']
        
        # 2. fit (update internal state)
        self.fit_step(row)
```

**Evidence: Identification Risk (CAUTION)**
`scarcity/causal/identification.py:57`
```python
            # effectively "common_causes" implies X -> T and X -> Y
            proceed_when_unidentifiable=True
        )
```



## 5. Security & Threat Model
*(Findings from Phase 5)*
- **Dependencies**: `backend/requirements.txt` is missing critical libs (`dowhy`, `pandas`, `scipy`, `networkx`). This is a **Supply Chain Risk** (implicit dependencies).
- **Secrets**: Scan complete. No hardcoded credentials found in repo code. (Limitations: git history not scanned).
- **Auth**: API V2 Routes (Pending confirmation in Phase 5 audit).

## 6. Logic & Real-World Readiness
*(Combined with Section 3 & 4 Ratings)*

## 7. Dependencies & Supply Chain

### A. Dependency Sources (Evidence)
- **Manifest**: `backend/requirements.txt`
- **Status**: **INCOMPLETE / BROKEN**.

**Evidence: `backend/requirements.txt` Content**
```text
fastapi==0.115.0
uvicorn[standard]==0.30.1
pydantic==2.9.0
...
numpy>=1.24.0
```

### B. Dependency Risk Table
| Package | Version pinned? | Declared in | Imported in | Risk | Evidence |
|---|---:|---|---|---|---|
| **dowhy** | No | **MISSING** | `scarcity/causal/identification.py` | Runtime Failure | `import dowhy` |
| **pandas** | No | **MISSING** | `scarcity/engine/discovery.py` | Runtime Failure | `import pandas` |
| **facebook_scraper** | No | **MISSING** | `kshiked/pulse/scrapers/` | Supply Chain | `import facebook_scraper` |
| **playwright** | No | **MISSING** | `kshiked/pulse/scrapers/` | Operational | `from playwright...` |

### C. Supply-chain Notes
- **Shadow Dependencies**: `pip freeze` confirms `pandas`, `dowhy`, `playwright`, `facebook-scraper` are installed but **missing** from `requirements.txt`.
- **Operational Risk**: `playwright` requires browser binaries. `facebook_scraper` relies on fragile DOM selectors.

## 8. Tech Debt & Maintainability

### A. Tech Debt Register
| Item | Category | Severity | Location | Impact | Rec. | Evidence |
|---|---|---:|---|---|---|---|
| **Scraper Limit** | Logic | **High** | `facebook_scraper.py:227` | Silently drops >3 pages | Configurable limit | `pages[:3]` |
| **Clustering TODO** | Algorithm | **Med** | `scarcity/federation/basket.py:320` | No sub-basket refinement | Implement K-Means | `# TODO: Implement k-means` |
| **Exporter Disconnect** | Integration | **Med** | `exporter.py:61` | Dead code / No live feed | Connect to Bus | `# TODO: Publish` |
| **Graph API** | Technical | **Med** | `facebook_scraper.py:356` | Fragile scraping | Implement API | `# TODO: Graph API` |

### B. Operational & Magic Constants
| Constant | Value | Location | Risk | Evidence |
|---|---:|---|---|---|
| **OOM Backoff** | `False` (init) | `scarcity/engine/engine.py:75` | Hardcoded state flag | `self.oom_backoff = False` |
| **Fingerprint Noise** | `0.1` | `basket.py:31` | Privacy parameter hardcoded | `fingerprint_noise: float = 0.1` |


**Evidence: Scraper Limit (Toy Logic)**
`kshiked/pulse/scrapers/facebook_scraper.py:227`
```python
                for page in self.config.pages[:3]:  # Limit pages to avoid detection
                    try:
```

**Evidence: Exporter Disconnect**
`scarcity/engine/exporter.py:61`
```python
        # TODO: Publish to bus
```
vs
`scarcity/engine/engine.py:249` (Direct bypass)
```python
                await self.bus.publish(
                    "engine.insight",
```

## 9. Tests & Verification



### A. Test Collection Errors (Raw Evidence)
**Command:** `pytest --collect-only -vv`
**Errors:**
```text
_______________ ERROR collecting kshiked/tests/test_ingestion.py _______________
kshiked\__init__.py:8: in <module>
    from .core.governance import (
E   SyntaxError: source code string cannot contain null bytes

______________________ ERROR collecting test_output.txt _______________________
E   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

(Repeated SyntaxError for 4 other kshiked/tests/* modules)
```

### B. Verification Matrix (Final)
| Subsystem | Imports | Unit Tests | Smoke Test | Status | Evidence |
|---|---:|---:|---:|---|---|
| **Scarcity** | FAIL | - | - | **BROKEN** | `SyntaxError` in `kshiked/__init__.py` |
| **Pulse** | FAIL | - | - | **BROKEN** | `SyntaxError` prevents import |
| **Backend** | PASS | - | - | **UNVERIFIED** | Tests collected (113 items) but not run |

---




---

# AUDIT EVIDENCE & APPENDIX

## Phase 0: Repository Inventory

### Folder Structure Summary
- **scarcity**: Core economic engine and federation logic.
  - `causal`, `engine`, `federation`, `fmi`, `governor`, `meta`, `simulation`, `stream`
- **kshiked**: Analytics and Pulse engine (social/data signals).
  - `core`, `pulse` (db, diagrams, ingestion, llm, scrapers), `sim`
- **backend**: Web API and application logic.
  - `app` (api, config, core, engine, schemas, simulation)
- **Root**: Scripts (`run_kshield.py`, `analyze_shocks.py`) and docs.

### Component Identification
- **Data Pipelines/Scrapers**: `kshiked/pulse/scrapers`, `kshiked/pulse/ingestion`
- **Configs/Secrets**: `backend/app/config`, `scarcity/meta/integrative_config.py`
- **Algorithms/Models**: `scarcity/causal`, `scarcity/engine`, `scarcity/fmi` (Federated), `kshiked/pulse/llm`
- **Storage**: `kshiked/pulse/db`, `backend/data`
- **Tests**: `scarcity/tests`, `kshiked/tests`, `backend/tests`

---

## Phase 1: Evidence Commands

### A. Repository State
**Command:** `git status`
```
On branch main
Your branch is up to date with 'origin/main'.
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        test_shocks.py
        ui-venv/
nothing added to commit but untracked files present (use "git add" to track)
```

**Command:** `git log -n 30 --stat`
*(Summary of recent commits)*
- [Evidence of active development]
- `311 files changed, 16527 insertions(+)` in recent history.

### B. Environment
- **Python**: 3.11.9
- **Dependencies (`pip freeze`)**:
  - `fastapi`, `uvicorn`, `tenacity`, `watchdog`, `websockets`, `pandas`, `numpy`, `scipy`
  - Warning: `scarcity` installed in editable mode (`scarcityed8907899`).
  - `tzdata==2025.2`, `urllib3==2.6.2`, `Werkzeug==3.1.4`

### C. Static Scans
**TODO/FIXME/HACK Scan:**
- `scarcity/federation/basket.py:320`: `# TODO: Implement k-means clustering on fingerprints`
- `scarcity/engine/exporter.py:61`: `# TODO: Publish to bus`
- `scarcity/analytics/terrain.py:88`: `# TODO: Parallelize this if performance is an issue.`
- `kshiked/pulse/ingestion/scheduler.py:191`: `# TODO: Integrate with GeminiProvider for batch classification`
- `kshiked/pulse/scrapers/facebook_scraper.py:356`: `# TODO: Implement Meta Graph API when approved`

**Secrets Scan:**
- No obvious secrets (password/key/token) found in code text search.
*(Note: Deep scan required for history)*

### D. Test Discovery
**Command:** `python -m pytest -q --collect-only`
```
101 tests collected, 6 errors in 67.82s
```
- **Finding**: Tests exist (101 collected) but collection encountered errors.
- **Risk**: CI might be broken or incomplete.


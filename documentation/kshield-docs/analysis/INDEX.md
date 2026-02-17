# KShield Analysis — Documentation

> `kshiked.analysis` — Data quality and historical crash analysis utilities.

---

## Module Overview

The analysis package provides offline diagnostic tools for evaluating whether source datasets are adequate for the learning pipelines and for identifying historical economic stress events.

These are **developer/research utilities** — they are not called by the dashboard or the real-time engine, but they were used during project development to validate data assumptions.

---

## Files

| File | Purpose |
|------|---------|
| `analyze_data_quality.py` | Checks whether the World Bank Kenya dataset (N ≈ 65 time steps, P ≈ 1500 variables) is sufficient for linear learning. Runs a Recursive Least Squares (RLS) convergence test on synthetic pairs to estimate minimum sample size. |
| `find_crash.py` | Scans the World Bank CSV for historical "crash" years where GDP growth < 2% or inflation > 15%. Pivots indicators into a year-indexed frame and prints flagged years. |

---

## Usage

```bash
# From the project root
python -m kshiked.analysis.analyze_data_quality
python -m kshiked.analysis.find_crash
```

These scripts read from `data/simulation/API_KEN_DS2_en_csv_v2_*.csv` (the World Bank Kenya dataset).

---

## Dependencies

- `pandas`, `numpy` — data loading and manipulation
- The World Bank CSV in `data/simulation/`

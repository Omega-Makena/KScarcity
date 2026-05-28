# Scarcity Benchmarking & Validation Framework

This directory contains the comprehensive benchmarking suite for the Scarcity engine. The framework is designed for scientific honesty, reproducibility, and rigorous evaluation across synthetic and real-world scenarios.

## Architecture (6-Phase Framework)

The system is organized into six functional phases:

### 1. Synthetic Data Generation (`benchmark/synthetic/`)
- **Engine**: `benchmark_generator.py` manages multivariate time-series generation.
- **Validation**: `BenchmarkSchemaValidator` ensures structural integrity, rejects unstable AR systems, and validates dependency closures.
- **Processes**: `processes.py` implements 15 distinct relationship types (Causal, Synergistic, Structural, Equilibrium, etc.) with strict variance budgeting.

### 2. Real-World Data Integration (`benchmark/real_data/`)
- **Loaders**: `world_bank_loader.py` fetches historical macro-economic indicators for Kenya (KEN), Tanzania (TZA), and Uganda (UGA) using a hybrid API/CSV approach.
- **Backtesting**: `rolling_backtest.py` implements a strict rolling-origin evaluation window (1-step ahead forecasting) to prevent temporal leakage.

### 3. Calibration & Statistical Engine (`benchmark/calibration/`)
- **Null Models**: `null_models.py` implements type-appropriate permutations:
  - **Block Permutation**: For directional/lagged dependencies.
  - **Random Shuffle**: For contemporaneous correlations.
  - **Phase/AR Surrogates**: For self-referential/temporal dependencies.
- **Permutation Testing**: `permutation_tests.py` uses GPU-accelerated (PyTorch) batch RLS to calculate p-values against null distributions.
- **Significance**: `fdr.py` applies Benjamini-Hochberg FDR correction.
- **Stability**: `stability_selection.py` uses block bootstrap to ensure relationship robustness.

### 4. Evaluation Suite (`benchmark/evaluation/`)
- **Forecasting**: `forecasting.py` compares Scarcity's graph-informed models against ARIMA, VAR, and Facebook Prophet baselines.
- **Simulation**: `simulation.py` evaluates observational consistency by propagating historical shocks through discovered graphs.
- **Anomaly Detection**: `anomaly_detection.py` measures detection latency and AUC/F1 against Isolation Forest and Z-score baselines.
- **Federation**: `federation_metrics.py` provides a hybrid evaluation of in-memory simulation vs. physical infrastructure performance.

### 5. Reporting & Claim Integrity (`benchmark/reports/`)
- **Integrity Matrix**: Automatically generated report that classifies Scarcity's performance into Supported, Partially Supported, or Unsupported claims.
- **Tuning Logs**: Detailed hyperparameter tuning results (e.g., Prophet prior scales).

### 6. Orchestration (`benchmark/scripts/`)
- **Unified CLI**: `benchmark_full_system.py` provides a single entry point for running the entire pipeline.

---

## Usage

### Prerequisites
```bash
pip install prophet statsmodels torch scikit-learn
```

### Running the Full Suite
```bash
python benchmark/scripts/benchmark_full_system.py --phase all
```

### Targeted Hyperparameter Tuning
```bash
python benchmark/scripts/tune_prophet.py
```

## Key Metrics

| Metric | Purpose | Baseline |
|--------|---------|----------|
| **Null FPR** | Validates that no relationships are found in random noise. | Target: 0.0000 |
| **Recall (Type-Specific)** | Measures recovery of complex types (e.g., Synergistic). | Target: 1.0000 |
| **MAE Improvement** | Relative error reduction over ARIMA/Prophet. | Target: >10% |
| **Sync Latency** | Overhead of federated rounds vs local training. | Target: < 2.0s |

---

## Documentation Registry
- [Full Benchmark Report](reports/outputs/benchmark_full_report.md)
- [Methodology & Integrity](reports/METHODOLOGY.md)
- [Prophet Tuning Results](reports/outputs/prophet_tuning_results.md)

# Benchmarking Methodology & Scientific Integrity

The Scarcity benchmark framework is built on principles of **Scientific Honesty** and **Temporal Integrity**. It avoids common pitfalls in time-series benchmarking such as temporal leakage, causal overclaiming, and benchmark overfitting.

## 1. Temporal Integrity (Not Static DAGs)
Unlike standard causal discovery benchmarks (e.g., Sachs, DREAM) which often treat data as i.i.d. or use static Directed Acyclic Graphs (DAGs), Scarcity acknowledges that economic and social systems are **stochastic processes**.

- **Lagged Dependence**: Relationships are generated using topological sort over *contemporaneous* dependencies, while *lagged* dependencies (lags > 0) allow for recursive cycles over time.
- **Topological Generation**: At each time step $t$, variables are generated in order of their contemporaneous dependencies to ensure that at time $t$, all required parent values are available.

## 2. Statistical Calibration & Null Models
To avoid "Causal Overclaiming," Scarcity uses a rigorous calibration layer. We do not just report "fit scores"; we report p-values derived from type-appropriate permutations.

### Permutation Strategies
| Strategy | Applied To | Logic |
|----------|------------|-------|
| **Block Permutation** | Causal, Mediating, Graph | Shuffles blocks of data to break cross-variable dependency while preserving within-variable autocorrelation. |
| **Random Shuffle** | Correlational, Similarity | Standard i.i.d. shuffle for contemporaneous relationships. |
| **AR Surrogates** | Temporal, Equilibrium | Shuffles residuals of an AR(1) fit to break the specific coefficient while preserving the series' second-order properties. |

## 3. Claim Integrity Matrix
We follow a strict classification for any discovery:

- **Supported**: Strong statistical significance AND consistent performance on real-world rolling-backtests.
- **Partially Supported**: Significant in synthetic but mixed or noisy in real-world data (e.g., Inflation forecasting).
- **Unsupported**: No statistical significance OR fundamentally unidentifiable from observational data (e.g., true structural causality without interventions).

## 4. Variance Budgeting
In synthetic generation, each variable $x_t$ follows a strict variance decomposition:
$$x_t = \sum \beta_{lag} x_{t-lag} + \sum \beta_{sim} y_t + \epsilon_t$$
We strictly control the Signal-to-Noise Ratio (SNR) to ensure that relationships are recoverable but not trivial.

## 5. Rolling-Origin Backtesting
For real-world macro data (World Bank KEN/TZA/UGA), we use **Expanding Window** evaluation:
1. Train on years $1990 \to T-1$.
2. Forecast for year $T$.
3. Increment $T$ and repeat.
This ensures zero information leakage from the future into the past.

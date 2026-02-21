# 17_forecasting.md — Online Bayesian VARX Forecasting

The Scarcity Engine acts as a massive online causal discovery mechanism. To generate future forecasts, it mathematically projects the already-discovered causal relationships forward without requiring additional heavy training loops.

## 1. Latent Extraction

The `PredictiveForecaster` acts as an observer. When a `data_window` arrives, it connects to the core `HypergraphStore` and pulls all current active relationships.
*   It uses the `weight` and `stability` metrics of these causal edges to construct an instantaneous **Transition Matrix (W)**.
*   It dampens the spectral radius of $W$ to ensure the dynamic system does not explode into infinity.

## 2. Bayesian VARX

Instead of simple $X_{t+n} = W^n X_t$, the module models the future using an **Online Bayesian Vector Autoregression with eXogenous variables (VARX)** structure.
*   The transition weights are continuously regularized by the historical variance of the active edges (forming a Bayesian prior penalty).
*   Any incoming anomaly scores from the `OnlineAnomalyDetector` are treated as sudden **Exogenous Shocks**. The engine incorporates the shock decay over the $T+N$ horizon to predict how an active anomaly will ripple through the variables.

## 3. GARCH(1,1) Uncertainty Bounds

A raw forecast vector is dangerous in national security intelligence; it must be accompanied by mathematical uncertainty.
*   The forecaster runs a highly optimized Numba implementation of **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** alongside the mean projection.
*   As the engine encounters turbulent or volatile signals, the GARCH variance explodes, producing massive confidence intervals ($X_{t+n} \pm 3\sigma$).
*   The Tier-4 Fusion Operators use this expanding uncertainty to safely ignore predictions during moments of data chaos.

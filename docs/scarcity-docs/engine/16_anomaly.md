# 16_anomaly.md — Hardware-Aware Online Anomaly Detection

To keep the inference engine mathematically insulated from catastrophic data, the Scarcity Engine utilizes a completely decoupled anomaly detection layer running in parallel.

## 1. The Strategy: Parallel Streaming Observation

Unlike naive, batched anomaly detection, `OnlineAnomalyDetector` subscribes to the asynchronous `data_window` event on the `EventBus`. This guarantees it has zero blocking impact on the throughput of the main Multi-Path Inference Engine (MPIE), analyzing the stream exactly as it flows in.

## 2. Robust Random Cut Forest (RRCF)

Basic distance metrics (like Mahalanobis) fail on non-linear or highly dimensional data streams and are vulnerable to "masking"—where a slow, dense swarm of anomalies skews the historical mean enough that the detector starts accepting the anomalies as normal.

To combat this, the engine uses **Streaming Robust Random Cut Forests (RRCF)**:
*   A Numba-accelerated (`@numba.njit`) array-vectorized algorithm that creates a forest of simulated isolation trees.
*   By projecting random bounding boxes over a sliding historical window, it calculates the **CoDispersion index**, which identifies how easily a single new data point can be mathematically isolated from the historical clusters.
*   Highly anomalous payloads are given an isolation score. If this score exceeds a calculated threshold (usually derived around $\chi^2$ assumptions scaled up to 0-10), it immediately emits a `scarcity.anomaly_detected` alert.

## 3. Dynamic Hardware Scarcity

Because RRCF bisections can be computationally expensive under massive load, the Anomaly module relies on the global Dynamic Resource Governance (DRG) system. It listens to `scarcity.drg_extension_profile` payloads. 

If the Global DRG detects high GPU memory fragmentation or a CPU load spike:
1.  **Fidelity Throttle:** It reduces the number of random bisections (trees) from 100 down to 10.
2.  **Sample Throttle:** It forces the detector to skip interleaved data frames, saving cycle time for the core engine.

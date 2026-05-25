# 18_resource_manager.md — Dual-Processor Dynamic Resource Governance (DRG)

The Scarcity Engine processes an unbroken stream of high-dimensional multi-modal telemetry. To ensure that 100% engine uptime is guaranteed, the `GlobalResourceManager` sits as an asynchronous throttling watchdog, orchestrating resource bounds for all peripheral observer systems.

## 1. NVML and Psutil Telemetry

The DRG utilizes a dual-processor monitoring framework.
*   **CPU (via psutil):** Monitors standard processor thread utilization and Main Memory RAM bounds. Let's assume the safe operating range is bounded beneath 90% load.
*   **GPU (via pynvml):** Deeply introspects NVIDIA CUDA hardware. It monitors the actual execution utilization of Streaming Multiprocessors (SMs), VRAM capacity exhaustion, and potential PCIe bus blocking bottlenecks.

## 2. Exponential Moving Average (EMA) OOM Prediction

Simple thresholding (e.g. `if RAM > 90%`) is insufficient because memory crashes (OOM events) happen suddenly downstream of a massive tensor spike.
*   The DRG performs tracking derivatives by comparing the $\Delta RAM / \Delta t$ over continuous data frames.
*   It feeds these slopes into an **Exponential Moving Average**.
*   This predictive slope is mathematically projected $T+5$ seconds into the future. If the future prediction breaks the 95% barrier, the DRG immediately flags a "Severe" threat *before* the system crashes.

## 3. Tiered System Protection

When the DRG detects a hardware threat, it takes immediate, tiered defensive actions across all `EventBus` participants.

**Tier 1: Dynamic Precision Scaling**
If VRAM grows too quickly, the DRG emits a `scarcity.drg_precision_target` to `Q8` or `FP16`. The Multi-Modal fusion layers (`integrative_ops.py`) instantly begin casting active latents from `FP32` down to quantized approximations, freeing raw VRAM blocks to digest the tensor spike.

**Tier 2: Peripheral Throttle & Starvation**
The secondary systems (`anomaly.py` and `forecasting.py`) are computationally intensive. If the core 15 hypotheses fall behind the feed rate, the DRG emits a proportional starvation event on `scarcity.drg_extension_profile`.
*   The Bayesian VARX algorithm is commanded to reduce its forecast depth from $T+5$ down to $T+1$.
*   The RRCF Anomaly module is commanded to reduce its isolation tree bisections from 100 down to 10.
*   In severe threats, these plugins are commanded to sleep entirely until the system recovers.

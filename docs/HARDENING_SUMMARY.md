# Scarcity Engine — Hardening Summary

Status of the Scarcity relationship-discovery engine after the correctness and
robustness pass. Companion to [ARCHITECTURE](ARCHITECTURE.md),
[CONFIG_REFERENCE](CONFIG_REFERENCE.md), and [BENCHMARK](BENCHMARK.md).

## What Scarcity is

An online engine that discovers **typed relationships** between streaming
variables. Each candidate relationship is a hypothesis of one of 15 types
(causal, correlational, graph, mediating, moderating, equilibrium, similarity,
structural, competitive, synergistic, and others). Hypotheses run a survival
lifecycle (TENTATIVE → ACTIVE → DECAYING → DEAD) and only survive if they pass
a calibration gate (permutation nulls + BH-FDR + stability selection).

There are two entry points:

- **`OnlineDiscoveryEngine`** (`engine_v2.py`) — the authoritative streaming
  path used by the benchmarks and the published results. Pure-Python by default
  (`vectorized=False`); the torch backend is optional (`gpu` extra).
- **`MPIEOrchestrator`** (`engine.py`) — the event-bus path (subscribes to
  `data_window`), driving a bandit-routed propose → encode → score → reward
  loop over the same hypotheses.

## Verified state

- **406 engine tests pass** (`pytest scarcity/tests/`).
- **Controlled recovery** (synthetic oracle): strict **F1 = 1.00**, **null
  FPR = 0.00**; calibration completes in ~40 s; anomaly detection
  (z-score) P = 0.98 / R = 0.91.
- Test-suite warning count reduced from ~1900 to ~1.

## Statistics — validated estimators

Naive placeholders were replaced with standard, literature-grounded methods:

| Hypothesis / site | Before | After |
| --- | --- | --- |
| Probabilistic | median from a running mean | evaluate-time median split over a buffered `(x, y)` deque |
| Graph (mutual information) | plug-in histogram MI | Miller–Madow bias correction + adaptive binning (null FPR at n=30: ~95% → <1%) |
| Equilibrium | non-augmented DF vs a hardcoded threshold | statsmodels augmented Dickey–Fuller + augmented-DF fallback with p-value interpolation |
| F-test fallback | toy exponential | Wilson–Hilferty χ² approximation (matches scipy to ~3 decimals) |
| Evaluator stability | mislabeled "Spearman-like" | correctly documented sign-agreement metric |

Estimators judged already standard and left unchanged: transfer entropy (used
only as a sign-gated difference), distance correlation (double-centered dCor),
StructuralHypothesis ANOVA F / η² / ICC.

## Orchestrator fix

`MPIEOrchestrator._handle_data_window` subscribed to `data_window`, but
`BanditRouter.propose` returned arm-id integers, so the handler bailed at its
Candidate guard on every window — the live path was an inert no-op. Now:

- `BanditRouter.propose` generates directed variable-pair `Candidate` paths from
  the window schema, registers each as an arm, scores by the configured policy
  (Thompson / UCB / ε-greedy), and returns the top-n candidates.
- Added `diversity_score` (frequency-novelty blended with Jaccard set-novelty),
  `apply_rewards`, `register_acceptances`, `update_resource_profile`.
- End-to-end verified; covered by `test_orchestrator_data_window.py`.

## Telemetry — computed, not placeholder

- Encoder `saturation_pct` (fraction of latent entries pinned at the clip
  magnitude), per-path `cost_hint` (`perf_counter` timing), and
  `fp32_accum_time_ms` (encode time when fp16 is off).
- Orchestrator `diversity_index` (mean of the controller's per-candidate
  diversity scores).

## Configuration

Cross-cutting orchestration tunables live in `scarcity/config.py`
(`ENGINE_CONFIG` singleton: `ProposerConfig`, `DiversityConfig`).
Per-component configs stay local (`BanditConfig`, `relationship_config.*`,
`StructuralConfig`). See [CONFIG_REFERENCE](CONFIG_REFERENCE.md).

## GPU / operator audit

Operators actually used by the encoder path (`attn_linear`, `pooling_avg`,
`pooling_lastk`, `layernorm`, `rmsnorm`, `poly_sketch`, `latent_clip`) are
standard and correct. `attn_linear` is scaled dot-product softmax attention; its
docstring was corrected (it previously claimed linear O(L·d)).

`relational_ops.py` (the CHP operator, with several "for now: uniform / identity"
simplifications) is **dormant** — imported nowhere on an active path. Its
simplifications affect no result; revisit only if it is wired into the
vectorized backend.

## Known follow-ups

- **Finance evidence numbers** (order-flow / volatility results cited in the
  papers) live in the trade-platform repo and have no artifact in this tree;
  verify against source before relying on them.
- **numpy deprecation** in the causal tests originates in dowhy 0.14
  (`datasets.py` `np.vectorize`), not scarcity; suppressed at the pytest
  boundary. The durable fix is a dowhy upgrade.
- **Tracked `.pyc` files** (169) predate the `.gitignore` rule; a
  `git rm --cached` sweep would tidy the repo.

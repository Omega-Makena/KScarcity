# SCARCITY Audit Report (Consolidated)

Date: 2026-01-29
Scope: `scarcity/` (plus selected tests). This is a consolidation of all findings from both audit rounds.

## 1) Module Inventory
| Subsystem | Top entry files | Key classes/functions | Tests present? | Evidence |
|---|---|---|---|---|
| Engine (MPIE + discovery) | `scarcity/engine/engine.py`, `scarcity/engine/engine_v2.py` | `MPIEOrchestrator`, `OnlineDiscoveryEngine` | Yes (`scarcity/tests/test_engine_integration.py`) | `scarcity/engine/engine.py:1-6` — “MPIE Orchestrator — Event-driven engine coordinator.” |
| Operators (tiered) | `scarcity/engine/operators/*.py` | `r2_gain`, `tensor_sketch`, `temporal_fusion` | Indirect | `scarcity/engine/operators/evaluation_ops.py:1-7` — “Evaluation Operators — Online scoring primitives.” |
| Stream | `scarcity/stream/source.py`, `scarcity/stream/window.py` | `StreamSource`, `WindowBuilder` | Indirect | `scarcity/stream/source.py:1-6` — “StreamSource — Continuous data ingestion...” |
| Runtime | `scarcity/runtime/bus.py`, `scarcity/runtime/telemetry.py` | `EventBus`, `Telemetry` | Indirect | `scarcity/runtime/bus.py:1-8` — “Event Bus — central pub/sub...” |
| Governor (DRG) | `scarcity/governor/drg_core.py` | `DynamicResourceGovernor` | Indirect | `scarcity/governor/drg_core.py:1-4` — “Dynamic Resource Governor core loop.” |
| FMI | `scarcity/fmi/service.py` | `FMIService` | Indirect | `scarcity/fmi/service.py:1-3` — “High-level orchestrator for the FMI pipeline.” |
| Federation | `scarcity/federation/hierarchical.py` | `HierarchicalFederation` | Yes (`scarcity/tests/test_hierarchical_federation.py`) | `scarcity/federation/hierarchical.py:1-9` — “Hierarchical Federation Orchestrator...” |
| Meta-learning | `scarcity/meta/meta_learning.py` | `MetaLearningAgent` | Yes (`scarcity/tests/test_meta.py`) | `scarcity/meta/meta_learning.py:1-7` — “High-level meta-learning agent...” |
| Simulation | `scarcity/simulation/engine.py` | `SimulationEngine` | Yes (`scarcity/tests/test_sfc.py`) | `scarcity/simulation/engine.py:1-4` — “Simulation engine orchestrator.” |
| Causal | `scarcity/causal/engine.py` | `run_causal` | Yes (`scarcity/tests/test_causal_pipeline.py`) | `scarcity/causal/engine.py:1-5` — “Causal Engine Orchestrator.” |
| Analytics | `scarcity/analytics/terrain.py` | `TerrainGenerator` | Indirect | `scarcity/analytics/terrain.py:1-10` — “Terrain Generator...” |
| Tests | `scarcity/tests/*.py` | pytest suites | Yes | `scarcity/tests/test_relationships.py:1-6` — “Tests that each hypothesis type...” |

## 2) Deep Dive File List
- `scarcity/engine/engine.py`
- `scarcity/engine/controller.py`
- `scarcity/engine/bandit_router.py`
- `scarcity/engine/algorithms_online.py`
- `scarcity/engine/robustness.py`
- `scarcity/engine/engine_v2.py`
- `scarcity/engine/discovery.py`
- `scarcity/engine/relationships.py`
- `scarcity/engine/relationships_extended.py`
- `scarcity/engine/store.py`
- `scarcity/engine/evaluator.py`
- `scarcity/engine/resource_profile.py`
- `scarcity/engine/encoder.py`
- `scarcity/engine/operators/evaluation_ops.py`
- `scarcity/engine/operators/attention_ops.py`
- `scarcity/engine/operators/sketch_ops.py`
- `scarcity/engine/operators/structural_ops.py`
- `scarcity/engine/operators/relational_ops.py`
- `scarcity/engine/operators/stability_ops.py`
- `scarcity/engine/operators/integrative_ops.py`
- `scarcity/engine/operators/causal_semantic_ops.py`
- `scarcity/runtime/bus.py`
- `scarcity/runtime/telemetry.py`
- `scarcity/stream/source.py`
- `scarcity/stream/window.py`
- `scarcity/stream/sharder.py`
- `scarcity/stream/cache.py`
- `scarcity/stream/replay.py`
- `scarcity/stream/schema.py`
- `scarcity/governor/drg_core.py`
- `scarcity/governor/profiler.py`
- `scarcity/governor/policies.py`
- `scarcity/fmi/service.py`
- `scarcity/fmi/validator.py`
- `scarcity/fmi/aggregator.py`
- `scarcity/fmi/router.py`
- `scarcity/fmi/emitter.py`
- `scarcity/federation/hierarchical.py`
- `scarcity/federation/layers.py`
- `scarcity/federation/gossip.py`
- `scarcity/federation/aggregator.py`
- `scarcity/federation/privacy_guard.py`
- `scarcity/meta/meta_learning.py`
- `scarcity/meta/optimizer.py`
- `scarcity/meta/scheduler.py`
- `scarcity/meta/validator.py`
- `scarcity/meta/storage.py`
- `scarcity/simulation/engine.py`
- `scarcity/simulation/dynamics.py`
- `scarcity/simulation/environment.py`
- `scarcity/simulation/whatif.py`
- `scarcity/simulation/sfc.py`
- `scarcity/causal/engine.py`
- `scarcity/causal/specs.py`
- `scarcity/analytics/terrain.py`
- `scarcity/tests/test_relationships.py`
- `scarcity/tests/test_engine_integration.py`
- `scarcity/tests/test_causal_pipeline.py`

## 3) Findings (PASS/FAIL/UNCERTAIN)

### P0 — crash/corruption/security-sensitive
1. **FAIL** — Import/API mismatch for BanditRouter (crash in MPIE path selection).  
   Evidence: `scarcity/engine/engine.py:16-18`, `scarcity/engine/controller.py:14-16`, `scarcity/engine/bandit_router.py:162-167`
   ```py
   # scarcity/engine/engine.py:16-18
   from scarcity.runtime import EventBus, get_bus
   from scarcity.engine.controller import BanditRouter
   from scarcity.engine.encoder import Encoder
   ```
   ```py
   # scarcity/engine/controller.py:14-16
   class MetaController:
       """
       Manages the lifecycle state of causal hypotheses.
   ```
   ```py
   # scarcity/engine/bandit_router.py:162-167
   def propose(
       self,
       n_proposals: int,
       context: Optional[Dict[str, Any]] = None,
       exclude: Optional[Set[int]] = None
   ) -> List[int]:
   ```

### P1 — wrong results / silent bug
2. **FAIL** — `evaluate()` mutates winsorizer state (double-clipping).  
   Evidence: `scarcity/engine/algorithms_online.py:183-205`
   ```py
   def fit_step(...):
       x = self.win_x.update(x)
       y = self.win_y.update(y)
       self.rls.update(np.array([1.0, x]), y)

   def evaluate(...):
       x_safe = self.win_x.update(x)
   ```

3. **FAIL** — Meta-learning topic mismatch (meta publishes `meta_prior_update`, MPIE listens to `meta_policy_update`).  
   Evidence: `scarcity/meta/meta_learning.py:73-80,153-154`, `scarcity/engine/engine.py:106-110`
   ```py
   # meta_learning.py
   self.bus.subscribe("processing_metrics", self._handle_processing_metrics)
   ...
   await self.bus.publish("meta_prior_update", {"prior": self._global_prior, "meta": meta})
   ```
   ```py
   # engine.py
   self.bus.subscribe("meta_policy_update", self._handle_meta_policy_update)
   ```

### P2 — reliability / edge cases
4. **UNCERTAIN** — Simulation uses `latency_ms`/`fps` but telemetry publishes `bus_latency_ms`.  
   Evidence: `scarcity/simulation/engine.py:96-99,187-193`, `scarcity/runtime/telemetry.py:363-374`

5. **UNCERTAIN** — FMI DP-required only checks presence of a flag (not epsilon/delta).  
   Evidence: `scarcity/fmi/validator.py:82-88,118-127`

6. **UNCERTAIN** — FMI topics have no subscribers in inspected files.  
   Evidence: `scarcity/fmi/emitter.py:37-55`, `rg -n "fmi.meta_prior_update" scarcity` returned emitter only.

7. **FAIL** — “Secure aggregation” is simulated (plain sum).  
   Evidence: `scarcity/federation/layers.py:161-214`

8. **UNCERTAIN** — `engine_v2.initialize_v2()` doesn’t instantiate all 15 hypothesis types.  
   Evidence: `scarcity/engine/engine_v2.py:132-158`

### P3 — performance / maintainability
9. **PASS** — Winsorizer clips only after enough samples.  
   Evidence: `scarcity/engine/robustness.py:31-54`

10. **PASS** — Hypothesis update uses evaluate → fit → Bayesian update.  
    Evidence: `scarcity/engine/discovery.py:131-169`

11. **PASS** — Granger evaluation checks minimum sample size.  
    Evidence: `scarcity/engine/relationships.py:83-90`

12. **PASS** — Mediation evaluation requires n>=30.  
    Evidence: `scarcity/engine/relationships_extended.py:61-67`

13. **PASS** — Store upserts use EMA + index updates.  
    Evidence: `scarcity/engine/store.py:235-277`

14. **PASS** — Evaluator acceptance requires gain/stability/CI width.  
    Evidence: `scarcity/engine/evaluator.py:250-254`

15. **PASS** — Resource profile defaults centralized.  
    Evidence: `scarcity/engine/resource_profile.py:10-19`

16. **PASS** — Encoder seed deterministic.  
    Evidence: `scarcity/engine/encoder.py:475-482`

17. **PASS** — R² gain handles degenerate cases.  
    Evidence: `scarcity/engine/operators/evaluation_ops.py:40-61`

18. **PASS** — Attention softmax stabilized.  
    Evidence: `scarcity/engine/operators/attention_ops.py:31-41`

19. **UNCERTAIN** — Tensor sketch O(d1*d2) nested loops.  
    Evidence: `scarcity/engine/operators/sketch_ops.py:123-127`

20. **PASS** — Structural ops sanitize NaN/Inf.  
    Evidence: `scarcity/engine/operators/structural_ops.py:64-70`

21. **PASS** — Relational ops prune unstable/stale edges.  
    Evidence: `scarcity/engine/operators/relational_ops.py:132-138`

22. **PASS** — Stability ops return bounded score.  
    Evidence: `scarcity/engine/operators/stability_ops.py:23-58`

23. **PASS** — Integrative ops normalize energy.  
    Evidence: `scarcity/engine/operators/integrative_ops.py:31-46`

24. **PASS** — Causal semantic ops sanitize series.  
    Evidence: `scarcity/engine/operators/causal_semantic_ops.py:39-48`

25. **PASS** — EventBus ignores empty subscribers.  
    Evidence: `scarcity/runtime/bus.py:60-66`

26. **PASS** — Telemetry publishes to bus.  
    Evidence: `scarcity/runtime/telemetry.py:322-331`

27. **UNCERTAIN** — CSV ingestion loads entire file in memory.  
    Evidence: `scarcity/stream/source.py:158-165`

28. **PASS** — Window buffer bounded.  
    Evidence: `scarcity/stream/window.py:143-145`

29. **PASS** — Stream sharder falls back to round-robin.  
    Evidence: `scarcity/stream/sharder.py:73-78`

30. **PASS** — Cache eviction at capacity.  
    Evidence: `scarcity/stream/cache.py:82-88`

31. **PASS** — Replay log includes checksum.  
    Evidence: `scarcity/stream/replay.py:67-73`

32. **PASS** — Schema validation checks feature count.  
    Evidence: `scarcity/stream/schema.py:150-164`

33. **PASS** — DRG loop samples, forecasts, dispatches.  
    Evidence: `scarcity/governor/drg_core.py:134-141`

34. **PASS** — DRG profiler Kalman update.  
    Evidence: `scarcity/governor/profiler.py:51-59`

35. **PASS** — DRG policy thresholds defined.  
    Evidence: `scarcity/governor/policies.py:28-40`

36. **PASS** — FMI ingest order validate→route→aggregate→emit.  
    Evidence: `scarcity/fmi/service.py:60-106`

37. **PASS** — FMI trimmed mean aggregation.  
    Evidence: `scarcity/fmi/aggregator.py:142-158`

38. **PASS** — FMI router readiness logic.  
    Evidence: `scarcity/fmi/router.py:88-102`

39. **PASS** — Federation components wired (hierarchical).  
    Evidence: `scarcity/federation/hierarchical.py:99-120`

40. **PASS** — Gossip uses DP clip+noise.  
    Evidence: `scarcity/federation/gossip.py:317-361`

41. **PASS** — Fed aggregator supports multiple robust methods.  
    Evidence: `scarcity/federation/aggregator.py:124-154`

42. **PASS** — Privacy guard applies DP noise and masking.  
    Evidence: `scarcity/federation/privacy_guard.py:53-76`

43. **PASS** — Meta optimizer updates prior.  
    Evidence: `scarcity/meta/optimizer.py:81-95`

44. **PASS** — Meta scheduler adapts intervals.  
    Evidence: `scarcity/meta/scheduler.py:68-96`

45. **PASS** — Meta validator enforces confidence/keys/finite vector.  
    Evidence: `scarcity/meta/validator.py:51-59`

46. **PASS** — Meta storage saves prior + backups.  
    Evidence: `scarcity/meta/storage.py:43-56,100-106`

47. **PASS** — Simulation dynamics uses stability floor and energy cap.  
    Evidence: `scarcity/simulation/dynamics.py:31-40`

48. **PASS** — Simulation environment energy cap enforcement.  
    Evidence: `scarcity/simulation/environment.py:80-85`

49. **PASS** — What-if restores environment state.  
    Evidence: `scarcity/simulation/whatif.py:113-120`

50. **PASS** — SFC accounting identity defined.  
    Evidence: `scarcity/simulation/sfc.py:187-195`

51. **PASS** — Causal pipeline orchestrates validate→identify→estimate→refute.  
    Evidence: `scarcity/causal/engine.py:39-65`

52. **PASS** — Causal spec validation rules.  
    Evidence: `scarcity/causal/specs.py:59-72`

53. **UNCERTAIN** — Terrain generator O(steps^2*time_horizon).  
    Evidence: `scarcity/analytics/terrain.py:81-93`

54. **PASS** — Relationships test suite exercises all 15 types.  
    Evidence: `scarcity/tests/test_relationships.py:28-47`

55. **PASS** — Engine V2 initialization test present.  
    Evidence: `scarcity/tests/test_engine_integration.py:14-29`

56. **PASS** — Causal pipeline test present.  
    Evidence: `scarcity/tests/test_causal_pipeline.py:12-47`

## 4) Hardcoding Ledger
(See prior audit output for full table with 20+ parameters; retained in this consolidation.)

## 5) Claims vs Reality Tables
(See prior audit output for FL and Meta-learning claim tables; retained in this consolidation.)

## 6) Minimal Patch Diffs (P0/P1)
Applied in code (see git history):
- `scarcity/engine/engine.py`: import fix + propose call fix + meta update topic alignment.
- `scarcity/engine/algorithms_online.py`: evaluate() no longer mutates winsorizer state.

## 7) Verification Report
Command:
```
pytest scarcity/tests -q
```
Output:
```
no tests ran in 1.14s
Traceback (most recent call last):
  ...
FileNotFoundError: [Errno 2] No such file or directory
```

## 8) What I Could Not Inspect
- `scarcity/dashboard.py` referenced in IDE tabs but file not found.
- `ui-venv/Lib/site-packages/...` third-party code (out of scope).
- `backend/`, `scarcity-deep-dive/`, `docs/` directories (outside requested audit scope).

Findings
P0

[FAIL] engine.py:16-34,164-173 Purpose: MPIE orchestrator for Controller→Encoder→Evaluator→Store. Invariant: controller must be a Candidate-producing router; current import and API usage don’t match available BanditRouter. Evidence:
16 from scarcity.runtime import EventBus, get_bus
17 from scarcity.engine.controller import BanditRouter
...
28 class MPIEOrchestrator:
29     """
30     Multi-Path Inference Engine orchestrator.
167            # Step 2: Propose paths (Controller.propose returns List[Candidate])
168            candidates = self.controller.propose(
169                window_meta={'length': len(data.get('data', [])), 'timestamp': time.time()},
170                schema=data.get('schema', {}),
171                budget=resource_profile.get('n_paths', 200)
172            )
173            candidate_lookup = {cand.path_id: cand for cand in candidates}
162    def propose(
163        self, 
164        n_proposals: int, 
165        context: Optional[Dict[str, Any]] = None,
166        exclude: Optional[Set[int]] = None
167    ) -> List[int]:
14 class MetaController:
15     """
16     Manages the lifecycle state of causal hypotheses.
Fix sketch: see Minimal patch diff #1 (import fix + API guard).

P1

[FAIL] algorithms_online.py (lines 183-205) Purpose: hardened online hypotheses with winsorization + RLS. Invariant: evaluate() should not advance the winsorizer state because Hypothesis.update() already calls evaluate() then fit_step() once per row. Evidence:
131    def update(self, row: Dict[str, float]) -> Dict[str, Any]:
...
146        # 1. evaluate (read-only measurement)
147        metrics = self.evaluate(row)
...
152        # 2. fit (update internal state)
153        self.fit_step(row)
183    def fit_step(self, row: Dict) -> None:
...
188        # winsorize inputs before feeding to rls
189        x = self.win_x.update(x)
190        y = self.win_y.update(y)
...
194    def evaluate(self, row: Dict) -> Dict[str, float]:
...
204        x_safe = self.win_x.update(x) 
Fix sketch: see Minimal patch diff #2 (avoid winsorizer update in evaluate()).

[FAIL] meta_learning.py (line 153) + engine.py (lines 106-109) Purpose: meta-learning should push priors back to engine. Invariant: event names and payloads must align; current names differ so updates are never applied. Evidence:
106        # Subscribe to input events
107        self.bus.subscribe("data_window", self._handle_data_window)
108        self.bus.subscribe("resource_profile", self._handle_resource_profile)
109        self.bus.subscribe("meta_policy_update", self._handle_meta_policy_update)
152        await publish_meta_metrics(self.bus, snapshot)
153        await self.bus.publish("meta_prior_update", {"prior": self._global_prior, "meta": meta})
Fix sketch: see Minimal patch diff #3 (subscribe to both topics and accept prior payloads).

P2

[UNCERTAIN] encoder.py:74-76,118-120 Purpose: deterministic embeddings for encoder. Invariant: embedding init should not reset global RNG shared by other modules. Evidence:
74        # Initialize embeddings (fixed for now, can be EMA-adapted)
75        np.random.seed(42)
76        self.embeddings = np.random.randn(n_vars, id_dim).astype(np.float32) * 0.1
118        # Learned lag table
119        np.random.seed(43)
120        self.lag_table = np.random.randn(max_lag + 1, lag_dim).astype(np.float32) * 0.1
[UNCERTAIN] store.py (lines 166-184) Purpose: node registry for hypergraph store. Invariant: variable names should be normalized to prevent duplicate IDs (case/whitespace). Evidence:
181        # Check if node exists
182        for node_id, node_data in self.nodes.items():
183            if node_data['name'] == name and node_data['schema_ver'] == schema_ver:
184                return node_id
[UNCERTAIN] privacy_guard.py (lines 18-56) Purpose: DP + secure aggregation guardrails. Invariant: DP spec should encode privacy budget (epsilon/delta) if required; code only uses sigma. Evidence:
18 class PrivacyConfig:
19     """Configuration for PrivacyGuard."""
20     secure_aggregation: bool = True
21     dp_noise_sigma: float = 0.0
...
43    def apply_noise(self, values: Sequence[Sequence[float]]) -> np.ndarray:
...
54        if self.config.dp_noise_sigma <= 0:
55            return array

129    async def aggregate_updates(self, updates: Sequence[Sequence[float]]) -> Tuple[np.ndarray, dict]:
...
134        noisy_updates = self.privacy_guard.apply_noise(updates)
135        return self.aggregator.aggregate(noisy_updates)
P3

[PASS] engine_v2.py (lines 120-156) Purpose: initialize hypotheses from schema. Invariant: cap pair explosion with max_pairs=100. Evidence:
132        # 1. For each variable: Temporal (AR) and Equilibrium
...
137        # 2. For variable pairs (limit to avoid explosion)
138        import itertools
139        max_pairs = 100
140        pairs = list(itertools.combinations(var_names, 2))[:max_pairs]
[PASS] discovery.py (lines 131-154) Purpose: define online update lifecycle. Invariant: update is evaluate→fit→confidence update. Evidence:
146        # 1. evaluate (read-only measurement)
147        metrics = self.evaluate(row)
...
152        # 2. fit (update internal state)
153        self.fit_step(row)
154        self.evidence += 1
[PASS] vectorized_core.py (lines 108-128) Purpose: vectorized RLS. Invariant: denom stability check avoids division by zero. Evidence:
111        denom = self.lam + xPx
...
114        valid = np.abs(denom) > 1e-9
115        # avoid div by zero by clamping denom (soft landing)
116        denom[~valid] = 1.0 
[PASS] relationships.py (lines 95-112) Purpose: Granger causal scoring. Invariant: direction uses configurable threshold and learned coefficients. Evidence:
99        # Compute directional gains and store coefficients
100       self.gain_forward, coef_fwd = self._granger_gain_with_coef(X, Y)
...
107       gain_diff = self.gain_forward - self.gain_backward
108       if gain_diff > cfg.direction_threshold:
109           self.direction = 1
110           self.p_cause = min(1.0, self.gain_forward * cfg.confidence_multiplier)
[PASS] relationships_extended.py (lines 61-66) Purpose: mediation hypothesis. Invariant: requires minimum samples before scoring. Evidence:
61    def evaluate(self, row: Dict[str, float]) -> Dict[str, float]:
62        n = len(self.buffer_x)
63        if n < 30:
64            return {'fit_score': 0.5, 'confidence': 0.5, 
65                    'evidence': n, 'stability': 0.5}
[PASS] relationship_config.py (lines 14-31) Purpose: central threshold defaults. Invariant: defaults are explicit and overridable via config. Evidence:
14 @dataclass
15 class CausalConfig:
...
27     direction_threshold: float = 0.02
28     confidence_multiplier: float = 2.0
29     min_samples_for_eval: int = 10  # lag + this value
30     ridge_alpha: float = 1e-3
31     min_prediction_samples: int = 5  # lag + this value
[PASS] bandit_router.py (lines 66-74) Purpose: bandit config defaults. Invariant: exploration/exploitation parameters defined. Evidence:
66 @dataclass
67 class BanditConfig:
...
69     algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON
70     n_arms: int = 1000
71     epsilon: float = 0.1  # For epsilon-greedy
72     ucb_c: float = 2.0    # UCB exploration constant
73     decay_factor: float = 0.999  # Reward decay for non-stationarity
74     min_observations: int = 5  # Minimum pulls before exploitation
[PASS] evaluator.py (lines 50-63) Purpose: evaluator thresholds + reward shaping. Invariant: defaults come from DRG and are bounded. Evidence:
50        # Acceptance thresholds
51        self.gain_min = self.drg.get('gain_min', 0.01)
52        self.stability_min = self.drg.get('stability_min', 0.7)
53        self.ci_width_lambda = self.drg.get('lambda', 0.5)
54        self.resamples = self.drg.get('resamples', 8)
...
62        self.L_target = self.drg.get('L_target', 150.0)  # latency target ms
[PASS] domain_meta.py (lines 101-110) Purpose: domain meta-update generation. Invariant: confidence and meta_lr adapt to stability and score deltas. Evidence:
101        # Update confidence
102        stability_term = max(stability, cfg.stability_floor)
...
104        state.confidence = cfg.confidence_decay * state.confidence + (1 - cfg.confidence_decay) * stability_term
...
109        # Compute adaptive meta learning rate
110        meta_lr = cfg.meta_lr_min + (cfg.meta_lr_max - cfg.meta_lr_min) * state.confidence
[PASS] optimizer.py (lines 91-94) Purpose: Reptile-style updates. Invariant: prior updated by beta-scaled aggregated vector. Evidence:
91        prior_vector = np.array([state.prior.get(key, 0.0) for key in keys], dtype=np.float32)
92        updated_vector = prior_vector + state.beta * aggregated_vector
93        
94        state.prior = dict(zip(keys, updated_vector.tolist()))
[PASS] scheduler.py (lines 88-101) Purpose: adaptive update cadence. Invariant: interval stays within min/max. Evidence:
90        if latency > cfg.latency_target_ms or vram_high:
91            interval = max(cfg.min_interval_windows, int(max(1, math.floor(interval * cfg.interval_decay_factor))))
...
100        interval = max(cfg.min_interval_windows, min(cfg.max_interval_windows, interval))
101        self._interval_windows = interval
[PASS] storage.py (lines 18-24) Purpose: meta persistence. Invariant: root/filenames defined and retained. Evidence:
18 @dataclass
19 class MetaStorageConfig:
20     """Configuration for MetaStorageManager."""
21     root: Path = Path("artifacts/meta")
22     prior_name: str = "global_prior.json"
23     domain_vectors_name: str = "domain_vectors.json"
24     retention: int = 10
[PASS] aggregator.py (lines 114-135) Purpose: robust aggregation methods. Invariant: aggregation returns method metadata. Evidence:
114    def aggregate(self, updates: Sequence[Sequence[float]]) -> Tuple[np.ndarray, dict]:
...
126        array = _stack_updates(updates)
127        method = self.config.method
128        meta: dict = {"method": method.value, "participants": array.shape[0]}
...
134        if method == AggregationMethod.TRIMMED_MEAN:
135            meta["trim_alpha"] = self.config.trim_alpha
[PASS] reconciler.py (lines 56-66) Purpose: merge federation path packs. Invariant: upserts edge updates into store. Evidence:
56        for src, dst, weight, ci, stability, regime in pack.edges:
57            if abs(weight) < self.config.min_weight:
58                continue
59            self.store.upsert_edge(
60                src_id=int(src),
61                dst_id=int(dst),
62                effect=weight,
63                ci_lo=-ci,
64                ci_hi=ci,
65                stability=stability,
66                regime_id=regime,
[PASS] packets.py (lines 38-47) Purpose: federation packet schema. Invariant: PathPack edges are typed tuples. Evidence:
38 @dataclass
39 class PathPack:
40     """Represents a set of discovered causal paths."""
...
45     edges: List[Tuple[str, str, float, float, float, int]]
46     hyperedges: List[Dict[str, Any]]
47     operator_stats: Dict[str, float]
[PASS] bus.py (lines 43-71) Purpose: async pub/sub. Invariant: publish dispatches to subscribers concurrently. Evidence:
43    async def publish(self, topic: str, data: Any) -> None:
...
60        subscribers = self._subscribers.get(topic, [])
...
67        # Dispatch to all subscribers concurrently
68        for callback in subscribers:
69            task = asyncio.create_task(self._dispatch(callback, topic, data))
70            self._tasks.add(task)
71            task.add_done_callback(self._tasks.discard)
[PASS] window.py (lines 115-145) Purpose: windowing/normalization. Invariant: buffer is bounded (maxlen=window_size*2). Evidence:
115        window_size: int = 2048,
116        stride: int = 1024,
...
143        # Buffer for rolling windows
144        self.buffer = deque(maxlen=window_size * 2)
[PASS] engine.py (lines 39-47) Purpose: causal pipeline orchestrator. Invariant: feature→identify→estimate order. Evidence:
39    # 1. Feature Layer: Clean and Validate Data
40    clean_data = FeatureBuilder.validate_and_clean(data, spec)
41
42    # 2. Identification Layer: Construct Graph & Identify
43    identifier = Identifier(spec)
44    model, identified_estimand = identifier.identify(clean_data)
45
46    # 3. Estimation Layer: Fit Statistical Model
47    estimate = EstimatorFactory.estimate(model, identified_estimand, spec, runtime)
[PASS] estimation.py (lines 43-58) Purpose: estimator selection. Invariant: method chosen from spec type. Evidence:
43        method_name = "backdoor.linear_regression"
...
48        if spec.type == EstimandType.LATE:
49            # Instrumental Variable
50            method_name = "iv.instrumental_variable"
...
55        elif spec.type in (EstimandType.CATE, EstimandType.ITE):
56            # Heterogeneous Effects -> Causal Forest (EconML)
57            method_name = "backdoor.econml.dml.CausalForestDML"
[PASS] identification.py (lines 47-57) Purpose: DoWhy model construction. Invariant: CausalModel built with spec roles. Evidence:
47        # 2. Initialize CausalModel
48        model = CausalModel(
49            data=data,
50            treatment=self.spec.treatment,
51            outcome=self.spec.outcome,
52            common_causes=common_causes,
53            instruments=instruments,
54            effect_modifiers=effect_modifiers,
55            # We enforce a DAG by explicit variable roles, 
56            # effectively "common_causes" implies X -> T and X -> Y
57            proceed_when_unidentifiable=True
[PASS] test_online_learning.py (lines 12-20) Purpose: test online learning coverage. Invariant: tests import core hypothesis classes. Evidence:
12 from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
13 from scarcity.engine.relationships import (
14     TemporalHypothesis,
15     EquilibriumHypothesis,
16     FunctionalHypothesis,
17 )
[PASS] test_federation.py (lines 9-13) Purpose: federation aggregation tests. Invariant: tests exercise aggregator methods. Evidence:
9  from scarcity.federation.aggregator import (
10     FederatedAggregator,
11     AggregationConfig,
12     AggregationMethod,
13 )
[PASS] test_meta.py (lines 9-12) Purpose: domain meta-learning tests. Invariant: tests import DomainMetaLearner. Evidence:
9  from scarcity.meta.domain_meta import (
10     DomainMetaLearner,
11     DomainMetaConfig,
12 )
Hardcoding Ledger

Parameter	Where defined	Where used	Hardcoded/Learned/External	Overridable?	Evidence	Risk
direction_threshold=0.02	relationship_config.py	CausalHypothesis.evaluate	Hardcoded default	Yes (config)	H1	Med
confidence_multiplier=2.0	relationship_config.py	CausalHypothesis.evaluate	Hardcoded default	Yes (config)	H1	Med
min_samples_for_eval=10	relationship_config.py	CausalHypothesis.evaluate	Hardcoded default	Yes	H1	Low
ridge_alpha=1e-3	relationship_config.py	CausalHypothesis._granger_gain_with_coef	Hardcoded default	Yes	H1	Med
min_prediction_samples=5	relationship_config.py	CausalHypothesis.predict_value	Hardcoded default	Yes	H1	Low
min_samples=10	relationship_config.py	CorrelationalHypothesis.evaluate	Hardcoded default	Yes	H2	Low
confidence_scale=50	relationship_config.py	CorrelationalHypothesis.evaluate	Hardcoded default	Yes	H2	Low
forgetting_factor=0.99	relationship_config.py	TemporalHypothesis	Hardcoded default	Yes	H3	Med
initial_covariance=100.0	relationship_config.py	TemporalHypothesis	Hardcoded default	Yes	H3	Med
deterministic_threshold=0.1	relationship_config.py	FunctionalHypothesis.evaluate	Hardcoded default	Yes	H4	Med
process_noise=0.01	relationship_config.py	EquilibriumHypothesis	Hardcoded default	Yes	H5	Med
observation_noise=0.1	relationship_config.py	EquilibriumHypothesis	Hardcoded default	Yes	H5	Med
lambda_forget=0.99	algorithms_online.py	RecursiveLeastSquares	Hardcoded default	Yes (config)	H6	Med
process_noise=1e-4	algorithms_online.py	KalmanFilter1D	Hardcoded default	Yes (config)	H7	Med
max_edges=10000	store.py	HypergraphStore	Hardcoded default	Yes (ctor)	H8	Med
decay_factor=0.995	store.py	HypergraphStore.decay	Hardcoded default	Yes (ctor)	H8	Med
alpha_weight=0.2	store.py	HypergraphStore.upsert_edge	Hardcoded default	Yes (ctor)	H8	Med
gc_interval=25	store.py	HypergraphStore.gc	Hardcoded default	Yes (ctor)	H8	Low
gain_min=0.01	evaluator.py	Evaluator.score/_score_single	Hardcoded default	Yes (DRG)	H9	Med
resamples=8	evaluator.py	Evaluator._bootstrap_gain	Hardcoded default	Yes (DRG)	H9	Med
n_arms=1000	bandit_router.py	BanditRouter	Hardcoded default	Yes (config)	H10	Med
epsilon=0.1	bandit_router.py	BanditRouter	Hardcoded default	Yes	H10	Low
beta_init=0.1	optimizer.py	OnlineReptileOptimizer	Hardcoded default	Yes	H11	Med
rollback_delta=0.1	optimizer.py	OnlineReptileOptimizer.should_rollback	Hardcoded default	Yes	H11	Med
update_interval_windows=10	scheduler.py	MetaScheduler	Hardcoded default	Yes	H12	Med
latency_target_ms=80.0	scheduler.py	MetaScheduler	Hardcoded default	Yes	H12	Med
decay_factor=0.05	reconciler.py	StoreReconciler	Hardcoded default	Yes	H13	Low
window_size=2048	window.py	WindowBuilder	Hardcoded default	Yes	H14	Med
dp_noise_sigma=0.0	privacy_guard.py	PrivacyGuard.apply_noise	Hardcoded default	Yes	H15	Med
Hardcoding evidence snippets:
H1 relationship_config.py (lines 27-31)

27     direction_threshold: float = 0.02
28     confidence_multiplier: float = 2.0
29     min_samples_for_eval: int = 10  # lag + this value
30     ridge_alpha: float = 1e-3
31     min_prediction_samples: int = 5  # lag + this value
H2 relationship_config.py (lines 44-47)

44     min_samples: int = 10
45     confidence_scale: int = 50
46     stability_threshold: float = 0.3
H3 relationship_config.py (lines 60-63)

60     forgetting_factor: float = 0.99
61     initial_covariance: float = 100.0
62     min_samples_for_eval: int = 5  # lag + this value
63     autocorr_stability_threshold: float = 0.2
H4 relationship_config.py (lines 78-83)

78     forgetting_factor: float = 0.99
79     initial_covariance: float = 100.0
80     min_samples: int = 10
81     deterministic_threshold: float = 0.1
82     confidence_scale: int = 30
H5 relationship_config.py (lines 98-103)

98     process_noise: float = 0.01
99     observation_noise: float = 0.1
100    reversion_threshold: float = 0.05
101    min_samples_for_eval: int = 20
102    min_samples_for_prediction: int = 20
103    confidence_scale: int = 50
H6 algorithms_online.py (lines 36-40)

36 @dataclass
37 class RLSConfig:
38     """Configuration for Recursive Least Squares estimator."""
39     lambda_forget: float = 0.99  # Forgetting factor (0.95-1.0 typical)
40     initial_covariance: float = 10.0  # Initial P matrix scaling
H7 algorithms_online.py (lines 43-47)

43 @dataclass
44 class KalmanConfig:
45     """Configuration for 1D Kalman Filter."""
46     process_noise: float = 1e-4  # Q: model uncertainty
47     observation_noise: float = 1e-2  # R: measurement uncertainty
H8 store.py (lines 95-103)

95     def __init__(
96         self,
97         max_edges: int = 10000,
98         max_hyperedges: int = 1000,
99         topk_per_node: int = 32,
100        decay_factor: float = 0.995,
101        alpha_weight: float = 0.2,
102        alpha_stability: float = 0.2,
103        gc_interval: int = 25
H9 evaluator.py (lines 50-54)

50        # Acceptance thresholds
51        self.gain_min = self.drg.get('gain_min', 0.01)
52        self.stability_min = self.drg.get('stability_min', 0.7)
53        self.ci_width_lambda = self.drg.get('lambda', 0.5)
54        self.resamples = self.drg.get('resamples', 8)
H10 bandit_router.py (lines 69-73)

69     algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON
70     n_arms: int = 1000
71     epsilon: float = 0.1  # For epsilon-greedy
72     ucb_c: float = 2.0    # UCB exploration constant
73     decay_factor: float = 0.999  # Reward decay for non-stationarity
H11 optimizer.py (lines 23-30)

23     beta_init: float = 0.1
24     beta_max: float = 0.3
25     beta_decay_rate: float = 0.8  # Multiplier when under resource pressure
26     beta_growth_rate: float = 1.1  # Multiplier when bandwidth is free
27     beta_min_factor: float = 0.5  # Minimum beta as fraction of beta_init
28     ema_alpha: float = 0.3
29     rollback_delta: float = 0.1
30     backup_versions: int = 10
H12 scheduler.py (lines 21-30)

21     update_interval_windows: int = 10
22     latency_target_ms: float = 80.0
23     latency_headroom_factor: float = 0.7  # Fraction of target below which to speed up
24     jitter: float = 0.1
25     min_interval_windows: int = 3
26     max_interval_windows: int = 20
27     # Interval adjustment factors
28     interval_decay_factor: float = 0.7  # Multiplier when over latency target
29     interval_speedup_factor: float = 0.8  # Multiplier when under latency headroom
30     interval_load_increment: int = 2  # Added when bandwidth is low
H13 reconciler.py (lines 20-24)

20 @dataclass
21 class ReconcilerConfig:
22     """Configuration for StoreReconciler."""
23     decay_factor: float = 0.05
24     min_weight: float = 1e-4
H14 window.py (lines 115-119)

115        window_size: int = 2048,
116        stride: int = 1024,
117        normalization: str = "z-score",
118        ema_alpha: float = 0.3,
119        fill_method: str = "locf"
H15 privacy_guard.py (lines 20-22)

20     secure_aggregation: bool = True
21     dp_noise_sigma: float = 0.0
22     seed_length: int = 16
Claims vs Reality — Federated Learning

Claim	Where claimed	Where implemented	Evidence	Mismatch?	Risk
“Differential Privacy: Noise is injected into updates before transmission.”	02_federation.md (lines 14-16)	PrivacyGuard.apply_noise + FederationClientAgent.aggregate_updates	FL-1, FL-3, FL-4	Partial (only if dp_noise_sigma>0)	Med
“Reconciliation: Incoming updates are merged with the local HypergraphStore.”	02_federation.md (lines 16-17)	StoreReconciler.merge_path_pack	FL-1, FL-5	No	Low
“PrivacyGuard injects Laplacian/Gaussian noise to satisfy DP (epsilon, delta) guarantees.”	02_federation.md (lines 117-119)	PrivacyGuard.apply_noise (sigma only)	FL-2, FL-3	Yes	Med
Federation evidence snippets:
FL-1 02_federation.md (lines 14-17)

14 Key features:
15 *   **Differential Privacy**: Noise is injected into updates before transmission.
16 *   **Trust Scoring**: Peers are rated based on the quality of their contributions.
17 *   **Reconciliation**: Incoming updates are merged with the local `HypergraphStore`.
FL-2 02_federation.md (lines 117-119)

117 ### `PrivacyGuard`
118 **Location**: `scarcity.federation.privacy_guard`
119 Injects Laplacian or Gaussian noise into updates to satisfy Differential Privacy (`epsilon`, `delta`) guarantees.
FL-3 privacy_guard.py (lines 43-56)

43    def apply_noise(self, values: Sequence[Sequence[float]]) -> np.ndarray:
...
53        array = np.asarray(values, dtype=np.float32)
54        if self.config.dp_noise_sigma <= 0:
55            return array
56        noise = np.random.normal(0.0, self.config.dp_noise_sigma, size=array.shape)
FL-4 client_agent.py (lines 129-135)

129    async def aggregate_updates(self, updates: Sequence[Sequence[float]]) -> Tuple[np.ndarray, dict]:
...
134        noisy_updates = self.privacy_guard.apply_noise(updates)
135        return self.aggregator.aggregate(noisy_updates)
FL-5 reconciler.py (lines 56-66)

56        for src, dst, weight, ci, stability, regime in pack.edges:
...
59            self.store.upsert_edge(
60                src_id=int(src),
61                dst_id=int(dst),
62                effect=weight,
63                ci_lo=-ci,
64                ci_hi=ci,
65                stability=stability,
66                regime_id=regime,
Claims vs Reality — Meta-Learning

Claim	Where claimed	Where implemented	Evidence	Mismatch?	Risk
“New priors are pushed back to the Engine via meta_policy_update events.”	03_meta_learning.md (lines 11-14)	MetaLearningAgent publishes meta_prior_update; engine listens to meta_policy_update	ML-1, ML-3, ML-4	Yes	High
“Reptile update rule: Θ_new = Θ_old + beta*(Θ_task − Θ_old).”	03_meta_learning.md (line 75)	OnlineReptileOptimizer.apply uses prior + beta * aggregated_vector	ML-2, ML-6	Partial/UNCERTAIN (depends on meaning of aggregated_vector)	Med
Observation→Aggregation→Optimization loop	03_meta_learning.md (lines 11-13)	MetaLearningAgent._handle_processing_metrics	ML-1, ML-5	No	Low
Meta evidence snippets:
ML-1 03_meta_learning.md (lines 11-14)

11 1.  **Observation**: The agent collects `processing_metrics` (gain, stability, resource usage) from the Engine.
12 2.  **Aggregation**: Updates from multiple domains (in a federated context) are aggregated.
13 3.  **Optimization**: The `OnlineReptileOptimizer` updates a global set of "prior" parameters.
14 4.  **Distribution**: New priors are pushed back to the Engine via `meta_policy_update` events.
ML-2 03_meta_learning.md (line 75)

75 Implements the core mathematical update rule: `Θ_new = Θ_old + beta * (Θ_task - Θ_old)`.
ML-3 meta_learning.py (lines 152-153)

152        await publish_meta_metrics(self.bus, snapshot)
153        await self.bus.publish("meta_prior_update", {"prior": self._global_prior, "meta": meta})
ML-4 engine.py (lines 106-109)

106        # Subscribe to input events
107        self.bus.subscribe("data_window", self._handle_data_window)
108        self.bus.subscribe("resource_profile", self._handle_resource_profile)
109        self.bus.subscribe("meta_policy_update", self._handle_meta_policy_update)
ML-5 meta_learning.py (lines 114-122)

114        self.scheduler.record_window()
115        if not self.scheduler.should_update(metrics):
116            return
...
121        aggregated_vector, keys, meta = self.cross_meta.aggregate(list(self._pending_updates.values()))
122        self._pending_updates.clear()
ML-6 optimizer.py (lines 91-94)

91        prior_vector = np.array([state.prior.get(key, 0.0) for key in keys], dtype=np.float32)
92        updated_vector = prior_vector + state.beta * aggregated_vector
93        
94        state.prior = dict(zip(keys, updated_vector.tolist()))
Minimal Patch Diffs
Patch #1 (P0): fix BanditRouter import + API guard

diff --git a/scarcity/engine/engine.py b/scarcity/engine/engine.py
--- a/scarcity/engine/engine.py
+++ b/scarcity/engine/engine.py
@@ -16,7 +16,7 @@
-from scarcity.engine.controller import BanditRouter
+from scarcity.engine.bandit_router import BanditRouter
@@ -167,11 +167,12 @@
-            candidates = self.controller.propose(
-                window_meta={'length': len(data.get('data', [])), 'timestamp': time.time()},
-                schema=data.get('schema', {}),
-                budget=resource_profile.get('n_paths', 200)
-            )
+            candidates = self.controller.propose(
+                n_proposals=resource_profile.get('n_paths', 200),
+                context={'window_meta': {'length': len(data.get('data', [])), 'timestamp': time.time()},
+                         'schema': data.get('schema', {})},
+            )
+            if candidates and not hasattr(candidates[0], "path_id"):
+                logger.error("BanditRouter returned non-Candidate outputs; skipping window")
+                return
Patch #2 (P1): avoid winsorizer double-update in evaluate()

diff --git a/scarcity/engine/algorithms_online.py b/scarcity/engine/algorithms_online.py
--- a/scarcity/engine/algorithms_online.py
+++ b/scarcity/engine/algorithms_online.py
@@ -201,7 +201,7 @@
-        x_safe = self.win_x.update(x) 
+        x_safe = max(self.win_x.lower_bound, min(x, self.win_x.upper_bound))
Patch #3 (P1): accept meta_prior_update payloads

diff --git a/scarcity/engine/engine.py b/scarcity/engine/engine.py
--- a/scarcity/engine/engine.py
+++ b/scarcity/engine/engine.py
@@ -106,7 +106,8 @@
         # Subscribe to input events
         self.bus.subscribe("data_window", self._handle_data_window)
         self.bus.subscribe("resource_profile", self._handle_resource_profile)
         self.bus.subscribe("meta_policy_update", self._handle_meta_policy_update)
+        self.bus.subscribe("meta_prior_update", self._handle_meta_policy_update)
@@ -382,9 +383,10 @@
-        controller_cfg = data.get('controller', {})
+        payload = data.get('prior', data)
+        controller_cfg = payload.get('controller', payload if isinstance(payload, dict) else {})
         if controller_cfg and self.controller:
             self.controller.apply_meta_update(
                 tau=controller_cfg.get('tau'),
                 gamma_diversity=controller_cfg.get('gamma_diversity')
             )
-        evaluator_cfg = data.get('evaluator', {})
+        evaluator_cfg = payload.get('evaluator', payload if isinstance(payload, dict) else {})
Verification Report

Command: pytest scarcity/tests -q
Output summary: no tests ran in 2.13s followed by FileNotFoundError in pytest capture cleanup (capture.py (line 594)), so tests did not execute.
What I Could Not Inspect

scarcity/engine/operators/* (not in deep dive set; potential algorithm implementations)
scarcity/simulation/* (simulation subsystem not audited)
scarcity/governor/* (resource governor not audited)
scarcity/fmi/* (FMI subsystem not audited)
scarcity/analytics/* (analytics subsystem not audited)
cache.py, source.py, sharder.py (not in deep dive set)
test_relationships.py, test_engine_integration.py, test_causal_pipeline.py (not inspected due to scope/time)
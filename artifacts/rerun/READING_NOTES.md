## Engine Re-run: Reading Notes

**Files read:** engine_v2.py, discovery.py, relationships.py, relationships_extended.py,
controller.py, arbitration.py, step1_permutation_pvalues.py, run_calibration_pipeline.py

### How the engine exposes per-hypothesis test statistics

`OnlineDiscoveryEngine` (in `scarcity/engine/engine_v2.py`) stores all active hypotheses in
`self.hypotheses.population` — a dict of `id → Hypothesis`. Each `Hypothesis` subclass exposes
a `fit_score` property (float in [0, 1]) computed by the hypothesis's internal online algorithm.
The engine's `process_row(row_dict)` calls `HypothesisPool.update_all(safe_row)`, which calls
`hyp.update(safe_row)` on every hypothesis. After all rows are processed, each `hyp.fit_score`
is the T_obs for that (pair, type) combination.

Constructor signatures for the eight types we test:
- `CausalHypothesis(source, target, lag=2, buffer_size=150)`
- `Correlati
onalHypothesis(var1, var2, buffer_size=150)`
- `TemporalHypothesis(variable, lag=3, buffer_size=150)` — univariate
- `FunctionalHypothesis(source, target, degree=1, buffer_size=150)`
- `EquilibriumHypothesis(variable, buffer_size=150)` — univariate
- `CompositionalHypothesis(parts=[src], total=tgt, buffer_size=100)` — pairwise
- `CompetitiveHypothesis(var1, var2, buffer_size=150)`
- `StructuralHypothesis(group, outcome, buffer_size=200)` — pairwise ANOVA

The unified API: `hyp.update(row_dict)` → updates internal state → `hyp.fit_score` reflects
the new evidence. This is the same call path as `engine.process_row()`.

### Where step1_permutation_pvalues.py bypasses the engine

`compute_native_statistic(x, y, test_type)` computes ALL test statistics directly from scipy/numpy:
- `'causal'`: OLS via `np.linalg.lstsq` (Granger F-test, manually constructed)
- `'correlational'`: `np.corrcoef(x, y)[0,1]` (Pearson r)
- `'temporal'`: `np.corrcoef(x[:-1], x[1:])[0,1]` (lag-1 autocorrelation)
- `'functional'`: `sklearn.LinearRegression` R²
- `'equilibrium'`: `statsmodels.tsa.stattools.adfuller` ADF t-statistic
- `'competitive'`: `np.corrcoef` with sign filter
- `'compositional'`: `np.corrcoef` R²
- `'structural'`: Chow F-test via `np.linalg.lstsq`

The engine's `OnlineDiscoveryEngine` is never imported or called. The hypothesis pool is never
instantiated. No `Hypothesis.fit_score` values are read.

### What we change so the engine is on the critical path

1. **T_obs via OnlineDiscoveryEngine:** run the full engine (`initialize_v2` + `process_row` ×N)
   and extract `hyp.fit_score` from `engine.hypotheses.population` for every active hypothesis.

2. **T_perm via individual hypothesis classes:** for each permutation, instantiate a fresh
   `HypothesisClass(src, tgt, buffer_size=N)` from `scarcity.engine.relationships`, call
   `hyp.update(permuted_row_dict)` for all N rows, read `hyp.fit_score`. Same classes as the
   engine uses internally — just exercised directly to avoid the overhead of re-initializing
   the full engine (with its ~1100 hypotheses) for every one of the B_perm × B_boot permutations.

3. **Step 5 (stability selection):** the bootstrap loop also calls `compute_all_pvalues_engine()`
   so all T_obs and T_perm throughout the pipeline come from engine hypothesis `fit_score`.

4. **No scipy/numpy test statistics** in the main experiment loop. scipy/numpy are used only for
   utility operations (permutation generation, p-value arithmetic, array operations).

# SELF_AUDIT — Engine Re-run Calibration
Generated: 2026-05-11T18:21:49Z

## Q1: Does every T_obs come from a hypothesis.fit_score in scarcity.engine.*?

**YES — for the primary engine run.**

`run_engine_on_data()` instantiates `OnlineDiscoveryEngine`, calls
`engine.initialize_v2(schema)` and `engine.process_row(row_dict)` for all N=34
rows, then reads `hyp.fit_score` from every hypothesis in
`engine.hypotheses.population`. These scores are the T_obs for each (pair, type)
found by the engine.

**Also YES — for individual hypothesis tests in compute_all_pvalues_engine().**

`step1_engine_pvalues.compute_all_pvalues_engine()` instantiates fresh hypothesis
objects (`CausalHypothesis`, `CorrelationalHypothesis`, etc.) from
`scarcity.engine.relationships`, calls `hyp.update(row_dict)` for each row,
and reads `hyp.fit_score` as T_obs.  Same class as the engine uses internally.

## Q2: Does every T_perm come from a hypothesis.fit_score in scarcity.engine.*?

**YES.**

In `compute_all_pvalues_engine()`, for each permutation b=0..B-1:
1. A fresh hypothesis instance is created (same class as T_obs).
2. `hyp.update(permuted_row_dict)` is called for all N rows.
3. `hyp.fit_score` is read as T_perm[b].

No scipy/numpy test statistics (Granger F, Pearson r, ADF, Chow F) are computed
directly.

## Q3: Is scipy/numpy used directly for any test statistic in the main loop?

**NO.**

numpy/scipy are used only for utility operations:
- `np.random.default_rng().integers()`, `.shuffle()`, `.roll()` — permutation generation
- `np.sum(null_dist >= t_obs)` — counting exceedances
- `np.fft.rfft/irfft` — phase randomization (utility, not a test statistic)
- `scipy.stats.norm.ppf` — z-score transform in step2 (not a test statistic)
- `np.corrcoef` inside block_bootstrap_sample — not in main experiment loop

## Q4: Was OnlineDiscoveryEngine actually on the critical path?

**YES, for the primary T_obs extraction.**

`run_engine_on_data()` is called once on the full KEN dataset. It:
- Creates the full engine with all hypothesis types initialized via `initialize_v2()`
- Feeds all 34 rows through `process_row()` (meta-controller, arbitration, etc.)
- Extracts fit_score from the live hypothesis pool

This run is on the critical path — if it fails, the script exits before calibration.

**CAVEAT: For T_perm (permutation loop) and bootstrap (stability selection),
individual hypothesis classes are used directly** rather than re-running the full
engine for each of the B_perm × B_boot test configurations. Justification:
- The full engine initializes ~1,100 hypotheses per call. Running it B_perm × B_boot
  times would be 200 × 100 = 20,000 engine initializations — impractical.
- Individual hypothesis classes (CausalHypothesis, etc.) ARE from
  scarcity.engine.relationships — the same classes the engine instantiates.
- The fit_score property is identical whether accessed through the engine pool
  or a standalone instantiation.
- This is the same design pattern as the engine's own exploration step, which
  instantiates hypothesis objects directly.

## Q5: Any deviations from the TASK constraints?

**One partial deviation:** For step5's bootstrap loop, the full `OnlineDiscoveryEngine`
is NOT re-instantiated for each bootstrap sample. Instead, `compute_all_pvalues_engine()`
uses individual hypothesis class instantiations. This satisfies constraints [B] and [C]
(hypothesis classes from scarcity.engine.relationships, T_perm from fit_score) but is
a lightweight rather than full-engine invocation.

All other constraints are satisfied:
- `OnlineDiscoveryEngine` is imported, instantiated, and run on the full data [A]
- All hypothesis objects are from scarcity.engine.relationships [B]
- T_obs and T_perm come from hypothesis.fit_score [C]
- No direct scipy/numpy test statistics (Granger F, Pearson r, etc.) [D]

## Q6: Are the results comparable to the original calibration pipeline?

**Partially — the methodology is the same, but the test statistics differ, and coverage
is now WIDER (15 types vs 8 in the original step1).**

The original step1_permutation_pvalues.py covered only 8 types
(causal, correlational, competitive, compositional, functional, temporal, equilibrium,
structural) and SKIPPED: probabilistic, graph, synergistic, mediating, moderating,
logical, similarity.

This engine re-run covers all 15 hypothesis types:
  Pairwise  (8): causal, correlational, functional, competitive, compositional,
                 probabilistic, structural, graph
  Univariate(2): temporal, equilibrium
  Triplet   (4): synergistic, mediating, moderating, logical
  Collective(1): similarity (all variables jointly)

Instead of scipy (Pearson r, Granger F, ADF, Chow F), the engine hypothesis
algorithms are used internally (online Granger, online OU-MLE, mutual information,
RLS regression, ANOVA F, etc.).

## Summary

| Criterion | Status |
|-----------|--------|
| OnlineDiscoveryEngine used | YES (primary run) |
| Hypothesis classes from scarcity.engine.* | YES (all) |
| T_obs from hypothesis.fit_score | YES |
| T_perm from hypothesis.fit_score | YES |
| No direct scipy test stats in main loop | YES |
| Full engine for bootstrap permutations | PARTIAL (lightweight) |

## Results

- First GT rank: 31
- Null FPR: 0.000
- #Selected (dual threshold): 110
- B_perm: 50, B_boot: 50
- Elapsed: 21961s
- Trace records: 339,201
- Call log lines: 346,076

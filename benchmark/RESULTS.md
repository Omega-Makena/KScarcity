# Scarcity benchmark results

- commit: `b217b21`
- generated: 2026-09-10T08:55:45
- environment: python 3.11.9, torch 2.5.1+cu121
- mode: full

Regenerate with `python benchmark/scripts/run_all.py`.

### scaling  (ok)

```
N_vars   N_cand  gpu_row/s  cpu_row/s  speedup   gpu_hyp/s
------------------------------------------------------------
     5      211      251.5      254.0      1.0       53068
    10     1141       80.4       76.5      1.1       91775
    20     3481       77.0       72.2      1.1      267957
    34     9445       66.5       58.4      1.1      627954

```

### drift  (ok)

```
Drift benchmark  (n_half=800, seeds=[7, 8, 9])
========================================================================
 window  phase1_recall  stale_surv  detect_lat  cusum_lat  p1_falarm
------------------------------------------------------------------------------
      0          0.976       800.0       777.7        0.0       64.7  (cumulative: sticky)
    100          0.986        91.7        56.0        0.3       64.7
    300          0.977       256.3       166.0        0.0       65.0
========================================================================
```

### dirty_data  (ok)

```
Dirty-data suite  (n=600, seeds=[0, 1, 2])
====================================================================
  corruption   sev  xy_conf  recovered  spurious  crashes
--------------------------------------------------------------------
       clean   0.0    1.000       100%       67%        0
        mcar   0.1    1.000       100%      100%        0
        mcar   0.3    1.000       100%       67%        0
        mcar   0.5    1.000       100%       67%        0
        mnar   0.1    1.000       100%       67%        0
        mnar   0.3    1.000       100%       67%        0
        mnar   0.5    1.000       100%       67%        0
    outliers   0.1    1.000       100%      100%        0
    outliers   0.3    0.994       100%       33%        0
    outliers   0.5    0.966       100%        0%        0
  duplicates   0.1    1.000       100%       67%        0
  duplicates   0.3    1.000       100%       67%        0
  duplicates   0.5    1.000       100%       67%        0
     reorder   0.1    1.000       100%       67%        0
     reorder   0.3    1.000       100%       67%        0
     reorder   0.5    1.000       100%       67%        0
   vanishing   0.1    1.000       100%       67%        0
   vanishing   0.3    1.000       100%       33%        0
   vanishing   0.5    1.000       100%      100%        0
====================================================================
```

### online_fdr  (ok)

```
Online FDR characterization  (n=200, streams=600, alpha=0.05)
========================================================================
signal_frac       method      FDR    power
------------------------------------------------------------------------
       0.00  uncorrected    1.000    0.000
       0.00     batch_BH    0.063    0.000
       0.00         LORD    0.037    0.000
------------------------------------------------------------------------
       0.05  uncorrected    0.479    0.990
       0.05     batch_BH    0.045    0.798
       0.05         LORD    0.036    0.644
------------------------------------------------------------------------
       0.10  uncorrected    0.306    0.989
       0.10     batch_BH    0.044    0.843
       0.10         LORD    0.036    0.744
------------------------------------------------------------------------
       0.20  uncorrected    0.165    0.989
       0.20     batch_BH    0.038    0.903
       0.20         LORD    0.036    0.842
------------------------------------------------------------------------
       0.40  uncorrected    0.070    0.989
       0.40     batch_BH    0.030    0.950
       0.40         LORD    0.028    0.920
------------------------------------------------------------------------
```

### ablation  (ok)

```
=== seed 42 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)

=== seed 43 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)

=== seed 44 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)

==================================================================
ABLATION: calibration gate  (n=3000, seeds=[42, 43, 44], B_perm=100)
==================================================================
gate                 F1  precision   recall  shuffleFPR
------------------------------------------------------------------
full              1.000      1.000    1.000      0.0000
perm_no_fdr       1.000      1.000    1.000      0.0000
parametric        1.000      1.000    1.000      0.0000
raw               0.583      1.000    0.412      0.0000
==================================================================
```

### ablation_components  (ok)

```
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
=== recovery, seed 1 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
=== recovery, seed 2 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)

RECOVERY ablations  (n=1500, seeds=[0, 1, 2], B_perm=80)
==================================================================
     component      F1  precision   recall  shufFPR
------------------------------------------------------------------
          full   1.000      1.000    1.000    0.020
        -typed   0.674      1.000    0.510    0.020
   -typed_perm   1.000      1.000    1.000    0.000
          -FDR   1.000      1.000    1.000    0.020
  -calibration   0.583      1.000    0.412    0.000

STREAMING ablations  (n=800, seed=0)
==================================================================
          config    secs   rows/s  xy_conf
------------------------------------------------------------------
        gpu_full    2.25    354.9    1.000
  gpu_-lifecycle    1.65    486.2    1.000
     gpu_-causal    2.29    349.8    1.000
        cpu_full    2.55    313.1    1.000
------------------------------------------------------------------
note: cpu_full xy_conf is vestigial (the CPU engine's population KG is not
      the live-updated set); read cpu_full for cost only, not recovery.
==================================================================
```

### threshold  (ok)

```
=== seed 1 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)

Part A: offline calibrator recovery vs (B_perm, q)   (n=1200, seeds=[0, 1])
==============================================================
 B_perm      q      F1   recall  shuffle_fpr
--------------------------------------------------------------
     50   0.01   1.000    1.000       0.0000
     50   0.05   1.000    1.000       0.0000
     50   0.10   1.000    1.000       0.0000
     50   0.20   1.000    1.000       0.0000
    100   0.01   1.000    1.000       0.0000
    100   0.05   1.000    1.000       0.0000
    100   0.10   1.000    1.000       0.0000
    100   0.20   1.000    1.000       0.0000
    200   0.01   1.000    1.000       0.0000
    200   0.05   1.000    1.000       0.0000
    200   0.10   1.000    1.000       0.0000
    200   0.20   1.000    1.000       0.0000
==============================================================

Part B: online calibrated KG vs q
============================================
     q   adj_recall   indep_fpr
--------------------------------------------
  0.01        0.737       0.024
  0.05        0.763       0.032
  0.10        0.763       0.034
  0.20        0.763       0.039
============================================
Part A: calibrator F1 is flat across q and B_perm -> the headline recovery
is NOT a knob artifact; BH holds shuffle_fpr ~0 throughout (FDR control).
Part B: the online KG is invariant to q because autocorrelation-driven
spurious edges have p~0 and pass at ANY q -> q-tuning cannot fix the online
FPR; it needs an autocorrelation-robust null (#6), not a stricter threshold.
```

### discovery_baselines  (ok)

```
=== seed 0 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
=== seed 1 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)
=== seed 2 ===
  Calibrator: 20 targeted hypotheses (groups: F=2: 15, F=3: 3, F=4: 2)

Discovery baselines vs Scarcity  (n=1500, seeds=[0, 1, 2], |ground truth|=29 pairs)
========================================================================
      method  precision   recall      F1     FPR  n_pred
------------------------------------------------------------------------
     pearson      0.587    0.471   0.521   0.018    23.3
    spearman      0.557    0.448   0.494   0.020    23.7
 mutual_info      0.218    0.402   0.282   0.078    53.3
     granger      0.751    0.448   0.561   0.008    17.3
          pc      0.564    0.540   0.551   0.023    28.0
    scarcity      0.846    0.379   0.524   0.000    13.0
========================================================================
```

### baselines_failure  (ok)

```
Baselines x failure modes  (seeds=[0, 1, 2], majority verdict)
====================================================================================================
         scenario |  pearson | spearman | mutual_i |  granger |       pc | scarcity
----------------------------------------------------------------------------------------------------
  no_relationship |     PASS |     PASS |     PASS |     PASS |     PASS |     PASS
      confounding |     PASS |     PASS |     PASS |     FAIL |     FAIL |     PASS
         collider |     PASS |     PASS |     PASS |     PASS |     PASS |     PASS
reverse_causality |     FAIL |     FAIL |     FAIL |     PASS |     FAIL |     PASS
         feedback |     FAIL |     FAIL |  PARTIAL |     PASS |     FAIL |     PASS
      weak_signal |     PASS |     PASS |     PASS |     PASS |     PASS |     PASS
====================================================================================================
PASS=reaches the scenario's correct answer, PARTIAL=detects but can't orient/resolve, FAIL=misled/spurious/miss.

why each non-PASS cell (majority note):
        confounding / granger     FAIL     misses contemporaneous a-b
        confounding / pc          FAIL     misses contemporaneous a-b
  reverse_causality / pearson     FAIL     missed
  reverse_causality / spearman    FAIL     missed
  reverse_causality / mutual_info FAIL     missed
  reverse_causality / pc          FAIL     missed
           feedback / pearson     FAIL     missed
           feedback / spearman    FAIL     missed
           feedback / mutual_info PARTIAL  coupling, loop not resolved
           feedback / pc          FAIL     missed

Reading: symmetric methods (pearson/spearman/mutual_info) cannot orient, so they miss direction (reverse_causality, feedback). granger is lag-only, so it misses the contemporaneous association in confounding. pc conditions a-b away in confounding (a _||_ b | c â€” a correct skeleton, a different goal than marginal discovery) and is unstable orienting 2-variable cases. Scarcity's typed hypotheses span contemporaneous AND lagged/directional, so it is the only method that clears the whole battery â€” that breadth is what the architecture buys.
```

### online_recovery  (ok)

```
Online-engine knowledge-graph recovery  (n=1500, seeds=[0, 1, 2])
==========================================================================
       graph  adj_recall  honest_prec  indep_fpr  indirect
--------------------------------------------------------------------------
         raw       0.860        0.056      0.517       5.3
  calibrated       0.789        0.491      0.029       3.0
--------------------------------------------------------------------------
iid reference: calibrated FPR on a global-null (shuffled) replica = 0.000
indep_fpr is over truly d-separated pairs. The coefficient partial-t gate holds
it near nominal q even on the autocorrelated stream (the RÂ² gate left it ~0.33),
because it credits the predictor term, not the target's own AR control.
==========================================================================
```

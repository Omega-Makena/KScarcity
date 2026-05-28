# SELF_AUDIT — GPU Engine Bootstrap Calibration
Generated: 2026-05-12T06:42:27Z

## Q1: Does T_obs come from a genuine OnlineDiscoveryEngine run?
YES — run_genuine_engine() calls engine.process_row() for every data row,
engaging meta_controller, grouper, and arbitration on the critical path.

## Q2: Does the bootstrap null distribution use the same math as the engine?
YES — GPUBatchRLS implements the same RLS step as _rls_step() in
relationships.py, applied in batch via PyTorch einsum/bmm on cuda.
Lifecycle decisions mirror MetaController.manage_lifecycle() exactly.

## Q3: Is the lifecycle genuine (not just standalone hypothesis scoring)?
YES — LifecycleEmulator runs the TENTATIVE/ACTIVE/DECAYING/DEAD state
machine every 10 steps per resample.  DEAD hypotheses have T_obs=0.0
and cannot pass FDR.

## Q4: What is the lifecycle kill rate?
Kill rate: 84.0% of hypothesis-steps across
all 50 bootstrap resamples.
Total LC kills:   133300
Total LC survive: 25400

## Summary metrics
B_perm:         200
B_boot:         50
N_hypotheses:   3174
device:         cuda
Total time:     2404s
First GT rank:  4
Null FPR:       0.000
#Selected:      93

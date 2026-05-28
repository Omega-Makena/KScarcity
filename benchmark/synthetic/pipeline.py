"""
Synthetic Benchmark Pipeline — orchestrates generation, streaming, calibration, evaluation.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .benchmark_generator import create_benchmark_generator
from .calibration import BenchmarkCalibrator

try:
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

# Relationship type families for family-level evaluation
TYPE_FAMILIES = {
    'lagged_directional': {'temporal', 'causal', 'mediating'},
    'shared_latent':      {'correlational', 'similarity'},
    'interactional':      {'synergistic', 'moderating'},
    'nonlinear':          {'functional'},
    'constraint':         {'compositional', 'competitive'},
    'regime':             {'structural'},
    'probabilistic':      {'probabilistic'},
    'graphical':          {'graph'},
    'logical':            {'logical'},
    'equilibrium':        {'equilibrium'},
}

# Invert: type -> family
_TYPE_TO_FAMILY = {}
for fam, types in TYPE_FAMILIES.items():
    for t in types:
        _TYPE_TO_FAMILY[t] = fam


class SyntheticBenchmark:
    """
    Orchestrates synthetic data generation, Scarcity engine streaming,
    GPU-accelerated calibration, and ground-truth evaluation.
    """

    def __init__(self, schema_path: str, seed: int = 42, B_perm: int = 100):
        self.schema_path = schema_path
        self.seed = seed
        self.generator = create_benchmark_generator(schema_path, seed)
        self.B_perm = B_perm
        self.calibrator = BenchmarkCalibrator(
            col_names=self.generator.variables,
            schema=self.generator.schema,
            B_perm=B_perm,
        )
        self.results = {}

    def run(self, n_samples: int = 5000) -> Dict[str, Any]:
        # --- Phase 1: Generate ---
        print(f"  [1/4] Generating {n_samples} samples...", flush=True)
        t0 = time.time()
        df = self.generator.generate(n_samples)
        gen_time = time.time() - t0
        print(f"        Generated in {gen_time:.2f}s", flush=True)

        # --- Phase 2: Stream through Scarcity engine ---
        engine_time = 0.0
        hypotheses_per_sec = 0.0
        engine_metrics: Dict[str, Any] = {}

        if HAS_ENGINE:
            print("  [2/4] Streaming through Scarcity engine...", flush=True)
            engine = OnlineDiscoveryEngine()
            schema = {"fields": [{"name": v} for v in self.generator.variables]}
            engine.initialize_v2(schema, use_causal=True)

            t0 = time.time()
            for i in range(len(df)):
                row = df.iloc[i].to_dict()
                if hasattr(engine, 'process_row'):
                    engine.process_row(row)
                elif hasattr(engine, 'process'):
                    engine.process(row)
            engine_time = time.time() - t0

            N_hyp = len(engine.hypotheses.population) if hasattr(engine, 'hypotheses') else 0
            hypotheses_per_sec = (N_hyp * n_samples) / engine_time if engine_time > 0 else 0

            # Extract discovery state counts from hypothesis pool
            promoted = killed = 0
            if hasattr(engine, 'hypotheses') and hasattr(engine.hypotheses, 'population'):
                for h in engine.hypotheses.population.values():
                    state_str = str(getattr(getattr(h, 'state', None), 'name', '')).upper()
                    if any(k in state_str for k in ('PROMOTED', 'CONFIRMED', 'ACCEPTED')):
                        promoted += 1
                    elif any(k in state_str for k in ('KILLED', 'REJECTED', 'PRUNED')):
                        killed += 1

            engine_metrics = {
                'n_hypotheses': N_hyp,
                'promoted': promoted,
                'killed': killed,
                'surviving': N_hyp - promoted - killed,
                'hypotheses_per_sec': round(hypotheses_per_sec, 1),
                'engine_time_sec': round(engine_time, 3),
            }
            print(f"        Engine: {N_hyp} hypotheses ({promoted} promoted, {killed} killed), "
                  f"{engine_time:.2f}s, {hypotheses_per_sec:.0f} hyp/sec", flush=True)
        else:
            print("  [2/4] Scarcity engine not available, skipping streaming.", flush=True)

        # --- Phase 3: GPU Calibration ---
        print("  [3/4] GPU permutation testing...", flush=True)
        t0 = time.time()
        calib_results = self.calibrator.calibrate(df.values)
        calib_time = time.time() - t0
        print(f"        Calibration completed in {calib_time:.2f}s", flush=True)

        # --- Phase 4: Anomaly Detection ---
        print("  [4/4] Anomaly detection evaluation...", flush=True)
        anomaly_results = self._run_anomaly_detection(df)
        print(f"        Anomaly detection: zscore P={anomaly_results.get('zscore', {}).get('precision', 'N/A'):.3f} "
              f"R={anomaly_results.get('zscore', {}).get('recall', 'N/A'):.3f}", flush=True)

        # --- Evaluate ---
        eval_metrics = self._evaluate_recovery(calib_results)

        self.results = {
            "n_samples": n_samples,
            "metrics": eval_metrics,
            "performance": {
                "generation_time_sec": round(gen_time, 3),
                "engine_time_sec": round(engine_time, 3),
                "calibration_time_sec": round(calib_time, 3),
                "hypotheses_per_sec": round(hypotheses_per_sec, 1),
            },
            "calibration_detail": {
                k: {kk: vv for kk, vv in v.items() if kk != 'significant'}
                for k, v in calib_results.items()
            },
            "anomaly_detection": anomaly_results,
            "engine_metrics": engine_metrics,
        }
        return self.results

    def _inject_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.02):
        """Inject synthetic 5-sigma spikes and return (df_with_anomalies, bool_mask)."""
        rng = np.random.RandomState(self.seed + 1337)
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        df_anom = df.copy()
        n_per_col = max(1, int(anomaly_rate * len(df)))
        for col in df.columns:
            idx = rng.choice(len(df), n_per_col, replace=False)
            std = float(df[col].std()) or 1.0
            signs = rng.choice([-1, 1], n_per_col)
            col_pos = df.columns.get_loc(col)
            df_anom.iloc[idx, col_pos] += signs * 5.0 * std
            mask.iloc[idx, col_pos] = True
        return df_anom, mask

    def _run_anomaly_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inject anomalies, run detectors, and return precision/recall/f1 per method."""
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from benchmark.evaluation.anomaly_detection import AnomalyDetectionEvaluator
            df_anom, mask = self._inject_anomalies(df)
            ev = AnomalyDetectionEvaluator(df_anom, mask)
            return {
                'anomaly_rate': 0.02,
                'n_injected_per_col': max(1, int(0.02 * len(df))),
                'zscore': ev.evaluate_zscore(threshold=3.0),
                'isolation_forest': ev.evaluate_isolation_forest(),
            }
        except Exception as e:
            return {'error': str(e)}

    def _evaluate_recovery(self, calib_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate against ground truth using calibration spec names directly."""

        # Ground truth: all spec names that are NOT null
        gt_names = set()
        gt_type_map = {}  # name -> type

        for name, res in calib_results.items():
            rt = res.get('rel_type', '')
            if rt != 'null':
                gt_names.add(name)
                gt_type_map[name] = rt

        # Discovered: significant non-null specs
        disc_names = set()
        for name, res in calib_results.items():
            if res.get('significant', False) and res.get('rel_type', '') != 'null':
                disc_names.add(name)

        # Strict: exact match on spec name
        def _f1(gt, disc):
            tp = len(gt & disc)
            fp = len(disc - gt)
            fn = len(gt - disc)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2*p*r / (p+r) if (p+r) > 0 else 0
            return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                    "tp": tp, "fp": fp, "fn": fn}

        strict = _f1(gt_names, disc_names)

        # Family: group by TYPE_FAMILY
        gt_family = set()
        disc_family = set()
        for name in gt_names:
            rt = gt_type_map[name]
            fam = _TYPE_TO_FAMILY.get(rt, rt)
            suffix = name[len(rt):] if name.startswith(rt) else name
            gt_family.add(f"{fam}{suffix}")
        for name in disc_names:
            rt = gt_type_map.get(name, '')
            fam = _TYPE_TO_FAMILY.get(rt, rt)
            suffix = name[len(rt):] if name.startswith(rt) else name
            disc_family.add(f"{fam}{suffix}")

        family = _f1(gt_family, disc_family)

        # Edge: extract variable pairs from spec names
        gt_edges = set()
        disc_edges = set()
        for name in gt_names:
            parts = name.split('_')
            if len(parts) >= 3:
                gt_edges.add((parts[-2], parts[-1]))
        for name in disc_names:
            parts = name.split('_')
            if len(parts) >= 3:
                disc_edges.add((parts[-2], parts[-1]))

        edge = _f1(gt_edges, disc_edges)

        # Per-type recall
        type_recall = {}
        all_types = set(gt_type_map.values())
        for rt in all_types:
            gt_of_type = {k for k, v in gt_type_map.items() if v == rt}
            disc_of_type = gt_of_type & disc_names
            type_recall[rt] = round(len(disc_of_type) / len(gt_of_type), 4) if gt_of_type else 0.0

        # Null FPR
        null_pairs = self.generator.schema.get("null_pairs", [])
        null_fp = 0
        for p in null_pairs:
            null_key = f"null_{p[0]}_{p[1]}"
            if calib_results.get(null_key, {}).get('significant', False):
                null_fp += 1
        fpr = null_fp / len(null_pairs) if null_pairs else 0.0

        return {
            "strict": strict,
            "family": family,
            "edge": edge,
            "per_type_recall": type_recall,
            "null_fpr": round(fpr, 4),
        }

    def run_sweep(
        self,
        sample_sizes: List[int],
        seeds: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run benchmark across multiple sample sizes, optionally multiple seeds."""
        if seeds is None:
            seeds = [self.seed]

        all_results = []
        for n in sample_sizes:
            for seed in seeds:
                print(f"\n{'='*60}", flush=True)
                print(f"Benchmark: N={n}, seed={seed}", flush=True)
                print(f"{'='*60}", flush=True)
                # Re-create generator with new seed
                self.generator = create_benchmark_generator(self.schema_path, seed)
                res = self.run(n_samples=n)
                res['seed'] = seed
                all_results.append(res)

        return all_results

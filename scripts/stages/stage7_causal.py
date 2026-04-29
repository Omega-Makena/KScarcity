"""
Stage 7 — Causal Pipeline.

Runs scarcity/causal/engine.py::run_causal on the ground-truth pairs
and compares sign accuracy to the online engine.
SKIP if run_causal cannot be called standalone.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stages.utils import (
    ALL_INDICATORS, ARTIFACTS_DIR, compute_baseline_means,
    compute_baseline_stds, fail_result, filter_pairs, load_ground_truth,
    make_result, make_structured_data, rows_to_yearly, save_artifact,
    skip_result,
)

logger = logging.getLogger(__name__)


def _try_import_causal():
    try:
        from scarcity.causal.engine import run_causal
        return run_causal, None
    except ImportError as e:
        return None, str(e)


def _try_import_causal_spec():
    try:
        from scarcity.causal.specs import EstimandSpec
        return EstimandSpec, None
    except ImportError:
        try:
            from scarcity.causal.engine import EstimandSpec
            return EstimandSpec, None
        except ImportError as e:
            return None, str(e)


def _build_dataframe(yearly: Dict) -> Any:
    import pandas as pd
    records = []
    for yr, row in sorted(yearly.items()):
        r = {"year": yr}
        r.update(row)
        records.append(r)
    return pd.DataFrame(records)


def _run_dowhy_pair(run_causal_fn, spec_cls, df, source: str, target: str) -> Dict[str, Any]:
    """Run DoWhy ATE estimation for one (source, target) pair."""
    try:
        from scarcity.causal.specs import RuntimeSpec
        spec = spec_cls(
            treatment=source,
            outcome=target,
        )
        runtime = RuntimeSpec(
            refute_random_common_cause=False,
            refute_placebo_treatment=False,
            refute_data_subset=False,
            parallelism="none",
            fail_policy="continue",
            artifact_root=str(ARTIFACTS_DIR / "causal"),
        )
        causal_result = run_causal_fn(df, [spec], runtime)
        if not causal_result.results:
            err = causal_result.errors[0].message if causal_result.errors else "no results"
            return {"error": err}
        effect = causal_result.results[0]
        ate = effect.estimate
        if ate is None:
            return {"error": "estimate is None"}
        ate_val = float(ate) if not isinstance(ate, (list, tuple)) else float(ate[0])
        sign = 1 if ate_val > 0 else -1
        return {
            "ate": round(ate_val, 6),
            "sign": sign,
        }
    except Exception as e:
        return {"error": str(e)}


def run_stage_7(seed: int = 42) -> Dict[str, Any]:
    start = time.time()

    run_causal, import_error = _try_import_causal()
    if run_causal is None:
        return skip_result("7", "dowhy_benchmark", f"run_causal not importable: {import_error}")

    spec_cls, spec_error = _try_import_causal_spec()
    if spec_cls is None:
        return skip_result("7", "dowhy_benchmark", f"EstimandSpec not importable: {spec_error}")

    try:
        import pandas as pd
    except ImportError:
        return skip_result("7", "dowhy_benchmark", "pandas not available")

    try:
        all_pairs = load_ground_truth()
        unambiguous = filter_pairs(all_pairs, "unambiguous")
    except Exception as e:
        return fail_result("7", "dowhy_benchmark", "causal sign accuracy", str(e))

    rows = make_structured_data(n_obs=34, seed=seed)
    yearly = rows_to_yearly(rows)
    df = _build_dataframe(yearly)

    pair_results = []
    correct, total = 0, 0

    for pair in unambiguous:
        src, tgt, exp_sign = pair["source"], pair["target"], pair["expected_sign"]
        if src not in df.columns or tgt not in df.columns:
            pair_results.append({"source": src, "target": tgt, "skipped": "not in df"})
            continue
        r = _run_dowhy_pair(run_causal, spec_cls, df, src, tgt)
        r["source"] = src
        r["target"] = tgt
        r["expected_sign"] = exp_sign
        if "sign" in r:
            r["sign_correct"] = r["sign"] == exp_sign
            total += 1
            if r["sign_correct"]:
                correct += 1
        pair_results.append(r)

    causal_sign_acc = correct / max(total, 1)
    status = "PASS" if total > 0 else "SKIP"

    return make_result(
        stage="7", name="dowhy_benchmark", status=status,
        target="DoWhy sign accuracy computed and comparable to online engine",
        result={
            "n_evaluated": total,
            "n_correct": correct,
            "causal_sign_accuracy": round(causal_sign_acc, 4),
            "pairs": pair_results,
        },
        wallclock_s=time.time() - start,
    )


def run_all(fast: bool = False) -> List[Dict[str, Any]]:
    results = [run_stage_7()]
    for r in results:
        save_artifact(f"stage7_{r['name']}.json", r)
    return results


if __name__ == "__main__":
    for r in run_all():
        print(f"  [{r['status']}] {r['stage']}: {r['name']}")

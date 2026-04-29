"""
benchmark_harness.py — Comprehensive Benchmarking Harness for K-Scarcity Core.

Covers the full architecture: Engine, Federation, Simulation, Meta-Learning, DRG, Causal.

Usage
-----
    python scripts/benchmark_harness.py                    # all stages
    python scripts/benchmark_harness.py --stage 0          # single stage
    python scripts/benchmark_harness.py --stage 1.2        # single substage
    python scripts/benchmark_harness.py --stage 1 2 3      # multiple stages
    python scripts/benchmark_harness.py --skip-slow        # skip stages > 10min
    python scripts/benchmark_harness.py --fast             # reduced trial counts
    python scripts/benchmark_harness.py --live             # enable WB API (where supported)
    python scripts/benchmark_harness.py --dry-run          # synthetic data only (default)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "harness"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("harness")

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

STAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "0": {
        "module": "scripts.stages.stage0_identity",
        "fn": "run_stage_0",
        "slow": False,
        "description": "Engine identity audit — resolves benchmark vs architecture discrepancy",
    },
    "1.1": {
        "module": "scripts.stages.stage1_foundation",
        "fn": "run_stage_1_1",
        "slow": False,
        "description": "Non-IID verification (Jensen-Shannon divergence)",
    },
    "1.2": {
        "module": "scripts.stages.stage1_foundation",
        "fn": "run_stage_1_2",
        "slow": True,
        "description": "Null data FPR (100 trials of pure noise)",
    },
    "1.3": {
        "module": "scripts.stages.stage1_foundation",
        "fn": "run_stage_1_3",
        "slow": True,
        "description": "Temporal ordering test (chronological vs reversed vs shuffled)",
    },
    "1.4": {
        "module": "scripts.stages.stage1_foundation",
        "fn": "run_stage_1_4",
        "slow": False,
        "description": "Correlation-sign baseline vs engine",
    },
    "2.1": {
        "module": "scripts.stages.stage2_discovery",
        "fn": "run_stage_2_1",
        "slow": True,
        "description": "Four-condition discovery matrix (cold/pretrain x no-fed/fed)",
    },
    "2.2": {
        "module": "scripts.stages.stage2_discovery",
        "fn": "run_stage_2_2",
        "slow": False,
        "description": "Discovery baselines (Pearson, Granger, VAR)",
    },
    "2.3": {
        "module": "scripts.stages.stage2_discovery",
        "fn": "run_stage_2_3",
        "slow": True,
        "description": "Cross-method comparison table",
    },
    "3.1": {
        "module": "scripts.stages.stage3_federation",
        "fn": "run_stage_3_1",
        "slow": True,
        "description": "Evidence-sharing ablation (isolated / fed / pooled)",
    },
    "3.2": {
        "module": "scripts.stages.stage3_federation",
        "fn": "run_stage_3_2",
        "slow": False,
        "description": "HierarchicalFederation vs simple hub",
    },
    "3.3": {
        "module": "scripts.stages.stage3_federation",
        "fn": "run_stage_3_3",
        "slow": False,
        "description": "DP utility-privacy tradeoff sweep",
    },
    "3.4": {
        "module": "scripts.stages.stage3_federation",
        "fn": "run_stage_3_4",
        "slow": False,
        "description": "Byzantine robustness (krum/bulyan/trimmed_mean)",
    },
    "4.1": {
        "module": "scripts.stages.stage4_simulation",
        "fn": "run_stage_4_1",
        "slow": False,
        "description": "SFC accounting identity check",
    },
    "4.2": {
        "module": "scripts.stages.stage4_simulation",
        "fn": "run_stage_4_2",
        "slow": False,
        "description": "Expanded directional validation (12 shocks)",
    },
    "4.3": {
        "module": "scripts.stages.stage4_simulation",
        "fn": "run_stage_4_3",
        "slow": False,
        "description": "Null shock falsification",
    },
    "5.1": {
        "module": "scripts.stages.stage5_meta",
        "fn": "run_stage_5_1",
        "slow": True,
        "description": "Pretrain inversion diagnosis (do inverted pairs get corrected?)",
    },
    "5.2": {
        "module": "scripts.stages.stage5_meta",
        "fn": "run_stage_5_2",
        "slow": True,
        "description": "Pioneer row sweep (accuracy vs n_pioneer_rows)",
    },
    "5.3": {
        "module": "scripts.stages.stage5_meta",
        "fn": "run_stage_5_3",
        "slow": False,
        "description": "MetaIntegrativeLayer policy verification",
    },
    "6.1": {
        "module": "scripts.stages.stage6_drg",
        "fn": "run_stage_6_1",
        "slow": False,
        "description": "DRG assurance level unit test",
    },
    "6.2": {
        "module": "scripts.stages.stage6_drg",
        "fn": "run_stage_6_2",
        "slow": True,
        "description": "Self-regulation loop (DRG -> MPIE -> Meta)",
    },
    "7": {
        "module": "scripts.stages.stage7_causal",
        "fn": "run_stage_7",
        "slow": True,
        "description": "DoWhy causal pipeline benchmark",
    },
    "8.1": {
        "module": "scripts.stages.stage8_integration",
        "fn": "run_stage_8_1",
        "slow": False,
        "description": "EventBus wiring audit (static + live)",
    },
    "9": {
        "module": "scripts.stages.stage9_prediction_mae",
        "fn": "run_stage_9",
        "slow": True,
        "description": "Rolling leave-one-year-out prediction MAE (Mean/AR1/FedAvg/Oracle/Scarcity)",
    },
    "10": {
        "module": "scripts.stages.stage10_regime_transfer",
        "fn": "run_stage_10",
        "slow": True,
        "description": "Regime transfer: post-2008 MAE for AR1-fixed vs rolling vs ScarcityEngine",
    },
    "11.1": {
        "module": "scripts.stages.stage11_sparsity_buffer",
        "fn": "run_stage_11_1",
        "slow": True,
        "description": "Sparsity sweep: MAE degradation at 0/20/40/60% data drop (local vs fed)",
    },
    "11.2": {
        "module": "scripts.stages.stage11_sparsity_buffer",
        "fn": "run_stage_11_2",
        "slow": False,
        "description": "Buffer size sweep: MAE vs buffer_size in [25, 50, 100, 200]",
    },
}

STAGE_ORDER = [
    "0",
    "1.1", "1.2", "1.3", "1.4",
    "2.1", "2.2", "2.3",
    "3.1", "3.2", "3.3", "3.4",
    "4.1", "4.2", "4.3",
    "5.1", "5.2", "5.3",
    "6.1", "6.2",
    "7",
    "8.1",
    "9",
    "10",
    "11.1", "11.2",
]

# ---------------------------------------------------------------------------
# Claim integrity matrix builder
# ---------------------------------------------------------------------------

CLAIM_MAP = {
    "Data heterogeneity (non-IID)":              ["1.1"],
    "Low false-positive rate on null data":       ["1.2"],
    "Temporal ordering sensitivity":             ["1.3"],
    "Engine outperforms naive Pearson baseline":  ["1.4"],
    "Correct sign discovery on GT pairs":         ["2.1", "2.2", "2.3"],
    "Federation improves discovery quality":      ["3.1", "3.2"],
    "Differential privacy utility tradeoff":      ["3.3"],
    "Byzantine robustness of aggregation":        ["3.4"],
    "SFC accounting identity holds":              ["4.1"],
    "Simulation directional validity":            ["4.2"],
    "Null shocks do not spuriously match":        ["4.3"],
    "Live data corrects pretrain inversions":     ["5.1"],
    "More data improves accuracy monotonically":  ["5.2"],
    "MetaIntegrativeLayer policy correctness":    ["5.3"],
    "DRG assurance levels correctly assigned":    ["6.1"],
    "System self-regulates under pressure":       ["6.2"],
    "Causal pipeline sign accuracy":              ["7"],
    "EventBus wiring completeness":               ["8.1"],
    "Federated prediction no worse than local":   ["9"],
    "Adaptive system beats frozen baseline":      ["10"],
    "Federation degrades gracefully under sparsity": ["11.1"],
    "Buffer size monotonically improves MAE":     ["11.2"],
}


def build_claim_matrix(results: List[Dict]) -> Dict[str, Any]:
    by_stage = {r["stage"]: r for r in results}
    matrix = {}
    for claim, stages in CLAIM_MAP.items():
        evidence = []
        statuses = []
        for stage_id in stages:
            r = by_stage.get(stage_id, {})
            statuses.append(r.get("status", "NOT_RUN"))
            evidence.append({
                "stage": stage_id,
                "status": r.get("status", "NOT_RUN"),
                "name": r.get("name", ""),
            })
        overall = (
            "PASS" if all(s == "PASS" for s in statuses)
            else "WARN" if any(s in ("PASS", "WARN") for s in statuses)
            else "SKIP" if all(s == "SKIP" for s in statuses)
            else "FAIL" if any(s == "FAIL" for s in statuses)
            else "NOT_RUN"
        )
        matrix[claim] = {"overall": overall, "evidence": evidence}
    return matrix


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(stage_id: str, fast: bool = False, live: bool = False) -> Dict[str, Any]:
    spec = STAGE_REGISTRY.get(stage_id)
    if spec is None:
        return {
            "stage": stage_id, "name": "unknown", "status": "SKIP",
            "target": "N/A", "result": {"reason": f"stage {stage_id!r} not in registry"},
            "wallclock_s": 0.0,
        }

    try:
        import importlib
        mod = importlib.import_module(spec["module"])
        fn = getattr(mod, spec["fn"])
    except Exception as e:
        return {
            "stage": stage_id, "name": spec.get("fn", "?"), "status": "FAIL",
            "target": "stage importable", "result": {"error": str(e)},
            "wallclock_s": 0.0,
        }

    # Inject fast/live flags where accepted
    import inspect
    sig = inspect.signature(fn)
    kwargs = {}
    if "fast" in sig.parameters:
        kwargs["fast"] = fast
    if "live" in sig.parameters:
        kwargs["live"] = live

    try:
        result = fn(**kwargs)
        return result
    except Exception as e:
        import traceback
        return {
            "stage": stage_id, "name": spec.get("fn", "?"), "status": "FAIL",
            "target": "stage runs without exception",
            "result": {"error": str(e), "traceback": traceback.format_exc()[-2000:]},
            "wallclock_s": 0.0,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_stages(requested: Optional[List[str]], skip_slow: bool) -> List[str]:
    if requested is None:
        stages = STAGE_ORDER
    else:
        # Accept "1" to mean all "1.*" stages
        stages = []
        for sid in STAGE_ORDER:
            if any(sid == r or sid.startswith(r + ".") or sid.startswith(r) for r in requested):
                stages.append(sid)
        if not stages:
            stages = [s for s in requested if s in STAGE_REGISTRY]

    if skip_slow:
        stages = [s for s in stages if not STAGE_REGISTRY.get(s, {}).get("slow", False)]

    return stages


def main():
    parser = argparse.ArgumentParser(description="K-Scarcity benchmark harness")
    parser.add_argument("--stage", nargs="*", metavar="ID",
                        help="Stage IDs to run (e.g. 0 1.2 3). Default: all.")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip stages marked slow (>5min)")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce trial counts for quick smoke-test")
    parser.add_argument("--live", action="store_true",
                        help="Enable World Bank API data (where supported)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic data only (default behaviour, flag for clarity)")
    parser.add_argument("--list", action="store_true",
                        help="List all stages and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'Stage':<8} {'Slow':<6} {'Description'}")
        print("-" * 70)
        for sid in STAGE_ORDER:
            spec = STAGE_REGISTRY[sid]
            slow = "yes" if spec["slow"] else "no"
            print(f"  {sid:<8} {slow:<6} {spec['description']}")
        return

    stages = _resolve_stages(args.stage, args.skip_slow)
    if not stages:
        print("No stages matched. Use --list to see available stages.")
        sys.exit(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nK-Scarcity Benchmark Harness — {len(stages)} stage(s)")
    print("=" * 60)

    results = []
    total_start = time.time()

    for stage_id in stages:
        spec = STAGE_REGISTRY.get(stage_id, {})
        desc = spec.get("description", "")
        print(f"\n[{stage_id}] {desc}")

        result = run_stage(stage_id, fast=args.fast, live=args.live)
        results.append(result)

        status = result.get("status", "UNKNOWN")
        wall = result.get("wallclock_s", 0.0)
        symbol = {"PASS": "+", "FAIL": "X", "WARN": "~", "SKIP": "-"}.get(status, "?")
        print(f"  {symbol} {status:<5}  {wall:.1f}s")

        # Gate on Stage 0
        if stage_id == "0" and status == "FAIL":
            print("\n  FATAL: Stage 0 failed. Resolve engine identity before proceeding.")
            break

    total_wall = time.time() - total_start

    # Save aggregated results
    with open(ARTIFACTS_DIR / "harness_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Build and save claim integrity matrix
    matrix = build_claim_matrix(results)
    with open(ARTIFACTS_DIR / "claim_integrity_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"claims": matrix, "n_stages_run": len(results)}, f, indent=2, default=str)

    # Summary
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_warn = sum(1 for r in results if r.get("status") == "WARN")
    n_skip = sum(1 for r in results if r.get("status") == "SKIP")

    print(f"\n{'=' * 60}")
    print(f"  PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIP={n_skip}  "
          f"total={len(results)}  wall={total_wall:.1f}s")
    print(f"  Artifacts: {ARTIFACTS_DIR}/")
    print()

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

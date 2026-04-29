"""benchmark_master_v2.py -Coverage-first master benchmark for the Scarcity system.

Enforces complete architectural coverage via COVERAGE_MANIFEST. Exits with code 2
if any manifest component has no passing stage. Stages 12–22 cover every component
that had zero prior coverage.

Exit codes:
  0 -all stages pass (or only WARNs)
  1 -at least one FAIL
  2 -coverage failure (manifest component uncovered / SKIP)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure project root is on sys.path so `scarcity` and `scripts` are importable
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# COVERAGE_MANIFEST -the enforcement contract.
# Every component listed here must map to a stage that runs (not SKIP/NOT_RUN).
# Exits code 2 if any entry is uncovered at end of run.
# ---------------------------------------------------------------------------

COVERAGE_MANIFEST: Dict[str, Optional[str]] = {
    # Layer 1 -Engine: all 15 hypothesis types
    "CausalHypothesis":             "12.1",
    "CorrelationalHypothesis":      "12.2",
    "TemporalHypothesis":           "12.3",
    "FunctionalHypothesis":         "12.4",
    "EquilibriumHypothesis":        "12.5",
    "CompositionalHypothesis":      "12.6",
    "CompetitiveHypothesis":        "12.7",
    "SynergisticHypothesis":        "12.8",
    "ProbabilisticHypothesis":      "12.9",
    "StructuralHypothesis":         "12.10",
    "MediatingHypothesis":          "13.1",
    "ModeratingHypothesis":         "13.2",
    "GraphHypothesis":              "13.3",
    "SimilarityHypothesis":         "13.4",
    "LogicalHypothesis":            "13.5",
    # Engine orchestration
    "MetaController":               "14.1",
    "HypothesisArbiter":            "14.2",
    "AdaptiveGrouper":              "14.3",
    "MPIEOrchestrator":             "14.4",
    # Routing + vectorization
    "BanditRouter_Thompson":        "15.1",
    "BanditRouter_UCB":             "15.2",
    "BanditRouter_EpsilonGreedy":   "15.3",
    "VectorizedRLS":                "16.1",
    "VectorizedHypothesisPool":     "16.2",
    # Layer 2 -Federation
    "GossipProtocol":               "17.1",
    "LocalDPMechanism":             "17.1",
    "MaterialityDetector":          "17.1",
    "Layer1Aggregator":             "17.2",
    "BasketManager":                "17.2",
    "Layer2Aggregator":             "17.3",
    "TrustScorer":                  "17.3",
    "HierarchicalFederation":       "17.4",
    "GlobalMetaMemory":             "17.4",
    "basket_isolation":             "17.5",
    # Layer 3 -Simulation
    "MultiSectorSFCEngine":         "18.1",
    "SFC_shock_agriculture":        "18.1",
    "SFC_shock_monetary_trade":     "18.2",
    "SFC_directional_coherence":    "18.3",
    # Layer 4 -Meta-learning
    "OnlineReptileOptimizer":       "19.1",
    "EpisodicMemory":               "19.1",
    "CrossDomainMetaAggregator":    "19.2",
    "MetaIntegrativeLayer":         "19.3",
    # Layer 5 -Governor
    "DynamicResourceGovernor":      "20.1",
    "DRG_compute_scarcity":         "20.2",
    # Layer 6 -Causal pipeline
    "run_causal_DoWhy":             "21.1",
    "run_causal_EconML":            "21.2",
    "Validator_refutation":         "21.3",
    # Cross-cutting scarcity dimensions
    "data_scarcity_N_sweep":        "22.1",
    "compute_scarcity_DRG_loop":    "22.2",
}

# ---------------------------------------------------------------------------
# Stage registry -lazy imports to avoid circular dependencies
# ---------------------------------------------------------------------------

def _load_stage_registry() -> Dict[str, Callable]:
    from scripts.stages.stage12_hyp_core import (
        run_stage_12_1, run_stage_12_2, run_stage_12_3, run_stage_12_4,
        run_stage_12_5, run_stage_12_6, run_stage_12_7, run_stage_12_8,
        run_stage_12_9, run_stage_12_10,
    )
    from scripts.stages.stage13_hyp_extended import (
        run_stage_13_1, run_stage_13_2, run_stage_13_3, run_stage_13_4, run_stage_13_5,
    )
    from scripts.stages.stage14_engine import (
        run_stage_14_1, run_stage_14_2, run_stage_14_3, run_stage_14_4,
    )
    from scripts.stages.stage15_bandit import (
        run_stage_15_1, run_stage_15_2, run_stage_15_3,
    )
    from scripts.stages.stage16_vectorized import (
        run_stage_16_1, run_stage_16_2,
    )
    from scripts.stages.stage17_federation import (
        run_stage_17_1, run_stage_17_2, run_stage_17_3, run_stage_17_4, run_stage_17_5,
    )
    from scripts.stages.stage18_simulation import (
        run_stage_18_1, run_stage_18_2, run_stage_18_3,
    )
    from scripts.stages.stage19_meta import (
        run_stage_19_1, run_stage_19_2, run_stage_19_3,
    )
    from scripts.stages.stage20_drg import (
        run_stage_20_1, run_stage_20_2,
    )
    from scripts.stages.stage21_causal import (
        run_stage_21_1, run_stage_21_2, run_stage_21_3,
    )
    from scripts.stages.stage22_scarcity import (
        run_stage_22_1, run_stage_22_2,
    )

    return {
        "12.1":  run_stage_12_1,
        "12.2":  run_stage_12_2,
        "12.3":  run_stage_12_3,
        "12.4":  run_stage_12_4,
        "12.5":  run_stage_12_5,
        "12.6":  run_stage_12_6,
        "12.7":  run_stage_12_7,
        "12.8":  run_stage_12_8,
        "12.9":  run_stage_12_9,
        "12.10": run_stage_12_10,
        "13.1":  run_stage_13_1,
        "13.2":  run_stage_13_2,
        "13.3":  run_stage_13_3,
        "13.4":  run_stage_13_4,
        "13.5":  run_stage_13_5,
        "14.1":  run_stage_14_1,
        "14.2":  run_stage_14_2,
        "14.3":  run_stage_14_3,
        "14.4":  run_stage_14_4,
        "15.1":  run_stage_15_1,
        "15.2":  run_stage_15_2,
        "15.3":  run_stage_15_3,
        "16.1":  run_stage_16_1,
        "16.2":  run_stage_16_2,
        "17.1":  run_stage_17_1,
        "17.2":  run_stage_17_2,
        "17.3":  run_stage_17_3,
        "17.4":  run_stage_17_4,
        "17.5":  run_stage_17_5,
        "18.1":  run_stage_18_1,
        "18.2":  run_stage_18_2,
        "18.3":  run_stage_18_3,
        "19.1":  run_stage_19_1,
        "19.2":  run_stage_19_2,
        "19.3":  run_stage_19_3,
        "20.1":  run_stage_20_1,
        "20.2":  run_stage_20_2,
        "21.1":  run_stage_21_1,
        "21.2":  run_stage_21_2,
        "21.3":  run_stage_21_3,
        "22.1":  run_stage_22_1,
        "22.2":  run_stage_22_2,
    }


# ---------------------------------------------------------------------------
# Coverage checker
# ---------------------------------------------------------------------------

def _check_coverage(
    results: List[Dict[str, Any]],
    manifest: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    by_stage = {r["stage"]: r["status"] for r in results}
    uncovered: List[Tuple[str, str, str]] = []
    for component, stage_id in manifest.items():
        if stage_id is None:
            uncovered.append((component, "NONE", "NO_STAGE"))
        else:
            status = by_stage.get(stage_id, "NOT_RUN")
            if status in ("SKIP", "NOT_RUN"):
                uncovered.append((component, stage_id, status))
    covered = len(manifest) - len(uncovered)
    coverage_pct = 100.0 * covered / max(len(manifest), 1)
    return {
        "total": len(manifest),
        "covered": covered,
        "uncovered_count": len(uncovered),
        "coverage_pct": round(coverage_pct, 1),
        "uncovered": [(c, s, st) for c, s, st in uncovered],
    }


# ---------------------------------------------------------------------------
# Stage selection helpers
# ---------------------------------------------------------------------------

SLOW_STAGES = {"14.4", "17.4", "21.1", "21.2", "21.3"}


def _select_stages(
    registry: Dict[str, Callable],
    stage_prefixes: Optional[List[str]],
    skip_slow: bool,
) -> Dict[str, Callable]:
    selected = dict(registry)
    if stage_prefixes:
        filtered = {}
        for sid, fn in registry.items():
            for prefix in stage_prefixes:
                if sid.startswith(prefix):
                    filtered[sid] = fn
                    break
        selected = filtered
    if skip_slow:
        selected = {k: v for k, v in selected.items() if k not in SLOW_STAGES}
    return selected


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

STATUS_SYMBOL = {"PASS": "+", "WARN": "~", "FAIL": "X", "SKIP": "-", "NOT_RUN": "?"}

def _print_result(r: Dict[str, Any]) -> None:
    sym = STATUS_SYMBOL.get(r["status"], "?")
    wall = r.get("wallclock_s", 0.0)
    print(f"  [{sym}] {r['stage']:>5}  {r['status']:<4}  {r['name']:<40}  {wall:.2f}s")


def _print_summary(results: List[Dict[str, Any]], coverage: Dict[str, Any], wall_total: float) -> None:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    total = len(results)
    print()
    print("=" * 72)
    print(f"K-Scarcity Master Benchmark v2 -{total} stage(s) in {wall_total:.1f}s")
    print(f"  PASS={counts['PASS']}  WARN={counts['WARN']}  FAIL={counts['FAIL']}  SKIP={counts['SKIP']}  total={total}")
    print(f"  Coverage: {coverage['covered']}/{coverage['total']} manifest items ({coverage['coverage_pct']}%)")
    if coverage["uncovered"]:
        print(f"  UNCOVERED ({coverage['uncovered_count']}):")
        for comp, sid, st in coverage["uncovered"][:10]:
            print(f"    {comp:40s} -> stage {sid} [{st}]")
        if coverage["uncovered_count"] > 10:
            print(f"    ... and {coverage['uncovered_count'] - 10} more")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-Scarcity Master Benchmark v2 -full architectural coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_master_v2.py                  # all stages
  python scripts/benchmark_master_v2.py --stage 12 13    # only hypothesis stages
  python scripts/benchmark_master_v2.py --fast            # shorter N/rounds
  python scripts/benchmark_master_v2.py --skip-slow       # skip slow stages (21.x, 17.4)
  python scripts/benchmark_master_v2.py --list            # list all stages + manifest
  python scripts/benchmark_master_v2.py --check-manifest  # print manifest without running

Exit codes: 0=pass, 1=stage FAIL, 2=coverage failure
        """,
    )
    parser.add_argument("--stage", nargs="+", metavar="PREFIX",
                        help="Only run stages whose ID starts with these prefixes (e.g. 12 13.3)")
    parser.add_argument("--fast", action="store_true",
                        help="Use shorter sequences for speed (n=40 instead of n=80 etc.)")
    parser.add_argument("--skip-slow", action="store_true",
                        help=f"Skip slow stages: {sorted(SLOW_STAGES)}")
    parser.add_argument("--list", action="store_true",
                        help="Print all stage IDs and manifest entries, then exit")
    parser.add_argument("--check-manifest", action="store_true",
                        help="Print COVERAGE_MANIFEST without running any stages")
    parser.add_argument("--json", dest="json_out", metavar="FILE",
                        help="Write full results JSON to FILE")
    parser.add_argument("--no-coverage-fail", action="store_true",
                        help="Do not exit 2 on coverage failure (useful in CI during development)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    if args.check_manifest:
        print(f"COVERAGE_MANIFEST -{len(COVERAGE_MANIFEST)} components")
        for comp, stage_id in COVERAGE_MANIFEST.items():
            print(f"  {comp:42s} -> {stage_id or 'NO_STAGE'}")
        return 0

    registry = _load_stage_registry()

    if args.list:
        print(f"Stage registry -{len(registry)} stages")
        for sid in sorted(registry):
            slow_marker = " [SLOW]" if sid in SLOW_STAGES else ""
            print(f"  {sid}{slow_marker}")
        print()
        print(f"Coverage manifest - {len(COVERAGE_MANIFEST)} components")
        for comp, sid in COVERAGE_MANIFEST.items():
            print(f"  {comp:42s} -> {sid or 'NO_STAGE'}")
        return 0

    selected = _select_stages(registry, args.stage, args.skip_slow)
    stage_ids_sorted = sorted(selected.keys(), key=lambda s: [int(x) for x in s.split(".")])

    print(f"K-Scarcity Master Benchmark v2 -{len(stage_ids_sorted)} stage(s)")
    if args.fast:
        print("  Mode: FAST (shorter sequences)")
    if args.skip_slow:
        print(f"  Skipping slow stages: {sorted(SLOW_STAGES)}")
    print()

    results: List[Dict[str, Any]] = []
    t_total = time.time()
    any_fail = False

    for sid in stage_ids_sorted:
        runner = selected[sid]
        try:
            result = runner(fast=args.fast)
        except Exception as exc:
            result = {
                "stage": sid,
                "name": f"stage_{sid}",
                "status": "FAIL",
                "target": "no crash",
                "result": {"error": str(exc)},
                "wallclock_s": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        results.append(result)
        _print_result(result)
        if result["status"] == "FAIL":
            any_fail = True

    wall_total = time.time() - t_total

    # Run coverage check against every stage that was in scope
    # (stages not selected get NOT_RUN treatment in the checker)
    coverage = _check_coverage(results, COVERAGE_MANIFEST)

    _print_summary(results, coverage, wall_total)

    if args.json_out:
        payload = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "fast": args.fast,
            "skip_slow": args.skip_slow,
            "n_stages": len(results),
            "wall_total_s": round(wall_total, 2),
            "coverage": coverage,
            "stages": results,
        }
        try:
            import pathlib
            pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            print(f"Results written to {args.json_out}")
        except Exception as exc:
            print(f"Warning: could not write JSON output: {exc}")

    # Exit code logic
    if coverage["uncovered"] and not args.no_coverage_fail:
        print(f"\nCOVERAGE FAILURE: {coverage['uncovered_count']} manifest item(s) not covered")
        return 2
    if any_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

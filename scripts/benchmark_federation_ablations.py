"""
Federation Ablation Benchmarks
================================
Four ablation studies that support the Scarcity preprint claims:

  A. Sparsity sweep     — drop 0/20/40/60% of annual observations randomly;
                          shows federation degrades more gracefully than local-only
  B. Federation size    — 1-node (local), 2-node, 3-node cross-training;
                          shows benefit curve as peers are added
  C. Buffer size        — buffer ∈ {25, 50, 100, 200};
                          shows DRG-style compute/accuracy trade-off
  D. Peer specificity   — Uganda paired with KEN-only, TZA-only, or both;
                          shows whether geographic proximity matters

All benchmarks reuse synthetic data by default (--dry-run) so they run
fully offline in CI.  Pass --live to hit the real World Bank API.

Outputs (artifacts/meta/):
  ablation_sparsity.csv
  ablation_federation_size.csv
  ablation_buffer_size.csv
  ablation_peer_specificity.csv
  ablation_summary.txt

Usage:
    python scripts/benchmark_federation_ablations.py
    python scripts/benchmark_federation_ablations.py --live
    python scripts/benchmark_federation_ablations.py --bench sparsity buffer
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.ablations")

# Re-use constants from the main experiment
from scripts.experiment_east_africa_federation import (
    WB_INDICATORS,
    COUNTRIES,
    fetch_country_data,
    _mock_country_data,
    _build_schema,
    _avg_confidence,
    _active_count,
)

CONF_THRESHOLD = 0.25   # cold-start gate in get_candidate_paths()
OUT_DIR = PROJECT_ROOT / "artifacts" / "meta"


# ---------------------------------------------------------------------------
# Shared engine factory
# ---------------------------------------------------------------------------

def _make_engine(field_names: List[str], buffer_size: int = 50):
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    eng = OnlineDiscoveryEngine(explore_interval=5, mode="balanced", buffer_size=buffer_size)
    eng.initialize(_build_schema(field_names))
    return eng


def _steps_to_threshold(engine, rows: List[Dict], threshold: float = CONF_THRESHOLD) -> int:
    """Feed rows one by one; return step count when avg_confidence first exceeds threshold."""
    for i, row in enumerate(rows, 1):
        engine.process_row(row)
        if _avg_confidence(engine) >= threshold:
            return i
    return -1   # never reached


def _run_stream(engine, rows: List[Dict]) -> float:
    """Feed all rows, return final avg_confidence."""
    for row in rows:
        engine.process_row(row)
    return _avg_confidence(engine)


# ---------------------------------------------------------------------------
# A. Sparsity sweep
# ---------------------------------------------------------------------------

def bench_sparsity(
    country_data: Dict[str, Dict[int, Dict]],
    all_fields: List[str],
    drop_fractions: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6),
    n_seeds: int = 5,
) -> List[Dict]:
    """
    For each drop fraction, randomly remove that fraction of years from each
    country before training.  Repeat over n_seeds random seeds to get variance.

    Compares local-only vs 3-node federated under each sparsity level.
    """
    logger.info("=== Benchmark A: Sparsity Sweep ===")
    records: List[Dict] = []

    for drop_frac in drop_fractions:
        for seed in range(n_seeds):
            rng = random.Random(seed)

            for scenario in ("local", "federated"):
                engines = {c: _make_engine(all_fields) for c in COUNTRIES}
                conf_sum = {c: 0.0 for c in COUNTRIES}

                for code in COUNTRIES:
                    years = sorted(country_data[code].keys())
                    # Apply drop
                    keep = [y for y in years if rng.random() > drop_frac]
                    if not keep:
                        keep = years[:1]

                    for year in keep:
                        own_row = country_data[code].get(year)
                        if own_row is None:
                            continue
                        engines[code].process_row(own_row)

                        if scenario == "federated":
                            for peer in COUNTRIES:
                                if peer == code:
                                    continue
                                peer_row = country_data[peer].get(year)
                                if peer_row is not None:
                                    engines[code].process_row(peer_row)

                    conf_sum[code] = _avg_confidence(engines[code])

                mean_conf = sum(conf_sum.values()) / len(conf_sum)
                records.append({
                    "benchmark":      "sparsity",
                    "scenario":       scenario,
                    "drop_fraction":  drop_frac,
                    "seed":           seed,
                    "mean_avg_confidence": round(mean_conf, 6),
                    "kenya_conf":     round(conf_sum["KEN"], 6),
                    "tanzania_conf":  round(conf_sum["TZA"], 6),
                    "uganda_conf":    round(conf_sum["UGA"], 6),
                })

    return records


# ---------------------------------------------------------------------------
# B. Federation size
# ---------------------------------------------------------------------------

def bench_federation_size(
    country_data: Dict[str, Dict[int, Dict]],
    all_fields: List[str],
) -> List[Dict]:
    """
    Train Uganda's engine with 0 / 1 / 2 peer countries contributing cross-data.

    N=0 : local only
    N=1 : Uganda + Kenya
    N=1 : Uganda + Tanzania
    N=2 : Uganda + Kenya + Tanzania
    """
    logger.info("=== Benchmark B: Federation Size ===")
    records: List[Dict] = []
    years = sorted({y for rows in country_data.values() for y in rows})

    def run_uganda_with_peers(peers: List[str]) -> Tuple[float, int, int]:
        eng = _make_engine(all_fields)
        for year in years:
            own = country_data["UGA"].get(year)
            if own is not None:
                eng.process_row(own)
            for peer in peers:
                row = country_data[peer].get(year)
                if row is not None:
                    eng.process_row(row)
        return _avg_confidence(eng), _active_count(eng), eng.step_count

    configs = [
        ("local",             "UGA", []),
        ("fed_ken",           "UGA", ["KEN"]),
        ("fed_tza",           "UGA", ["TZA"]),
        ("fed_ken_tza",       "UGA", ["KEN", "TZA"]),
    ]

    for label, focus, peers in configs:
        conf, active, steps = run_uganda_with_peers(peers)
        records.append({
            "benchmark":      "federation_size",
            "config":         label,
            "focus_country":  focus,
            "peers":          "+".join(peers) if peers else "none",
            "n_peers":        len(peers),
            "avg_confidence": round(conf, 6),
            "active_hyp":     active,
            "total_steps":    steps,
        })

    # Same experiment repeated for Kenya (as a stable anchor)
    ken_configs = [
        ("local",      "KEN", []),
        ("fed_tza",    "KEN", ["TZA"]),
        ("fed_uga",    "KEN", ["UGA"]),
        ("fed_all",    "KEN", ["TZA", "UGA"]),
    ]
    for label, focus, peers in ken_configs:
        eng = _make_engine(all_fields)
        for year in years:
            own = country_data[focus].get(year)
            if own is not None:
                eng.process_row(own)
            for peer in peers:
                row = country_data[peer].get(year)
                if row is not None:
                    eng.process_row(row)
        conf = _avg_confidence(eng)
        records.append({
            "benchmark":      "federation_size",
            "config":         label,
            "focus_country":  focus,
            "peers":          "+".join(peers) if peers else "none",
            "n_peers":        len(peers),
            "avg_confidence": round(conf, 6),
            "active_hyp":     _active_count(eng),
            "total_steps":    eng.step_count,
        })

    return records


# ---------------------------------------------------------------------------
# C. Buffer size sensitivity
# ---------------------------------------------------------------------------

def bench_buffer_size(
    country_data: Dict[str, Dict[int, Dict]],
    all_fields: List[str],
    buffer_sizes: Tuple[int, ...] = (25, 50, 100, 200),
) -> List[Dict]:
    """
    Vary buffer_size (the sliding window for window-batch hypotheses).

    Smaller buffers → less memory, faster adapt, noisier estimates.
    Larger buffers → more accurate, slower cold-start.

    This directly demonstrates the DRG compute-budget trade-off.
    Records:
      - final avg_confidence for each country
      - steps to cross CONF_THRESHOLD for each country
      - total hypotheses spawned
    """
    logger.info("=== Benchmark C: Buffer Size Sensitivity ===")
    records: List[Dict] = []
    years = sorted({y for rows in country_data.values() for y in rows})

    for buf in buffer_sizes:
        for code in COUNTRIES:
            rows = [country_data[code][y] for y in years if y in country_data[code]]

            # Convergence speed: steps to threshold
            eng_conv = _make_engine(all_fields, buffer_size=buf)
            steps_to_thresh = _steps_to_threshold(eng_conv, rows)

            # Final confidence: full stream
            eng_final = _make_engine(all_fields, buffer_size=buf)
            final_conf = _run_stream(eng_final, rows)

            records.append({
                "benchmark":        "buffer_size",
                "buffer_size":      buf,
                "country":          COUNTRIES[code],
                "country_code":     code,
                "final_confidence": round(final_conf, 6),
                "steps_to_threshold": steps_to_thresh,
                "threshold":        CONF_THRESHOLD,
                "total_observations": len(rows),
                "total_hypotheses": len(eng_final.hypotheses.population),
                "active_hyp":       _active_count(eng_final),
            })

    return records


# ---------------------------------------------------------------------------
# D. Peer specificity
# ---------------------------------------------------------------------------

def bench_peer_specificity(
    country_data: Dict[str, Dict[int, Dict]],
    all_fields: List[str],
) -> List[Dict]:
    """
    For each country as focus node, measure contribution of each individual peer
    by computing the confidence gain over local baseline.

    gain(peer) = conf(focus + peer) - conf(focus alone)

    This tests whether cross-domain transfer is uniform or peer-specific,
    relevant to the heterogeneity claim.
    """
    logger.info("=== Benchmark D: Peer Specificity ===")
    records: List[Dict] = []
    years = sorted({y for rows in country_data.values() for y in rows})

    def run_with_peers(focus: str, peers: List[str]) -> float:
        eng = _make_engine(all_fields)
        for year in years:
            own = country_data[focus].get(year)
            if own is not None:
                eng.process_row(own)
            for peer in peers:
                row = country_data[peer].get(year)
                if row is not None:
                    eng.process_row(row)
        return _avg_confidence(eng)

    for focus in COUNTRIES:
        peers = [c for c in COUNTRIES if c != focus]
        baseline = run_with_peers(focus, [])

        records.append({
            "benchmark":      "peer_specificity",
            "focus":          COUNTRIES[focus],
            "focus_code":     focus,
            "peers_added":    "none",
            "n_peers":        0,
            "avg_confidence": round(baseline, 6),
            "gain_over_local": 0.0,
        })

        for peer in peers:
            conf_with_peer = run_with_peers(focus, [peer])
            gain = conf_with_peer - baseline
            records.append({
                "benchmark":       "peer_specificity",
                "focus":           COUNTRIES[focus],
                "focus_code":      focus,
                "peers_added":     COUNTRIES[peer],
                "n_peers":         1,
                "avg_confidence":  round(conf_with_peer, 6),
                "gain_over_local": round(gain, 6),
            })

        # All peers together
        conf_all = run_with_peers(focus, peers)
        records.append({
            "benchmark":       "peer_specificity",
            "focus":           COUNTRIES[focus],
            "focus_code":      focus,
            "peers_added":     "+".join(COUNTRIES[p] for p in peers),
            "n_peers":         len(peers),
            "avg_confidence":  round(conf_all, 6),
            "gain_over_local": round(conf_all - baseline, 6),
        })

    return records


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(
    sparsity: List[Dict],
    fed_size: List[Dict],
    buf_size: List[Dict],
    peer_spec: List[Dict],
    path: Path,
) -> None:
    lines = []
    lines.append("ABLATION BENCHMARK SUMMARY")
    lines.append("=" * 60)

    # A. Sparsity
    lines.append("\nA. SPARSITY SWEEP  (mean avg_confidence across 3 countries × 5 seeds)")
    lines.append(f"  {'Drop %':<10} {'Local':>10} {'Federated':>12} {'Fed Advantage':>15}")
    lines.append("  " + "-" * 50)
    from collections import defaultdict
    by_drop_scenario: Dict = defaultdict(lambda: defaultdict(list))
    for r in sparsity:
        by_drop_scenario[r["drop_fraction"]][r["scenario"]].append(r["mean_avg_confidence"])
    for drop in sorted(by_drop_scenario):
        loc = sum(by_drop_scenario[drop]["local"]) / max(1, len(by_drop_scenario[drop]["local"]))
        fed = sum(by_drop_scenario[drop]["federated"]) / max(1, len(by_drop_scenario[drop]["federated"]))
        adv = fed - loc
        lines.append(f"  {int(drop*100):>5}%      {loc:>10.4f} {fed:>12.4f} {adv:>+15.4f}")

    # B. Federation size (Uganda focus)
    lines.append("\nB. FEDERATION SIZE  (Uganda as focus node)")
    lines.append(f"  {'Config':<15} {'N peers':>8} {'Confidence':>12}")
    lines.append("  " + "-" * 38)
    uga_rows = [r for r in fed_size if r["focus_country"] == "UGA"]
    for r in sorted(uga_rows, key=lambda x: x["n_peers"]):
        lines.append(f"  {r['config']:<15} {r['n_peers']:>8} {r['avg_confidence']:>12.4f}")

    # C. Buffer size
    lines.append("\nC. BUFFER SIZE SENSITIVITY  (mean across 3 countries)")
    lines.append(f"  {'Buffer':>8} {'Final Conf':>12} {'Steps→0.25':>12}")
    lines.append("  " + "-" * 36)
    by_buf: Dict = defaultdict(list)
    by_buf_steps: Dict = defaultdict(list)
    for r in buf_size:
        by_buf[r["buffer_size"]].append(r["final_confidence"])
        if r["steps_to_threshold"] > 0:
            by_buf_steps[r["buffer_size"]].append(r["steps_to_threshold"])
    for buf in sorted(by_buf):
        mean_conf = sum(by_buf[buf]) / len(by_buf[buf])
        steps_list = by_buf_steps.get(buf, [])
        mean_steps = sum(steps_list) / len(steps_list) if steps_list else float("nan")
        steps_str = f"{mean_steps:.1f}" if not math.isnan(mean_steps) else "never"
        lines.append(f"  {buf:>8} {mean_conf:>12.4f} {steps_str:>12}")

    # D. Peer specificity
    lines.append("\nD. PEER SPECIFICITY  (confidence gain over local baseline)")
    lines.append(f"  {'Focus':<12} {'Peer':<12} {'Confidence':>12} {'Gain':>8}")
    lines.append("  " + "-" * 48)
    for r in sorted(peer_spec, key=lambda x: (x["focus"], x["n_peers"])):
        lines.append(
            f"  {r['focus']:<12} {r['peers_added']:<12}"
            f" {r['avg_confidence']:>12.4f} {r['gain_over_local']:>+8.4f}"
        )

    lines.append("")
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        print("\n" + text, flush=True)
    except UnicodeEncodeError:
        print("\n" + text.encode("ascii", "replace").decode("ascii"), flush=True)
    logger.info(f"Summary written -> {path}")


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def save_csv(records: List[Dict], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved {len(records)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",   type=int, default=1995)
    p.add_argument("--end",     type=int, default=2023)
    p.add_argument("--seeds",   type=int, default=5, help="Seeds for sparsity sweep")
    p.add_argument("--live",    action="store_true", help="Fetch real World Bank data")
    p.add_argument("--bench",   nargs="*",
                   choices=["sparsity", "size", "buffer", "peer"],
                   help="Run only named benchmarks (default: all)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_all = not args.bench

    logger.info("Federation Ablation Benchmarks")
    logger.info(f"Mode: {'LIVE (World Bank API)' if args.live else 'DRY RUN (synthetic)'}")

    # ---- Load data ---------------------------------------------------------
    country_data: Dict[str, Dict[int, Dict]] = {}
    for code in COUNTRIES:
        if args.live:
            country_data[code] = fetch_country_data(code, args.start, args.end)
        else:
            country_data[code] = _mock_country_data(code, args.start, args.end)
        if not country_data[code]:
            logger.error(f"No data for {code}. Use --live or check mock data.")
            sys.exit(1)

    all_fields = sorted({
        field
        for rows in country_data.values()
        for row in rows.values()
        for field in row
    })
    logger.info(f"Fields: {len(all_fields)}   Countries: {list(COUNTRIES)}")

    # ---- Run benchmarks ----------------------------------------------------
    sparsity_records: List[Dict] = []
    size_records:     List[Dict] = []
    buffer_records:   List[Dict] = []
    peer_records:     List[Dict] = []

    if run_all or "sparsity" in (args.bench or []):
        sparsity_records = bench_sparsity(country_data, all_fields, n_seeds=args.seeds)
        save_csv(sparsity_records, args.out_dir / "ablation_sparsity.csv")

    if run_all or "size" in (args.bench or []):
        size_records = bench_federation_size(country_data, all_fields)
        save_csv(size_records, args.out_dir / "ablation_federation_size.csv")

    if run_all or "buffer" in (args.bench or []):
        buffer_records = bench_buffer_size(country_data, all_fields)
        save_csv(buffer_records, args.out_dir / "ablation_buffer_size.csv")

    if run_all or "peer" in (args.bench or []):
        peer_records = bench_peer_specificity(country_data, all_fields)
        save_csv(peer_records, args.out_dir / "ablation_peer_specificity.csv")

    # ---- Summary -----------------------------------------------------------
    if sparsity_records or size_records or buffer_records or peer_records:
        write_summary(
            sparsity_records, size_records, buffer_records, peer_records,
            args.out_dir / "ablation_summary.txt",
        )

    logger.info("All ablation benchmarks complete.")


if __name__ == "__main__":
    main()

"""
Comprehensive Benchmark — Scarcity Robustness & Depth Suite
============================================================
Covers gaps not addressed by existing benchmark scripts:

  A. ABLATION STUDIES
     A1. No lifecycle management (pruning disabled)
     A2. No confidence gate (all hypotheses feed simulation)
     A3. Federation mechanism: evidence-sharing vs pooled vs isolated
     A4. Peer count: full federation vs single peer vs no peers

  B. STRESS TESTS
     B1. Permutation test — shuffled time series → discovery should degrade
     B2. Time reversal — reversed chronology → causal relationships should break
     B3. Synthetic null world — random data → false positive rate
     B4. Shock falsification — shocks on structurally disconnected variables

  C. FAILURE MODES
     C1. Early overconfidence — confidence at step 5 vs step 34
     C2. Conflict oscillation — hypotheses cycling ACTIVE↔DECAYING
     C3. Structural break — mid-stream regime change

  D. CALIBRATION
     D1. Confidence bins vs fit_score (reliability proxy)

  E. HYPOTHESIS LIFECYCLE
     E1. Creation, pruning, lifetime distribution per run

  F. DRG PERFORMANCE
     F1. Throughput (obs/sec), latency distribution, memory delta

Outputs:  artifacts/meta/ablation_*.csv
          artifacts/meta/stress_*.csv
          artifacts/meta/failure_*.csv
          artifacts/meta/calibration_*.csv
          artifacts/meta/lifecycle_*.csv
          artifacts/meta/drg_perf.csv
          artifacts/meta/comprehensive_summary.txt

Usage:
    python scripts/benchmark_comprehensive.py
    python scripts/benchmark_comprehensive.py --sections A B C  # specific sections
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import random
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.comprehensive")

OUT_DIR = PROJECT_ROOT / "artifacts" / "meta"

# Re-use mock data and helpers from existing scripts
from scripts.experiment_east_africa_federation import (
    WB_INDICATORS, COUNTRIES, _mock_country_data,
    _build_schema, _avg_confidence, _active_count,
)

COUNTRIES_3 = {k: v for k, v in COUNTRIES.items()}  # KEN, TZA, UGA
START_YEAR, END_YEAR = 1990, 2023
ALL_FIELDS = list(WB_INDICATORS.values())
CONF_GATE   = 0.25   # simulation activation threshold


# ===========================================================================
# Engine factory helpers
# ===========================================================================

def _make_engine(
    conf_thresh: float = 0.25,
    stab_thresh: float = 0.25,
    min_evidence: int  = 5,
    buffer_size: int   = 50,
    explore_interval: int = 2,
) -> "OnlineDiscoveryEngine":
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    engine = OnlineDiscoveryEngine(
        explore_interval=explore_interval,
        mode="balanced",
        buffer_size=buffer_size,
    )
    engine.initialize(_build_schema(ALL_FIELDS))
    engine.meta_controller.conf_thresh  = conf_thresh
    engine.meta_controller.stab_thresh  = stab_thresh
    engine.meta_controller.min_evidence = min_evidence
    return engine


def _train_engine(engine, rows: List[Dict[str, float]]) -> None:
    for row in rows:
        try:
            engine.process_row(row)
        except Exception:
            pass


def _engine_stats(engine) -> Dict[str, float]:
    """Return summary stats from a trained engine."""
    hyps = list(engine.hypotheses.population.values())
    if not hyps:
        return {"avg_conf": 0.0, "n_active": 0, "n_total": 0,
                "n_dead": 0, "n_tentative": 0, "n_decaying": 0}

    active    = [h for h in hyps if _state_str(h) == "active"]
    tentative = [h for h in hyps if _state_str(h) == "tentative"]
    decaying  = [h for h in hyps if _state_str(h) == "decaying"]
    dead      = [h for h in hyps if _state_str(h) == "dead"]

    confs  = [float(getattr(h, "confidence", 0)) for h in active if float(getattr(h, "confidence", 0)) > 0]
    avg_c  = sum(confs) / len(confs) if confs else 0.0
    graveyard_n = len(getattr(engine.hypotheses, "graveyard", []))

    return {
        "avg_conf":   round(avg_c, 4),
        "n_active":   len(active),
        "n_tentative":len(tentative),
        "n_decaying": len(decaying),
        "n_dead":     len(dead) + graveyard_n,
        "n_total":    len(hyps),
    }


def _state_str(hyp) -> str:
    s = getattr(hyp, "state", None) or getattr(hyp.meta, "state", None)
    if s is None:
        return "unknown"
    return str(s.value) if hasattr(s, "value") else str(s).lower()


def _load_mock_rows(seed: int = 42) -> Dict[str, List[Dict[str, float]]]:
    """Load synthetic rows for KEN, TZA, UGA sorted by year."""
    out = {}
    for code in COUNTRIES_3:
        raw = _mock_country_data(code, START_YEAR, END_YEAR, seed=seed)
        out[code] = [raw[yr] for yr in sorted(raw)]
    return out


def _write_csv(path: Path, records: List[Dict]) -> None:
    if not records:
        logger.warning(f"No records for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"  -> {path.name} ({len(records)} rows)")


# ===========================================================================
# A. ABLATION STUDIES
# ===========================================================================

def section_A_ablations(rows_by_country: Dict[str, List[Dict]]) -> List[str]:
    logger.info("=== SECTION A: ABLATION STUDIES ===")
    summary = []

    # ---- A1: No lifecycle management ----------------------------------------
    logger.info("  A1: No lifecycle management (pruning disabled)")
    configs = {
        "standard":    dict(conf_thresh=0.25, stab_thresh=0.25, min_evidence=5),
        "no_pruning":  dict(conf_thresh=0.0,  stab_thresh=0.0,  min_evidence=1),
        "tight":       dict(conf_thresh=0.5,  stab_thresh=0.5,  min_evidence=15),
    }
    a1_records = []
    for label, cfg in configs.items():
        for code in COUNTRIES_3:
            engine = _make_engine(**cfg)
            _train_engine(engine, rows_by_country[code])
            st = _engine_stats(engine)
            a1_records.append({
                "config": label, "country": COUNTRIES_3[code],
                **st, "can_simulate": int(st["avg_conf"] >= CONF_GATE),
            })
    _write_csv(OUT_DIR / "ablation_A1_pruning.csv", a1_records)

    a1_std = [r for r in a1_records if r["config"] == "standard"]
    a1_np  = [r for r in a1_records if r["config"] == "no_pruning"]
    avg_std = sum(r["avg_conf"] for r in a1_std) / max(1, len(a1_std))
    avg_np  = sum(r["avg_conf"] for r in a1_np)  / max(1, len(a1_np))
    summary.append(
        f"A1 Lifecycle mgmt: standard avg_conf={avg_std:.3f} | no-pruning avg_conf={avg_np:.3f} | "
        f"delta={avg_np - avg_std:+.3f}"
    )

    # ---- A2: Confidence gate comparison --------------------------------------
    logger.info("  A2: Confidence gate — which hypotheses qualify for simulation")
    a2_records = []
    for code in COUNTRIES_3:
        engine = _make_engine(conf_thresh=0.25, stab_thresh=0.25, min_evidence=5)
        _train_engine(engine, rows_by_country[code])
        hyps = list(engine.hypotheses.population.values())
        for gate in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            eligible = [h for h in hyps if float(getattr(h, "confidence", 0)) >= gate]
            a2_records.append({
                "country": COUNTRIES_3[code],
                "conf_gate": gate,
                "n_eligible": len(eligible),
                "n_total": len(hyps),
                "pct_eligible": round(100 * len(eligible) / max(1, len(hyps)), 1),
                "avg_conf_eligible": round(
                    sum(float(getattr(h, "confidence", 0)) for h in eligible) / max(1, len(eligible)), 4
                ),
            })
    _write_csv(OUT_DIR / "ablation_A2_conf_gate.csv", a2_records)
    gate_25 = [r for r in a2_records if r["conf_gate"] == 0.25]
    pct_25  = sum(r["pct_eligible"] for r in gate_25) / max(1, len(gate_25))
    summary.append(f"A2 At gate=0.25: {pct_25:.1f}% of hypotheses qualify for simulation")

    # ---- A3: Federation mechanism comparison ---------------------------------
    logger.info("  A3: Federation mechanism — evidence-sharing vs pooled vs isolated")
    a3_records = []
    all_rows_pooled = []
    for code in COUNTRIES_3:
        all_rows_pooled.extend(rows_by_country[code])
    random.shuffle(all_rows_pooled)

    # Isolated: each engine trains only on its own rows
    for code in COUNTRIES_3:
        engine = _make_engine()
        _train_engine(engine, rows_by_country[code])
        st = _engine_stats(engine)
        a3_records.append({"mechanism": "isolated", "country": COUNTRIES_3[code], **st})

    # Evidence-sharing (federated): each engine trains on its own rows + peer rows
    for code in COUNTRIES_3:
        engine = _make_engine()
        own   = rows_by_country[code]
        peers = [r for c, rows in rows_by_country.items() if c != code for r in rows]
        random.shuffle(peers)
        # Interleave: 1 own row then 1 peer row
        interleaved = []
        p_idx = 0
        for own_row in own:
            interleaved.append(own_row)
            if p_idx < len(peers):
                interleaved.append(peers[p_idx]); p_idx += 1
        _train_engine(engine, interleaved)
        st = _engine_stats(engine)
        a3_records.append({"mechanism": "evidence_sharing", "country": COUNTRIES_3[code], **st})

    # Pooled: one engine sees all rows (centralised upper bound)
    engine_pooled = _make_engine()
    _train_engine(engine_pooled, all_rows_pooled)
    st_pooled = _engine_stats(engine_pooled)
    for code in COUNTRIES_3:
        a3_records.append({"mechanism": "pooled_centralised", "country": COUNTRIES_3[code],
                           **st_pooled})

    _write_csv(OUT_DIR / "ablation_A3_mechanism.csv", a3_records)
    by_mech = defaultdict(list)
    for r in a3_records:
        by_mech[r["mechanism"]].append(r["avg_conf"])
    for mech, confs in sorted(by_mech.items()):
        avg = sum(confs) / len(confs)
        summary.append(f"A3 Mechanism [{mech}]: avg_conf={avg:.3f}")

    # ---- A4: Peer count (Uganda focus) ---------------------------------------
    logger.info("  A4: Peer count — Uganda with 0 / 1 / 2 peers")
    a4_records = []
    peer_combos = {
        "no_peers":     [],
        "one_peer_KEN": ["KEN"],
        "one_peer_TZA": ["TZA"],
        "two_peers":    ["KEN", "TZA"],
    }
    for label, peer_codes in peer_combos.items():
        engine = _make_engine()
        rows = list(rows_by_country["UGA"])
        for pc in peer_codes:
            rows = rows + list(rows_by_country[pc])
        random.shuffle(rows)
        _train_engine(engine, rows)
        st = _engine_stats(engine)
        a4_records.append({"variant": label, "n_peers": len(peer_codes), **st})
    _write_csv(OUT_DIR / "ablation_A4_peer_count.csv", a4_records)
    for r in a4_records:
        summary.append(
            f"A4 Peers={r['n_peers']} [{r['variant']}]: avg_conf={r['avg_conf']:.3f}, "
            f"n_active={r['n_active']}"
        )

    return summary


# ===========================================================================
# B. STRESS TESTS
# ===========================================================================

def section_B_stress(rows_by_country: Dict[str, List[Dict]]) -> List[str]:
    logger.info("=== SECTION B: STRESS TESTS ===")
    summary = []
    rng = random.Random(42)

    # Use Kenya as the test country
    ken_rows = list(rows_by_country["KEN"])

    # ---- B1: Permutation test -----------------------------------------------
    logger.info("  B1: Permutation test — shuffled vs ordered time series")
    b1_records = []
    for trial in range(5):
        shuffled = list(ken_rows)
        rng.shuffle(shuffled)
        # Ordered
        e_ordered   = _make_engine(); _train_engine(e_ordered, ken_rows)
        e_shuffled  = _make_engine(); _train_engine(e_shuffled, shuffled)
        so = _engine_stats(e_ordered)
        ss = _engine_stats(e_shuffled)
        b1_records.append({
            "trial": trial,
            "ordered_conf":  so["avg_conf"], "ordered_active":  so["n_active"],
            "shuffled_conf": ss["avg_conf"], "shuffled_active": ss["n_active"],
            "conf_delta": round(ss["avg_conf"] - so["avg_conf"], 4),
        })
    _write_csv(OUT_DIR / "stress_B1_permutation.csv", b1_records)
    avg_delta = sum(r["conf_delta"] for r in b1_records) / len(b1_records)
    summary.append(
        f"B1 Permutation: mean conf drop={avg_delta:+.3f} when time order destroyed "
        f"(negative = real structure, positive = artefact)"
    )

    # ---- B2: Time reversal --------------------------------------------------
    logger.info("  B2: Time reversal — reverse chronological order")
    b2_records = []
    reversed_rows = list(reversed(ken_rows))
    e_forward  = _make_engine(); _train_engine(e_forward, ken_rows)
    e_reversed = _make_engine(); _train_engine(e_reversed, reversed_rows)
    sf = _engine_stats(e_forward)
    sr = _engine_stats(e_reversed)
    b2_records.append({
        "variant": "forward",  **sf})
    b2_records.append({
        "variant": "reversed", **sr})
    _write_csv(OUT_DIR / "stress_B2_time_reversal.csv", b2_records)
    delta_rev = sr["avg_conf"] - sf["avg_conf"]
    summary.append(
        f"B2 Time reversal: forward={sf['avg_conf']:.3f}, reversed={sr['avg_conf']:.3f}, "
        f"delta={delta_rev:+.3f}"
    )

    # ---- B3: Synthetic null — false positive rate ---------------------------
    logger.info("  B3: Synthetic null world — random data, no causal structure")
    b3_records = []
    for trial in range(5):
        # Generate truly random data (no autocorrelation, no structure)
        null_rows = []
        for _ in range(len(ken_rows)):
            row = {field: rng.gauss(0, 1) for field in ALL_FIELDS}
            null_rows.append(row)
        e_null = _make_engine(); _train_engine(e_null, null_rows)
        sn = _engine_stats(e_null)
        b3_records.append({
            "trial": trial,
            "n_hypotheses_created": sn["n_total"],
            "n_active_false_positives": sn["n_active"],
            "false_positive_rate": round(sn["n_active"] / max(1, sn["n_total"]), 3),
            "avg_conf_null": sn["avg_conf"],
        })
    _write_csv(OUT_DIR / "stress_B3_null_world.csv", b3_records)
    avg_fpr = sum(r["false_positive_rate"] for r in b3_records) / len(b3_records)
    avg_null_conf = sum(r["avg_conf_null"] for r in b3_records) / len(b3_records)
    summary.append(
        f"B3 Null world: false positive rate={avg_fpr:.3f}, avg_conf={avg_null_conf:.3f} "
        f"(should be near 0 if well-calibrated)"
    )

    # ---- B4: Shock falsification -------------------------------------------
    logger.info("  B4: Shock falsification — shocks on disconnected variables")
    try:
        from scarcity.engine.economic_engine import EconomicDiscoveryEngine
        from scarcity.engine.simulation import PolicySimulator

        # Train economic engine
        eco_engine = EconomicDiscoveryEngine()
        eco_engine.core.meta_controller.conf_thresh  = 0.25
        eco_engine.core.meta_controller.stab_thresh  = 0.25
        eco_engine.core.meta_controller.min_evidence = 5
        eco_engine.core.explore_interval = 2

        for row in ken_rows:
            try:
                eco_engine.process_row_raw(row)
            except Exception:
                pass

        initial_state = ken_rows[-1].copy() if ken_rows else {}
        b4_records = []

        if initial_state and hasattr(eco_engine, "get_simulation_handle"):
            sim = eco_engine.get_simulation_handle()
            if hasattr(sim, "step") and hasattr(sim, "set_initial_state"):
                # Real shock (S3 inflation — known to propagate)
                sim.set_initial_state(initial_state)
                sim.perturb("inflation_cpi", initial_state.get("inflation_cpi", 7.0) + 5.0)
                real_traj = []
                for _ in range(3):
                    s = sim.step()
                    if isinstance(s, dict): real_traj.append(s)

                # Falsified shock: perturb 'life_expectancy' by +10 (no causal path to economic vars)
                sim2 = eco_engine.get_simulation_handle()
                sim2.set_initial_state(initial_state)
                sim2.perturb("life_expectancy", initial_state.get("life_expectancy", 60.0) + 10.0)
                false_traj = []
                for _ in range(3):
                    s = sim2.step()
                    if isinstance(s, dict): false_traj.append(s)

                real_propagation  = len(real_traj)
                false_propagation = len(false_traj)
                b4_records.append({
                    "shock": "real_inflation+5pp",
                    "steps_propagated": real_propagation,
                    "expected": "propagation",
                })
                b4_records.append({
                    "shock": "falsified_life_expectancy+10",
                    "steps_propagated": false_propagation,
                    "expected": "no_propagation",
                })
                summary.append(
                    f"B4 Shock falsification: real shock propagated {real_propagation} steps, "
                    f"falsified shock propagated {false_propagation} steps"
                )
    except Exception as exc:
        logger.warning(f"  B4 skipped: {exc}")
        b4_records = [{"shock": "skipped", "steps_propagated": 0, "expected": "n/a", "error": str(exc)}]
        summary.append("B4 Shock falsification: skipped (EconomicDiscoveryEngine unavailable)")

    _write_csv(OUT_DIR / "stress_B4_shock_falsification.csv", b4_records)

    return summary


# ===========================================================================
# C. FAILURE MODES
# ===========================================================================

def section_C_failure_modes(rows_by_country: Dict[str, List[Dict]]) -> List[str]:
    logger.info("=== SECTION C: FAILURE MODES ===")
    summary = []
    ken_rows = list(rows_by_country["KEN"])

    # ---- C1: Early overconfidence ------------------------------------------
    logger.info("  C1: Early overconfidence — confidence trajectory over time")
    c1_records = []
    engine = _make_engine()
    for step_idx, row in enumerate(ken_rows):
        try:
            engine.process_row(row)
        except Exception:
            pass
        if step_idx >= 2:  # skip first 2 (cold start)
            st = _engine_stats(engine)
            c1_records.append({
                "step": step_idx + 1,
                "n_years_seen": step_idx + 1,
                "avg_conf": st["avg_conf"],
                "n_active": st["n_active"],
                "n_total":  st["n_total"],
                "above_gate": int(st["avg_conf"] >= CONF_GATE),
            })
    _write_csv(OUT_DIR / "failure_C1_early_overconfidence.csv", c1_records)

    if c1_records:
        early5  = [r["avg_conf"] for r in c1_records[:5]]
        late5   = [r["avg_conf"] for r in c1_records[-5:]]
        avg_early = sum(early5) / len(early5) if early5 else 0
        avg_late  = sum(late5)  / len(late5)  if late5  else 0
        summary.append(
            f"C1 Early overconfidence: conf@steps3-7={avg_early:.3f} vs conf@final5={avg_late:.3f} "
            f"({'overconfident early' if avg_early > avg_late + 0.02 else 'monotone increasing' if avg_late > avg_early else 'stable'})"
        )

    # ---- C2: Conflict oscillation ------------------------------------------
    logger.info("  C2: Conflict oscillation — hypotheses cycling ACTIVE↔DECAYING")
    c2_records = []
    engine2 = _make_engine(conf_thresh=0.25, stab_thresh=0.25, min_evidence=3)
    state_history: Dict[str, List[str]] = defaultdict(list)

    for row in ken_rows:
        try:
            engine2.process_row(row)
        except Exception:
            pass
        for hid, hyp in engine2.hypotheses.population.items():
            state_history[hid].append(_state_str(hyp))

    # Count oscillations per hypothesis
    for hid, states in state_history.items():
        n_oscillations = 0
        for i in range(1, len(states)):
            if (states[i] != states[i-1] and
                    states[i] in ("active", "decaying") and
                    states[i-1] in ("active", "decaying")):
                n_oscillations += 1
        hyp = engine2.hypotheses.population.get(hid)
        if hyp is None:
            continue
        c2_records.append({
            "hypothesis_id": hid[:8],
            "rel_type": str(getattr(hyp, "rel_type", "?")).split(".")[-1],
            "final_state": states[-1] if states else "?",
            "n_state_changes": n_oscillations,
            "final_conf": round(float(getattr(hyp, "confidence", 0)), 4),
            "evidence": int(getattr(hyp, "evidence", 0)),
        })
    _write_csv(OUT_DIR / "failure_C2_oscillation.csv", c2_records)

    if c2_records:
        n_oscillating = sum(1 for r in c2_records if r["n_state_changes"] > 1)
        summary.append(
            f"C2 Conflict oscillation: {n_oscillating}/{len(c2_records)} hypotheses "
            f"oscillated ACTIVE↔DECAYING (>{1} state change)"
        )

    # ---- C3: Structural break -----------------------------------------------
    logger.info("  C3: Structural break — mid-stream regime change")
    half = len(ken_rows) // 2
    normal_rows = ken_rows[:half]
    rng = random.Random(7)

    # Create a regime-change block: 5 rows with very different distribution
    break_rows = []
    for i in range(5):
        shock_row = {k: v * rng.uniform(3.0, 5.0) if rng.random() > 0.5 else v * rng.uniform(0.1, 0.3)
                     for k, v in ken_rows[half].items()}
        break_rows.append(shock_row)
    post_break_rows = ken_rows[half:]

    c3_records = []
    engine3 = _make_engine()
    for step_idx, row in enumerate(normal_rows + break_rows + post_break_rows):
        phase = ("normal" if step_idx < half
                 else "structural_break" if step_idx < half + 5
                 else "post_break")
        try:
            engine3.process_row(row)
        except Exception:
            pass
        if step_idx % 3 == 0 or phase == "structural_break":
            st = _engine_stats(engine3)
            c3_records.append({
                "step": step_idx + 1, "phase": phase,
                "avg_conf": st["avg_conf"], "n_active": st["n_active"],
                "n_dead_cumulative": st["n_dead"],
            })
    _write_csv(OUT_DIR / "failure_C3_structural_break.csv", c3_records)

    pre_break  = [r["avg_conf"] for r in c3_records if r["phase"] == "normal"]
    during     = [r["avg_conf"] for r in c3_records if r["phase"] == "structural_break"]
    post_break = [r["avg_conf"] for r in c3_records if r["phase"] == "post_break"]
    avg_pre  = sum(pre_break)  / max(1, len(pre_break))
    avg_dur  = sum(during)     / max(1, len(during))
    avg_post = sum(post_break) / max(1, len(post_break))
    summary.append(
        f"C3 Structural break: pre={avg_pre:.3f} | during_break={avg_dur:.3f} | "
        f"post={avg_post:.3f} — conf {'collapses' if avg_dur < avg_pre * 0.8 else 'resilient'}"
    )

    return summary


# ===========================================================================
# D. CALIBRATION
# ===========================================================================

def section_D_calibration(rows_by_country: Dict[str, List[Dict]]) -> List[str]:
    logger.info("=== SECTION D: CALIBRATION ===")
    summary = []
    ken_rows = list(rows_by_country["KEN"])

    # Collect (confidence_bin, fit_score) per hypothesis per step
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
            (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
    bin_fit_scores: Dict[Tuple, List[float]] = {b: [] for b in bins}
    bin_survival:   Dict[Tuple, List[int]]   = {b: [] for b in bins}  # survived to next step?

    engine = _make_engine()
    prev_active_ids: set = set()

    for step_idx, row in enumerate(ken_rows):
        # Record pre-step confidence for each hypothesis
        pre_conf: Dict[str, float] = {
            hid: float(getattr(h, "confidence", 0))
            for hid, h in engine.hypotheses.population.items()
        }
        try:
            engine.process_row(row)
        except Exception:
            pass

        post_active = {
            hid for hid, h in engine.hypotheses.population.items()
            if _state_str(h) in ("active", "tentative")
        }

        # For each hypothesis that existed before this step
        for hid, conf in pre_conf.items():
            hyp = engine.hypotheses.population.get(hid)
            if hyp is None:
                continue
            fit_s = float(getattr(hyp, "fit_score", 0.5))
            survived = int(hid in post_active)

            # Bin
            for lo, hi in bins:
                if lo <= conf < hi:
                    bin_fit_scores[(lo, hi)].append(fit_s)
                    bin_survival[(lo, hi)].append(survived)
                    break

        prev_active_ids = post_active

    d1_records = []
    for (lo, hi) in bins:
        fits     = bin_fit_scores[(lo, hi)]
        survivals = bin_survival[(lo, hi)]
        d1_records.append({
            "conf_bin_lo": lo, "conf_bin_hi": hi,
            "n_samples": len(fits),
            "mean_fit_score": round(sum(fits) / len(fits), 4) if fits else float("nan"),
            "survival_rate": round(sum(survivals) / len(survivals), 4) if survivals else float("nan"),
            "calibration_gap": round(
                abs((lo + hi) / 2 - (sum(survivals) / max(1, len(survivals)))), 4
            ) if survivals else float("nan"),
        })
    _write_csv(OUT_DIR / "calibration_D1_reliability.csv", d1_records)

    well_cal = [r for r in d1_records if not math.isnan(r.get("calibration_gap", float("nan")))
                and r["calibration_gap"] < 0.10]
    summary.append(
        f"D1 Calibration: {len(well_cal)}/{len(d1_records)} confidence bins well-calibrated "
        f"(|conf_midpoint - survival_rate| < 0.10)"
    )

    # Brier score analog: mean squared error between confidence and survival outcome
    all_confs, all_outcomes = [], []
    for (lo, hi), survivals in bin_survival.items():
        mid_conf = (lo + hi) / 2
        all_confs.extend([mid_conf] * len(survivals))
        all_outcomes.extend(survivals)
    if all_confs:
        brier = sum((c - o)**2 for c, o in zip(all_confs, all_outcomes)) / len(all_confs)
        summary.append(f"D1 Brier score analog = {brier:.4f} (0=perfect, 0.25=random)")

    return summary


# ===========================================================================
# E. HYPOTHESIS LIFECYCLE
# ===========================================================================

def section_E_lifecycle(rows_by_country: Dict[str, List[Dict]]) -> List[str]:
    logger.info("=== SECTION E: HYPOTHESIS LIFECYCLE ===")
    summary = []
    ken_rows = list(rows_by_country["KEN"])

    engine = _make_engine()
    creation_times: Dict[str, int] = {}
    death_times:    Dict[str, int] = {}
    state_snapshots: Dict[str, List[str]] = defaultdict(list)

    for step_idx, row in enumerate(ken_rows):
        prev_ids = set(engine.hypotheses.population.keys())
        try:
            engine.process_row(row)
        except Exception:
            pass
        curr_ids = set(engine.hypotheses.population.keys())

        for new_id in curr_ids - prev_ids:
            creation_times[new_id] = step_idx
        for dead_id in prev_ids - curr_ids:
            death_times[dead_id] = step_idx

        for hid, hyp in engine.hypotheses.population.items():
            state_snapshots[hid].append(_state_str(hyp))

    # Also track graveyard
    for dead_hyp in getattr(engine.hypotheses, "graveyard", []):
        if isinstance(dead_hyp, tuple) and len(dead_hyp) >= 1:
            hid = getattr(dead_hyp[0], "meta", None)
            hid = getattr(hid, "id", None) if hid else None
            if hid and hid not in death_times:
                death_times[hid] = len(ken_rows)
        elif hasattr(dead_hyp, "meta"):
            hid = dead_hyp.meta.id
            if hid not in death_times:
                death_times[hid] = len(ken_rows)

    e1_records = []
    all_ids = set(creation_times) | set(state_snapshots)
    for hid in all_ids:
        birth = creation_times.get(hid, 0)
        death = death_times.get(hid, len(ken_rows))
        lifetime = death - birth

        states = state_snapshots.get(hid, [])
        n_transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        final_state = states[-1] if states else "unknown"

        hyp = engine.hypotheses.population.get(hid)
        conf  = round(float(getattr(hyp, "confidence", 0)), 4) if hyp else 0.0
        evid  = int(getattr(hyp, "evidence", 0)) if hyp else 0
        rtype = str(getattr(hyp, "rel_type", "?")).split(".")[-1] if hyp else "pruned"

        e1_records.append({
            "hypothesis_id": hid[:8],
            "rel_type": rtype,
            "birth_step": birth,
            "death_step": death,
            "lifetime_steps": lifetime,
            "n_state_transitions": n_transitions,
            "final_state": final_state,
            "final_conf": conf,
            "evidence": evid,
        })
    _write_csv(OUT_DIR / "lifecycle_E1_distribution.csv", e1_records)

    if e1_records:
        lifetimes = [r["lifetime_steps"] for r in e1_records]
        avg_lt = sum(lifetimes) / len(lifetimes)
        max_lt = max(lifetimes)
        long_lived = sum(1 for lt in lifetimes if lt == max(lifetimes))
        pruned = sum(1 for r in e1_records if r["final_state"] in ("dead", "unknown"))
        summary.append(
            f"E1 Lifecycle: {len(e1_records)} hypotheses created, {pruned} pruned, "
            f"avg lifetime={avg_lt:.1f} steps, max={max_lt} steps"
        )

    return summary


# ===========================================================================
# F. DRG PERFORMANCE (timing & throughput)
# ===========================================================================

def section_F_drg_performance(rows_by_country: Dict[str, List[Dict]]) -> List[str]:
    logger.info("=== SECTION F: DRG PERFORMANCE ===")
    summary = []
    ken_rows = list(rows_by_country["KEN"])

    # Extend to a larger synthetic stream for better throughput measurement
    rng = random.Random(99)
    stream_sizes = [10, 34, 100, 500]
    f1_records = []

    for n_obs in stream_sizes:
        # Generate stream of n_obs rows
        if n_obs <= len(ken_rows):
            stream = ken_rows[:n_obs]
        else:
            base = ken_rows[-1] if ken_rows else {f: 0.0 for f in ALL_FIELDS}
            stream = list(ken_rows)
            while len(stream) < n_obs:
                row = {k: v + rng.gauss(0, abs(v) * 0.03) for k, v in base.items()}
                stream.append(row)
                base = row

        engine = _make_engine()

        # Measure memory before
        tracemalloc.start()
        gc.collect()
        snap1 = tracemalloc.take_snapshot()

        latencies_ms = []
        t_total_start = time.perf_counter()

        for row in stream:
            t0 = time.perf_counter()
            try:
                engine.process_row(row)
            except Exception:
                pass
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        t_total = time.perf_counter() - t_total_start

        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snap2.compare_to(snap1, "lineno")
        mem_delta_kb = sum(stat.size_diff for stat in top_stats) / 1024

        throughput = n_obs / t_total if t_total > 0 else float("inf")
        f1_records.append({
            "n_obs": n_obs,
            "total_time_s": round(t_total, 4),
            "throughput_obs_per_s": round(throughput, 1),
            "latency_mean_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
            "latency_p50_ms":  round(sorted(latencies_ms)[len(latencies_ms)//2], 3),
            "latency_p95_ms":  round(sorted(latencies_ms)[int(len(latencies_ms)*0.95)], 3),
            "latency_max_ms":  round(max(latencies_ms), 3),
            "mem_delta_kb":    round(mem_delta_kb, 1),
            "n_hypotheses_final": len(engine.hypotheses.population),
        })
        logger.info(
            f"  n={n_obs}: {throughput:.0f} obs/s, "
            f"p95={f1_records[-1]['latency_p95_ms']:.1f}ms, "
            f"mem_delta={mem_delta_kb:.0f}KB"
        )

    _write_csv(OUT_DIR / "drg_perf_F1.csv", f1_records)

    # Summarize
    for r in f1_records:
        summary.append(
            f"F1 n={r['n_obs']:4d}: {r['throughput_obs_per_s']:.0f} obs/s | "
            f"p95={r['latency_p95_ms']:.1f}ms | mem_delta={r['mem_delta_kb']:.0f}KB"
        )

    return summary


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sections", nargs="*",
        choices=["A", "B", "C", "D", "E", "F"],
        default=["A", "B", "C", "D", "E", "F"],
        help="Which sections to run (default: all)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading mock country data ...")
    rows_by_country = _load_mock_rows(seed=42)
    for code, rows in rows_by_country.items():
        logger.info(f"  {code}: {len(rows)} year-rows")

    all_summary: List[str] = [
        "=" * 68,
        "SCARCITY — COMPREHENSIVE BENCHMARK RESULTS",
        "Sections: Ablations / Stress Tests / Failure Modes / Calibration / Lifecycle / DRG",
        "=" * 68,
    ]

    section_fns = {
        "A": section_A_ablations,
        "B": section_B_stress,
        "C": section_C_failure_modes,
        "D": section_D_calibration,
        "E": section_E_lifecycle,
        "F": section_F_drg_performance,
    }

    for sec in args.sections:
        try:
            lines = section_fns[sec](rows_by_country)
            all_summary.append(f"\n--- Section {sec} ---")
            all_summary.extend(lines)
        except Exception as exc:
            logger.error(f"Section {sec} failed: {exc}")
            all_summary.append(f"Section {sec}: ERROR — {exc}")

    summary_text = "\n".join(all_summary)
    summary_path = OUT_DIR / "comprehensive_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        print("\n" + summary_text)
    except UnicodeEncodeError:
        print("\n" + summary_text.encode("ascii", "replace").decode("ascii"))

    logger.info("Comprehensive benchmark complete.")


if __name__ == "__main__":
    main()

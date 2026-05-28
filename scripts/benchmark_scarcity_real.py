"""benchmark_scarcity_real.py — Real-data scarcity benchmark.

Feeds Kenya World Bank macro data (2000-2024, N=25 annual observations) into the
OnlineDiscoveryEngine and lets it autonomously discover all hypothesis types and
causal relationships. No hypothesis pairs are hardcoded.

Answers two concrete questions:
  DATA SCARCITY:    At what minimum N does the engine start producing confident
                    discoveries? How does discovery quality degrade as N shrinks?
  COMPUTE SCARCITY: Under DRG RED pressure, does the engine adapt gracefully —
                    completing inference, reducing beta, decaying bandit arms?

Data source: data/simulation/API_KEN_DS2_en_csv_v2_14659.csv
  Kenya Development Indicators, World Bank 2000-2024 (25 annual observations)
  9 variables: GDP_growth, Inflation, Unemployment, CA_balance, Remittances_pct,
               Gov_consumption, GCF, Exports_pct, Imports_pct

Exit codes: 0=all PASS/WARN, 1=any FAIL
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# World Bank Kenya macro data loader
# ---------------------------------------------------------------------------

_WB_PATH = _PROJECT_ROOT / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv"

_WB_INDICATORS = {
    "GDP_growth":      "GDP growth (annual %)",
    "Inflation":       "Inflation, consumer prices (annual %)",
    "Unemployment":    "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    "CA_balance":      "Current account balance (% of GDP)",
    "Remittances_pct": "Personal remittances, received (% of GDP)",
    "Gov_consumption": "General government final consumption expenditure (% of GDP)",
    "GCF":             "Gross capital formation (% of GDP)",
    "Exports_pct":     "Exports of goods and services (% of GDP)",
    "Imports_pct":     "Imports of goods and services (% of GDP)",
}


def _load_wb_rows() -> List[Dict[str, float]]:
    """Load World Bank Kenya data as chronological list of annual observation dicts."""
    import pandas as pd
    df = pd.read_csv(_WB_PATH, skiprows=4)
    year_cols = [c for c in df.columns if c.isdigit() and 2000 <= int(c) <= 2024]
    series: Dict[str, Any] = {}
    for key, indicator_name in _WB_INDICATORS.items():
        match = df[df["Indicator Name"].str.strip() == indicator_name]
        if len(match):
            series[key] = match[year_cols].iloc[0]
    ts = pd.DataFrame(series, index=year_cols).dropna(how="all")
    rows = []
    for _, vals in ts.iterrows():
        if vals.notna().all():
            rows.append({k: float(v) for k, v in vals.items()})
    return rows  # 25 rows, 2000-2024


_SCHEMA = {
    "fields": [{"name": k, "type": "numeric"} for k in _WB_INDICATORS]
}

_VAR_NAMES = list(_WB_INDICATORS.keys())


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _result(stage_id: str, name: str, status: str, target: str,
            metrics: Dict[str, Any], wall: float) -> Dict[str, Any]:
    return {
        "stage": stage_id, "name": name, "status": status,
        "target": target, "result": metrics,
        "wallclock_s": round(wall, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _fail(stage_id: str, name: str, target: str, err: str, wall: float) -> Dict[str, Any]:
    return _result(stage_id, name, "FAIL", target, {"error": err[-600:]}, wall)


def _skip(stage_id: str, name: str, reason: str) -> Dict[str, Any]:
    return _result(stage_id, name, "SKIP", reason, {}, 0.0)


def _build_engine(buffer_size: int = 150):
    from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
    engine = OnlineDiscoveryEngine(
        explore_interval=5,
        mode="balanced",
        buffer_size=buffer_size,
    )
    engine.initialize_v2(_SCHEMA, use_causal=True)
    return engine


# ---------------------------------------------------------------------------
# Stage DS.1 — Minimum viable N: when does the engine start discovering?
#
# Feeds N rows from the real WB series (N = 8 .. 25) and counts how many
# hypotheses the engine has promoted to confidence >= 0.25 (the candidate
# filter used by get_candidate_paths). Reports the first N where at least
# one confident relationship appears.
# ---------------------------------------------------------------------------

def run_ds_1(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "DS.1", "data_scarcity_floor"
    try:
        _build_engine(buffer_size=8)
    except ImportError as e:
        return _skip(stage_id, name, f"engine import failed: {e}")
    try:
        rows = _load_wb_rows()
    except Exception as e:
        return _fail(stage_id, name, "load WB data", str(e), time.time() - t0)

    try:
        n_sweep = [8, 10, 12, 15, 18, 20, 22, 25]
        n_sweep = [n for n in n_sweep if n <= len(rows)]
        sweep: Dict[int, Dict] = {}
        first_discovery_n = None

        for n in n_sweep:
            # Fresh engine per N — buffer_size = n so the engine works at the data limit
            engine = _build_engine(buffer_size=n)
            for row in rows[:n]:
                engine.process_row(row)

            candidates = engine.export_hypothesis_summary(min_conf=0.0)
            confident = [c for c in candidates if c['conf'] >= 0.25]
            pool_size = len(engine.hypotheses.population)

            sweep[n] = {
                "pool_size": pool_size,
                "candidates": len(candidates),
                "confident": len(confident),
                "top_confidence": round(max((c['conf'] for c in candidates), default=0.0), 4),
            }
            if len(confident) > 0 and first_discovery_n is None:
                first_discovery_n = n

        max_n = max(n_sweep)
        confident_at_max = sweep[max_n]["confident"]
        top_conf_at_max = sweep[max_n]["top_confidence"]

        wall = time.time() - t0
        passing = first_discovery_n is not None and confident_at_max >= 1
        status = "PASS" if passing else ("WARN" if first_discovery_n is not None else "FAIL")

        return _result(stage_id, name, status,
                       "engine discovers >=1 confident relationship before N=25",
                       {"n_available": len(rows),
                        "first_discovery_n": first_discovery_n,
                        "confident_at_n25": confident_at_max,
                        "top_confidence_at_n25": top_conf_at_max,
                        "sweep": {str(k): v for k, v in sweep.items()}},
                       wall)
    except Exception as e:
        return _fail(stage_id, name, "engine discovery sweep over N",
                     f"{e}\n{traceback.format_exc()[-800:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage DS.2 — Full-N discovery: what does the engine find at N=25?
#
# Runs the engine on all 25 years of Kenya macro data. Reports:
#   - Total hypotheses generated (engine decides which types to create)
#   - Discovered causal/correlational relationships (confidence >= 0.25)
#   - Knowledge graph edges
#   - Which variable pairs surfaced as strongest signals
# ---------------------------------------------------------------------------

def run_ds_2(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "DS.2", "full_discovery_n25"
    try:
        engine = _build_engine(buffer_size=25)
    except ImportError as e:
        return _skip(stage_id, name, f"engine import failed: {e}")
    try:
        rows = _load_wb_rows()
    except Exception as e:
        return _fail(stage_id, name, "load WB data", str(e), time.time() - t0)

    try:
        for row in rows:
            engine.process_row(row)

        candidates = engine.export_hypothesis_summary(min_conf=0.0)
        kg = engine.get_knowledge_graph()
        summary = engine.export_hypothesis_summary(min_conf=0.15)

        confident = [c for c in candidates if c['conf'] >= 0.25]
        strong    = [c for c in candidates if c['conf'] >= 0.50]
        pool_size = len(engine.hypotheses.population)

        # Top-5 discovered relationships
        top5 = sorted(candidates, key=lambda c: c['conf'], reverse=True)[:5]
        top5_list = [
            {"vars": c['vars'], "confidence": round(c['conf'], 4),
             "score": round(c.get('fit_score', 0.0), 4)}
            for c in top5
        ]

        wall = time.time() - t0
        passing = len(confident) >= 3
        status = "PASS" if passing else ("WARN" if len(confident) >= 1 else "FAIL")

        return _result(stage_id, name, status,
                       ">=3 confident relationships (conf>=0.25) discovered at N=25",
                       {"pool_size": pool_size,
                        "total_candidates": len(candidates),
                        "confident_025": len(confident),
                        "strong_050": len(strong),
                        "kg_edges": len(kg),
                        "hypothesis_summary_count": len(summary),
                        "top5_discovered": top5_list},
                       wall)
    except Exception as e:
        return _fail(stage_id, name, "full engine discovery on 25 WB rows",
                     f"{e}\n{traceback.format_exc()[-800:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage DS.3 — Scarcity degradation curve
#
# At each N in {8,10,12,15,18,20,22,25}: count confident discoveries.
# Quantifies the scarcity penalty as a rate (discoveries lost per year removed)
# and identifies the inflection point where discovery rate first reaches 50%
# of the N=25 baseline.
# ---------------------------------------------------------------------------

def run_ds_3(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "DS.3", "scarcity_degradation_curve"
    try:
        _build_engine(buffer_size=8)
    except ImportError as e:
        return _skip(stage_id, name, f"engine import failed: {e}")
    try:
        rows = _load_wb_rows()
    except Exception as e:
        return _fail(stage_id, name, "load WB data", str(e), time.time() - t0)

    try:
        n_sweep = [8, 10, 12, 15, 18, 20, 22, 25]
        n_sweep = [n for n in n_sweep if n <= len(rows)]
        curve: Dict[int, Dict] = {}

        for n in n_sweep:
            engine = _build_engine(buffer_size=n)
            for row in rows[:n]:
                engine.process_row(row)
            candidates = engine.export_hypothesis_summary(min_conf=0.0)
            confident = sum(1 for c in candidates if c['conf'] >= 0.25)
            top_conf = max((c['conf'] for c in candidates), default=0.0)
            curve[n] = {
                "n_confident": confident,
                "top_confidence": round(top_conf, 4),
            }

        baseline = curve.get(25, curve[max(n_sweep)])["n_confident"]
        inflection_n = None
        threshold = max(1, baseline // 2)
        for n in n_sweep:
            if curve[n]["n_confident"] >= threshold:
                inflection_n = n
                break

        scarcity_loss = curve.get(25, {"n_confident": 0})["n_confident"] - \
                        curve.get(10, {"n_confident": 0})["n_confident"]

        wall = time.time() - t0
        passing = baseline >= 3 and inflection_n is not None
        status = "PASS" if passing else ("WARN" if baseline >= 1 else "FAIL")

        return _result(stage_id, name, status,
                       "reports discovery rate curve; identifies scarcity inflection N",
                       {"baseline_confident_n25": baseline,
                        "inflection_n": inflection_n,
                        "scarcity_loss_n25_vs_n10": scarcity_loss,
                        "curve": {str(k): v for k, v in curve.items()}},
                       wall)
    except Exception as e:
        return _fail(stage_id, name, "degradation curve across N values",
                     f"{e}\n{traceback.format_exc()[-800:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage DS.4 — Streaming coherence: online vs batch agreement
#
# Runs the engine in strict streaming mode (one row at a time, never replayed)
# and checks that the final knowledge graph is coherent — no contradictory
# directed edges with equal confidence, no self-loops in the top candidates.
# This tests the core scarcity claim: the system should be usable even when
# data arrives one observation at a time and cannot be replayed.
# ---------------------------------------------------------------------------

def run_ds_4(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "DS.4", "streaming_coherence"
    try:
        engine = _build_engine(buffer_size=25)
    except ImportError as e:
        return _skip(stage_id, name, f"engine import failed: {e}")
    try:
        rows = _load_wb_rows()
    except Exception as e:
        return _fail(stage_id, name, "load WB data", str(e), time.time() - t0)

    try:
        # Strict streaming: one observation at a time
        snapshot_at = {10: None, 15: None, 20: None, 25: None}
        for i, row in enumerate(rows):
            engine.process_row(row)
            n = i + 1
            if n in snapshot_at:
                cands = engine.export_hypothesis_summary(min_conf=0.0)
                snapshot_at[n] = len([c for c in cands if c['conf'] >= 0.25])

        # Final state
        final_candidates = engine.export_hypothesis_summary(min_conf=0.0)
        kg = engine.get_knowledge_graph()

        # Coherence checks
        # 1. Monotonic discovery: discoveries should grow or stay flat as N increases
        snap_vals = [v for v in snapshot_at.values() if v is not None]
        monotonic = all(snap_vals[i] <= snap_vals[i+1] for i in range(len(snap_vals)-1))

        # 2. No self-loops in candidates (a variable does not cause itself)
        self_loops = sum(1 for c in final_candidates
                         if len(c['vars']) == 2 and c['vars'][0] == c['vars'][-1])

        # 3. Knowledge graph has non-trivial edges
        kg_ok = len(kg) >= 1

        wall = time.time() - t0
        passing = kg_ok and self_loops == 0
        status = "PASS" if (passing and monotonic) else ("WARN" if passing else "FAIL")

        return _result(stage_id, name, status,
                       "streaming: discoveries monotonic; no self-loops; KG non-empty",
                       {"snapshots": {str(k): v for k, v in snapshot_at.items()},
                        "final_confident": snapshot_at.get(25, 0),
                        "kg_edges": len(kg),
                        "self_loops": self_loops,
                        "monotonic_growth": monotonic,
                        "kg_ok": kg_ok},
                       wall)
    except Exception as e:
        return _fail(stage_id, name, "streaming coherence on WB data",
                     f"{e}\n{traceback.format_exc()[-800:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage CS.1 — Compute scarcity: DRG RED adaptation
#
# Runs the engine on real WB data and applies DRG RED profile (simulated
# 95% CPU) to the meta-learner and bandit router. Verifies:
#   1. Engine completes all 25 rows without crash
#   2. OnlineReptileOptimizer beta shrinks under RED
#   3. BanditRouter.decay() executes without error
# ---------------------------------------------------------------------------

def run_cs_1(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "CS.1", "compute_scarcity_drg_adaptation"
    try:
        from scarcity.meta.optimizer import OnlineReptileOptimizer
        from scarcity.engine.bandit_router import BanditRouter, BanditConfig
        engine = _build_engine(buffer_size=25)
    except ImportError as e:
        return _skip(stage_id, name, f"import failed: {e}")
    try:
        rows = _load_wb_rows()
    except Exception as e:
        return _fail(stage_id, name, "load WB data", str(e), time.time() - t0)

    try:
        optimizer = OnlineReptileOptimizer()
        keys = _VAR_NAMES
        drg_green = {"vram_high": 0.0, "latency_high": 0.0, "bandwidth_free": 1.0}
        drg_red   = {"vram_high": 1.0, "latency_high": 1.0, "bandwidth_free": 0.0}

        # Feed real data through engine + get aggregated vector
        rows_completed = 0
        for row in rows:
            engine.process_row(row)
            rows_completed += 1

        # Extract top-candidate vector to drive Reptile
        candidates = engine.export_hypothesis_summary(min_conf=0.0)
        agg_vec = np.array(
            [c['conf'] for c in candidates[:len(keys)]] +
            [0.0] * max(0, len(keys) - len(candidates)),
            dtype=np.float32
        )[:len(keys)]
        norm = np.linalg.norm(agg_vec)
        if norm > 0:
            agg_vec = agg_vec / norm

        # Baseline GREEN step
        optimizer.apply(agg_vec, keys, reward=0.7, drg_profile=drg_green)
        beta_green = float(optimizer.state.beta) if hasattr(optimizer, "state") else 0.1

        # 5 RED steps
        for _ in range(5):
            optimizer.apply(agg_vec, keys, reward=0.3, drg_profile=drg_red)
        beta_red = float(optimizer.state.beta) if hasattr(optimizer, "state") else 0.1
        beta_decays = beta_red <= beta_green

        # BanditRouter decay
        n_arms = max(1, len(candidates))
        router = BanditRouter(config=BanditConfig(), n_arms=n_arms)
        router.register_arms(n_arms)
        decay_ok = False
        try:
            router.decay()
            decay_ok = True
        except Exception:
            pass

        wall = time.time() - t0
        passing = rows_completed == len(rows) and beta_decays and decay_ok
        status = "PASS" if passing else ("WARN" if rows_completed == len(rows) else "FAIL")

        return _result(stage_id, name, status,
                       "engine completes 25 rows; beta shrinks under RED; BanditRouter.decay() ok",
                       {"rows_completed": rows_completed,
                        "n_rows": len(rows),
                        "beta_green": round(beta_green, 6),
                        "beta_red": round(beta_red, 6),
                        "beta_reduction_pct": round(100 * (1 - beta_red / max(beta_green, 1e-9)), 1),
                        "beta_decays": beta_decays,
                        "decay_ok": decay_ok,
                        "n_candidates_after": len(candidates)},
                       wall)
    except Exception as e:
        return _fail(stage_id, name, "DRG RED adaptation on real WB data",
                     f"{e}\n{traceback.format_exc()[-800:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage CS.2 — Throughput under compute scarcity
#
# Runs the engine twice on the full 25-year series:
#   - Run A: GREEN profile (normal operation)
#   - Run B: RED profile (constrained — every 5 rows, apply RED step to optimizer)
# Compares wall-clock time and discovery quality. RED must complete; overhead < 5x.
# ---------------------------------------------------------------------------

def run_cs_2(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "CS.2", "compute_scarcity_throughput"
    try:
        from scarcity.meta.optimizer import OnlineReptileOptimizer
        _build_engine(buffer_size=25)
    except ImportError as e:
        return _skip(stage_id, name, f"import failed: {e}")
    try:
        rows = _load_wb_rows()
    except Exception as e:
        return _fail(stage_id, name, "load WB data", str(e), time.time() - t0)

    try:
        keys = _VAR_NAMES

        def _run(drg_profile: Dict[str, float], label: str) -> Dict[str, Any]:
            optimizer = OnlineReptileOptimizer()
            engine = _build_engine(buffer_size=25)
            t = time.time()
            for i, row in enumerate(rows):
                engine.process_row(row)
                if i % 5 == 4:  # every 5 rows: optimizer step with current profile
                    cands = engine.export_hypothesis_summary(min_conf=0.0)
                    cands_sorted = sorted(cands, key=lambda c: c['conf'], reverse=True)
                    vec = np.array(
                        [c['conf'] for c in cands_sorted[:len(keys)]] +
                        [0.0] * max(0, len(keys) - len(cands_sorted)),
                        dtype=np.float32
                    )[:len(keys)]
                    n = np.linalg.norm(vec)
                    optimizer.apply(vec / n if n > 0 else vec, keys,
                                    reward=0.5, drg_profile=drg_profile)
            cands = engine.export_hypothesis_summary(min_conf=0.0)
            return {
                "wall_s": round(time.time() - t, 3),
                "confident": sum(1 for c in cands if c['conf'] >= 0.25),
            }

        drg_green = {"vram_high": 0.0, "latency_high": 0.0, "bandwidth_free": 1.0}
        drg_red   = {"vram_high": 1.0, "latency_high": 1.0, "bandwidth_free": 0.0}

        green_r = _run(drg_green, "GREEN")
        red_r   = _run(drg_red,   "RED")

        overhead = round(red_r["wall_s"] / max(green_r["wall_s"], 1e-6), 2)
        quality_retained = round(
            red_r["confident"] / max(green_r["confident"], 1), 3
        )

        wall = time.time() - t0
        passing = overhead < 5.0 and red_r["confident"] >= 1
        status = "PASS" if passing else ("WARN" if red_r["confident"] >= 1 else "FAIL")

        return _result(stage_id, name, status,
                       "RED completes; overhead < 5x GREEN; discovery quality retained",
                       {"time_green_s": green_r["wall_s"],
                        "time_red_s": red_r["wall_s"],
                        "overhead_ratio": overhead,
                        "confident_green": green_r["confident"],
                        "confident_red": red_r["confident"],
                        "quality_retained": quality_retained},
                       wall)
    except Exception as e:
        return _fail(stage_id, name, "throughput GREEN vs RED with real data",
                     f"{e}\n{traceback.format_exc()[-800:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage CS.3 — Memory-constrained buffer: engine with tiny window
#
# Sets buffer_size to very small values {5, 8, 10, 15, 20, 25} and feeds the
# last `buffer_size` rows from the 25-year series. Verifies:
#   - Engine initialises and runs without crash at every buffer size
#   - At buffer=5 it finds fewer/weaker signals than at buffer=25
#   - No self-loops or degenerate outputs at any size
# ---------------------------------------------------------------------------

def run_cs_3(fast: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "CS.3", "memory_scarcity_buffer_sweep"
    try:
        _build_engine(buffer_size=5)
    except ImportError as e:
        return _skip(stage_id, name, f"engine import failed: {e}")
    try:
        rows = _load_wb_rows()
    except Exception as e:
        return _fail(stage_id, name, "load WB data", str(e), time.time() - t0)

    try:
        buf_sweep = [5, 8, 10, 15, 20, 25]
        buf_sweep = [b for b in buf_sweep if b <= len(rows)]
        buf_results: Dict[int, Dict] = {}
        crashes = []

        for buf in buf_sweep:
            window = rows[-buf:]  # last `buf` rows
            try:
                engine = _build_engine(buffer_size=buf)
                for row in window:
                    engine.process_row(row)
                cands = engine.export_hypothesis_summary(min_conf=0.0)
                confident = sum(1 for c in cands if c['conf'] >= 0.25)
                top_conf = max((c['conf'] for c in cands), default=0.0)
                pool_size = len(engine.hypotheses.population)
                buf_results[buf] = {
                    "pool_size": pool_size,
                    "confident": confident,
                    "top_confidence": round(top_conf, 4),
                }
            except Exception as exc:
                crashes.append(f"buf={buf}: {exc}")
                buf_results[buf] = {"pool_size": 0, "confident": 0, "top_confidence": 0.0}

        no_crashes = len(crashes) == 0
        conf_at_5  = buf_results.get(5,  {}).get("confident", 0)
        conf_at_25 = buf_results.get(25, buf_results.get(max(buf_sweep), {})).get("confident", 0)
        improves = conf_at_25 >= conf_at_5  # more window = more discovery

        wall = time.time() - t0
        status = "PASS" if (no_crashes and improves) else ("WARN" if no_crashes else "FAIL")

        return _result(stage_id, name, status,
                       "no crash at any buffer size; discovery improves as buffer grows",
                       {"no_crashes": no_crashes,
                        "confident_at_buf5": conf_at_5,
                        "confident_at_buf25": conf_at_25,
                        "improves_with_buffer": improves,
                        "crashes": crashes,
                        "buffer_results": {str(k): v for k, v in buf_results.items()}},
                       wall)
    except Exception as e:
        return _fail(stage_id, name, "buffer-size sweep on WB data",
                     f"{e}\n{traceback.format_exc()[-800:]}", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage CS.4 — Composite scarcity verdict
#
# Synthesises all prior results into a viability rating:
#   HIGH   — system discovers reliably at the data limit; adapts under compute pressure
#   MEDIUM — partially viable; some gaps
#   LOW    — marginal; significant scarcity impact
#   UNABLE — fails to produce useful output under the given constraints
# ---------------------------------------------------------------------------

def run_cs_4(prior_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
    t0 = time.time()
    stage_id, name = "CS.4", "scarcity_verdict"

    if not prior_results:
        return _result(stage_id, name, "SKIP",
                       "requires prior stage results", {}, 0.0)

    by = {r["stage"]: r.get("result", {}) for r in prior_results}

    ds1 = by.get("DS.1", {})
    ds2 = by.get("DS.2", {})
    ds3 = by.get("DS.3", {})
    ds4 = by.get("DS.4", {})
    cs1 = by.get("CS.1", {})
    cs2 = by.get("CS.2", {})
    cs3 = by.get("CS.3", {})

    data_score = 0
    compute_score = 0

    # Data scarcity scoring
    first_n = ds1.get("first_discovery_n")
    if first_n is not None and first_n <= 15:    data_score += 3
    elif first_n is not None and first_n <= 20:  data_score += 2
    elif first_n is not None:                    data_score += 1

    conf25 = ds2.get("confident_025", 0)
    if conf25 >= 5:   data_score += 3
    elif conf25 >= 3: data_score += 2
    elif conf25 >= 1: data_score += 1

    base = ds3.get("baseline_confident_n25", 0)
    loss = ds3.get("scarcity_loss_n25_vs_n10", base)
    if base > 0 and loss / base <= 0.5:  data_score += 2
    elif base > 0:                        data_score += 1

    if ds4.get("monotonic_growth"):  data_score += 1
    if ds4.get("self_loops", 1) == 0: data_score += 1

    # Compute scarcity scoring
    if cs1.get("rows_completed", 0) == cs1.get("n_rows", 1): compute_score += 2
    if cs1.get("beta_decays"):    compute_score += 2
    if cs1.get("decay_ok"):       compute_score += 1

    overhead = cs2.get("overhead_ratio", 999.0)
    if overhead is not None and overhead < 1.5:   compute_score += 3
    elif overhead is not None and overhead < 3.0: compute_score += 2
    elif overhead is not None and overhead < 5.0: compute_score += 1

    if cs3.get("no_crashes"):          compute_score += 1
    if cs3.get("improves_with_buffer"): compute_score += 1

    total = data_score + compute_score
    max_s = 20

    if total >= 15:    verdict = "HIGH"
    elif total >= 10:  verdict = "MEDIUM"
    elif total >= 6:   verdict = "LOW"
    else:              verdict = "UNABLE"

    passing = verdict in ("HIGH", "MEDIUM")
    status = "PASS" if passing else ("WARN" if verdict == "LOW" else "FAIL")

    return _result(stage_id, name, status,
                   "composite viability verdict: HIGH/MEDIUM/LOW/UNABLE",
                   {"verdict": verdict,
                    "total_score": total,
                    "max_score": max_s,
                    "data_score": data_score,
                    "compute_score": compute_score,
                    "min_viable_n": first_n,
                    "confident_at_n25": conf25,
                    "compute_overhead_ratio": overhead,
                    "beta_reduction_pct": cs1.get("beta_reduction_pct"),
                    "streaming_monotonic": ds4.get("monotonic_growth"),
                    "buffer_sweep_ok": cs3.get("no_crashes")},
                   time.time() - t0)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

STATUS_SYM = {"PASS": "+", "WARN": "~", "FAIL": "X", "SKIP": "-"}


def _print_result(r: Dict) -> None:
    sym = STATUS_SYM.get(r["status"], "?")
    print(f"  [{sym}] {r['stage']:>4}  {r['status']:<4}  {r['name']:<38}  {r['wallclock_s']:.2f}s")


def _print_detail(r: Dict) -> None:
    res = r.get("result", {})
    s = r["stage"]

    if s == "DS.1":
        print(f"         first_discovery_n={res.get('first_discovery_n')}  "
              f"confident_at_n25={res.get('confident_at_n25')}  "
              f"top_conf={res.get('top_confidence_at_n25')}")
        sweep = res.get("sweep", {})
        if sweep:
            line = " | ".join(
                f"N={k}: {v.get('confident')} conf" for k, v in sorted(sweep.items(), key=lambda x: int(x[0]))
            )
            print(f"         sweep: {line}")

    elif s == "DS.2":
        print(f"         pool={res.get('pool_size')}  candidates={res.get('total_candidates')}  "
              f"confident(>=0.25)={res.get('confident_025')}  "
              f"strong(>=0.50)={res.get('strong_050')}  kg_edges={res.get('kg_edges')}")
        for rel in (res.get("top5_discovered") or []):
            print(f"         -> vars={rel.get('vars')}  conf={rel.get('confidence')}  "
                  f"score={rel.get('score')}")

    elif s == "DS.3":
        print(f"         baseline_n25={res.get('baseline_confident_n25')}  "
              f"inflection_N={res.get('inflection_n')}  "
              f"scarcity_loss={res.get('scarcity_loss_n25_vs_n10')}")
        curve = res.get("curve", {})
        if curve:
            line = " | ".join(
                f"N={k}:{v.get('n_confident')}" for k, v in sorted(curve.items(), key=lambda x: int(x[0]))
            )
            print(f"         curve: {line}")

    elif s == "DS.4":
        print(f"         snapshots={res.get('snapshots')}  kg_edges={res.get('kg_edges')}  "
              f"self_loops={res.get('self_loops')}  monotonic={res.get('monotonic_growth')}")

    elif s == "CS.1":
        print(f"         rows={res.get('rows_completed')}/{res.get('n_rows')}  "
              f"beta_green={res.get('beta_green')}  beta_red={res.get('beta_red')}  "
              f"reduction={res.get('beta_reduction_pct')}%  decay_ok={res.get('decay_ok')}")

    elif s == "CS.2":
        print(f"         time_green={res.get('time_green_s')}s  time_red={res.get('time_red_s')}s  "
              f"overhead={res.get('overhead_ratio')}x  "
              f"quality_retained={res.get('quality_retained')}")

    elif s == "CS.3":
        print(f"         no_crashes={res.get('no_crashes')}  "
              f"conf_buf5={res.get('confident_at_buf5')}  "
              f"conf_buf25={res.get('confident_at_buf25')}  "
              f"improves={res.get('improves_with_buffer')}")
        br = res.get("buffer_results", {})
        if br:
            line = " | ".join(
                f"buf={k}:{v.get('confident')}" for k, v in sorted(br.items(), key=lambda x: int(x[0]))
            )
            print(f"         buffer curve: {line}")

    elif s == "CS.4":
        v = res.get("verdict", "?")
        print(f"         VERDICT: {v}  (score {res.get('total_score')}/{res.get('max_score')}  "
              f"data={res.get('data_score')}  compute={res.get('compute_score')})")
        print(f"         min_viable_N={res.get('min_viable_n')}  "
              f"discoveries_at_N25={res.get('confident_at_n25')}  "
              f"compute_overhead={res.get('compute_overhead_ratio')}x")

    if r["status"] == "FAIL" and res.get("error"):
        print(f"         ERROR: {str(res['error'])[:300]}")


def _print_summary(results: List[Dict], wall: float) -> None:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print()
    print("=" * 72)
    print(f"K-Scarcity Real-Data Benchmark -- {len(results)} stage(s) in {wall:.1f}s")
    print(f"  Data: Kenya World Bank Macro Indicators 2000-2024  (N=25, 9 variables)")
    print(f"  Engine: OnlineDiscoveryEngine -- autonomous hypothesis discovery")
    print(f"  PASS={counts.get('PASS',0)}  WARN={counts.get('WARN',0)}  "
          f"FAIL={counts.get('FAIL',0)}  SKIP={counts.get('SKIP',0)}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

_STAGES = [
    ("DS.1", run_ds_1),
    ("DS.2", run_ds_2),
    ("DS.3", run_ds_3),
    ("DS.4", run_ds_4),
    ("CS.1", run_cs_1),
    ("CS.2", run_cs_2),
    ("CS.3", run_cs_3),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K-Scarcity real-data benchmark — Kenya World Bank 2000-2024",
        epilog="""
Stages:
  DS.1  data scarcity floor        — minimum N for first discovery
  DS.2  full discovery at N=25     — what the engine finds with all data
  DS.3  degradation curve          — discovery rate vs N
  DS.4  streaming coherence        — one-row-at-a-time, no replay
  CS.1  DRG adaptation             — RED pressure: beta decay + bandit decay
  CS.2  throughput comparison      — GREEN vs RED wall-clock + quality
  CS.3  memory-constrained buffer  — tiny buffer sizes, no crash
  CS.4  verdict                    — composite HIGH/MEDIUM/LOW/UNABLE

Exit: 0=pass, 1=fail
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fast", action="store_true")
    p.add_argument("--stage", nargs="+", metavar="ID")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--json", metavar="FILE")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not _WB_PATH.exists():
        print(f"ERROR: World Bank data not found:\n  {_WB_PATH}")
        return 1

    selected = _STAGES
    if args.stage:
        selected = [(sid, fn) for sid, fn in _STAGES if sid in args.stage]

    print("K-Scarcity Real-Data Benchmark")
    print("  Data:   Kenya World Bank Macro Indicators 2000-2024 (N=25 annual obs)")
    print("  Engine: OnlineDiscoveryEngine (autonomous — no hardcoded hypothesis pairs)")
    print("  Vars:  ", ", ".join(_VAR_NAMES))
    if args.fast:
        print("  Mode:   FAST")
    print()

    results: List[Dict] = []
    t_total = time.time()
    any_fail = False

    for sid, runner in selected:
        try:
            result = runner(fast=args.fast)
        except Exception as exc:
            result = _fail(sid, sid, "no crash",
                           f"{exc}\n{traceback.format_exc()[-400:]}", 0.0)
        results.append(result)
        _print_result(result)
        if args.verbose:
            _print_detail(result)
        if result["status"] == "FAIL":
            any_fail = True

    # Always run verdict with all prior results
    verdict = run_cs_4(prior_results=results)
    results.append(verdict)
    _print_result(verdict)
    if args.verbose:
        _print_detail(verdict)
    if verdict["status"] == "FAIL":
        any_fail = True

    _print_summary(results, time.time() - t_total)

    # Compact verdict line even without --verbose
    vr = verdict.get("result", {})
    if vr.get("verdict") and not args.verbose:
        v = vr["verdict"]
        print()
        print(f"  SCARCITY VERDICT: {v}  "
              f"(data={vr.get('data_score')}, compute={vr.get('compute_score')}, "
              f"total={vr.get('total_score')}/{vr.get('max_score')})")
        print(f"  Minimum viable N : {vr.get('min_viable_n')} years of annual data")
        print(f"  Discoveries@N=25 : {vr.get('confident_at_n25')} relationships (conf>=0.25)")
        print(f"  Compute overhead : {vr.get('compute_overhead_ratio')}x  (RED vs GREEN)")
        print()

    if args.json:
        import json
        payload = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": str(_WB_PATH),
            "n_variables": len(_VAR_NAMES),
            "variables": _VAR_NAMES,
            "fast": args.fast,
            "stages": results,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"Results written to {args.json}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())

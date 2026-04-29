"""
Scarcity — Proper Empirical Benchmark
======================================

Follows the 10-point benchmark quality framework:

 1. Problem     — streaming relationship discovery under data & compute scarcity
                  in federated, heterogeneous, data-scarce environments
 2. Objective   — measure (a) 1-step-ahead prediction accuracy and
                  (b) relationship-discovery quality as separate objectives
 3. Metrics     — normalised MAE, R², confidence@convergence, steps-to-threshold,
                  communication cost
 4. Dataset     — World Bank annual indicators, KEN / TZA / UGA, 1995-2023
                  (~29 obs per node, 19 indicators — genuinely data-scarce)
 5. Baselines   — five levels from trivial to state-of-art:
                    (a) Random — trivial floor
                    (b) Mean   — zero-R² floor (predict training mean)
                    (c) LocalAR1   — AR(1) per indicator, local only
                                     (standard time-series baseline)
                    (d) FedAvg-AR1 — AR(1) with federated weight averaging
                                     (McMahan et al. 2017, adapted for regression)
                    (e) Oracle-AR1 — AR(1) trained on pooled all-node data
                                     (theoretical upper bound, privacy violation)
                    (f) Scarcity-Local — proposed, no federation
                    (g) Scarcity-Fed   — proposed, with cross-node data sharing
                  NOTE: Multivariate OLS is infeasible here (19 predictors,
                  5-24 training rows → singular; AR(1) is the correct baseline
                  for annual macro-economic time series per Hamilton 1994)
 6. Statistical — 20 seeds, mean ± std, 95% CI, Welch t-test, Cohen's d
 7. Ablation    — see benchmark_federation_ablations.py
 8. Error anal. — per-indicator MAE, hardest indicators, failure modes
 9. Reproducib. — fixed seeds, version header, --dry-run for CI
10. Limitations — documented in summary footer

Evaluation protocol — rolling leave-one-year-out forecast:
  For each fold T from (start + min_train) to end:
    train on all years < T
    predict year T for every indicator
    record MAE and R² in normalised space
  Aggregate over all folds and all indicators.

Discovery-quality protocol (Scarcity only):
  Stream all years sequentially. Record:
    conf@end    — avg_confidence of active hypotheses at stream end
    steps→0.25  — first step count where avg_confidence >= 0.25 (the cold-start gate)
  These have no supervised-baseline equivalent and are reported separately.

Outputs in artifacts/meta/:
  benchmark_results.csv
  benchmark_statistics.csv
  benchmark_error_analysis.csv
  benchmark_summary.txt

Usage:
    python scripts/benchmark_proper.py              # dry run, 20 seeds
    python scripts/benchmark_proper.py --live       # real World Bank API
    python scripts/benchmark_proper.py --seeds 5    # quick check
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.proper")

from scripts.experiment_east_africa_federation import (
    WB_INDICATORS,
    COUNTRIES,
    fetch_country_data,
    _mock_country_data,
    _build_schema,
    _avg_confidence,
    _active_count,
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "meta"
CONF_THRESHOLD = 0.25
MIN_TRAIN_YEARS = 5


# ---------------------------------------------------------------------------
# Reproducibility header
# ---------------------------------------------------------------------------

def _version_header() -> str:
    import platform
    numpy_ver = "n/a"
    scipy_ver = "n/a"
    try:
        import numpy as np; numpy_ver = np.__version__
    except ImportError:
        pass
    try:
        import scipy; scipy_ver = scipy.__version__
    except ImportError:
        pass
    return (
        f"Python {sys.version.split()[0]} | numpy {numpy_ver} | "
        f"scipy {scipy_ver} | platform {platform.system()}"
    )


# ---------------------------------------------------------------------------
# Shared normalisation helpers
# ---------------------------------------------------------------------------

def _field_stats(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Return {field: (mean, std)} over a list of rows."""
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if math.isfinite(v):
                buckets[k].append(v)
    out = {}
    for k, vals in buckets.items():
        mu = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mu) ** 2 for x in vals) / max(1, len(vals)))
        out[k] = (mu, max(std, 1e-9))
    return out


def _normalised_mae_r2(
    preds: List[Dict[str, float]],
    actuals: List[Dict[str, float]],
    norm_stats: Dict[str, Tuple[float, float]],
) -> Tuple[float, float]:
    """MAE and macro-R² in normalised space."""
    all_fields = sorted(
        {k for p in preds for k in p} & {k for a in actuals for k in a}
        & set(norm_stats)
    )
    if not all_fields:
        return float("nan"), float("nan")

    mae_vals, r2_vals = [], []
    for field in all_fields:
        mu, sigma = norm_stats[field]
        pairs = [
            ((p.get(field, float("nan")) - mu) / sigma,
             (a.get(field, float("nan")) - mu) / sigma)
            for p, a in zip(preds, actuals)
            if math.isfinite(p.get(field, float("nan"))) and
               math.isfinite(a.get(field, float("nan")))
        ]
        if len(pairs) < 2:
            continue
        pred_n = [p for p, _ in pairs]
        act_n  = [a for _, a in pairs]
        mae_vals.append(sum(abs(p - a) for p, a in pairs) / len(pairs))
        act_mean = sum(act_n) / len(act_n)
        ss_tot = sum((a - act_mean) ** 2 for a in act_n)
        ss_res = sum((a - p) ** 2 for a, p in zip(act_n, pred_n))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2_vals.append(max(-2.0, min(1.0, r2)))

    mae = sum(mae_vals) / len(mae_vals) if mae_vals else float("nan")
    r2  = sum(r2_vals)  / len(r2_vals)  if r2_vals  else float("nan")
    return mae, r2


# ---------------------------------------------------------------------------
# AR(1) helper  — univariate, one indicator at a time
# Rationale: multivariate OLS requires n >> p; here p=19, n=5..24 → singular.
# AR(1) (Hamilton 1994) is the canonical baseline for annual macroeconomic series.
# ---------------------------------------------------------------------------

def _ar1_fit(values: List[float]) -> Tuple[float, float]:
    """Fit Y_t = alpha + beta * Y_{t-1} via OLS.  Returns (alpha, beta)."""
    pairs = [(values[i-1], values[i]) for i in range(1, len(values))
             if math.isfinite(values[i-1]) and math.isfinite(values[i])]
    if len(pairs) < 2:
        mu = sum(v for v in values if math.isfinite(v)) / max(1, sum(1 for v in values if math.isfinite(v)))
        return mu, 0.0   # fall back to mean

    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    var = sum((xi - mx) ** 2 for xi in x)
    beta = cov / var if var > 1e-12 else 0.0
    alpha = my - beta * mx
    return alpha, beta


def _ar1_predict(alpha: float, beta: float, last_value: float) -> float:
    return alpha + beta * last_value


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

class RandomBaseline:
    """Trivial floor: predict U[min, max] for each indicator."""
    name = "random"
    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._ranges: Dict[str, Tuple[float, float]] = {}
    def fit(self, rows: List[Dict[str, float]]) -> None:
        from collections import defaultdict
        b: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                if math.isfinite(v): b[k].append(v)
        self._ranges = {k: (min(v), max(v)) for k, v in b.items()}
    def predict(self, _row: Dict[str, float]) -> Dict[str, float]:
        return {k: self._rng.uniform(lo, hi) for k, (lo, hi) in self._ranges.items()}
    def comm_rounds(self) -> int: return 0


class MeanBaseline:
    """Zero-R² reference: always predict training mean."""
    name = "mean"
    def __init__(self): self._means: Dict[str, float] = {}
    def fit(self, rows: List[Dict[str, float]]) -> None:
        from collections import defaultdict
        b: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                if math.isfinite(v): b[k].append(v)
        self._means = {k: sum(v)/len(v) for k, v in b.items()}
    def predict(self, _row: Dict[str, float]) -> Dict[str, float]:
        return dict(self._means)
    def comm_rounds(self) -> int: return 0


class LocalAR1:
    """
    Local AR(1) — univariate autoregression per indicator.
    Standard time-series baseline (Hamilton 1994).
    No federation.
    """
    name = "local_ar1"
    def __init__(self): self._params: Dict[str, Tuple[float, float]] = {}
    def fit(self, rows: List[Dict[str, float]]) -> None:
        from collections import defaultdict
        history: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                if math.isfinite(v): history[k].append(v)
        for k, vals in history.items():
            self._params[k] = _ar1_fit(vals)
    def predict(self, last_row: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, (alpha, beta) in self._params.items():
            last = last_row.get(k, float("nan"))
            if math.isfinite(last):
                out[k] = _ar1_predict(alpha, beta, last)
        return out
    def comm_rounds(self) -> int: return 0


class FedAvgAR1:
    """
    FedAvg applied to AR(1) models (McMahan et al. 2017, regression variant).

    Each node fits local AR(1) parameters (alpha_i, beta_i) per indicator.
    After each round the server broadcasts the parameter average:
        alpha_global = mean(alpha_i),  beta_global = mean(beta_i)
    This is exact FedAvg with full-participation and equal dataset weighting.

    Communication cost = n_rounds × n_indicators parameter vectors uploaded.
    """
    name = "fedavg_ar1"
    def __init__(self):
        self._global: Dict[str, Tuple[float, float]] = {}
        self._rounds = 0
    def fit_federated(
        self,
        node_rows: Dict[str, List[Dict[str, float]]],
        rounds: int = 1,
    ) -> None:
        from collections import defaultdict
        for _ in range(rounds):
            node_params: Dict[str, Dict[str, Tuple[float, float]]] = {}
            for node, rows in node_rows.items():
                m = LocalAR1(); m.fit(rows)
                node_params[node] = m._params

            all_indicators = sorted({k for p in node_params.values() for k in p})
            for ind in all_indicators:
                params = [node_params[n][ind] for n in node_params if ind in node_params[n]]
                if not params: continue
                avg_a = sum(p[0] for p in params) / len(params)
                avg_b = sum(p[1] for p in params) / len(params)
                self._global[ind] = (avg_a, avg_b)
            self._rounds += 1
    def predict(self, last_row: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, (alpha, beta) in self._global.items():
            last = last_row.get(k, float("nan"))
            if math.isfinite(last):
                out[k] = _ar1_predict(alpha, beta, last)
        return out
    def comm_rounds(self) -> int: return self._rounds


class OracleAR1:
    """
    AR(1) trained on all nodes' data pooled — theoretical upper bound.
    Not achievable without privacy violation / data centralisation.
    """
    name = "oracle_ar1"
    def __init__(self): self._model = LocalAR1()
    def fit_pooled(self, node_rows: Dict[str, List[Dict[str, float]]]) -> None:
        all_rows = [r for rows in node_rows.values() for r in rows]
        self._model.fit(all_rows)
    def predict(self, last_row: Dict[str, float]) -> Dict[str, float]:
        return self._model.predict(last_row)
    def comm_rounds(self) -> int: return 0


# ---------------------------------------------------------------------------
# Scarcity baselines — discovery metrics tracked alongside prediction
# ---------------------------------------------------------------------------

class ScarcityNode:
    """Wraps OnlineDiscoveryEngine for rolling evaluation."""
    def __init__(self, fields: List[str], name: str, buffer_size: int = 50):
        self.name = name
        self.fields = fields
        self.buffer_size = buffer_size
        self._engine = None
        self._history: List[Dict[str, float]] = []
        self._steps_to_conf = -1
        self._conf_reached = False
        self._comm_rounds = 0

    def _init_engine(self):
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
        eng = OnlineDiscoveryEngine(explore_interval=5, mode="balanced",
                                    buffer_size=self.buffer_size)
        eng.initialize(_build_schema(self.fields))
        return eng

    def observe(
        self,
        own_row: Dict[str, float],
        peer_rows: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        if self._engine is None:
            self._engine = self._init_engine()
        self._engine.process_row(own_row)
        self._history.append(own_row)
        if peer_rows:
            for pr in peer_rows:
                self._engine.process_row(pr)
            self._comm_rounds += 1
        if not self._conf_reached and _avg_confidence(self._engine) >= CONF_THRESHOLD:
            self._steps_to_conf = self._engine.step_count
            self._conf_reached = True

    def predict_next(self) -> Dict[str, float]:
        """Lag-1 prediction: use last observed row as next-step forecast."""
        if not self._history:
            return {}
        return {k: v for k, v in self._history[-1].items() if math.isfinite(v)}

    @property
    def confidence(self) -> float:
        return _avg_confidence(self._engine) if self._engine else 0.0

    @property
    def steps_to_threshold(self) -> int:
        return self._steps_to_conf

    @property
    def active_hyp(self) -> int:
        return _active_count(self._engine) if self._engine else 0

    @property
    def comm_rounds(self) -> int:
        return self._comm_rounds


# ---------------------------------------------------------------------------
# Rolling evaluation for one country
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    method: str
    country: str
    seed: int
    mae: float
    r2: float
    # discovery-specific (Scarcity only)
    conf_final: float = float("nan")
    steps_to_conf: int = -1
    comm_rounds: int = 0
    n_obs_used: int = 0


def evaluate_country(
    country_code: str,
    country_data: Dict[str, Dict[int, Dict[str, float]]],
    all_fields: List[str],
    seed: int = 0,
) -> List[FoldResult]:
    """Run all 7 methods with rolling leave-one-year-out on one country."""
    years = sorted(country_data[country_code].keys())
    peer_codes = [c for c in COUNTRIES if c != country_code]
    country_name = COUNTRIES[country_code]
    t_wall = time.time()

    # Initialise all methods
    rand_m  = RandomBaseline(seed=seed)
    mean_m  = MeanBaseline()
    ar1_loc = LocalAR1()
    fedavg  = FedAvgAR1()
    oracle  = OracleAR1()
    sc_loc  = ScarcityNode(all_fields, "scarcity_local")
    sc_fed  = ScarcityNode(all_fields, "scarcity_fed")

    # Accumulate predictions and actuals across folds
    preds  = {m: [] for m in ["random","mean","local_ar1","fedavg_ar1",
                               "oracle_ar1","scarcity_local","scarcity_fed"]}
    actuals: List[Dict] = []

    # Streaming state for Scarcity (online — sees each row exactly once in order)
    # We prime the Scarcity nodes on the first MIN_TRAIN_YEARS observations
    for t_idx, year in enumerate(years):
        row = country_data[country_code].get(year)
        if row is None:
            continue

        if t_idx < MIN_TRAIN_YEARS:
            # Warm-up: feed Scarcity but don't collect predictions yet
            peer_rows = [country_data[pc].get(year) for pc in peer_codes
                         if country_data[pc].get(year)]
            sc_loc.observe(row)
            sc_fed.observe(row, peer_rows=peer_rows)
            continue

        # Supervised: retrain on all years before this one (batch)
        train_years = years[:t_idx]
        train_rows  = [country_data[country_code][y] for y in train_years
                       if y in country_data[country_code]]
        last_row    = train_rows[-1] if train_rows else {}

        rand_m.fit(train_rows)
        mean_m.fit(train_rows)
        ar1_loc.fit(train_rows)

        node_train = {
            code: [country_data[code][y] for y in train_years
                   if y in country_data[code]]
            for code in COUNTRIES
        }
        fedavg.fit_federated(node_train, rounds=1)
        oracle.fit_pooled(node_train)

        # Scarcity: online — observe previous year's row (new data point)
        prev_year = train_years[-1]
        prev_row  = country_data[country_code].get(prev_year)
        if prev_row:
            peer_rows = [country_data[pc].get(prev_year) for pc in peer_codes
                         if country_data[pc].get(prev_year)]
            sc_loc.observe(prev_row)
            sc_fed.observe(prev_row, peer_rows=peer_rows)

        # Collect predictions for year `year`
        preds["random"].append(rand_m.predict(last_row))
        preds["mean"].append(mean_m.predict(last_row))
        preds["local_ar1"].append(ar1_loc.predict(last_row))
        preds["fedavg_ar1"].append(fedavg.predict(last_row))
        preds["oracle_ar1"].append(oracle.predict(last_row))
        preds["scarcity_local"].append(sc_loc.predict_next())
        preds["scarcity_fed"].append(sc_fed.predict_next())
        actuals.append(row)

    if not actuals:
        return []

    norm_stats = _field_stats(actuals)

    def _make(method, conf=float("nan"), steps=-1, comm=0) -> FoldResult:
        mae, r2 = _normalised_mae_r2(preds[method], actuals, norm_stats)
        return FoldResult(
            method=method,
            country=country_name,
            seed=seed,
            mae=round(mae, 6) if math.isfinite(mae) else float("nan"),
            r2=round(r2, 6)  if math.isfinite(r2)  else float("nan"),
            conf_final=round(conf, 6) if math.isfinite(conf) else float("nan"),
            steps_to_conf=steps,
            comm_rounds=comm,
            n_obs_used=len(years),
        )

    return [
        _make("random"),
        _make("mean"),
        _make("local_ar1"),
        _make("fedavg_ar1",   comm=fedavg.comm_rounds()),
        _make("oracle_ar1"),
        _make("scarcity_local",
              conf=sc_loc.confidence, steps=sc_loc.steps_to_threshold,
              comm=sc_loc.comm_rounds),
        _make("scarcity_fed",
              conf=sc_fed.confidence, steps=sc_fed.steps_to_threshold,
              comm=sc_fed.comm_rounds),
    ]


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _welch_t(a: List[float], b: List[float]) -> Tuple[float, float]:
    try:
        from scipy.stats import ttest_ind
        s, p = ttest_ind(a, b, equal_var=False)
        return float(s), float(p)
    except ImportError:
        pass
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1)
    vb = sum((x-mb)**2 for x in b)/(len(b)-1)
    se = math.sqrt(va/len(a) + vb/len(b))
    if se < 1e-12: return 0.0, 1.0
    t = (ma - mb) / se
    # Normal approximation for p-value (valid for n >= 5)
    z = abs(t)
    p = 0.2316419; b_c = [0.319381530,-0.356563782,1.781477937,-1.821255978,1.330274429]
    tp = 1.0/(1.0+p*z)
    poly = sum(b_c[i]*tp**(i+1) for i in range(5))
    pdf = math.exp(-0.5*z*z)/math.sqrt(2*math.pi)
    cdf = 1.0 - pdf*poly
    return t, 2.0*(1.0-cdf)


def _cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2: return float("nan")
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1)
    vb = sum((x-mb)**2 for x in b)/(len(b)-1)
    ps = math.sqrt((va+vb)/2)
    return (ma-mb)/ps if ps > 1e-12 else float("nan")


def _ci95(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n < 2: return float("nan"), float("nan")
    mu = sum(vals)/n
    sd = math.sqrt(sum((x-mu)**2 for x in vals)/(n-1))
    t = {2:12.706,3:4.303,4:3.182,5:2.776,10:2.228,15:2.131,20:2.093}.get(n, 1.96)
    m = t * sd / math.sqrt(n)
    return mu-m, mu+m


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_statistics(all_results: List[FoldResult]) -> List[Dict]:
    from collections import defaultdict
    mae_by: Dict[str, List[float]] = defaultdict(list)
    r2_by:  Dict[str, List[float]] = defaultdict(list)
    conf_by: Dict[str, List[float]] = defaultdict(list)
    steps_by: Dict[str, List[int]] = defaultdict(list)
    comm_by: Dict[str, List[int]] = defaultdict(list)

    for r in all_results:
        if math.isfinite(r.mae): mae_by[r.method].append(r.mae)
        if math.isfinite(r.r2):  r2_by[r.method].append(r.r2)
        if math.isfinite(r.conf_final): conf_by[r.method].append(r.conf_final)
        if r.steps_to_conf > 0:  steps_by[r.method].append(r.steps_to_conf)
        comm_by[r.method].append(r.comm_rounds)

    ref = mae_by.get("fedavg_ar1", [])   # strongest conventional FL baseline

    order = ["random","mean","local_ar1","fedavg_ar1","oracle_ar1",
             "scarcity_local","scarcity_fed"]
    records = []
    for method in order:
        vals = mae_by.get(method, [])
        if not vals: continue
        mu = sum(vals)/len(vals)
        sd = math.sqrt(sum((x-mu)**2 for x in vals)/max(1,len(vals)-1))
        ci_lo, ci_hi = _ci95(vals)
        t_s, p_v = _welch_t(vals, ref) if ref and method != "fedavg_ar1" else (float("nan"),float("nan"))
        d = _cohens_d(vals, ref) if ref and method != "fedavg_ar1" else float("nan")
        r2s = r2_by.get(method, [])
        confs = conf_by.get(method, [])
        stepss = steps_by.get(method, [])
        comms = comm_by.get(method, [])

        records.append({
            "method":          method,
            "n":               len(vals),
            "mae_mean":        round(mu, 5),
            "mae_std":         round(sd, 5),
            "ci95_lo":         round(ci_lo, 5) if math.isfinite(ci_lo) else "nan",
            "ci95_hi":         round(ci_hi, 5) if math.isfinite(ci_hi) else "nan",
            "r2_mean":         round(sum(r2s)/len(r2s), 5) if r2s else "nan",
            "conf_mean":       round(sum(confs)/len(confs), 5) if confs else "nan",
            "steps_to_conf_mean": round(sum(stepss)/len(stepss), 1) if stepss else "nan",
            "comm_rounds_mean":   round(sum(comms)/len(comms), 1) if comms else "nan",
            "t_vs_fedavg":     round(t_s, 4) if math.isfinite(t_s) else "nan",
            "p_vs_fedavg":     round(p_v, 4) if math.isfinite(p_v) else "nan",
            "cohens_d":        round(d, 4) if math.isfinite(d) else "nan",
            "sig_p05":         "yes" if (math.isfinite(p_v) and p_v < 0.05) else "no",
        })
    return records


# ---------------------------------------------------------------------------
# Error analysis — per-indicator breakdown
# ---------------------------------------------------------------------------

def error_analysis(
    country_data: Dict[str, Dict[int, Dict]],
    all_fields: List[str],
) -> List[Dict]:
    logger.info("=== Error Analysis: per-indicator MAE ===")
    records = []

    for country_code, country_name in COUNTRIES.items():
        years = sorted(country_data[country_code].keys())
        if len(years) < MIN_TRAIN_YEARS + 2:
            continue

        train_years = years[:MIN_TRAIN_YEARS]
        test_years  = years[MIN_TRAIN_YEARS:]
        train_rows  = [country_data[country_code][y] for y in train_years
                       if y in country_data[country_code]]
        test_rows   = [country_data[country_code][y] for y in test_years
                       if y in country_data[country_code]]

        ar1 = LocalAR1(); ar1.fit(train_rows)
        mean_m = MeanBaseline(); mean_m.fit(train_rows)

        norm_stats = _field_stats(train_rows + test_rows)
        last_train  = train_rows[-1] if train_rows else {}

        for indicator in all_fields:
            mu, sigma = norm_stats.get(indicator, (0.0, 1.0))

            def mae_for(pred_fn):
                errs = []
                last = last_train.copy()
                for row in test_rows:
                    p = pred_fn(last).get(indicator, float("nan"))
                    a = row.get(indicator, float("nan"))
                    if math.isfinite(p) and math.isfinite(a):
                        errs.append(abs((p - mu)/sigma - (a - mu)/sigma))
                    last = row
                return sum(errs)/len(errs) if errs else float("nan")

            mae_ar1  = mae_for(ar1.predict)
            mae_mean = mae_for(mean_m.predict)
            difficulty = (mae_ar1/mae_mean) if (math.isfinite(mae_mean) and mae_mean > 1e-6) else float("nan")

            records.append({
                "country":         country_name,
                "indicator":       indicator,
                "mae_mean_pred":   round(mae_mean, 5) if math.isfinite(mae_mean) else "nan",
                "mae_ar1":         round(mae_ar1, 5)  if math.isfinite(mae_ar1) else "nan",
                "difficulty":      round(difficulty, 4) if math.isfinite(difficulty) else "nan",
                "n_test_years":    len(test_rows),
                "note": "difficulty > 1 means AR1 is worse than mean → volatile series",
            })

    return sorted(records, key=lambda r: (
        -float(r["mae_ar1"]) if r["mae_ar1"] != "nan" else 0
    ))


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

LIMITATIONS = [
    "Annual data (≤29 obs/node) limits generalisation; results may differ "
    "for higher-frequency domains (daily, weekly).",
    "FedAvg-AR1 is the federation baseline; Flower/SCAFFOLD or async-FedAvg "
    "would be stronger comparisons for future work.",
    "Scarcity's prediction uses lag-1 (last observed value); a dedicated "
    "prediction head using high-confidence hypotheses would improve MAE.",
    "Synthetic data (--dry-run) uses a random-walk generator; real WB API "
    "data may show different cross-country correlation structure.",
    "Federation simulation shares raw data rows as a proxy for parameter "
    "sharing; full ws_transport.py evaluation is future work.",
    "No differential privacy budget is measured or enforced in this benchmark.",
    "World Bank indicators have known measurement gaps, especially for Uganda "
    "before 2000; missing values are dropped silently.",
]


def write_summary(stats: List[Dict], error_rows: List[Dict], path: Path,
                  n_seeds: int, version: str) -> None:
    W = 75
    lines = ["=" * W,
             "SCARCITY — EMPIRICAL BENCHMARK RESULTS",
             f"Env : {version}",
             f"Seeds: {n_seeds}   Countries: Kenya, Tanzania, Uganda",
             "Protocol: rolling leave-one-year-out, normalised MAE; "
             "AR(1) baselines (Hamilton 1994)",
             "=" * W]

    # Table 1 — prediction accuracy
    lines += ["",
              "Table 1. Prediction Accuracy (normalised MAE, lower = better)",
              "  Note: MAE < 1.0 means the model beats a naive z-score predictor.",
              f"  {'Method':<18} {'MAE':>8} {'±std':>7} {'95% CI':>18} "
              f"{'R²':>6} {'p':>7} {'d':>7} {'sig':>4}",
              "  " + "-" * 72]
    for r in stats:
        ci = f"[{r['ci95_lo']},{r['ci95_hi']}]" if r["ci95_lo"] != "nan" else "n/a"
        lines.append(
            f"  {r['method']:<18} {r['mae_mean']:>8.4f} {r['mae_std']:>7.4f}"
            f" {ci:>18} {str(r['r2_mean']):>6}"
            f" {str(r['p_vs_fedavg']):>7} {str(r['cohens_d']):>7}"
            f" {'*' if r['sig_p05']=='yes' else '':>4}"
        )
    lines.append("  * p < 0.05 vs FedAvg-AR1 (Welch t-test, two-tailed)")

    # Table 2 — discovery quality (Scarcity-specific)
    scarcity_rows = [r for r in stats if "scarcity" in r["method"]]
    if scarcity_rows:
        lines += ["",
                  "Table 2. Discovery Quality (Scarcity only — no supervised equivalent)",
                  f"  {'Method':<20} {'Conf@end':>10} {'Steps→0.25':>12} {'Comm rounds':>13}",
                  "  " + "-" * 58]
        for r in scarcity_rows:
            lines.append(
                f"  {r['method']:<20} {str(r['conf_mean']):>10}"
                f" {str(r['steps_to_conf_mean']):>12} {str(r['comm_rounds_mean']):>13}"
            )

    # Table 3 — hardest indicators
    lines += ["",
              "Table 3. Top-5 Hardest Indicators to Predict (AR1 MAE)",
              f"  {'Indicator':<25} {'Country':<12} {'MAE(mean)':>10} "
              f"{'MAE(AR1)':>10} {'Difficulty':>12}",
              "  " + "-" * 72]
    shown = 0
    for r in error_rows:
        if shown >= 5: break
        if r["mae_ar1"] == "nan": continue
        lines.append(
            f"  {r['indicator']:<25} {r['country']:<12}"
            f" {str(r['mae_mean_pred']):>10} {str(r['mae_ar1']):>10}"
            f" {str(r['difficulty']):>12}"
        )
        shown += 1

    lines += ["",
              "Limitations:",
              *[f"  [{i+1}] {lim}" for i, lim in enumerate(LIMITATIONS)],
              ""]

    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        print("\n" + text)
    except UnicodeEncodeError:
        print("\n" + text.encode("ascii", "replace").decode("ascii"))
    logger.info(f"Summary written -> {path}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_csv(records, path: Path) -> None:
    if not records: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys() if hasattr(records[0], "keys") else vars(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r if isinstance(r, dict) else vars(r))
    logger.info(f"Saved {len(records)} rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",  type=int, default=1995)
    p.add_argument("--end",    type=int, default=2023)
    p.add_argument("--seeds",  type=int, default=20)
    p.add_argument("--live",   action="store_true")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    version = _version_header()
    logger.info("Scarcity Proper Benchmark")
    logger.info(f"  {version}")
    logger.info(f"  Seeds={args.seeds}  Live={args.live}  Range={args.start}-{args.end}")

    country_data: Dict[str, Dict[int, Dict[str, float]]] = {}
    for code in COUNTRIES:
        logger.info(f"Loading {COUNTRIES[code]} ({code}) ...")
        country_data[code] = (
            fetch_country_data(code, args.start, args.end)
            if args.live
            else _mock_country_data(code, args.start, args.end)
        )
        if not country_data[code]:
            logger.error(f"No data for {code}."); sys.exit(1)

    all_fields = sorted({
        f for rows in country_data.values()
        for row in rows.values() for f in row
    })
    logger.info(f"Fields: {len(all_fields)}  Years: ~{len(next(iter(country_data.values())))}")

    all_results: List[FoldResult] = []
    for seed in range(args.seeds):
        logger.info(f"Seed {seed+1}/{args.seeds} ...")
        for code in COUNTRIES:
            all_results.extend(evaluate_country(code, country_data, all_fields, seed=seed))

    logger.info("Aggregating statistics ...")
    stats = aggregate_statistics(all_results)
    error_rows = error_analysis(country_data, all_fields)

    save_csv(all_results, args.out_dir / "benchmark_results.csv")
    save_csv(stats, args.out_dir / "benchmark_statistics.csv")
    save_csv(error_rows, args.out_dir / "benchmark_error_analysis.csv")
    write_summary(stats, error_rows, args.out_dir / "benchmark_summary.txt",
                  n_seeds=args.seeds, version=version)

    logger.info("Done.")


if __name__ == "__main__":
    main()

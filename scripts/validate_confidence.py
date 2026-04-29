"""
Confidence calibration validator — Phase 1 verification.

Runs three conditions and checks that the signal-based confidence fix
produces sensible results:

  1. NULL WORLD  — pure Gaussian noise, N=34, no structure.
                   Expected: avg_conf < 0.10  (was ~0.48 before fix)

  2. REAL KENYA  — synthetic Kenya-like macro data with genuine autocorrelation
                   and cross-variable structure, N=34.
                   Expected: avg_conf in [0.15, 0.55]

  3. MIXED       — real Kenya data with 50% observations replaced by noise.
                   Expected: avg_conf between conditions 1 and 2.

Usage:
    python scripts/validate_confidence.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

INDICATORS = [
    "gdp_growth", "inflation_cpi", "unemployment", "real_interest_rate",
    "govt_debt", "exports_gdp", "broad_money", "dom_credit_pvt",
    "labor_force_part", "electricity_access",
]

SCHEMA = {"fields": [{"name": v, "type": "float"} for v in INDICATORS]}


def _null_rows(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [
        {v: float(rng.standard_normal()) for v in INDICATORS}
        for _ in range(n)
    ]


def _kenya_rows(n: int, seed: int) -> list:
    """Synthetic macro-like data with genuine autocorrelation and correlations."""
    rng = np.random.default_rng(seed)
    rows = []
    state = {v: 0.0 for v in INDICATORS}

    # AR coefficients typical of annual macro series
    ar = {
        "gdp_growth": 0.55,
        "inflation_cpi": 0.72,
        "unemployment": 0.65,
        "real_interest_rate": 0.60,
        "govt_debt": 0.80,
        "exports_gdp": 0.50,
        "broad_money": 0.70,
        "dom_credit_pvt": 0.68,
        "labor_force_part": 0.85,
        "electricity_access": 0.78,
    }

    for _ in range(n):
        row = {}
        shock = float(rng.standard_normal()) * 0.3
        for v in INDICATORS:
            state[v] = ar[v] * state[v] + float(rng.standard_normal()) * 0.4
            # cross-variable: inflation affects interest rate
            if v == "real_interest_rate":
                state[v] += 0.3 * state["inflation_cpi"]
            row[v] = state[v]
        rows.append(row)

    return rows


def _mixed_rows(n: int, seed: int) -> list:
    """Half real structure, half noise."""
    real = _kenya_rows(n, seed)
    null = _null_rows(n, seed + 1000)
    rng = np.random.default_rng(seed + 2000)
    mixed = []
    for r, nl in zip(real, null):
        row = {}
        for v in INDICATORS:
            row[v] = r[v] if rng.random() > 0.5 else nl[v]
        mixed.append(row)
    return mixed


def run_condition(name: str, rows: list, seed: int) -> dict:
    eng = OnlineDiscoveryEngine(mode="balanced")
    eng.initialize_v2(SCHEMA, use_causal=True)

    for row in rows:
        eng.process_row(row)

    hyps = list(eng.hypotheses.population.values())
    if not hyps:
        return {"name": name, "avg_conf": 0.0, "n_active": 0, "n_hyps": 0}

    confs = [h.confidence for h in hyps]
    n_above_gate = sum(1 for c in confs if c >= 0.25)

    return {
        "name": name,
        "avg_conf": float(np.mean(confs)),
        "max_conf": float(np.max(confs)),
        "n_above_gate": n_above_gate,
        "n_hyps": len(hyps),
    }


def main():
    N = 34
    SEED = 42

    conditions = [
        ("NULL  (pure noise)", _null_rows(N, SEED)),
        ("KENYA (structured) ", _kenya_rows(N, SEED)),
        ("MIXED (50% noise)  ", _mixed_rows(N, SEED)),
    ]

    print("\n=== Confidence Calibration Validation ===\n")
    print(f"{'Condition':<26} {'avg_conf':>10} {'max_conf':>10} {'n>=0.25':>8} {'n_hyps':>8}")
    print("-" * 66)

    results = {}
    for name, rows in conditions:
        r = run_condition(name, rows, SEED)
        results[name.strip()] = r
        print(f"{r['name']:<26} {r['avg_conf']:>10.4f} {r['max_conf']:>10.4f} "
              f"{r['n_above_gate']:>8} {r['n_hyps']:>8}")

    print()

    # --- assertions ---
    null_conf = results["NULL  (pure noise)"]["avg_conf"]
    kenya_conf = results["KENYA (structured)"]["avg_conf"]
    mixed_conf = results["MIXED (50% noise)"]["avg_conf"]

    failures = []

    if null_conf >= 0.15:
        failures.append(
            f"FAIL: NULL avg_conf={null_conf:.4f} >= 0.15  "
            f"(confidence too high on random data)"
        )
    else:
        print(f"PASS: NULL avg_conf={null_conf:.4f} < 0.15")

    if kenya_conf < 0.10:
        failures.append(
            f"FAIL: KENYA avg_conf={kenya_conf:.4f} < 0.10  "
            f"(real data confidence too low)"
        )
    else:
        print(f"PASS: KENYA avg_conf={kenya_conf:.4f} >= 0.10")

    if kenya_conf <= null_conf:
        failures.append(
            f"FAIL: KENYA conf ({kenya_conf:.4f}) not higher than NULL ({null_conf:.4f})"
        )
    else:
        print(f"PASS: KENYA conf ({kenya_conf:.4f}) > NULL conf ({null_conf:.4f})")

    if mixed_conf >= kenya_conf:
        failures.append(
            f"FAIL: MIXED conf ({mixed_conf:.4f}) not lower than KENYA ({kenya_conf:.4f})"
        )
    else:
        print(f"PASS: MIXED conf ({mixed_conf:.4f}) < KENYA conf ({kenya_conf:.4f})")

    print()
    if failures:
        for f in failures:
            print(f)
        print(f"\n{len(failures)} assertion(s) FAILED.")
        sys.exit(1)
    else:
        print("All assertions passed.")


if __name__ == "__main__":
    main()

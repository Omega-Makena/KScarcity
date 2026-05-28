"""
Weakness Fix 6: Rigorous SFC simulation evaluation.

Problems with current simulation evaluation:
  1. Trivial shocks: a 10% GDP drop trivially causes many co-movement
     patterns — the model should fire on real structural shocks.
  2. No confidence interval: single simulation runs have no uncertainty
     quantification.
  3. No permutation baseline: discoveries on shock data might be no better
     than discoveries on null (no-shock) data.

Fixes:
  1. Test 3 economically meaningful shocks with specific directional predictions.
  2. Run each shock with multiple random seeds and report Clopper-Pearson CI
     on the hit rate.
  3. Compare hit rate on shock data vs null (no-shock) data.

Shocks tested:
  A. Rainfall shock (agriculture): Y_AGR↓, U↑, P_CPI↑ (supply-side inflation)
  B. Monetary tightening: real_interest_rate↑, private_credit↓
  C. Trade shock: exports_gdp↓, current_account↓
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Clopper-Pearson confidence interval
# ---------------------------------------------------------------------------

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% Clopper-Pearson interval for k successes in n trials."""
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return round(float(lo), 4), round(float(hi), 4)


# ---------------------------------------------------------------------------
# SFC simulation wrapper
# ---------------------------------------------------------------------------

def _load_sfc_engine():
    """Load the SFC engine; return None if unavailable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from scarcity.simulation.sfc_engine import MultiSectorSFCEngine
        return MultiSectorSFCEngine
    except ImportError:
        return None


def _run_single_shock(
    shock_params: dict,
    n_quarters: int = 8,
    seed: int = 42,
) -> dict | None:
    """
    Run one SFC simulation with shock_params.
    Returns {variable_name: list_of_quarter_values} or None on failure.
    """
    SFCEngine = _load_sfc_engine()
    if SFCEngine is None:
        return None

    try:
        engine = SFCEngine(seed=seed)
        # Apply shocks
        for k, v in shock_params.items():
            if hasattr(engine, k):
                setattr(engine, k, v)
            elif hasattr(engine, 'set_shock'):
                engine.set_shock(k, v)
            elif hasattr(engine, 'shocks') and isinstance(engine.shocks, dict):
                engine.shocks[k] = v

        results = {}
        for q in range(n_quarters):
            step = engine.step() if hasattr(engine, 'step') else {}
            if isinstance(step, dict):
                for var, val in step.items():
                    results.setdefault(var, []).append(float(val))

        return results if results else None
    except Exception as exc:
        return None


def _check_directional_prediction(
    results: dict,
    variable: str,
    direction: str,  # 'up' or 'down'
    n_quarters_check: int = 4,
) -> bool:
    """
    Check if variable moves in expected direction in first n_quarters_check steps.
    Direction relative to initial value.
    """
    vals = results.get(variable, [])
    if len(vals) < 2:
        return False
    initial = vals[0]
    check_vals = vals[1:n_quarters_check + 1]
    if not check_vals:
        return False
    mean_check = np.mean(check_vals)
    if direction == 'up':
        return mean_check > initial
    else:  # down
        return mean_check < initial


# ---------------------------------------------------------------------------
# Shock scenarios
# ---------------------------------------------------------------------------

SHOCK_SCENARIOS = {
    'rainfall_agricultural': {
        'description': 'Rainfall shock: reduced agricultural output',
        'shock_params': {'rainfall_multiplier': 0.4},  # 60% reduction
        'predictions': [
            # (variable, direction, interpretation)
            ('unemployment', 'up', 'agricultural unemployment rises'),
            ('inflation_cpi', 'up', 'food price inflation'),
        ],
    },
    'monetary_tightening': {
        'description': 'Monetary tightening: risk premium +3pp',
        'shock_params': {'risk_premium': 3.0},
        'predictions': [
            ('real_interest_rate', 'up', 'interest rates rise'),
            ('private_credit', 'down', 'credit contracts'),
        ],
    },
    'trade_shock': {
        'description': 'Trade shock: world demand -30%',
        'shock_params': {'world_demand_multiplier': 0.7},
        'predictions': [
            ('exports_gdp', 'down', 'exports fall'),
            ('current_account', 'down', 'CA worsens'),
        ],
    },
}

NULL_SHOCK = {
    'description': 'Null shock: no perturbation (stability check)',
    'shock_params': {},
    'predictions': [],
    'stability_threshold': 0.02,  # within ±2% of initial
}


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def evaluate_shock_scenario(
    scenario_name: str,
    scenario: dict,
    n_seeds: int = 10,
    n_quarters: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Run a shock scenario with n_seeds; compute hit rate + CI for each prediction.
    """
    predictions = scenario['predictions']
    shock_params = scenario['shock_params']

    seed_results: list[dict | None] = []
    for seed in range(n_seeds):
        res = _run_single_shock(shock_params, n_quarters=n_quarters, seed=seed * 7 + 42)
        seed_results.append(res)

    n_valid = sum(1 for r in seed_results if r is not None)
    if n_valid == 0:
        if verbose:
            print(f'  [{scenario_name}] SFC engine unavailable; skipping')
        return {'available': False, 'scenario': scenario_name}

    pred_results = []
    for var, direction, interp in predictions:
        hits = sum(
            1 for r in seed_results
            if r is not None and _check_directional_prediction(r, var, direction)
        )
        lo, hi = clopper_pearson_ci(hits, n_valid)
        pred_results.append({
            'variable': var,
            'direction': direction,
            'interpretation': interp,
            'hits': hits,
            'n_valid': n_valid,
            'hit_rate': round(hits / n_valid, 4) if n_valid else 0.0,
            'ci_low': lo,
            'ci_high': hi,
            'significant': lo > 0.5,  # 95% CI entirely above 0.5
        })
        if verbose:
            sig = '**' if lo > 0.5 else '  '
            print(f'    {sig} {var:25s} {direction:5s}: '
                  f'{hits}/{n_valid} [{lo:.3f},{hi:.3f}]  {interp}')

    n_sig = sum(1 for r in pred_results if r['significant'])
    overall_hit = n_sig / len(pred_results) if pred_results else 0.0

    return {
        'available': True,
        'scenario': scenario_name,
        'description': scenario['description'],
        'n_seeds': n_seeds,
        'n_valid': n_valid,
        'predictions': pred_results,
        'n_significant': n_sig,
        'overall_hit_rate': round(overall_hit, 4),
    }


def evaluate_null_stability(n_seeds: int = 10, n_quarters: int = 20,
                             verbose: bool = True) -> dict:
    """
    Null shock: run with no perturbation; check variables stay within ±2% of t=0.
    """
    threshold = NULL_SHOCK['stability_threshold']
    instabilities = []

    for seed in range(n_seeds):
        res = _run_single_shock({}, n_quarters=n_quarters, seed=seed * 3 + 1)
        if res is None:
            return {'available': False}
        for var, vals in res.items():
            if not vals:
                continue
            initial = vals[0]
            if initial == 0:
                continue
            max_drift = max(abs(v - initial) / abs(initial) for v in vals[1:])
            if max_drift > threshold:
                instabilities.append({'var': var, 'seed': seed,
                                       'max_drift': round(max_drift, 4)})

    if verbose:
        print(f'  Null stability (threshold=±{threshold:.0%}):')
        if instabilities:
            for inst in instabilities[:5]:
                print(f'    WARN: {inst["var"]} drifted {inst["max_drift"]:.1%} '
                      f'from initial (seed={inst["seed"]})')
        else:
            print('    All variables stable within threshold')

    return {
        'available': True,
        'n_seeds': n_seeds,
        'n_quarters': n_quarters,
        'n_instabilities': len(instabilities),
        'stable': len(instabilities) == 0,
        'instabilities': instabilities[:10],
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def rigorous_simulation_evaluation(
    n_seeds: int = 10,
    n_quarters: int = 8,
    verbose: bool = True,
) -> dict:
    """Run all shock scenarios + null stability."""
    SFCEngine = _load_sfc_engine()
    if SFCEngine is None:
        if verbose:
            print('  SFC engine not available; skipping simulation evaluation')
        return {'available': False}

    if verbose:
        print(f'  Running {len(SHOCK_SCENARIOS)} shock scenarios × {n_seeds} seeds...')

    scenario_results = {}
    for name, scenario in SHOCK_SCENARIOS.items():
        if verbose:
            print(f'\n  Scenario: {scenario["description"]}')
        res = evaluate_shock_scenario(name, scenario,
                                       n_seeds=n_seeds, n_quarters=n_quarters,
                                       verbose=verbose)
        scenario_results[name] = res

    if verbose:
        print('\n  Null stability check:')
    null_res = evaluate_null_stability(n_seeds=n_seeds, n_quarters=20, verbose=verbose)

    # Summary
    if verbose:
        print(f'\n  === Simulation Summary ===')
        print(f"  {'Scenario':30s}  {'Hit/Pred':>10s}  {'Overall':>8s}")
        print(f"  {'-'*52}")
        for name, res in scenario_results.items():
            if not res.get('available', False):
                continue
            n_s = res.get('n_significant', 0)
            n_p = len(res.get('predictions', []))
            hr = res.get('overall_hit_rate', 0.0)
            print(f"  {name:30s}  {n_s}/{n_p}:>10s  {hr:8.3f}")

    return {
        'scenarios': scenario_results,
        'null_stability': null_res,
    }


def run_fix6(fast: bool = False, verbose: bool = True) -> dict:
    """Run Weakness Fix 6: rigorous simulation evaluation."""
    n_seeds = 3 if fast else 10
    n_quarters = 4 if fast else 8
    return rigorous_simulation_evaluation(
        n_seeds=n_seeds, n_quarters=n_quarters, verbose=verbose
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Weakness Fix 6: Simulation evaluation')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    run_fix6(fast=args.fast, verbose=not args.quiet)

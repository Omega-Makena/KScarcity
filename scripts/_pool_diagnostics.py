"""
Diagnose the hypothesis pool: what confidence is each of the 15 types reaching?
Helps understand why some types don't surface in the extracted graph.
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
warnings.filterwarnings('ignore')

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.discovery import HypothesisState
from benchmark.real_data.world_bank_loader import prepare_multi_country_data

COUNTRIES = ['KEN', 'TZA', 'UGA']

def build_engine(var_names):
    e = OnlineDiscoveryEngine(mode='balanced', small_dataset_mode=True)
    e.initialize_v2({'fields': [{'name': v} for v in var_names]}, use_causal=True)
    return e

def stream(engine, df, aux_dfs, label=''):
    all_vars = sorted(df.columns.tolist())
    years = sorted(df.index.tolist())
    for i, yr in enumerate(years):
        engine.process_row(df.loc[yr].to_dict())
        for cc, aux in aux_dfs.items():
            if yr in aux.index:
                engine.process_row(aux.loc[yr].reindex(all_vars).to_dict())
        if label and (i+1) % 10 == 0:
            print(f'  {label}: {i+1}/{len(years)} rows', flush=True)

def pool_summary(engine, label, conf_threshold=0.45, min_evidence=5):
    pop = engine.hypotheses.population
    by_type = defaultdict(list)
    for h in pop.values():
        by_type[h.rel_type.value].append(h)

    print(f'\n{"="*70}')
    print(f'HYPOTHESIS POOL — {label}')
    print(f'{"="*70}')
    print(f'  Total hypotheses in pool: {len(pop)}')
    print(f'  Extraction threshold: conf>={conf_threshold}, evidence>={min_evidence}')
    print()
    print(f'  {"Type":<18} {"Count":>6} {"Max conf":>9} {"Med conf":>9} '
          f'{"Above thresh":>13} {"ACTIVE":>7} {"Best variables"}')
    print(f'  {"-"*85}')

    extractable_total = 0
    for rel_type in sorted(by_type.keys()):
        hyps = by_type[rel_type]
        confs = [h.confidence for h in hyps]
        evids = [h.evidence  for h in hyps]
        states = [h.meta.state.value for h in hyps]

        above = sum(1 for h in hyps
                    if h.confidence >= conf_threshold and h.evidence >= min_evidence
                    and h.meta.state != HypothesisState.DEAD)
        active = states.count('active')
        extractable_total += above

        # Find best hypothesis for this type
        best = max(hyps, key=lambda h: h.confidence)
        best_vars = ' & '.join(best.variables[:3])

        print(f'  {rel_type:<18} {len(hyps):>6} {max(confs):>9.3f} '
              f'{np.median(confs):>9.3f} {above:>13} {active:>7}   {best_vars}')

    print(f'\n  Total extractable edges (conf>={conf_threshold}, evid>={min_evidence}): {extractable_total}')

    # Show best hypothesis per under-represented type
    SPARSE_TYPES = {'temporal', 'equilibrium', 'compositional', 'competitive',
                    'synergistic', 'probabilistic', 'structural', 'mediating',
                    'moderating', 'graph', 'similarity', 'logical'}
    print(f'\n  Best hypothesis per under-represented type:')
    print(f'  {"Type":<18} {"Conf":>6} {"Fit":>6} {"Evid":>5} {"State":<10} {"Variables"}')
    print(f'  {"-"*75}')
    for rel_type in sorted(SPARSE_TYPES):
        if rel_type not in by_type:
            print(f'  {rel_type:<18}  (no hypotheses)')
            continue
        hyps = by_type[rel_type]
        best = max(hyps, key=lambda h: h.confidence)
        vars_str = ' & '.join(best.variables[:3])
        print(f'  {rel_type:<18} {best.confidence:>6.3f} {best.fit_score:>6.3f} '
              f'{best.evidence:>5} {best.meta.state.value:<10} {vars_str}')

    return extractable_total


print('[Loading data]')
data = prepare_multi_country_data(COUNTRIES)
ken_df  = data['KEN']
aux_dfs = {cc: data[cc].reindex(columns=sorted(ken_df.columns)) for cc in COUNTRIES[1:]}
all_vars = sorted(ken_df.columns.tolist())

# ── Single-country engine ──────────────────────────────────────────────────────
print('\n[Single-country engine: KEN only, 34 rows]')
e_single = build_engine(all_vars)
stream(e_single, ken_df, {}, label='KEN')
pool_summary(e_single, 'Single-country (KEN, 34 rows)')

# ── Federated engine ───────────────────────────────────────────────────────────
print('\n[Federated engine: KEN+TZA+UGA, ~102 rows]')
e_fed = build_engine(all_vars)
stream(e_fed, ken_df, aux_dfs, label='KEN+TZA+UGA')
pool_summary(e_fed, 'Federated (KEN+TZA+UGA, ~102 rows)')

# ── What threshold would surface all types? ───────────────────────────────────
print(f'\n{"="*70}')
print('THRESHOLD SENSITIVITY: how low does conf need to go to surface each type?')
print(f'{"="*70}')
pop = e_fed.hypotheses.population
for rel_type in sorted(set(h.rel_type.value for h in pop.values())):
    hyps = [h for h in pop.values() if h.rel_type.value == rel_type]
    best = max(hyps, key=lambda h: h.confidence)
    needed = f'{best.confidence:.3f}' if best.confidence > 0 else 'never (conf=0)'
    print(f'  {rel_type:<18}  best conf={needed}  (evidence={best.evidence})')

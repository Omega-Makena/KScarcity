"""
Step 5: Stability selection via block bootstrap.

Repeat Steps 1-4 on B_boot block-bootstrap resamples of the data.
For each hypothesis, compute the selection frequency (fraction of resamples
where it was declared significant at the given FDR level).

CRITICAL: Block bootstrap (Künsch 1989), not iid. Annual macro data has
temporal autocorrelation. iid bootstrap destroys autocorrelation structure
and produces anti-conservative results.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from scripts.experiments.calibration.step1_permutation_pvalues import compute_all_pvalues
from scripts.experiments.calibration.step2_zscore_transform import add_zscores
from scripts.experiments.calibration.step3_per_pair_selection import select_best_type_per_pair
from scripts.experiments.calibration.step4_fdr_control import apply_fdr, benjamini_hochberg


def block_bootstrap_sample(
    df: pd.DataFrame,
    block_size: int = 4,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Moving block bootstrap resample of df.

    Procedure:
    1. Build all overlapping blocks of length block_size (starting at each row)
    2. Sample ceil(N / block_size) blocks WITH replacement
    3. Concatenate and trim to N rows

    Returns a DataFrame with N rows, same columns, resampled in blocks.
    The resampled DataFrame will have repeated row-blocks — correct behavior.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(df)
    if n <= block_size:
        return df.copy()

    # All possible starting positions for a block of length block_size
    max_start = n - block_size + 1
    n_blocks_needed = ceil(n / block_size)

    starts = rng.integers(0, max_start, size=n_blocks_needed)
    blocks = [df.iloc[s: s + block_size].values for s in starts]
    resampled = np.concatenate(blocks, axis=0)[:n]
    return pd.DataFrame(resampled, columns=df.columns)


def _run_steps_1_to_4(
    df: pd.DataFrame,
    B_perm: int,
    fdr_q: float,
    include_types: Optional[list[str]],
    seed: int,
) -> list[dict]:
    """
    Run Steps 1-4 on a single (possibly resampled) DataFrame.
    Returns the per-pair-best-type list with FDR significance annotated.
    """
    pval_results = compute_all_pvalues(df, B=B_perm, seed=seed,
                                       include_types=include_types, verbose=False)
    add_zscores(pval_results)
    selected = select_best_type_per_pair(pval_results, max_types_per_pair=1)

    # Suppress print from apply_fdr in bootstrap loop
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        apply_fdr(selected, q_levels=[fdr_q])
    finally:
        sys.stdout = old_stdout

    return selected


def stability_selection(
    df: pd.DataFrame,
    B_boot: int = 50,
    B_perm: int = 100,
    fdr_q: float = 0.10,
    block_size: int = 4,
    seed: int = 42,
    include_types: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Full stability selection procedure.

    For each bootstrap resample b in 1..B_boot:
        1. Block-bootstrap sample df_b
        2. Run Steps 1-4 on df_b
        3. Record which hypotheses were declared significant

    For each hypothesis H: selection_frequency = (times significant) / B_boot

    Returns list of dicts with all fields including:
        'selection_frequency': float
        'score': float (z_score * selection_frequency, from original data)
        'stable': bool (selection_frequency >= 0.6)
        'significant_and_stable': bool
    """
    rng_master = np.random.default_rng(seed)

    # Step 1-4 on original data (for z_score and p_value baseline)
    if verbose:
        print(f'\nRunning Steps 1-4 on original data...')
    base_results = compute_all_pvalues(df, B=B_perm, seed=int(rng_master.integers(0, 2**31)),
                                       include_types=include_types, verbose=verbose)
    add_zscores(base_results)
    base_selected = select_best_type_per_pair(base_results, max_types_per_pair=1)
    apply_fdr(base_selected, q_levels=[fdr_q])

    # Build index: (pair, test_type) → result
    base_index: dict[tuple, dict] = {}
    for r in base_selected:
        key = (r['pair'], r['test_type'])
        base_index[key] = r

    # Bootstrap loop: track selection counts
    selection_counts: dict[tuple, int] = defaultdict(int)

    if verbose:
        print(f'\nStability selection: {B_boot} resamples, {B_perm} permutations each')

    for b in range(B_boot):
        boot_seed = int(rng_master.integers(0, 2**31))
        boot_df = block_bootstrap_sample(df, block_size=block_size,
                                         rng=np.random.default_rng(boot_seed))
        boot_results = _run_steps_1_to_4(
            boot_df, B_perm=B_perm, fdr_q=fdr_q,
            include_types=include_types, seed=boot_seed,
        )
        n_sig = 0
        for r in boot_results:
            if r.get('fdr_significant', False):
                key = (r['pair'], r['test_type'])
                selection_counts[key] += 1
                n_sig += 1
        if verbose:
            print(f'  Resample {b + 1}/{B_boot} ... {n_sig} significant at q={fdr_q:.2f}')

    # Compute selection frequencies and final scores
    final_results = []
    for r in base_selected:
        key = (r['pair'], r['test_type'])
        freq = selection_counts[key] / B_boot if B_boot > 0 else 0.0
        z = r.get('z_score', 0.0)

        entry = dict(r)
        entry['selection_frequency'] = freq
        entry['score'] = float(z * freq)
        entry['stable'] = freq >= 0.6
        entry['significant_and_stable'] = (r.get('fdr_significant', False) and freq >= 0.6)
        final_results.append(entry)

    # Sort by score descending
    final_results.sort(key=lambda r: r['score'], reverse=True)

    if verbose:
        n_stable = sum(1 for r in final_results if r['stable'])
        n_sig_stable = sum(1 for r in final_results if r['significant_and_stable'])
        m = len(final_results)
        print(f'\nStability selection complete ({B_boot} resamples, {B_perm} permutations each)')
        print(f'Hypotheses with pi >= 0.6: {n_stable}/{m} ({100 * n_stable / max(m, 1):.1f}%)')
        print(f'Hypotheses significant AND stable: {n_sig_stable}/{m} '
              f'({100 * n_sig_stable / max(m, 1):.1f}%)')

        top10 = [r for r in final_results if r['score'] > 0][:10]
        if top10:
            print(f'\nTop {len(top10)} by Score(H) = Z × pi:')
            print(f"  {'Rank':>4}  {'Source':16s}  {'Target':16s}  "
                  f"{'Type':14s}  {'Z':>6}  {'pi':>6}  {'Score':>7}")
            print(f"  {'-' * 80}")
            for rank, r in enumerate(top10, start=1):
                src = r['source'][:15]
                tgt = r['target'][:15]
                tt = r['test_type'][:13]
                z = r['z_score']
                pi = r['selection_frequency']
                sc = r['score']
                print(f"  {rank:>4}  {src:16s}  {tgt:16s}  {tt:14s}  "
                      f"{z:>6.3f}  {pi:>6.3f}  {sc:>7.4f}")

    return final_results

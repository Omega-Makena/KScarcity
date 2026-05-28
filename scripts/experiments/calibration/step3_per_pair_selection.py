"""
Step 3: For each variable pair, select the BEST hypothesis type.

WHY NOT STOUFFER AGGREGATION:
Stouffer's method assumes independent Z-scores. For the SAME variable pair,
different hypothesis types (correlational, causal, competitive, etc.) all
operate on the same two data columns. Their test statistics are correlated
because they share the same data. Aggregating correlated Z-scores with Stouffer
inflates the combined Z, producing false significance.

INSTEAD: For each pair (X, Y), select the type with the lowest p-value.
The winning type becomes the pair's label. This is cleaner, avoids the
independence assumption, and naturally produces typed output.

Univariate tests (temporal, equilibrium, structural) are keyed by (X, X)
and do not compete with cross-variable tests.
"""
from __future__ import annotations

from collections import defaultdict


def select_best_type_per_pair(
    results: list[dict],
    max_types_per_pair: int = 1,
) -> list[dict]:
    """
    Group results by variable pair. For each pair, select the type(s) with
    the lowest p-value (highest z-score).

    Args:
        results: Output of step 2 (with z_scores added)
        max_types_per_pair: How many types to keep per pair.
            1 = strict best-type (one per pair)
            2 = report top-2 if both significant

    Returns:
        Filtered list with at most max_types_per_pair entries per pair, each
        having additional fields:
            'type_rank': int (1 = best for this pair)
            'competing_types': list of (type, p_value, z_score) for all types
                tested on this pair
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        key = r['pair']
        grouped[key].append(r)

    output = []
    for pair, items in grouped.items():
        # Sort ascending by p-value (best = lowest p = highest z)
        sorted_items = sorted(items, key=lambda r: r['p_value'])
        competing = [(r['test_type'], r['p_value'], r['z_score']) for r in sorted_items]

        for rank, item in enumerate(sorted_items[:max_types_per_pair], start=1):
            entry = dict(item)
            entry['type_rank'] = rank
            entry['competing_types'] = competing
            # Flag as multi-type if both top-2 are significant
            if len(sorted_items) >= 2:
                top2_both_sig = (sorted_items[0]['z_significant']
                                 and sorted_items[1]['z_significant'])
                entry['multi_type_pair'] = top2_both_sig
            else:
                entry['multi_type_pair'] = False
            output.append(entry)

    if __debug__:
        # Print type distribution summary
        type_counts: dict[str, int] = defaultdict(int)
        for r in output:
            if r['type_rank'] == 1:
                type_counts[r['test_type']] += 1
        total_pairs = sum(1 for r in output if r['type_rank'] == 1)
        pairwise_pairs = sum(1 for k in grouped if k[0] != k[1])
        univar_pairs = sum(1 for k in grouped if k[0] == k[1])
        print(f'  Per-pair selection: {pairwise_pairs} pairwise + {univar_pairs} univariate '
              f'-> {total_pairs} representatives')
        print(f'  Type distribution of winners:')
        for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f'    {t}: {cnt}')

    return output

"""Phase 2 — Evaluation harness.

Compares discovered relationships against ground truth edges.
All metrics are computed across 10 seeds (mean ± std).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

# ---------------------------------------------------------------------------
# Type normalisation
# ---------------------------------------------------------------------------

KSCARCITY_TYPE_MAP: dict[str, str] = {
    'CausalHypothesis': 'causal',
    'CorrelationalHypothesis': 'correlational',
    'TemporalHypothesis': 'temporal',
    'FunctionalHypothesis': 'functional',
    'EquilibriumHypothesis': 'equilibrium',
    'CompositionalHypothesis': 'compositional',
    'CompetitiveHypothesis': 'competitive',
    'SynergisticHypothesis': 'synergistic',
    'ProbabilisticHypothesis': 'probabilistic',
    'StructuralHypothesis': 'structural',
    'MediatingHypothesis': 'mediating',
    'ModeratingHypothesis': 'moderating',
    'GraphHypothesis': 'graph',
    'SimilarityHypothesis': 'similarity',
    'LogicalHypothesis': 'logical',
    # also accept already-normalised names
    'causal': 'causal',
    'correlational': 'correlational',
    'temporal': 'temporal',
    'functional': 'functional',
    'equilibrium': 'equilibrium',
    'compositional': 'compositional',
    'competitive': 'competitive',
    'synergistic': 'synergistic',
    'probabilistic': 'probabilistic',
    'structural': 'structural',
    'mediating': 'mediating',
    'moderating': 'moderating',
    'graph': 'graph',
    'similarity': 'similarity',
    'logical': 'logical',
}


def normalize_type(kscarcity_type: str) -> str:
    """Map K-Scarcity class name or enum value to ground truth type label."""
    return KSCARCITY_TYPE_MAP.get(
        kscarcity_type,
        kscarcity_type.lower().replace('hypothesis', ''),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical_pairs_from_discovered(d: dict) -> list[tuple]:
    """
    Return a list of (pair_key, type_str) tuples for one discovered entry.

    pair_key:
      - directed pair:   (source, target)          e.g. ('V1', 'V2')
      - undirected pair: frozenset({var1, var2})    e.g. frozenset({'V5','V6'})
      - self-loop:       ('V7',)                    for equilibrium single-var
    """
    vars_ = d.get('vars') or d.get('variables') or []
    # accept source/target format from baselines
    if not vars_:
        src = d.get('source', '')
        tgt = d.get('target', src)
        vars_ = [src, tgt] if src != tgt else [src]

    typ = normalize_type(d.get('type', ''))
    directed_types = {'causal', 'synergistic', 'mediating', 'moderating', 'functional', 'logical'}
    is_directed = typ in directed_types

    pairs = []
    if len(vars_) == 0:
        return pairs
    if len(vars_) == 1:
        # equilibrium / temporal self-loop
        pairs.append((vars_[0],))
    elif len(vars_) == 2:
        if is_directed:
            pairs.append((vars_[0], vars_[1]))
        else:
            pairs.append(frozenset(vars_))
    else:
        # triplet: compositional, synergistic, mediating — match any sub-pair
        for i in range(len(vars_)):
            for j in range(i + 1, len(vars_)):
                a, b = vars_[i], vars_[j]
                if is_directed:
                    pairs.append((a, b))
                    pairs.append((b, a))
                else:
                    pairs.append(frozenset({a, b}))
    return [(p, typ) for p in pairs]


def _gt_pair_key(gt_edge: dict) -> tuple:
    """Canonical pair key for a ground truth edge."""
    src, tgt = gt_edge['source'], gt_edge['target']
    if src == tgt:
        return (src,)   # self-loop (equilibrium)
    if gt_edge['directed']:
        return (src, tgt)
    return frozenset({src, tgt})


def _is_null_pair(vars_: list[str], null_pairs: list[tuple[str, str]]) -> bool:
    """Return True if the discovered vars contain a known-null pair."""
    null_set = {frozenset(p) for p in null_pairs}
    if len(vars_) >= 2:
        for i in range(len(vars_)):
            for j in range(i + 1, len(vars_)):
                if frozenset({vars_[i], vars_[j]}) in null_set:
                    return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.25   # K-Scarcity "confident" gate


def match_discovered_to_ground_truth(
    discovered: list[dict],
    ground_truth: list[dict],
    null_pairs: list[tuple[str, str]],
    mode: str = 'typed',
) -> dict:
    """Match discovered relationships against ground truth.

    Args:
        discovered: Engine output dicts (keys: vars/source/target, type, confidence/conf).
        ground_truth: From get_ground_truth_edges().
        null_pairs: From get_known_null_pairs().
        mode: 'typed' — TP only if pair AND type match.
              'edge_only' — TP if pair matches (any type).

    Returns:
        Dict with precision, recall, f1, type_accuracy, null_rejection_rate, and lists.
    """
    # Normalise confidence key
    def _conf(d: dict) -> float:
        return float(d.get('confidence', d.get('conf', 0.0)))

    # Filter to confident discoveries only
    confident = [d for d in discovered if _conf(d) >= CONFIDENCE_THRESHOLD]

    # Build GT lookup: pair_key -> list of GT edges
    gt_by_pair: dict[tuple, list[dict]] = {}
    for gt in ground_truth:
        key = _gt_pair_key(gt)
        gt_by_pair.setdefault(key, []).append(gt)

    # Track matched GT edges (to avoid double-counting)
    matched_gt_ids: set[int] = set()   # index into ground_truth list
    true_positives: list[tuple[dict, dict]] = []
    false_positives: list[dict] = []
    null_violations: list[dict] = []

    # For each discovered edge, try to match a GT edge
    # Sort by confidence descending so high-conf discoveries get first pick
    for disc in sorted(confident, key=_conf, reverse=True):
        vars_ = disc.get('vars') or disc.get('variables') or []
        if not vars_:
            src = disc.get('source', '')
            tgt = disc.get('target', src)
            vars_ = [src, tgt] if src != tgt else [src]

        disc_type = normalize_type(disc.get('type', ''))

        # Check null pair violation
        if _is_null_pair(vars_, null_pairs):
            null_violations.append(disc)

        # Generate candidate pair keys from this discovery
        candidate_pairs = _canonical_pairs_from_discovered(disc)

        best_gt_idx = None
        best_type_match = False

        for pair_key, _ in candidate_pairs:
            if pair_key not in gt_by_pair:
                continue
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt_ids:
                    continue
                if _gt_pair_key(gt) != pair_key:
                    continue
                type_matches = (normalize_type(gt['type']) == disc_type)
                if mode == 'edge_only' or type_matches:
                    # Prefer type-matching GT edges when available
                    if best_gt_idx is None or (not best_type_match and type_matches):
                        best_gt_idx = gt_idx
                        best_type_match = type_matches

        if best_gt_idx is not None:
            matched_gt_ids.add(best_gt_idx)
            true_positives.append((disc, ground_truth[best_gt_idx]))
        else:
            false_positives.append(disc)

    # False negatives = GT edges not matched
    false_negatives = [gt for i, gt in enumerate(ground_truth) if i not in matched_gt_ids]

    n_tp = len(true_positives)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Type accuracy: among TPs, fraction with exact type match (typed mode)
    if mode == 'typed' and n_tp > 0:
        exact_type = sum(
            1 for disc, gt in true_positives
            if normalize_type(disc.get('type', '')) == normalize_type(gt['type'])
        )
        type_accuracy = exact_type / n_tp
    else:
        type_accuracy = 1.0 if n_tp > 0 else 0.0

    # Null rejection rate: fraction of null pairs NOT discovered
    null_pair_set = {frozenset(p) for p in null_pairs}
    violated_null_pairs: set[frozenset] = set()
    for disc in confident:
        vars_ = disc.get('vars') or disc.get('variables') or []
        if not vars_:
            src = disc.get('source', '')
            tgt = disc.get('target', src)
            vars_ = [src, tgt]
        if len(vars_) >= 2:
            for i in range(len(vars_)):
                for j in range(i + 1, len(vars_)):
                    key = frozenset({vars_[i], vars_[j]})
                    if key in null_pair_set:
                        violated_null_pairs.add(key)

    n_null = len(null_pairs)
    null_rejection_rate = 1.0 - (len(violated_null_pairs) / n_null) if n_null > 0 else 1.0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'null_violations': null_violations,
        'n_discovered': len(confident),
        'n_tp': n_tp,
        'n_fp': n_fp,
        'n_fn': n_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'type_accuracy': type_accuracy,
        'null_rejection_rate': null_rejection_rate,
    }


def compute_n_sweep_metrics(
    results_by_n: dict[int, list[dict]],
    ground_truth: list[dict],
    null_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Compute evaluation metrics across the full N-sweep.

    Args:
        results_by_n: N -> list of per-seed discovery lists (list of edge dicts).
        ground_truth: From get_ground_truth_edges().
        null_pairs: From get_known_null_pairs().

    Returns:
        DataFrame with mean/std of all metrics, one row per N.
    """
    rows = []
    for n in sorted(results_by_n.keys()):
        seed_results = results_by_n[n]
        seed_metrics: list[dict] = []
        for discoveries in seed_results:
            m = match_discovered_to_ground_truth(discoveries, ground_truth, null_pairs, mode='typed')
            seed_metrics.append(m)

        def _mean(key: str) -> float:
            return float(np.mean([s[key] for s in seed_metrics]))

        def _std(key: str) -> float:
            return float(np.std([s[key] for s in seed_metrics]))

        rows.append({
            'N': n,
            'precision_mean': _mean('precision'),
            'precision_std': _std('precision'),
            'recall_mean': _mean('recall'),
            'recall_std': _std('recall'),
            'f1_mean': _mean('f1'),
            'f1_std': _std('f1'),
            'type_accuracy_mean': _mean('type_accuracy'),
            'type_accuracy_std': _std('type_accuracy'),
            'null_rejection_mean': _mean('null_rejection_rate'),
            'null_rejection_std': _std('null_rejection_rate'),
            'n_discovered_mean': _mean('n_discovered'),
            'n_discovered_std': _std('n_discovered'),
            'n_true_positives_mean': _mean('n_tp'),
            'n_false_positives_mean': _mean('n_fp'),
        })
    return pd.DataFrame(rows)


def compute_n_sweep_metrics_edge_only(
    results_by_n: dict[int, list[dict]],
    ground_truth: list[dict],
    null_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Same as compute_n_sweep_metrics but in edge_only mode."""
    rows = []
    for n in sorted(results_by_n.keys()):
        seed_results = results_by_n[n]
        seed_metrics: list[dict] = []
        for discoveries in seed_results:
            m = match_discovered_to_ground_truth(discoveries, ground_truth, null_pairs, mode='edge_only')
            seed_metrics.append(m)

        def _mean(key: str) -> float:
            return float(np.mean([s[key] for s in seed_metrics]))

        def _std(key: str) -> float:
            return float(np.std([s[key] for s in seed_metrics]))

        rows.append({
            'N': n,
            'precision_mean': _mean('precision'),
            'precision_std': _std('precision'),
            'recall_mean': _mean('recall'),
            'recall_std': _std('recall'),
            'f1_mean': _mean('f1'),
            'f1_std': _std('f1'),
            'type_accuracy_mean': _mean('type_accuracy'),
            'type_accuracy_std': _std('type_accuracy'),
            'null_rejection_mean': _mean('null_rejection_rate'),
            'null_rejection_std': _std('null_rejection_rate'),
            'n_discovered_mean': _mean('n_discovered'),
            'n_discovered_std': _std('n_discovered'),
            'n_true_positives_mean': _mean('n_tp'),
            'n_false_positives_mean': _mean('n_fp'),
        })
    return pd.DataFrame(rows)


def compute_scarcity_gap(
    kscarcity_metrics: pd.DataFrame,
    baseline_metrics: dict[str, pd.DataFrame],
) -> dict:
    """Compute the scarcity gap: area between K-Scarcity F1 and each baseline's F1.

    Uses trapezoidal integration over N values.

    Returns:
        Dict mapping baseline_name -> {scarcity_gap, gap_at_n10, gap_at_n25, crossover_n}
    """
    gaps: dict[str, dict] = {}
    ks_ns = kscarcity_metrics['N'].values
    ks_f1 = kscarcity_metrics['f1_mean'].values

    for name, bm in baseline_metrics.items():
        # Align on common N values
        common_ns = sorted(set(ks_ns) & set(bm['N'].values))
        if len(common_ns) < 2:
            gaps[name] = {
                'scarcity_gap': 0.0,
                'gap_at_n10': None,
                'gap_at_n25': None,
                'crossover_n': None,
            }
            continue

        ks_f1_aligned = np.interp(common_ns, ks_ns, ks_f1)
        bm_f1_aligned = np.interp(common_ns, bm['N'].values, bm['f1_mean'].values)
        diff = ks_f1_aligned - bm_f1_aligned

        # Trapezoidal integration (positive = K-Scarcity better)
        scarcity_gap = float(np.trapz(diff, x=common_ns))

        # Gap at specific N values
        def _gap_at(n_target: int) -> float | None:
            if n_target in common_ns:
                idx = common_ns.index(n_target)
                return float(diff[idx])
            return None

        # Crossover: smallest N where baseline catches up (diff becomes <= 0)
        crossover_n = None
        for i, n in enumerate(common_ns):
            if diff[i] <= 0:
                crossover_n = int(n)
                break

        gaps[name] = {
            'scarcity_gap': scarcity_gap,
            'gap_at_n10': _gap_at(10),
            'gap_at_n25': _gap_at(25),
            'crossover_n': crossover_n,
        }
    return gaps


# ---------------------------------------------------------------------------
# Self-test with hand-crafted examples
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from synthetic_data import get_ground_truth_edges, get_known_null_pairs

    gt = get_ground_truth_edges()
    null = get_known_null_pairs()

    # Perfect recall scenario: hand-craft all GT edges as discovered
    perfect = [
        {'vars': ['V1', 'V2'], 'type': 'causal', 'confidence': 0.8},
        {'vars': ['V1', 'V3'], 'type': 'causal', 'confidence': 0.7},
        {'vars': ['V3', 'V4'], 'type': 'causal', 'confidence': 0.6},
        {'vars': ['V1', 'V5', 'V4'], 'type': 'synergistic', 'confidence': 0.5},
        {'vars': ['V5', 'V6'], 'type': 'correlational', 'confidence': 0.9},
        {'vars': ['V7'], 'type': 'equilibrium', 'confidence': 0.7},
        {'vars': ['V8', 'V9'], 'type': 'competitive', 'confidence': 0.8},
        {'vars': ['V5', 'V6', 'V10'], 'type': 'compositional', 'confidence': 0.6},
        {'vars': ['V1', 'V3', 'V4'], 'type': 'mediating', 'confidence': 0.5},
    ]

    m = match_discovered_to_ground_truth(perfect, gt, null, mode='typed')
    print(f"Perfect recall test:")
    print(f"  TP={m['n_tp']}  FP={m['n_fp']}  FN={m['n_fn']}")
    print(f"  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"  Type accuracy={m['type_accuracy']:.3f}")
    print(f"  Null rejection rate={m['null_rejection_rate']:.3f}")
    assert m['recall'] > 0.5, f"Expected recall > 0.5, got {m['recall']}"

    # Null violation test
    with_null = perfect + [
        {'vars': ['V8', 'V2'], 'type': 'causal', 'confidence': 0.6},
    ]
    m2 = match_discovered_to_ground_truth(with_null, gt, null, mode='edge_only')
    assert len(m2['null_violations']) == 1, "Should detect 1 null violation"
    print(f"\nNull violation test: detected {len(m2['null_violations'])} violation(s) OK")

    # N-sweep metrics test
    results_by_n = {
        10: [perfect],
        25: [perfect, perfect],
    }
    df = compute_n_sweep_metrics(results_by_n, gt, null)
    print(f"\nN-sweep metrics (typed mode):\n{df[['N','f1_mean','f1_std','recall_mean']].to_string(index=False)}")
    assert df.shape[0] == 2, "Should have 2 rows"
    print("\nAll evaluation harness checks passed.")

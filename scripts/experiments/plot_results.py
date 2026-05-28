"""Phase 7 — Plotting and LaTeX table generation.

All figures saved as both PDF and PNG. Tables in booktabs LaTeX format.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Colour palette — accessible to colour-blind readers
METHOD_COLORS = {
    'K-Scarcity': '#1f77b4',
    'PC': '#ff7f0e',
    'FCI': '#2ca02c',
    'GES': '#d62728',
    'NOTEARS': '#9467bd',
    'DirectLiNGAM': '#8c564b',
    'CorrThreshold': '#7f7f7f',
    'typed': '#1f77b4',
    'edge_only': '#aec7e8',
    'full_system': '#1f77b4',
    'no_meta_learning': '#ff7f0e',
    'no_bandit_routing': '#2ca02c',
    'no_vectorized_rls': '#d62728',
    'causal_only': '#9467bd',
    'no_federation': '#8c564b',
}
METHOD_LWIDTHS = {
    'K-Scarcity': 2.5,
}


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    png_path = path.replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)


def _plot_method_line(
    ax,
    df: pd.DataFrame,
    method_name: str,
    metric_col: str = 'f1_mean',
    std_col: str = 'f1_std',
    color: str | None = None,
    lw: float = 1.5,
    alpha_band: float = 0.15,
) -> None:
    ns = df['N'].values
    means = df[metric_col].values
    stds = df[std_col].values
    c = color or METHOD_COLORS.get(method_name, '#333333')
    ax.plot(ns, means, label=method_name, color=c, linewidth=lw)
    ax.fill_between(ns, means - stds, means + stds, alpha=alpha_band, color=c)


# ---------------------------------------------------------------------------
# Figure 1 — N-sweep F1 (main figure)
# ---------------------------------------------------------------------------

def plot_n_sweep_f1(
    kscarcity_metrics: pd.DataFrame,
    baseline_metrics: dict[str, pd.DataFrame],
    output_path: str = 'experiments/results/figures/n_sweep_f1.pdf',
    mode: str = 'edge_only',
) -> None:
    """X: N (log scale), Y: F1. K-Scarcity vs all baselines with error bands."""
    fig, ax = plt.subplots()

    _plot_method_line(ax, kscarcity_metrics, 'K-Scarcity', lw=2.5)
    for name, bm in baseline_metrics.items():
        _plot_method_line(ax, bm, name, lw=1.5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of observations (N)')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'Discovery F1 vs. Data Scarcity ({mode} mode)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left')

    # Mark N=25 (Kenya benchmark)
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.text(25 * 1.05, 0.02, 'N=25\n(Kenya)', fontsize=9, color='gray')

    _save(fig, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Precision and Recall side by side
# ---------------------------------------------------------------------------

def plot_n_sweep_precision_recall(
    kscarcity_metrics: pd.DataFrame,
    baseline_metrics: dict[str, pd.DataFrame],
    output_path: str = 'experiments/results/figures/n_sweep_pr.pdf',
) -> None:
    """Two-panel: Precision (left) and Recall (right) vs N."""
    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, std_col, title in [
        (ax_p, 'precision_mean', 'precision_std', 'Precision'),
        (ax_r, 'recall_mean', 'recall_std', 'Recall'),
    ]:
        _plot_method_line(ax, kscarcity_metrics, 'K-Scarcity',
                          metric_col=metric, std_col=std_col, lw=2.5)
        for name, bm in baseline_metrics.items():
            _plot_method_line(ax, bm, name,
                              metric_col=metric, std_col=std_col, lw=1.5)

        ax.set_xscale('log')
        ax.set_xlabel('Number of observations (N)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs. Data Scarcity')
        ax.set_ylim(0, 1.05)
        ax.grid(True, which='both', alpha=0.3)
        ax.axvline(x=25, color='gray', linestyle='--', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout()
    _save(fig, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Typed vs Edge-only F1
# ---------------------------------------------------------------------------

def plot_typed_vs_edge_only(
    kscarcity_typed: pd.DataFrame,
    kscarcity_edge_only: pd.DataFrame,
    output_path: str = 'experiments/results/figures/typed_vs_edge.pdf',
) -> None:
    """K-Scarcity F1: typed mode vs edge-only mode."""
    fig, ax = plt.subplots()

    _plot_method_line(ax, kscarcity_typed, 'K-Scarcity (typed)',
                      color=METHOD_COLORS['typed'], lw=2.5)
    _plot_method_line(ax, kscarcity_edge_only, 'K-Scarcity (edge-only)',
                      color=METHOD_COLORS['edge_only'], lw=2.5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of observations (N)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Relationship Type Identification: Typed vs Edge-Only F1')
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', alpha=0.3)
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.6)
    ax.legend()

    _save(fig, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Ablation heatmap
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(
    ablation_f1: dict[str, dict[int, float]],
    output_path: str = 'experiments/results/figures/ablation_heatmap.pdf',
) -> None:
    """Heatmap: rows=ablation variants, columns=N values, cells=F1."""
    variants = list(ablation_f1.keys())
    n_values = sorted({n for v in ablation_f1.values() for n in v.keys()})
    matrix = np.array([
        [ablation_f1[v].get(n, 0.0) for n in n_values]
        for v in variants
    ])

    fig, ax = plt.subplots(figsize=(10, max(4, len(variants) * 0.8)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([f'N={n}' for n in n_values])
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_title('Ablation Study — F1 Score (typed mode)')

    for i in range(len(variants)):
        for j in range(len(n_values)):
            ax.text(j, i, f'{matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='black' if 0.2 < matrix[i, j] < 0.8 else 'white')

    fig.colorbar(im, ax=ax, label='F1 Score')
    fig.tight_layout()
    _save(fig, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 5 — Compute budget degradation
# ---------------------------------------------------------------------------

def plot_compute_budget_degradation(
    budget_results: dict,
    reference_n_discoveries: int,
    output_path: str = 'experiments/results/figures/compute_budget.pdf',
) -> None:
    """X: compute budget (log), Y: discovery F1 relative to unconstrained."""
    budgets = sorted(budget_results.keys())

    fig, ax = plt.subplots()

    for drg_key, label, color in [
        ('with_drg', 'With DRG', '#1f77b4'),
        ('without_drg', 'Without DRG', '#d62728'),
    ]:
        means, stds = [], []
        for budget in budgets:
            seed_results = budget_results[budget][drg_key]
            disc_counts = [r['n_discoveries'] for r in seed_results]
            relative = [c / max(reference_n_discoveries, 1) for c in disc_counts]
            means.append(np.mean(relative))
            stds.append(np.std(relative))

        means = np.array(means)
        stds = np.array(stds)
        ax.plot(budgets, means, label=label, color=color, linewidth=2.0, marker='o')
        ax.fill_between(budgets, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Compute budget per observation (seconds)')
    ax.set_ylabel('Relative discovery count (vs unconstrained)')
    ax.set_title('Compute Scarcity: Discovery Under Time Budget')
    ax.set_ylim(0, 1.3)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Unconstrained')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    _save(fig, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def generate_latex_tables(
    kscarcity_metrics: pd.DataFrame,
    baseline_metrics: dict[str, pd.DataFrame],
    scarcity_gaps: dict,
    ablation_f1: dict[str, dict[int, float]] | None = None,
    output_path: str = 'experiments/results/figures/tables.tex',
) -> None:
    """Generate publication-ready LaTeX tables (booktabs style)."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    lines: list[str] = []
    all_n = sorted(kscarcity_metrics['N'].values)

    # ---- Table 1: F1 at each N for all methods ----
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{F1 Score at Each Data Size (typed mode)}')
    lines.append(r'\label{tab:f1_sweep}')
    cols = 'l' + 'r' * len(all_n)
    lines.append(r'\begin{tabular}{' + cols + '}')
    lines.append(r'\toprule')
    header = 'Method & ' + ' & '.join([f'N={n}' for n in all_n]) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    def _f1_row(name: str, df: pd.DataFrame) -> str:
        cells = []
        for n in all_n:
            row = df[df['N'] == n]
            if len(row) == 0:
                cells.append('---')
            else:
                m = float(row['f1_mean'].iloc[0])
                s = float(row['f1_std'].iloc[0])
                cells.append(f'{m:.3f}$\\pm${s:.3f}')
        return name + ' & ' + ' & '.join(cells) + r' \\'

    lines.append(_f1_row('K-Scarcity', kscarcity_metrics))
    lines.append(r'\midrule')
    for name, bm in baseline_metrics.items():
        lines.append(_f1_row(name, bm))

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    # ---- Table 2: Scarcity gap summary ----
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Scarcity Gap: K-Scarcity vs Baselines (positive = K-Scarcity better)}')
    lines.append(r'\label{tab:scarcity_gap}')
    lines.append(r'\begin{tabular}{lrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Baseline & Integrated Gap & $\Delta$F1@N=10 & $\Delta$F1@N=25 & Crossover N \\')
    lines.append(r'\midrule')
    for name, gap in scarcity_gaps.items():
        g = gap['scarcity_gap']
        g10 = gap['gap_at_n10']
        g25 = gap['gap_at_n25']
        cx = gap['crossover_n']
        g10_s = f'{g10:.3f}' if g10 is not None else '---'
        g25_s = f'{g25:.3f}' if g25 is not None else '---'
        cx_s = str(cx) if cx is not None else 'never'
        lines.append(f'{name} & {g:.3f} & {g10_s} & {g25_s} & {cx_s} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    # ---- Table 3: Ablation at N=25 ----
    if ablation_f1:
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(r'\caption{Ablation Study at N=25 (typed mode F1)}')
        lines.append(r'\label{tab:ablation}')
        lines.append(r'\begin{tabular}{lr}')
        lines.append(r'\toprule')
        lines.append(r'Variant & F1@N=25 \\')
        lines.append(r'\midrule')
        for variant, n_f1 in ablation_f1.items():
            f1_25 = n_f1.get(25, 0.0)
            lines.append(f'{variant.replace("_", " ")} & {f1_25:.3f} \\\\')
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"  Saved LaTeX tables: {output_path}")

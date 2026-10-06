import pathlib

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

repo_root = pathlib.Path(__file__).resolve().parents[1]
results_dir = repo_root / 'results'
figures_dir = repo_root / 'docs' / 'figures'

metrics_path = results_dir / 'metrics.csv'
precision_path = results_dir / 'precision.csv'

# validated colorblind-safe pair (worst protan delta E 24.7); identity is also carried by the legend
model_colors = {'fashion_clip': '#2a78d6', 'clip_vit_b32': '#eb6834'}
model_labels = {'fashion_clip': 'FashionCLIP', 'clip_vit_b32': 'CLIP ViT-B/32'}
model_keys = list(model_colors)

# protocol section 4 thresholds on capped recall@10
thresholds = {'A': 0.80, 'B': 0.50, 'C': 0.30}
tier_names = {'A': 'A\nsingle category', 'B': 'B\ncategory + attribute', 'C': 'C\ncompositional', 'D': 'D\nout of ontology'}

ink = '#2b2b2b'
muted = '#6b6b6b'
grid = '#e4e4e1'

plt.rcParams.update({
    'font.size': 10,
    'axes.edgecolor': muted,
    'axes.labelcolor': ink,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlecolor': ink,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': muted,
    'ytick.color': muted,
    'legend.frameon': False,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

def style_value_axis(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=grid, linewidth=0.8)

def label_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, color=ink)

def plot_tier_recall(metrics_df):
    unfiltered = metrics_df[metrics_df['condition'] == 'unfiltered']
    tier_means = unfiltered.groupby(['tier', 'model_key'])['capped_recall@10'].mean().unstack('model_key')

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.36
    for i, model_key in enumerate(model_keys):
        positions = [t + (i - 0.5) * (width + 0.02) for t in range(len(tier_means))]
        bars = ax.bar(positions, tier_means[model_key], width=width, color=model_colors[model_key], label=model_labels[model_key])
        label_bars(ax, bars, tier_means[model_key])

    # each tier's pass line spans only that tier's bars
    for t, tier in enumerate(tier_means.index):
        ax.hlines(thresholds[tier], t - 0.45, t + 0.45, colors=ink, linestyles='--', linewidth=1.2)

    ax.set_xticks(range(len(tier_means)), [tier_names[tier] for tier in tier_means.index])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('mean capped recall@10')
    ax.set_title('Headline metric by tier, against the pre-registered thresholds', loc='left')
    style_value_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=ink, linestyle='--', linewidth=1.2))
    labels.append('pass threshold (0.80 / 0.50 / 0.30)')
    ax.legend(handles, labels, loc='upper right')

    fig.savefig(figures_dir / 'tier_recall.png')
    plt.close(fig)

def plot_manual_precision(precision_df):
    unfiltered = precision_df[precision_df['condition'] == 'unfiltered']
    tier_means = unfiltered.groupby(['tier', 'model_key'])[['strict_precision@10', 'lenient_precision@10']].mean()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.36
    tiers = sorted(unfiltered['tier'].unique())
    for i, model_key in enumerate(model_keys):
        positions = [t + (i - 0.5) * (width + 0.02) for t in range(len(tiers))]
        strict = [tier_means.loc[(tier, model_key), 'strict_precision@10'] for tier in tiers]
        lenient = [tier_means.loc[(tier, model_key), 'lenient_precision@10'] for tier in tiers]

        # lenient as a pale bar behind the strict bar, so the gap between them reads as the share of 1s
        ax.bar(positions, lenient, width=width, color=model_colors[model_key], alpha=0.28)
        bars = ax.bar(positions, strict, width=width, color=model_colors[model_key], label=f"{model_labels[model_key]}, strict")
        label_bars(ax, bars, strict)

    ax.set_xticks(range(len(tiers)), [tier_names[tier] for tier in tiers])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('mean manual precision@10')
    ax.set_title('Hand-judged precision by tier, strict and lenient', loc='left')
    style_value_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=muted, alpha=0.28))
    labels.append('pale extension: lenient (2s and 1s)')
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)

    fig.savefig(figures_dir / 'manual_precision.png')
    plt.close(fig)

def plot_sparsity_gap(precision_df):
    gap = precision_df[(precision_df['condition'] == 'unfiltered') & (precision_df['tier'] != 'D')]
    gap_wide = gap.pivot(index='query_id', columns='model_key', values='strict_gap')
    gap_wide = gap_wide.loc[gap_wide.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    height = 0.38
    for i, model_key in enumerate(model_keys):
        positions = [q + (i - 0.5) * (height + 0.02) for q in range(len(gap_wide))]
        ax.barh(positions, gap_wide[model_key], height=height, color=model_colors[model_key], label=model_labels[model_key])

    ax.axvline(0, color=ink, linewidth=1)
    ax.set_yticks(range(len(gap_wide)), gap_wide.index)
    ax.set_xlim(-0.4, 1.1)
    ax.set_xlabel('strict precision@10 minus capped recall@10')
    ax.set_title('Label sparsity gap by query (protocol section 7.2)', loc='left')
    style_value_axis(ax, axis='x')
    ax.legend(loc='lower right')

    fig.savefig(figures_dir / 'sparsity_gap.png')
    plt.close(fig)

def plot_tier_d(precision_df):
    tier_d = precision_df[(precision_df['condition'] == 'unfiltered') & (precision_df['tier'] == 'D')]
    tier_d = tier_d.pivot(index='query_id', columns='model_key', values='strict_precision@10')
    query_names = {'D1': 'D1 all black\n(color)', 'D2': 'D2 business casual\n(occasion)', 'D3': 'D3 beach\n(occasion)', 'D4': 'D4 goth\n(style/genre)'}

    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.36
    for i, model_key in enumerate(model_keys):
        positions = [q + (i - 0.5) * (width + 0.02) for q in range(len(tier_d))]
        bars = ax.bar(positions, tier_d[model_key], width=width, color=model_colors[model_key], label=model_labels[model_key])
        label_bars(ax, bars, tier_d[model_key])

    ax.set_xticks(range(len(tier_d)), [query_names[query_id] for query_id in tier_d.index])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('strict manual precision@10')
    ax.set_title('Tier D by query, strict manual precision@10', loc='left')
    style_value_axis(ax)
    ax.legend(loc='upper right')

    fig.savefig(figures_dir / 'tier_d.png')
    plt.close(fig)

if __name__ == '__main__':
    metrics_df = pd.read_csv(metrics_path)
    precision_df = pd.read_csv(precision_path)
    figures_dir.mkdir(exist_ok=True)

    plot_tier_recall(metrics_df)
    plot_manual_precision(precision_df)
    plot_sparsity_gap(precision_df)
    plot_tier_d(precision_df)

    print(f"figures: n = {len(list(figures_dir.glob('*.png')))}, saved to {figures_dir}")

#!/usr/bin/env python3
"""Score distribution plots: individual points (strip plot), no bins."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/v4"
PRED_DIR = os.path.join(OUT_DIR, "predictions")
SUMMARY = os.path.join(OUT_DIR, "experiment_summary.tsv")

routes = [
    "advanced_xgb", "xgboost_weighted", "weighted_tmb",
    "unstable_xgb", "sensitive_xgb", "mahal_vs_cosine",
    "cosine_advanced", "full_upgrade_v2", "no_locus_filter",
]

summary = pd.read_csv(SUMMARY, sep='\t')


def plot_panel(ax, scores, y_true, thr_lines, subtitle, rng_seed=42):
    """Strip plot: each sample is a dot, jittered vertically."""
    rng = np.random.RandomState(rng_seed)

    mss_mask = y_true == 'MSS'
    msih_mask = y_true == 'MSI-H'
    mss_scores = scores[mss_mask]
    msih_scores = scores[msih_mask]

    if len(mss_scores) == 0 and len(msih_scores) == 0:
        ax.set_title(subtitle, fontsize=11)
        return 1

    # Jitter: random y offset to avoid overplotting
    jitter_h = 0.2
    y_mss = rng.uniform(-jitter_h, jitter_h, size=len(mss_scores))
    y_msih = rng.uniform(-jitter_h, jitter_h, size=len(msih_scores))

    # Plot points
    ax.scatter(mss_scores, y_mss, c='#3498db', alpha=0.5, s=15,
               label=f'MSS (n={len(mss_scores)})', edgecolors='none', zorder=3)
    if len(msih_scores) > 0:
        ax.scatter(msih_scores, y_msih, c='#e74c3c', alpha=0.5, s=15,
                   label=f'MSI-H (n={len(msih_scores)})', edgecolors='none', zorder=3)

    # Threshold lines
    styles = ['--', ':', '-.']
    for i, (val, color, _) in enumerate(thr_lines):
        ax.axvline(val, color=color, linestyle=styles[i % 3], linewidth=2, zorder=5)

    ax.set_xlabel('Score', fontsize=11)
    ax.set_yticks([])
    ax.set_title(subtitle, fontsize=11, pad=25)
    ax.legend(fontsize=8, frameon=False, loc='upper right',
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0,
              markerscale=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-0.8, 0.8)
    return


for route in routes:
    print(f"Processing {route}...", flush=True)
    rd = summary[summary['route'] == route]
    if len(rd) == 0:
        continue

    v3_thr = rd[rd['experiment'] == 'v3→v4 (v3 thr)']['threshold'].values[0]
    youden_thr = rd[rd['experiment'] == 'v3→v4 (Youden)']['threshold'].values[0]
    v4_thr = rd[rd['experiment'] == 'v4 self-eval']['threshold'].values[0]

    paths = {k: os.path.join(PRED_DIR, f"{route}_{k}.tsv") for k in ['v3test', 'v3to4', 'v4self']}
    if not all(os.path.isfile(p) for p in paths.values()):
        print(f"  Skipping")
        continue

    dfs = {k: pd.read_csv(v, sep='\t') for k, v in paths.items()}
    v3test_thr = dfs['v3test']['threshold'].iloc[0] if 'threshold' in dfs['v3test'].columns else v3_thr

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(route, fontsize=16, fontweight='bold', y=1.01)

    # Panel 1: v3 test
    plot_panel(axes[0], dfs['v3test']['score'].values, dfs['v3test']['MSI_status'].values,
               [(v3test_thr, '#2ecc71', f'thr={v3test_thr:.3f}')], 'v3 test set')
    axes[0].annotate(f'thr={v3test_thr:.3f}', xy=(v3test_thr, 0.85), xycoords=('data', 'axes fraction'),
                     fontsize=10, fontweight='bold', color='#2ecc71', ha='center', va='bottom')

    # Panel 2: v3→v4
    plot_panel(axes[1], dfs['v3to4']['score'].values, dfs['v3to4']['MSI_status'].values,
               [(v3_thr, '#2ecc71', ''), (youden_thr, '#e67e22', '')], 'v3→v4')
    axes[1].annotate(f'v3 thr={v3_thr:.3f}', xy=(v3_thr, 0.85), xycoords=('data', 'axes fraction'),
                     fontsize=10, fontweight='bold', color='#2ecc71', ha='center', va='bottom')
    axes[1].annotate(f'Youden={youden_thr:.3f}', xy=(youden_thr, 0.92), xycoords=('data', 'axes fraction'),
                     fontsize=10, fontweight='bold', color='#e67e22', ha='center', va='bottom')

    # Panel 3: v4 self-eval
    plot_panel(axes[2], dfs['v4self']['score'].values, dfs['v4self']['MSI_status'].values,
               [(v4_thr, '#9b59b6', f'thr={v4_thr:.3f}')], 'v4 self-eval')
    axes[2].annotate(f'thr={v4_thr:.3f}', xy=(v4_thr, 0.85), xycoords=('data', 'axes fraction'),
                     fontsize=10, fontweight='bold', color='#9b59b6', ha='center', va='bottom')

    plt.tight_layout(pad=1.5)
    out_path = os.path.join(OUT_DIR, "score_distributions", f"{route}_score_dist.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")

print("Done.")

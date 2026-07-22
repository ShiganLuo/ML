# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Visualize MSI% distribution across groups."""

import os
import sys
import argparse
import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

INPUT_FILE = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/msisensor_pro_merged.tsv"
OUTPUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/figures"

# Consistent color palette
COLORS = {
    "MSI-H": "#e74c3c",
    "MSS":   "#3498db",
}


def plot_violin(df, output_dir, group_col="MSI_STATUS", value_col="msi_pct"):
    """Violin + box + strip plot of MSI% by group.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe.
    output_dir : str
        Directory to save the figure.
    group_col : str
        Column name for grouping.
    value_col : str
        Column name for MSI% values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: violin + box ---
    ax = axes[0]
    groups = sorted(df[group_col].unique())
    data = [df.loc[df[group_col] == g, value_col].dropna().values for g in groups]
    parts = ax.violinplot(data, showmeans=False, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS.get(groups[i], "#95a5a6"))
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # Overlay strip (jittered points, subsample if too many)
    for i, g in enumerate(groups):
        vals = data[i]
        if len(vals) > 500:
            vals = np.random.choice(vals, 500, replace=False)
        jitter = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(jitter, vals, s=3, alpha=0.3,
                   color=COLORS.get(g, "#95a5a6"), zorder=2)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_ylabel("MSI%")
    ax.set_title("MSI% distribution by group")

    # --- Right: overlapping histogram ---
    ax = axes[1]
    for g in groups:
        subset = df.loc[df[group_col] == g, value_col].dropna()
        ax.hist(subset, bins=50, alpha=0.5, label=f"{g} (n={len(subset)})",
                color=COLORS.get(g, "#95a5a6"), edgecolor="white", linewidth=0.3)
    ax.set_xlabel("MSI%")
    ax.set_ylabel("Count")
    ax.set_title("MSI% histogram by group")
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(output_dir, "msi_pct_distribution.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")


def plot_box_by_origin(df, output_dir, group_col="MSI_STATUS", value_col="msi_pct", origin_col="origin"):
    """Box plot of MSI% by origin × group.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe.
    output_dir : str
        Directory to save the figure.
    group_col, value_col, origin_col : str
        Column names.
    """
    origins = sorted(df[origin_col].dropna().unique())
    groups = sorted(df[group_col].unique())
    n_origins = len(origins)
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(max(8, n_origins * 2), 5))
    positions = []
    box_data = []
    labels = []
    colors = []
    spacing = 1.5
    for i, origin in enumerate(origins):
        for j, g in enumerate(groups):
            subset = df.loc[(df[origin_col] == origin) & (df[group_col] == g), value_col].dropna()
            pos = i * spacing + j * 0.4
            positions.append(pos)
            box_data.append(subset.values)
            labels.append(f"{origin}\n{g}")
            colors.append(COLORS.get(g, "#95a5a6"))

    bp = ax.boxplot(box_data, positions=positions, widths=0.35, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # X tick labels at group center
    tick_pos = [i * spacing + 0.2 for i in range(n_origins)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(origins, rotation=0)
    ax.set_ylabel("MSI%")
    ax.set_title("MSI% by origin and MSI status")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS.get(g, "#95a5a6"), alpha=0.6, label=g) for g in groups]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    outpath = os.path.join(output_dir, "msi_pct_by_origin.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")


def plot_roc(df, output_dir, score_col="msi_pct", label_col="MSI_STATUS"):
    """ROC curve for MSI% predicting MSI-H status.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with score and label columns.
    output_dir : str
        Directory to save the figure.
    score_col : str
        Column with MSI% scores.
    label_col : str
        Column with true labels (MSI-H / MSS).
    """
    from scipy.integrate import trapezoid as _trapz

    subset = df[[score_col, label_col]].dropna()
    y_true = (subset[label_col] == "MSI-H").astype(int).values
    y_score = subset[score_col].values

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present, skipping ROC")
        return

    # Manual ROC computation
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    total_pos = tps[-1]
    total_neg = fps[-1]
    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps)
    thresholds = y_score[sorted_idx]

    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])

    roc_auc = _trapz(tpr, fpr)

    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.scatter(fpr[best_idx], tpr[best_idx], marker="o", s=80, color="#2ecc71",
               zorder=5, label=f"Best threshold = {best_thr:.2f}%")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: MSI% predicting MSI-H status")
    ax.legend(loc="lower right")

    plt.tight_layout()
    outpath = os.path.join(output_dir, "msi_pct_roc.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")
    logger.info(f"AUC = {roc_auc:.4f}, best threshold = {best_thr:.4f}%")


def print_stats(df, group_col="MSI_STATUS", value_col="msi_pct"):
    """Print descriptive statistics per group.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe.
    group_col, value_col : str
        Column names.
    """
    print("\n" + "=" * 60)
    print("MSI% Descriptive Statistics by Group")
    print("=" * 60)
    for g in sorted(df[group_col].unique()):
        vals = df.loc[df[group_col] == g, value_col].dropna()
        print(f"\n  {g} (n={len(vals)}):")
        print(f"    mean   = {vals.mean():.4f}")
        print(f"    median = {vals.median():.4f}")
        print(f"    std    = {vals.std():.4f}")
        print(f"    min    = {vals.min():.4f}")
        print(f"    max    = {vals.max():.4f}")
        print(f"    25%    = {vals.quantile(0.25):.4f}")
        print(f"    75%    = {vals.quantile(0.75):.4f}")

    # Mann-Whitney U test
    from scipy.stats import mannwhitneyu
    groups = sorted(df[group_col].unique())
    if len(groups) == 2:
        a = df.loc[df[group_col] == groups[0], value_col].dropna()
        b = df.loc[df[group_col] == groups[1], value_col].dropna()
        stat, pval = mannwhitneyu(a, b, alternative="two-sided")
        print(f"\n  Mann-Whitney U test ({groups[0]} vs {groups[1]}):")
        print(f"    U = {stat:.0f}, p = {pval:.2e}")
    print("=" * 60 + "\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Plot MSI% distribution")
    parser.add_argument("--input", default=INPUT_FILE, help="Input merged TSV")
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR, help="Output directory for figures")
    parser.add_argument("--group-col", default="MSI_status", help="Column for grouping")
    parser.add_argument("--value-col", default="msi_pct", help="Column for MSI% values")
    parser.add_argument("--origin-col", default="origin", help="Column for data origin")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    # Plot
    plot_violin(df, args.output_dir, group_col=args.group_col, value_col=args.value_col)
    plot_box_by_origin(df, args.output_dir, group_col=args.group_col,
                       value_col=args.value_col, origin_col=args.origin_col)
    plot_roc(df, args.output_dir, score_col=args.value_col, label_col=args.group_col)

    # Stats
    print_stats(df, group_col=args.group_col, value_col=args.value_col)
    print_stats(df, group_col=args.origin_col, value_col=args.value_col)

    logger.info(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()

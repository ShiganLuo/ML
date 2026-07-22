# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""MSI% threshold study stratified by cancer type."""

import os
import sys
import argparse
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)

INPUT_FILE = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/msisensor_pro_merged.tsv"
OUTPUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/threshold_by_cancer"

COLORS = {"MSI-H": "#e74c3c", "MSS": "#3498db"}
CANCER_EN = {
    "结直肠癌": "CRC",
    "胃癌": "Gastric",
    "子宫内膜癌": "Endometrial",
    "肺癌": "Lung",
    "乳腺癌": "Breast",
    "肝癌": "Liver",
}


# ---------------------------------------------------------------------------
# ROC helpers
# ---------------------------------------------------------------------------
def compute_roc(y_true, y_score):
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    total_pos = tps[-1]
    total_neg = fps[-1]
    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps)
    thresholds = y_score[sorted_idx]
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    return fpr, tpr, thresholds


def evaluate(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / total if total > 0 else 0
    return {"threshold": threshold, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": sens, "specificity": spec, "accuracy": acc}


# ---------------------------------------------------------------------------
# Per-cancer analysis
# ---------------------------------------------------------------------------
def analyze_cancer(cancer_df, cancer_name, output_dir):
    """Threshold study for a single cancer type.

    Parameters
    ----------
    cancer_df : pd.DataFrame
        Subset for one cancer type (BL data with labels).
    cancer_name : str
        Cancer type name.
    output_dir : str
        Output directory.

    Returns
    -------
    dict
        Summary with cancer name, n, AUC, threshold, metrics.
    """
    y_true = (cancer_df["MSI_status"] == "MSI-H").astype(int).values
    y_score = cancer_df["msi_pct"].values

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos < 2 or n_neg < 2:
        logger.warning(f"{cancer_name}: too few samples (MSI-H={n_pos}, MSS={n_neg}), skipping")
        return None

    fpr, tpr, thresholds = compute_roc(y_true, y_score)
    auc = float(trapezoid(tpr, fpr))
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thr = float(thresholds[best_idx])
    m = evaluate(y_true, y_score, best_thr)

    # Mann-Whitney U
    a = y_score[y_true == 1]
    b = y_score[y_true == 0]
    u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")

    # Print
    print(f"\n{'='*70}")
    print(f"  {cancer_name} (n={len(cancer_df)}, MSI-H={n_pos}, MSS={n_neg})")
    print(f"{'='*70}")
    print(f"  AUC       = {auc:.4f}")
    print(f"  Threshold = {best_thr:.2f}%")
    print(f"  Sensitivity = {m['sensitivity']:.4f}")
    print(f"  Specificity = {m['specificity']:.4f}")
    print(f"  Accuracy    = {m['accuracy']:.4f}")
    print(f"  Mann-Whitney p = {p_val:.2e}")
    print(f"  MSI-H mean = {a.mean():.2f}%, MSS mean = {b.mean():.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ROC
    ax = axes[0]
    cancer_en = CANCER_EN.get(cancer_name, cancer_name)
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.scatter(fpr[best_idx], tpr[best_idx], marker="o", s=60, color="#2ecc71", zorder=5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{cancer_en} ROC")
    ax.legend(loc="lower right")

    # Violin
    ax = axes[1]
    groups = ["MSS", "MSI-H"]
    data = [cancer_df.loc[cancer_df["MSI_status"] == g, "msi_pct"].values for g in groups]
    parts = ax.violinplot(data, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[groups[i]])
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel("MSI%")
    ax.axhline(best_thr, color="#2ecc71", linestyle="--", lw=1.5)
    ax.set_title(f"{cancer_en} MSI%")

    # Sens/Spec curve
    ax = axes[2]
    thr_range = np.linspace(max(thresholds.min(), 0), min(thresholds.max(), 30), 200)
    sens_l, spec_l = [], []
    for t in thr_range:
        em = evaluate(y_true, y_score, t)
        sens_l.append(em["sensitivity"])
        spec_l.append(em["specificity"])
    ax.plot(thr_range, sens_l, color=COLORS["MSI-H"], lw=2, label="Sensitivity")
    ax.plot(thr_range, spec_l, color=COLORS["MSS"], lw=2, label="Specificity")
    ax.axvline(best_thr, color="#2ecc71", linestyle="--", lw=1.5, label=f"Best={best_thr:.1f}%")
    ax.set_xlabel("Threshold (%)")
    ax.set_title(f"{cancer_en} Sens/Spec")
    ax.legend()

    plt.tight_layout()
    safe_name = cancer_name.replace("/", "_")
    outpath = os.path.join(output_dir, f"{safe_name}_analysis.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")

    return {
        "cancer": cancer_name,
        "n": len(cancer_df),
        "n_msih": n_pos,
        "n_mss": n_neg,
        "msih_mean": float(a.mean()),
        "mss_mean": float(b.mean()),
        "delta": float(a.mean() - b.mean()),
        "auc": auc,
        "threshold": best_thr,
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "accuracy": m["accuracy"],
        "mannwhitney_p": p_val,
    }


def plot_summary(summary_list, output_dir):
    """Bar chart comparing AUC and threshold across cancer types.

    Parameters
    ----------
    summary_list : list[dict]
        Per-cancer summary dicts.
    output_dir : str
        Output directory.
    """
    df = pd.DataFrame(summary_list)
    df["cancer_en"] = df["cancer"].map(CANCER_EN).fillna(df["cancer"])
    df = df.sort_values("auc", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # AUC comparison
    ax = axes[0]
    bars = ax.barh(df["cancer_en"], df["auc"], color="#e74c3c", alpha=0.8)
    ax.set_xlabel("AUC")
    ax.set_title("AUC by Cancer Type")
    ax.set_xlim(0.4, 1.0)
    for bar, val in zip(bars, df["auc"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", fontsize=9)

    # Threshold comparison
    ax = axes[1]
    bars = ax.barh(df["cancer_en"], df["threshold"], color="#f39c12", alpha=0.8)
    ax.set_xlabel("Optimal Threshold (%)")
    ax.set_title("Threshold by Cancer Type")
    for bar, val in zip(bars, df["threshold"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                va="center", fontsize=9)

    # Mean MSI% comparison
    ax = axes[2]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df["msih_mean"], width, label="MSI-H", color=COLORS["MSI-H"], alpha=0.7)
    ax.bar(x + width/2, df["mss_mean"], width, label="MSS", color=COLORS["MSS"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["cancer_en"], rotation=30, ha="right")
    ax.set_ylabel("Mean MSI%")
    ax.set_title("Mean MSI% by Cancer Type")
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(output_dir, "cancer_comparison.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="MSI% threshold study by cancer type")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Loaded {len(df)} rows")

    # Split by origin
    bl = df[df["origin"] == "BL"].copy()
    pcr = df[df["origin"] == "PCR"].copy()
    renqun = df[df["origin"] == "renqun"].copy()
    logger.info(f"BL={len(bl)}, PCR={len(pcr)}, renqun={len(renqun)}")

    # Per-cancer analysis on BL
    print("\n" + "#" * 70)
    print("  BL Training Set - Per Cancer Type Analysis")
    print("#" * 70)

    summary_list = []
    cancer_types = bl["cancertype"].dropna().unique()
    for ct in sorted(cancer_types):
        subset = bl[bl["cancertype"] == ct]
        result = analyze_cancer(subset, ct, args.output_dir)
        if result is not None:
            summary_list.append(result)

    # Summary table
    if summary_list:
        plot_summary(summary_list, args.output_dir)
        summary_df = pd.DataFrame(summary_list)
        summary_path = os.path.join(args.output_dir, "cancer_threshold_summary.tsv")
        summary_df.to_csv(summary_path, sep='\t', index=False)
        logger.info(f"Saved: {summary_path}")

        print("\n" + "=" * 90)
        print("  Summary: Cancer Type Comparison")
        print("=" * 90)
        print(f"{'Cancer':>10} {'n':>5} {'MSI-H%':>8} {'MSI-H_mean':>12} {'MSS_mean':>10} {'Delta':>8} {'AUC':>6} {'Thr%':>6} {'Sens':>6} {'Spec':>6}")
        print("-" * 90)
        for r in sorted(summary_list, key=lambda x: -x["auc"]):
            print(f"{r['cancer']:>10} {r['n']:>5} {r['n_msih']/r['n']*100:>7.1f}% "
                  f"{r['msih_mean']:>11.2f} {r['mss_mean']:>10.2f} {r['delta']:>8.2f} "
                  f"{r['auc']:>6.3f} {r['threshold']:>5.1f} {r['sensitivity']:>6.3f} {r['specificity']:>6.3f}")
        print("=" * 90)

    # Predict renqun per cancer type (only for cancers with BL labels)
    print("\n" + "#" * 70)
    print("  Renqun Prediction (per cancer type, BL-labeled only)")
    print("#" * 70)

    labeled_cancers = {r["cancer"] for r in summary_list}
    renqun_labeled = renqun[renqun["cancertype"].isin(labeled_cancers)].copy()

    renqun_results = []
    for r in summary_list:
        ct = r["cancer"]
        thr = r["threshold"]
        sub = renqun_labeled[renqun_labeled["cancertype"] == ct].copy()
        if len(sub) == 0:
            continue
        sub["predicted_status"] = np.where(sub["msi_pct"] >= thr, "MSI-H", "MSS")
        n_pred_msih = (sub["predicted_status"] == "MSI-H").sum()
        n_pred_mss = (sub["predicted_status"] == "MSS").sum()
        renqun_results.append({
            "cancer": ct,
            "threshold": thr,
            "n": len(sub),
            "n_pred_msih": n_pred_msih,
            "n_pred_mss": n_pred_mss,
            "pred_msih_pct": n_pred_msih / len(sub) * 100,
        })
        print(f"  {ct}: n={len(sub)}, threshold={thr:.2f}%, "
              f"MSI-H={n_pred_msih} ({n_pred_msih/len(sub)*100:.1f}%), "
              f"MSS={n_pred_mss}")

    # Skipped cancers
    skipped = renqun[~renqun["cancertype"].isin(labeled_cancers)]
    if len(skipped) > 0:
        print(f"\n  Skipped (no BL labels): {skipped['cancertype'].value_counts().to_dict()}")

    # Save renqun predictions
    if renqun_results:
        renqun_pred_path = os.path.join(args.output_dir, "renqun_predictions_by_cancer.tsv")
        pred_df = pd.DataFrame(renqun_results)
        pred_df.to_csv(renqun_pred_path, sep='\t', index=False)
        logger.info(f"Saved: {renqun_pred_path}")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""MSI% threshold study using BL as training set, PCR as validation, renqun for prediction."""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

INPUT_FILE = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/msisensor_pro_merged.tsv"
OUTPUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/threshold_study"

COLORS = {"MSI-H": "#e74c3c", "MSS": "#3498db"}


# ---------------------------------------------------------------------------
# ROC helpers (no sklearn dependency)
# ---------------------------------------------------------------------------
def compute_roc(y_true, y_score):
    """Compute ROC curve from scratch.

    Parameters
    ----------
    y_true : array-like
        Binary labels (1 = positive).
    y_score : array-like
        Higher score = more likely positive.

    Returns
    -------
    fpr, tpr, thresholds : np.ndarray
    """
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


def compute_auc(fpr, tpr):
    """Compute AUC using trapezoidal rule."""
    from scipy.integrate import trapezoid
    return trapezoid(tpr, fpr)


def find_optimal_threshold(fpr, tpr, thresholds):
    """Find threshold that maximizes Youden's J statistic."""
    j = tpr - fpr
    best_idx = np.argmax(j)
    return thresholds[best_idx], best_idx


def evaluate(y_true, y_score, threshold):
    """Compute classification metrics at a given threshold.

    Parameters
    ----------
    y_true : array-like
        Binary labels (1 = MSI-H).
    y_score : array-like
        MSI% scores.
    threshold : float
        Decision threshold.

    Returns
    -------
    dict
        Sensitivity, specificity, accuracy, PPV, NPV, etc.
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    total = tp + tn + fp + fn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "threshold": threshold,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "PPV": ppv,
        "NPV": npv,
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def split_data(df):
    """Split dataframe into BL (train), PCR (val), renqun (predict).

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe.

    Returns
    -------
    bl, pcr, renqun : pd.DataFrame
    """
    bl = df[df["origin"] == "BL"].copy()
    pcr = df[df["origin"] == "PCR"].copy()
    renqun = df[df["origin"] == "renqun"].copy()
    logger.info(f"Split: BL={len(bl)}, PCR={len(pcr)}, renqun={len(renqun)}")
    return bl, pcr, renqun


def threshold_study(bl_df, output_dir):
    """Determine optimal threshold from BL training set.

    Parameters
    ----------
    bl_df : pd.DataFrame
        BL dataset with MSI_status and msi_pct columns.
    output_dir : str
        Output directory.

    Returns
    -------
    float
        Optimal threshold.
    """
    y_true = (bl_df["MSI_status"] == "MSI-H").astype(int).values
    y_score = bl_df["msi_pct"].values

    fpr, tpr, thresholds = compute_roc(y_true, y_score)
    auc = compute_auc(fpr, tpr)
    best_thr, best_idx = find_optimal_threshold(fpr, tpr, thresholds)

    # Evaluate at multiple candidate thresholds
    candidates = sorted(set([best_thr, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]))
    print("\n" + "=" * 80)
    print("BL Training Set - Threshold Evaluation")
    print("=" * 80)
    print(f"{'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} {'Accuracy':>10} {'PPV':>8} {'NPV':>8} {'J':>8}")
    print("-" * 80)

    results = []
    for thr in candidates:
        m = evaluate(y_true, y_score, thr)
        j = m["sensitivity"] + m["specificity"] - 1
        results.append(m)
        print(f"{thr:>10.2f} {m['sensitivity']:>12.4f} {m['specificity']:>12.4f} "
              f"{m['accuracy']:>10.4f} {m['PPV']:>8.4f} {m['NPV']:>8.4f} {j:>8.4f}")

    print("=" * 80)
    print(f"AUC = {auc:.4f}, Optimal threshold (Youden's J) = {best_thr:.4f}%")

    # Plot ROC
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.scatter(fpr[best_idx], tpr[best_idx], marker="o", s=80, color="#2ecc71",
               zorder=5, label=f"Best={best_thr:.2f}%")
    ax.set_xlabel("FPR (1 - Specificity)")
    ax.set_ylabel("TPR (Sensitivity)")
    ax.set_title("BL Training Set ROC")
    ax.legend(loc="lower right")

    # Plot sensitivity/specificity vs threshold
    ax = axes[1]
    thr_range = np.linspace(thresholds.min(), min(thresholds.max(), 30), 200)
    sens_list, spec_list = [], []
    for t in thr_range:
        m = evaluate(y_true, y_score, t)
        sens_list.append(m["sensitivity"])
        spec_list.append(m["specificity"])
    ax.plot(thr_range, sens_list, color=COLORS["MSI-H"], lw=2, label="Sensitivity")
    ax.plot(thr_range, spec_list, color=COLORS["MSS"], lw=2, label="Specificity")
    ax.axvline(best_thr, color="#2ecc71", linestyle="--", lw=1.5, label=f"Best={best_thr:.2f}%")
    ax.set_xlabel("MSI% Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Sensitivity / Specificity vs Threshold")
    ax.legend()
    ax.set_xlim(thr_range[0], thr_range[-1])

    plt.tight_layout()
    outpath = os.path.join(output_dir, "bl_threshold_study.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")

    return best_thr


def validate(pcr_df, threshold, output_dir):
    """Validate threshold on PCR independent set.

    Parameters
    ----------
    pcr_df : pd.DataFrame
        PCR dataset.
    threshold : float
        Threshold from training.
    output_dir : str
        Output directory.
    """
    y_true = (pcr_df["MSI_status"] == "MSI-H").astype(int).values
    y_score = pcr_df["msi_pct"].values

    m = evaluate(y_true, y_score, threshold)

    print("\n" + "=" * 80)
    print(f"PCR Validation Set (threshold = {threshold:.2f}%)")
    print("=" * 80)
    print(f"  n = {len(pcr_df)}")
    print(f"  TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}")
    print(f"  Sensitivity = {m['sensitivity']:.4f}")
    print(f"  Specificity = {m['specificity']:.4f}")
    print(f"  Accuracy    = {m['accuracy']:.4f}")
    print(f"  PPV         = {m['PPV']:.4f}")
    print(f"  NPV         = {m['NPV']:.4f}")
    print("=" * 80)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color="white")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred MSS", "Pred MSI-H"])
    ax.set_yticklabels(["True MSS", "True MSI-H"])
    ax.set_title(f"PCR Validation (thr={threshold:.2f}%)")
    plt.colorbar(im)
    plt.tight_layout()
    outpath = os.path.join(output_dir, "pcr_confusion_matrix.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")


def predict_renqun(renqun_df, threshold, output_dir):
    """Predict MSI status for renqun samples and save results.

    Parameters
    ----------
    renqun_df : pd.DataFrame
        Renqun dataset (labels unreliable).
    threshold : float
        Decision threshold.
    output_dir : str
        Output directory.
    """
    renqun_df = renqun_df.copy()
    renqun_df["predicted_status"] = np.where(
        renqun_df["msi_pct"] >= threshold, "MSI-H", "MSS"
    )

    n_msih = (renqun_df["predicted_status"] == "MSI-H").sum()
    n_mss = (renqun_df["predicted_status"] == "MSS").sum()

    print("\n" + "=" * 80)
    print(f"Renqun Prediction (threshold = {threshold:.2f}%)")
    print("=" * 80)
    print(f"  Total samples: {len(renqun_df)}")
    print(f"  Predicted MSI-H: {n_msih} ({n_msih/len(renqun_df)*100:.2f}%)")
    print(f"  Predicted MSS:   {n_mss} ({n_mss/len(renqun_df)*100:.2f}%)")
    print("=" * 80)

    # Save
    outpath = os.path.join(output_dir, "renqun_predictions.tsv")
    renqun_df.to_csv(outpath, sep='\t', index=False)
    logger.info(f"Saved: {outpath}")

    # Distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(renqun_df.loc[renqun_df["predicted_status"] == "MSS", "msi_pct"],
            bins=50, alpha=0.6, color=COLORS["MSS"], label=f"MSS (n={n_mss})")
    ax.hist(renqun_df.loc[renqun_df["predicted_status"] == "MSI-H", "msi_pct"],
            bins=50, alpha=0.6, color=COLORS["MSI-H"], label=f"MSI-H (n={n_msih})")
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold={threshold:.2f}%")
    ax.set_xlabel("MSI%")
    ax.set_ylabel("Count")
    ax.set_title("Renqun Predicted MSI Status")
    ax.legend()
    plt.tight_layout()
    figpath = os.path.join(output_dir, "renqun_prediction_dist.png")
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {figpath}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="MSI% threshold study")
    parser.add_argument("--input", default=INPUT_FILE, help="Input merged TSV")
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (skip training)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Loaded {len(df)} rows")

    bl, pcr, renqun = split_data(df)

    # Step 1: Threshold from BL training
    if args.threshold is not None:
        best_thr = args.threshold
        logger.info(f"Using user-specified threshold: {best_thr}")
    else:
        best_thr = threshold_study(bl, args.output_dir)

    # Step 2: Validate on PCR
    validate(pcr, best_thr, args.output_dir)

    # Step 3: Predict renqun
    predict_renqun(renqun, best_thr, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

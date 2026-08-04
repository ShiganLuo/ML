# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Metrics utilities for MSI detection experiments."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def calc_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute classification metrics at a given threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True labels ('MSI-H' or 'MSS').
    scores : np.ndarray
        Continuous scores from detector.
    threshold : float
        Decision threshold.

    Returns
    -------
    dict
        auc, sens, spec, acc, tp, fp, fn, tn, threshold, fpr, tpr, n_msih, n_mss.
    """
    predictions = np.where(scores >= threshold, "MSI-H", "MSS")
    tp = int(((predictions == "MSI-H") & (y_true == "MSI-H")).sum())
    fp = int(((predictions == "MSI-H") & (y_true == "MSS")).sum())
    fn = int(((predictions == "MSS") & (y_true == "MSI-H")).sum())
    tn = int(((predictions == "MSS") & (y_true == "MSS")).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / len(y_true)
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores, pos_label="MSI-H")
    auc_val = auc(fpr_arr, tpr_arr)
    return {
        "auc": auc_val, "sens": sens, "spec": spec, "acc": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold, "fpr": fpr_arr, "tpr": tpr_arr,
        "n_msih": tp + fn, "n_mss": tn + fp,
    }


def find_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold that maximizes TPR - FPR.

    Parameters
    ----------
    y_true : np.ndarray
        True labels ('MSI-H' or 'MSS').
    scores : np.ndarray
        Continuous scores from detector.

    Returns
    -------
    float
        Optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label="MSI-H")
    j_scores = tpr - fpr
    return float(thresholds[int(np.argmax(j_scores))])


def collect_summary(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Collect experiment results into a summary DataFrame.

    Parameters
    ----------
    all_results : dict
        {route_name: result_dict} from experiment runners.

    Returns
    -------
    pd.DataFrame
        Columns: route, experiment, auc, sens, spec, acc, threshold, n_features.
    """
    rows: List[Dict[str, Any]] = []
    for route_name, res in all_results.items():
        if "v3to4" in res and "error" not in res["v3to4"]:
            r = res["v3to4"]
            rows.append({
                "route": route_name, "experiment": "v3→v4 (v3 thr)",
                "auc": r["v3to4_v3thr"]["auc"], "sens": r["v3to4_v3thr"]["sens"],
                "spec": r["v3to4_v3thr"]["spec"], "acc": r["v3to4_v3thr"]["acc"],
                "threshold": r["threshold"], "n_features": r["n_features_used"],
            })
            rows.append({
                "route": route_name, "experiment": "v3→v4 (Youden)",
                "auc": r["v3to4_youden"]["auc"], "sens": r["v3to4_youden"]["sens"],
                "spec": r["v3to4_youden"]["spec"], "acc": r["v3to4_youden"]["acc"],
                "threshold": r["youden_threshold"], "n_features": r["n_features_used"],
            })
        if "v4self" in res and "error" not in res["v4self"]:
            r = res["v4self"]
            rows.append({
                "route": route_name, "experiment": "v4 self-eval",
                "auc": r["v4test"]["auc"], "sens": r["v4test"]["sens"],
                "spec": r["v4test"]["spec"], "acc": r["v4test"]["acc"],
                "threshold": r["threshold"], "n_features": len(r["features"]),
            })
    return pd.DataFrame(rows)


def print_summary_table(summary_df: pd.DataFrame, n_routes: int) -> None:
    """Pretty-print summary table to console."""
    print(f"\n\n{'#' * 80}")
    print(f"  SUMMARY: {n_routes} routes evaluated")
    print(f"{'#' * 80}")

    if summary_df.empty:
        print("  No successful experiments.\n")
        return

    for exp in ["v3→v4 (v3 thr)", "v3→v4 (Youden)", "v4 self-eval"]:
        sub = summary_df[summary_df["experiment"] == exp]
        if sub.empty:
            continue
        print(f"\n  {exp}:")
        print(f"  {'Route':<25} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'Acc':>8} {'Thr':>8} {'Feat':>6}")
        print(f"  {'-' * 72}")
        for _, row in sub.sort_values("auc", ascending=False).iterrows():
            print(f"  {row['route']:<25} {row['auc']:>8.4f} {row['sens']:>8.4f} "
                  f"{row['spec']:>8.4f} {row['acc']:>8.4f} {row['threshold']:>8.4f} "
                  f"{row['n_features']:>6.0f}")

    print()

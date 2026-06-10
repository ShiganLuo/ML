"""Diagnose group-based train/test split quality for SV frequency modeling."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit


def _build_group_key(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    """Build a composite group key from multiple columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    group_cols : list[str]
        Columns to concatenate as the group key.

    Returns
    -------
    pandas.Series
        Composite group key.
    """
    key = df[group_cols[0]].astype(str)
    for col in group_cols[1:]:
        key = key + "__" + df[col].astype(str)
    return key


def _summarize_distribution(values: pd.Series) -> dict:
    """Summarize distribution statistics for a numeric series.

    Parameters
    ----------
    values : pandas.Series
        Numeric series to summarize.

    Returns
    -------
    dict
        Summary statistics.
    """
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p10": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(values.median()),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def _low_freq_stats(values: pd.Series, thresholds: List[float]) -> dict:
    """Compute low-frequency proportions for thresholds.

    Parameters
    ----------
    values : pandas.Series
        Numeric series of ddPCR_AF values.
    thresholds : list[float]
        Thresholds to compute proportions for.

    Returns
    -------
    dict
        Mapping threshold to proportion.
    """
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {f"lt_{t}": np.nan for t in thresholds}
    return {f"lt_{t}": float((values < t).mean()) for t in thresholds}


def _write_text_report(
    out_path: str,
    total_rows: int,
    train_rows: int,
    test_rows: int,
    group_sizes: pd.Series,
    train_groups: int,
    test_groups: int,
    overlap_groups: int,
) -> None:
    """Write a concise text report for split diagnostics."""
    stats = {
        "min": int(group_sizes.min()) if not group_sizes.empty else 0,
        "median": float(group_sizes.median()) if not group_sizes.empty else np.nan,
        "mean": float(group_sizes.mean()) if not group_sizes.empty else np.nan,
        "max": int(group_sizes.max()) if not group_sizes.empty else 0,
    }
    lines = [
        "Split Diagnostics",
        "=" * 40,
        f"Total rows: {total_rows}",
        f"Train rows: {train_rows}",
        f"Test rows: {test_rows}",
        "",
        "Group sizes (rows per group):",
        f"  min={stats['min']}",
        f"  median={stats['median']:.2f}",
        f"  mean={stats['mean']:.2f}",
        f"  max={stats['max']}",
        f"  groups size=1: {(group_sizes == 1).sum()}",
        "",
        f"Unique groups (train): {train_groups}",
        f"Unique groups (test): {test_groups}",
        f"Overlap groups (train & test): {overlap_groups}",
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def diagnose_split(
    features_path: Optional[str],
    split_assignments_path: Optional[str],
    out_dir: str,
    group_cols: List[str],
    label_col: str,
    test_size: float,
    random_state: int,
) -> None:
    """Diagnose train/test split and ddPCR_AF distribution.

    Parameters
    ----------
    features_path : str or None
        Path to the features TSV file. Required if split assignments are absent.
    split_assignments_path : str or None
        Path to split assignments TSV. If provided, this file drives the analysis.
    out_dir : str
        Output directory for diagnostic artifacts.
    group_cols : list[str]
        Columns used to build the group key.
    label_col : str
        Target column name.
    test_size : float
        Test size used for splitting if split assignments are not provided.
    random_state : int
        Random seed used for splitting if split assignments are not provided.
    """
    os.makedirs(out_dir, exist_ok=True)

    if split_assignments_path:
        df = pd.read_csv(split_assignments_path, sep="\t")
    elif features_path:
        df = pd.read_csv(features_path, sep="\t")
    else:
        raise ValueError("Provide either --split-assignments or --features")

    missing_cols = [col for col in group_cols + [label_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if "split" not in df.columns:
        groups = _build_group_key(df, group_cols)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, df[label_col], groups=groups))
        df = df.copy()
        df["split"] = "train"
        df.loc[test_idx, "split"] = "test"
        df.to_csv(os.path.join(out_dir, "split_assignments.tsv"), sep="\t", index=False)

    total_rows = int(df.shape[0])
    train_rows = int((df["split"] == "train").sum())
    test_rows = int((df["split"] == "test").sum())

    group_key = _build_group_key(df, group_cols)
    group_sizes = group_key.value_counts()

    train_groups = int(group_key[df["split"] == "train"].nunique())
    test_groups = int(group_key[df["split"] == "test"].nunique())
    overlap_groups = int(
        len(set(group_key[df["split"] == "train"]) & set(group_key[df["split"] == "test"]))
    )

    _write_text_report(
        out_path=os.path.join(out_dir, "split_diagnostics.txt"),
        total_rows=total_rows,
        train_rows=train_rows,
        test_rows=test_rows,
        group_sizes=group_sizes,
        train_groups=train_groups,
        test_groups=test_groups,
        overlap_groups=overlap_groups,
    )

    group_size_df = group_sizes.reset_index()
    group_size_df.columns = ["group", "size"]
    group_size_df.to_csv(os.path.join(out_dir, "group_size_distribution.tsv"), sep="\t", index=False)

    label_stats = []
    thresholds = [1.0, 5.0, 10.0]
    for split in ["train", "test"]:
        split_vals = df.loc[df["split"] == split, label_col]
        stats = _summarize_distribution(split_vals)
        stats.update(_low_freq_stats(split_vals, thresholds))
        stats["split"] = split
        label_stats.append(stats)

    stats_df = pd.DataFrame(label_stats)
    stats_df.to_csv(os.path.join(out_dir, "ddpcr_af_summary.tsv"), sep="\t", index=False)

    # Plot ddPCR_AF distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split, color in zip(axes, ["train", "test"], ["#4C72B0", "#DD8452"]):
        vals = pd.to_numeric(df.loc[df["split"] == split, label_col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=30, color=color, alpha=0.8)
        ax.set_title(f"ddPCR_AF distribution ({split})")
        ax.set_xlabel(label_col)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ddpcr_af_hist.png"), dpi=200)
    plt.close(fig)

    # Plot group size histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(group_sizes.values, bins=min(30, max(5, int(np.sqrt(len(group_sizes))))), color="#55A868", alpha=0.85)
    ax.set_title("Group size distribution")
    ax.set_xlabel("Rows per group")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "group_size_hist.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose group train/test split quality")
    parser.add_argument("--features", default=None, help="Feature TSV (used when split assignments are absent)")
    parser.add_argument("--split-assignments", default=None, help="Split assignments TSV (preferred)")
    parser.add_argument("-o", "--out-dir", required=True, help="Output directory")
    parser.add_argument("--group-cols", nargs="+", default=["Original_ID", "mutation"])
    parser.add_argument("--label-col", default="ddPCR_AF")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    diagnose_split(
        features_path=args.features,
        split_assignments_path=args.split_assignments,
        out_dir=args.out_dir,
        group_cols=args.group_cols,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Plotting utilities for MSI detection experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_combined_roc(all_results: Dict[str, Dict], out_path: Path) -> None:
    """Plot ROC curves for all routes (2 panels: v3→v4 and v4 self-eval).

    Parameters
    ----------
    all_results : dict
        {route_name: result_dict} from experiment runners.
    out_path : Path
        Output PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for exp_idx, (exp_key, title) in enumerate([
        ("v3to4", "v3→v4 (Youden threshold)"),
        ("v4self", "v4 Self-Eval (Test Set)"),
    ]):
        ax = axes[exp_idx]
        ci = 0
        for route_name, res in all_results.items():
            if exp_key not in res or "error" in res[exp_key]:
                continue
            r = res[exp_key]
            m = r[f"{exp_key}_youden"] if exp_key == "v3to4" else r["v4test"]
            label = f"{route_name} (AUC={m['auc']:.3f})"
            ax.plot(m["fpr"], m["tpr"], color=colors[ci % 10], lw=1.5, label=label)
            ci += 1

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR (1 - Specificity)")
        ax.set_ylabel("TPR (Sensitivity)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    route_name: str,
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Strip plot: 3 panels (v3 test, v3→v4, v4 self-eval) for one route.

    Parameters
    ----------
    route_name : str
        Route name (must match summary_df and prediction file names).
    summary_df : pd.DataFrame
        Summary DataFrame with thresholds.
    out_dir : Path
        Output directory (predictions/ subdirectory expected).
    """
    pred_dir = out_dir / "predictions"
    paths = {k: pred_dir / f"{route_name}_{k}.tsv" for k in ["v3test", "v3to4", "v4self"]}
    if not all(p.is_file() for p in paths.values()):
        logger.warning(f"Skipping score dist for {route_name}: missing prediction files")
        return

    rd = summary_df[summary_df["route"] == route_name]
    if rd.empty:
        return

    # Extract thresholds
    def _get_thr(experiment: str) -> float:
        rows = rd[rd["experiment"] == experiment]
        return float(rows["threshold"].values[0]) if len(rows) > 0 else 0.0

    v3_thr = _get_thr("v3→v4 (v3 thr)")
    youden_thr = _get_thr("v3→v4 (Youden)")
    v4_thr = _get_thr("v4 self-eval")

    dfs = {k: pd.read_csv(v, sep="\t") for k, v in paths.items()}
    v3test_thr = float(dfs["v3test"]["threshold"].iloc[0]) if "threshold" in dfs["v3test"].columns else v3_thr

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(route_name, fontsize=16, fontweight="bold", y=1.01)

    # Panel 1: v3 test
    _plot_strip(axes[0], dfs["v3test"]["score"].values, dfs["v3test"]["MSI_status"].values,
                [(v3test_thr, "#2ecc71")], "v3 test set")
    _annotate_thr(axes[0], v3test_thr, f"thr={v3test_thr:.3f}", "#2ecc71")

    # Panel 2: v3→v4
    _plot_strip(axes[1], dfs["v3to4"]["score"].values, dfs["v3to4"]["MSI_status"].values,
                [(v3_thr, "#2ecc71"), (youden_thr, "#e67e22")], "v3→v4")
    _annotate_thr(axes[1], v3_thr, f"v3 thr={v3_thr:.3f}", "#2ecc71", y_pos=0.85)
    _annotate_thr(axes[1], youden_thr, f"Youden={youden_thr:.3f}", "#e67e22", y_pos=0.92)

    # Panel 3: v4 self-eval
    _plot_strip(axes[2], dfs["v4self"]["score"].values, dfs["v4self"]["MSI_status"].values,
                [(v4_thr, "#9b59b6")], "v4 self-eval")
    _annotate_thr(axes[2], v4_thr, f"thr={v4_thr:.3f}", "#9b59b6")

    plt.tight_layout(pad=1.5)
    out_path = out_dir / "score_distributions" / f"{route_name}_score_dist.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_strip(ax, scores: np.ndarray, y_true: np.ndarray,
                thr_lines: List[Tuple[float, str]], subtitle: str,
                jitter_h: float = 0.2, rng_seed: int = 42) -> None:
    """Strip plot: each sample is a dot, jittered vertically."""
    rng = np.random.RandomState(rng_seed)
    mss_mask = y_true == "MSS"
    msih_mask = y_true == "MSI-H"
    mss_scores = scores[mss_mask]
    msih_scores = scores[msih_mask]

    if len(mss_scores) == 0 and len(msih_scores) == 0:
        ax.set_title(subtitle, fontsize=11)
        return

    y_mss = rng.uniform(-jitter_h, jitter_h, size=len(mss_scores))
    y_msih = rng.uniform(-jitter_h, jitter_h, size=len(msih_scores))

    ax.scatter(mss_scores, y_mss, c="#3498db", alpha=0.5, s=15,
               label=f"MSS (n={len(mss_scores)})", edgecolors="none", zorder=3)
    if len(msih_scores) > 0:
        ax.scatter(msih_scores, y_msih, c="#e74c3c", alpha=0.5, s=15,
                   label=f"MSI-H (n={len(msih_scores)})", edgecolors="none", zorder=3)

    styles = ["--", ":", "-."]
    for i, (val, color) in enumerate(thr_lines):
        ax.axvline(val, color=color, linestyle=styles[i % 3], linewidth=2, zorder=5)

    ax.set_xlabel("Score", fontsize=11)
    ax.set_yticks([])
    ax.set_title(subtitle, fontsize=11, pad=25)
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0, markerscale=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.8, 0.8)


def _annotate_thr(ax, x: float, text: str, color: str, y_pos: float = 0.85) -> None:
    ax.annotate(text, xy=(x, y_pos), xycoords=("data", "axes fraction"),
                fontsize=10, fontweight="bold", color=color, ha="center", va="bottom")


def plot_individual_roc(
    route_name: str,
    results: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Plot individual ROC for one route: single panel, 3 curves overlaid.

    Curves:
      - v3 test: blue solid
      - v3→v4 (Youden): green solid
      - v4 self-eval: purple dashed

    Parameters
    ----------
    route_name : str
        Route name for title and filename.
    results : dict
        Result dict from run_one_route (keys: v3to4, v4self).
    out_dir : Path
        Output directory (roc_individual/ subdirectory will be created).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)

    # v3 test: blue solid
    if "v3to4" in results and "error" not in results["v3to4"]:
        m = results["v3to4"]["v3test"]
        ax.plot(m["fpr"], m["tpr"], color="#1f77b4", lw=2,
                label=f"v3 test (AUC={m['auc']:.3f})")

    # v3→v4 (Youden): green solid
    if "v3to4" in results and "error" not in results["v3to4"]:
        m = results["v3to4"]["v3to4_youden"]
        ax.plot(m["fpr"], m["tpr"], color="#2ca02c", lw=2,
                label=f"v3→v4 (AUC={m['auc']:.3f})")

    # v4 self-eval: purple dashed
    if "v4self" in results and "error" not in results["v4self"]:
        m = results["v4self"]["v4test"]
        ax.plot(m["fpr"], m["tpr"], color="#9467bd", lw=2, linestyle="--",
                label=f"v4 self-eval (AUC={m['auc']:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("FPR (1 - Specificity)", fontsize=12)
    ax.set_ylabel("TPR (Sensitivity)", fontsize=12)
    ax.set_title(route_name, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "roc_individual" / f"{route_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")

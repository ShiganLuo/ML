#!/usr/bin/env python3
# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Run multiple MSI routes on v4 data.

Experiments per route:
  1. v3 train → v4 test (cross-reagent evaluation)
     - v3 threshold applied to v4
     - Youden threshold on v4
  2. v4 self-evaluation (80/20 split)

Outputs (under --out-dir):
  experiment_summary.tsv          All routes × experiments metrics
  roc_all_routes.png              Combined ROC (2 panels)
  score_distributions/<route>.png Score strip plots (3 panels)
  predictions/<route>_*.tsv       Per-sample predictions
  features/<route>*.txt           Selected features
  config/<route>.json             Route configuration
  logs/<route>.log                Per-route detailed log

Usage:
  python run_v4_experiments.py
  python run_v4_experiments.py --routes xgboost_robust logistic_robust
  python run_v4_experiments.py --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from MSI.experiment_runners import build_pipeline, extract_features, train_and_eval, self_eval
from MSI.metrics import calc_metrics, collect_summary, print_summary_table
from MSI.plotting import plot_combined_roc, plot_score_distribution, plot_individual_roc
from compare_features import load_config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class V4Config:
    """All paths and defaults in one place."""
    config: str = (
        "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/"
        "luoshg_15179660974/Data/sta/20260615_MSI/config/compare_features.json"
    )
    v3_data: str = (
        "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/"
        "luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info_dedup.tsv"
    )
    v4_data: str = (
        "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/"
        "luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/v4_newsequencing.tsv"
    )
    out_dir: str = (
        "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/"
        "luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/v4"
    )
    default_routes: List[str] = field(default_factory=lambda: ["xgboost_robust"])


CFG = V4Config()

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

def setup_route_logger(route_name: str, out_dir: Path) -> tuple[logging.Logger, logging.FileHandler]:
    """Create per-route logger with file handler attached to root for bubble-up."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{route_name}.log"

    route_logger = logging.getLogger(f"route.{route_name}")
    route_logger.setLevel(logging.DEBUG)
    route_logger.handlers.clear()
    route_logger.propagate = False

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logging.getLogger().addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(f"[%(name)s] %(message)s"))
    route_logger.addHandler(ch)

    return route_logger, fh


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_v3_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    bl = df[df["origin"] == "BL"].copy()
    bl["sample_id"] = bl["site_feature"].apply(
        lambda x: os.path.basename(x).split("_cancer")[0] if isinstance(x, str) else None
    )
    bl = bl.set_index("sample_id")
    bl = bl[bl["MSI_real"].notna()]
    return bl


def load_v4_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["sample_id"] = df["样本编号"].astype(str).str.strip()
    df = df.set_index("sample_id")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# File I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_features(out_dir: Path, route_name: str, selected_cols: List[str],
                   strat_cfg: Dict, det_cfg: Dict, threshold: float) -> None:
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    with open(feat_dir / f"{route_name}.txt", "w") as f:
        f.write(f"Route: {route_name}\n")
        f.write(f"Strategy: {strat_cfg['name']}\n")
        f.write(f"Detector: {det_cfg['name']}\n")
        f.write(f"Locus selector: {strat_cfg.get('locus_selector', {}).get('type', 'none')}\n")
        f.write(f"Feature selector: {strat_cfg.get('feature_selector', {}).get('type', 'none')}\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        f.write(f"Features ({len(selected_cols)}):\n")
        for feat in selected_cols:
            f.write(f"  {feat}\n")


def _save_predictions(out_dir: Path, route_name: str, suffix: str, df: pd.DataFrame) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_dir / f"{route_name}_{suffix}.tsv", sep="\t")


# ══════════════════════════════════════════════════════════════════════════════
# Per-route orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_one_route(
    route_name: str,
    v3_meta: pd.DataFrame,
    v4_meta: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run all experiments for one route. Returns result dict."""
    route_logger, fh = setup_route_logger(route_name, out_dir)

    # Load route config
    route_configs = load_config(CFG.config, route_names=[route_name])
    cfg = route_configs[route_name]
    strat_cfg = cfg["strategies"][0]
    det_cfg = cfg["detectors"][0]
    run_cfg = cfg.get("pipeline", {})

    # Log + save config
    route_logger.info(f"{'=' * 60}")
    route_logger.info(f"Route: {route_name}")
    route_logger.info(f"Strategy: {strat_cfg['name']}")
    route_logger.info(f"Detector: {det_cfg['name']}")
    route_logger.info(f"Locus selector: {strat_cfg.get('locus_selector', {}).get('type', 'none')}")
    route_logger.info(f"Feature selector: {strat_cfg.get('feature_selector', {}).get('type', 'none')}")
    route_logger.info(f"Pipeline config: {json.dumps(run_cfg, indent=2)}")
    route_logger.info(f"{'=' * 60}")

    config_dir = out_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{route_name}.json").write_text(json.dumps(cfg, indent=2))

    results: Dict[str, Any] = {}

    # ── Experiment 1: v3 train → v4 test ──
    route_logger.info("\n--- Experiment 1: v3 train → v4 test ---")
    try:
        pipeline = build_pipeline(strat_cfg, det_cfg)
        exp1 = train_and_eval(
            pipeline, v3_meta, v4_meta, run_cfg,
            site_col_train="site_feature",
            site_col_test="site_path",
            cache_dir=cfg.get("cache_dir"),
            exp_logger=route_logger,
        )

        # Save features
        _save_features(out_dir, route_name, exp1["selected_cols"], strat_cfg, det_cfg, exp1["threshold"])

        # Save v4 predictions
        pred_df = pd.DataFrame({
            "MSI_status": exp1["y_true"],
            "score": exp1["test_scores"],
            "predicted_v3thr": np.where(exp1["test_scores"] >= exp1["threshold"], "MSI-H", "MSS"),
            "predicted_youden": np.where(exp1["test_scores"] >= exp1["youden_threshold"], "MSI-H", "MSS"),
        })
        pred_df["correct_v3thr"] = pred_df["MSI_status"] == pred_df["predicted_v3thr"]
        pred_df["correct_youden"] = pred_df["MSI_status"] == pred_df["predicted_youden"]
        _save_predictions(out_dir, route_name, "v3to4", pred_df)

        # Save v3 test predictions
        v3test_df = exp1["train_result"]["test"]["df"]
        v3test_scores = exp1["train_result"]["test"]["scores"]
        v3test_pred = pd.DataFrame({
            "MSI_status": v3test_df["MSI_status"].values,
            "score": v3test_scores,
            "predicted": np.where(v3test_scores >= exp1["threshold"], "MSI-H", "MSS"),
            "threshold": exp1["threshold"],
        })
        _save_predictions(out_dir, route_name, "v3test", v3test_pred)

        # v3 test metrics
        m_v3test = calc_metrics(
            v3test_df["MSI_status"].values, v3test_scores, exp1["threshold"],
        )

        results["v3to4"] = {
            "v3test": m_v3test,
            "v3to4_v3thr": exp1["test_metrics"],
            "v3to4_youden": exp1["youden_metrics"],
            "features": exp1["selected_cols"],
            "threshold": exp1["threshold"],
            "youden_threshold": exp1["youden_threshold"],
            "n_features_used": exp1["n_features_used"],
            "n_features_missing": exp1["n_features_missing"],
        }

    except Exception as e:
        route_logger.error(f"Experiment 1 failed: {e}", exc_info=True)
        results["v3to4"] = {"error": str(e)}

    # ── Experiment 2: v4 self-eval (80/20) ──
    route_logger.info("\n--- Experiment 2: v4 self-eval (80/20) ---")
    try:
        pipeline2 = build_pipeline(strat_cfg, det_cfg)

        # Prepare v4 metadata
        v4_meta2 = v4_meta.copy()
        v4_meta2["site_feature"] = v4_meta2["site_path"]
        if "cancertype" not in v4_meta2.columns:
            v4_meta2["cancertype"] = "unknown"
        if "MSI_CNC" not in v4_meta2.columns:
            v4_meta2["MSI_CNC"] = v4_meta2["MSI_real"]

        exp2 = self_eval(
            pipeline2, v4_meta2, run_cfg,
            site_col="site_feature",
            exp_logger=route_logger,
        )

        # Save features
        feat_dir = out_dir / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        with open(feat_dir / f"{route_name}_v4self.txt", "w") as f:
            f.write(f"Route: {route_name} (v4 self-eval)\n")
            f.write(f"Threshold: {exp2['threshold']:.4f}\n")
            f.write(f"Features ({len(exp2['selected_cols'])}):\n")
            for feat in exp2["selected_cols"]:
                f.write(f"  {feat}\n")

        # Save test predictions (from pipeline's own 80/20 split)
        test_df = exp2["train_result"]["test"]["df"]
        test_scores = exp2["train_result"]["test"]["scores"]
        pred_df2 = pd.DataFrame({
            "MSI_status": test_df["MSI_status"].values,
            "score": test_scores,
            "predicted": np.where(test_scores >= exp2["threshold"], "MSI-H", "MSS"),
        })
        pred_df2["correct"] = pred_df2["MSI_status"] == pred_df2["predicted"]
        _save_predictions(out_dir, route_name, "v4self", pred_df2)

        m_v4test = calc_metrics(
            test_df["MSI_status"].values, test_scores, exp2["threshold"],
        )

        results["v4self"] = {
            "v4test": m_v4test,
            "features": exp2["selected_cols"],
            "threshold": exp2["threshold"],
        }

    except Exception as e:
        route_logger.error(f"Experiment 2 failed: {e}", exc_info=True)
        results["v4self"] = {"error": str(e)}

    # Cleanup
    logging.getLogger().removeHandler(fh)
    fh.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4 MSI experiments")
    parser.add_argument("--routes", nargs="*", default=None,
                        help="Route names to evaluate (default: from config)")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--v3-data", default=None, help="v3 data path")
    parser.add_argument("--v4-data", default=None, help="v4 data path")
    parser.add_argument("--config", default=None, help="Config JSON path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.config:
        CFG.config = args.config
    if args.v3_data:
        CFG.v3_data = args.v3_data
    if args.v4_data:
        CFG.v4_data = args.v4_data
    if args.out_dir:
        CFG.out_dir = args.out_dir

    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    routes = args.routes if args.routes else CFG.default_routes

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load data
    logger.info("Loading data...")
    v3_meta = load_v3_data(CFG.v3_data)
    v4_meta = load_v4_data(CFG.v4_data)
    logger.info(f"v3: {len(v3_meta)} samples, v4: {len(v4_meta)} samples")
    logger.info(f"Routes to evaluate: {len(routes)}")

    # Run routes
    all_results: Dict[str, Dict] = {}
    for i, route_name in enumerate(routes):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[{i + 1}/{len(routes)}] Running route: {route_name}")
        logger.info(f"{'=' * 60}")
        all_results[route_name] = run_one_route(route_name, v3_meta, v4_meta, out_dir)

    # Collect + save summary
    summary_df = collect_summary(all_results)
    summary_path = out_dir / "experiment_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    # Plot ROC
    plot_combined_roc(all_results, out_dir / "roc_all_routes.png")

    # Plot score distributions + individual ROC
    for route_name in routes:
        plot_score_distribution(route_name, summary_df, out_dir)
        if route_name in all_results:
            plot_individual_roc(route_name, all_results[route_name], out_dir)

    # Print summary
    print_summary_table(summary_df, len(routes))
    print(f"  Output: {out_dir}")
    print(f"    experiment_summary.tsv")
    print(f"    roc_all_routes.png")
    print(f"    roc_individual/<route>.png")
    print(f"    score_distributions/<route>_score_dist.png")
    print(f"    predictions/<route>_*.tsv")
    print(f"    features/<route>*.txt")
    print(f"    config/<route>.json")
    print(f"    logs/<route>.log")
    print()


if __name__ == "__main__":
    main()

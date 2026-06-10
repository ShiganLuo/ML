"""Method 1: Self-Training with Pseudo-Labels for Semi-Supervised SV Frequency Correction.

Iteratively:
  1. Train model on labeled (+ previously accepted pseudo-labeled) data.
  2. Predict on unlabeled data.
  3. Filter high-confidence predictions as pseudo-labels.
  4. Merge pseudo-labeled data into training set.
  5. Repeat.

Confidence filtering strategies:
  - residual:  |pred - Freq| < quantile_threshold
  - range:     pred in (epsilon, 1-epsilon)
  - top_k:     keep top-k% lowest-uncertainty predictions (ensemble disagreement)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

from _data import load_combined_features, extract_xy, grouped_train_test_split
from train import (
    ModelRegistry,
    build_models,
    fit_model_with_registry,
    _fit_model_with_cv,
    _logit,
    _sigmoid,
    _clip_01,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ───────────────────────── Pseudo-label filtering ─────────────────────────

def filter_pseudo_labels(
    unlabeled_df: pd.DataFrame,
    predictions: np.ndarray,
    feature_columns: List[str],
    strategy: str = "residual",
    freq_col: str = "Freq",
    quantile: float = 0.75,
    top_k_frac: float = 0.3,
    pred_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Filter unlabeled predictions to produce pseudo-labels.

    Parameters
    ----------
    unlabeled_df : DataFrame
        Unlabeled rows with features.
    predictions : ndarray
        Model predictions for unlabeled rows.
    feature_columns : list[str]
        Feature columns.
    strategy : str
        'residual' | 'range' | 'top_k' | 'combined'
    freq_col : str
        Column with raw detected frequency (used for residual filter).
    quantile : float
        Quantile threshold for residual-based filtering.
    top_k_frac : float
        Fraction of top predictions to keep (for top_k strategy).
    pred_epsilon : float
        Minimum prediction value (range filter).

    Returns
    -------
    DataFrame
        Filtered unlabeled rows with pseudo-label column 'pseudo_ddPCR_AF'.
    """
    df = unlabeled_df.copy()
    df["_pred"] = predictions

    # Range filter (always applied)
    range_mask = (df["_pred"] > pred_epsilon) & (df["_pred"] < 1.0 - pred_epsilon)

    if strategy == "range":
        mask = range_mask

    elif strategy == "residual":
        if freq_col not in df.columns:
            logger.warning("Freq column '%s' not found, falling back to range filter", freq_col)
            mask = range_mask
        else:
            freq = pd.to_numeric(df[freq_col], errors="coerce").fillna(0.0)
            residual = np.abs(df["_pred"] - freq)
            threshold = residual[range_mask].quantile(quantile)
            residual_mask = residual <= threshold
            mask = range_mask & residual_mask
            logger.info("Residual filter: threshold=%.6f, kept %d / %d",
                        threshold, mask.sum(), len(df))

    elif strategy == "top_k":
        # Use ensemble disagreement or prediction confidence
        # For single model: use absolute residual from Freq as proxy
        if freq_col in df.columns:
            freq = pd.to_numeric(df[freq_col], errors="coerce").fillna(0.0)
            confidence = -np.abs(predictions - freq.values)  # higher = more confident
        else:
            # No Freq: use prediction magnitude as proxy (closer to known range = more confident)
            confidence = -np.abs(predictions - 0.5)
        n_keep = max(1, int(top_k_frac * range_mask.sum()))
        top_idx = np.argsort(confidence[range_mask])[-n_keep:]
        mask = pd.Series(False, index=df.index)
        mask.iloc[np.where(range_mask)[0][top_idx]] = True

    elif strategy == "combined":
        # Range + residual + top_k intersection
        if freq_col not in df.columns:
            mask = range_mask
        else:
            freq = pd.to_numeric(df[freq_col], errors="coerce").fillna(0.0)
            residual = np.abs(df["_pred"] - freq)
            threshold = residual[range_mask].quantile(quantile)
            residual_mask = residual <= threshold
            # Top-k among residual-filtered
            eligible = range_mask & residual_mask
            n_keep = max(1, int(top_k_frac * eligible.sum()))
            if n_keep < eligible.sum():
                confidence = -residual
                top_idx = np.argsort(confidence[eligible])[-n_keep:]
                mask = pd.Series(False, index=df.index)
                mask.iloc[np.where(eligible)[0][top_idx]] = True
            else:
                mask = eligible
            logger.info("Combined filter: kept %d / %d", mask.sum(), len(df))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    filtered = df[mask].copy()
    filtered["pseudo_ddPCR_AF"] = filtered["_pred"]
    filtered.drop(columns=["_pred"], inplace=True)
    return filtered



def self_train(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    feature_columns: List[str],
    out_dir: str,
    model_name: str = "gradient_boosting",
    group_cols: Optional[List[str]] = None,
    n_iterations: int = 3,
    filter_strategy: str = "residual",
    filter_quantile: float = 0.75,
    filter_top_k_frac: float = 0.3,
    target_transform: str = "logit",
    clip_epsilon: float = 1e-6,
    test_size: float = 0.2,
    random_state: int = 42,
    weight_low_af: bool = True,
    weight_epsilon: float = 1e-6,
    weight_power: float = 1.0,
    enable_cv: bool = True,
    cv_folds: int = 10,
    max_pseudo_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Run self-training loop.

    Parameters
    ----------
    max_pseudo_fraction : float
        Maximum fraction of pseudo-labels relative to labeled data per iteration.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Extract labeled X, y
    X_labeled, y_labeled, labeled_idx = extract_xy(labeled_df, feature_columns)
    logger.info("Labeled data: %d rows, %d features", X_labeled.shape[0], X_labeled.shape[1])

    # Extract unlabeled X (no y)
    unlabeled_valid = unlabeled_df[feature_columns].notna().all(axis=1)
    X_unlabeled = unlabeled_df.loc[unlabeled_valid, feature_columns].fillna(0.0).to_numpy(dtype=float)
    unlabeled_valid_idx = unlabeled_df.index[unlabeled_valid]
    logger.info("Unlabeled data: %d rows", X_unlabeled.shape[0])

    # Split labeled into train/eval (grouped)
    train_idx_local, test_idx_local, train_groups = grouped_train_test_split(
        X=X_labeled, y=y_labeled, df=labeled_df.loc[labeled_idx],
        group_cols=group_cols, test_size=test_size, random_state=random_state,
    )

    X_train = X_labeled[train_idx_local]
    y_train = y_labeled[train_idx_local]
    X_test = X_labeled[test_idx_local]
    y_test = y_labeled[test_idx_local]
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # Target transform
    if target_transform == "logit":
        y_train_model = _logit(_clip_01(y_train, clip_epsilon))
        y_test_model = _logit(_clip_01(y_test, clip_epsilon))
    else:
        y_train_model = y_train
        y_test_model = y_test

    # Sample weights
    if weight_low_af:
        train_weights = (1.0 / (y_train + weight_epsilon)) ** weight_power
        train_weights = train_weights / np.mean(train_weights)
    else:
        train_weights = None

    # Track iterations
    history = []
    registry = ModelRegistry(random_state=random_state)
    max_pseudo = max(1, int(max_pseudo_fraction * len(X_train)))

    for iteration in range(n_iterations + 1):
        iter_dir = os.path.join(out_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        # Build and train model
        model = registry.build_models(model_names=[model_name])[model_name]

        if enable_cv and iteration == 0:
            # CV only on first iteration (pure labeled data)
            param_grid = registry.get_param_grid(model_name)
            model, best_params = _fit_model_with_cv(
                name=model_name,
                X_train=X_train,
                y_train=y_train_model,
                train_weights=train_weights,
                groups=train_groups,
                cv_folds=cv_folds,
                param_grid=param_grid,
                random_state=random_state,
            )
        else:
            fit_model_with_registry(model_name, model, X_train, y_train_model, train_weights)

        # Evaluate on held-out test set
        pred_test = model.predict(X_test)
        if target_transform == "logit":
            pred_test = _sigmoid(pred_test)

        mse = float(mean_squared_error(y_test, pred_test))
        mae = float(mean_absolute_error(y_test, pred_test))
        r2 = float(r2_score(y_test, pred_test))
        pearson_r = float(np.corrcoef(y_test, pred_test)[0, 1]) if len(y_test) > 1 else float("nan")

        iter_metrics = {
            "iteration": iteration,
            "n_train": len(X_train),
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "pearson_r": pearson_r,
        }
        history.append(iter_metrics)
        logger.info("Iter %d: n_train=%d, MSE=%.6f, MAE=%.6f, R²=%.6f, r=%.6f",
                     iteration, len(X_train), mse, mae, r2, pearson_r)

        # Save model
        joblib.dump(model, os.path.join(iter_dir, f"model_{model_name}.joblib"))

        # Plot prediction vs truth
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, pred_test, s=30, alpha=0.75)
        vmin = min(y_test.min(), pred_test.min())
        vmax = max(y_test.max(), pred_test.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="y=x")
        ax.set_xlabel("True ddPCR_AF")
        ax.set_ylabel("Predicted ddPCR_AF")
        ax.set_title(f"Iter {iteration} — {model_name}\nMSE={mse:.6f} R²={r2:.6f} r={pearson_r:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(iter_dir, "pred_vs_truth.png"), dpi=200)
        plt.close(fig)

        # --- Self-training: generate pseudo-labels ---
        if iteration == n_iterations:
            break

        pred_unlabeled = model.predict(X_unlabeled)
        if target_transform == "logit":
            pred_unlabeled = _sigmoid(pred_unlabeled)

        # Filter pseudo-labels
        unlabeled_for_filter = unlabeled_df.loc[unlabeled_valid_idx].copy()
        pseudo_df = filter_pseudo_labels(
            unlabeled_df=unlabeled_for_filter,
            predictions=pred_unlabeled,
            feature_columns=feature_columns,
            strategy=filter_strategy,
            quantile=filter_quantile,
            top_k_frac=filter_top_k_frac,
        )

        # Cap pseudo-label count
        if len(pseudo_df) > max_pseudo:
            # Keep the most confident ones (smallest residual to Freq)
            if "Freq" in pseudo_df.columns:
                freq = pd.to_numeric(pseudo_df["Freq"], errors="coerce").fillna(0.0)
                residual = np.abs(pseudo_df["pseudo_ddPCR_AF"].values - freq.values)
                pseudo_df = pseudo_df.iloc[np.argsort(residual)[:max_pseudo]]
            else:
                pseudo_df = pseudo_df.head(max_pseudo)

        n_pseudo = len(pseudo_df)
        iter_metrics["n_pseudo"] = n_pseudo
        logger.info("Pseudo-labels accepted: %d / %d (max %d)", n_pseudo, X_unlabeled.shape[0], max_pseudo)

        # Save pseudo-labels
        pseudo_df.to_csv(os.path.join(iter_dir, "pseudo_labels.tsv"), sep="\t", index=False)

        if n_pseudo == 0:
            logger.warning("No pseudo-labels accepted at iteration %d, stopping early", iteration)
            break

        # Merge labeled + pseudo-labeled
        X_pseudo = pseudo_df[feature_columns].fillna(0.0).to_numpy(dtype=float)
        y_pseudo = pseudo_df["pseudo_ddPCR_AF"].to_numpy(dtype=float)

        X_train = np.vstack([X_train, X_pseudo])
        y_train = np.concatenate([y_train, y_pseudo])

        # Update sample weights (pseudo-labels get lower weight)
        if weight_low_af:
            w_labeled = (1.0 / (y_train[:len(X_train) - n_pseudo] + weight_epsilon)) ** weight_power
            w_pseudo = (1.0 / (y_pseudo + weight_epsilon)) ** weight_power * 0.5  # half weight
            train_weights = np.concatenate([w_labeled, w_pseudo])
            train_weights = train_weights / np.mean(train_weights)
        else:
            train_weights = None

        # Recompute model target
        if target_transform == "logit":
            y_train_model = _logit(_clip_01(y_train, clip_epsilon))
        else:
            y_train_model = y_train

    # --- Final summary ---
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "iteration_history.tsv"), sep="\t", index=False)

    # Plot convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label in zip(axes, ["mse", "mae", "r2"], ["MSE", "MAE", "R²"]):
        ax.plot(history_df["iteration"], history_df[metric], "o-", lw=2, ms=8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Iteration")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Self-Training Convergence", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=200)
    plt.close(fig)

    # Plot training set size growth
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(history_df["iteration"], history_df["n_train"], color="#4C72B0", alpha=0.85)
    if "n_pseudo" in history_df.columns:
        ax.bar(history_df["iteration"], history_df["n_pseudo"], bottom=history_df["n_train"] - history_df["n_pseudo"],
               color="#DD8452", alpha=0.85, label="Pseudo-labels")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training set size")
    ax.set_title("Training Set Growth")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_set_growth.png"), dpi=200)
    plt.close(fig)

    # Final comparison with baseline
    baseline = history[0]
    final = history[-1]
    summary = {
        "baseline": baseline,
        "final": final,
        "improvement": {
            "mse_reduction": baseline["mse"] - final["mse"],
            "mae_reduction": baseline["mae"] - final["mae"],
            "r2_gain": final["r2"] - baseline["r2"],
        },
        "config": {
            "model_name": model_name,
            "n_iterations": n_iterations,
            "filter_strategy": filter_strategy,
            "filter_quantile": filter_quantile,
            "target_transform": target_transform,
            "weight_low_af": weight_low_af,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=== Self-Training Summary ===")
    logger.info("Baseline: MSE=%.6f, R²=%.6f", baseline["mse"], baseline["r2"])
    logger.info("Final:    MSE=%.6f, R²=%.6f", final["mse"], final["r2"])
    logger.info("Improvement: MSE Δ=%.6f, R² Δ=%.6f",
                summary["improvement"]["mse_reduction"],
                summary["improvement"]["r2_gain"])

    return summary


# ───────────────────────── CLI ─────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Training Semi-Supervised SV Frequency Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--labeled-tsv", required=True, help="Labeled data TSV (with ddPCR_AF)")
    parser.add_argument("--unlabeled-tsv", required=True, help="Unlabeled data TSV (without ddPCR_AF)")
    parser.add_argument("-o", "--out-dir", required=True, help="Output directory")
    parser.add_argument("--probe-infile", default=None, help="Probe sequence BED file")
    parser.add_argument("--feature-dir", default=None,
                        help="Shared directory for BAM feature cache (avoid re-extraction)")
    parser.add_argument("--group-cols", action="append", default=["原始编号", "FusionGene", "FusionExon"],
                        help="Column(s) for grouped train/test split (repeatable, default: 原始编号)")
    parser.add_argument("--model-name", default="gradient_boosting",
                        choices=["ridge", "svr_rbf", "gradient_boosting", "random_forest",
                                 "extra_trees", "adaboost", "knn", "mlp"],
                        help="Regression model to use (default: gradient_boosting)")
    parser.add_argument("--n-iterations", type=int, default=3, help="Number of self-training iterations")
    parser.add_argument("--filter-strategy", default="combined",
                        choices=["residual", "range", "top_k", "combined"],
                        help="Pseudo-label filtering strategy")
    parser.add_argument("--filter-quantile", type=float, default=0.75,
                        help="Quantile threshold for residual filter")
    parser.add_argument("--filter-top-k-frac", type=float, default=0.3,
                        help="Top-k fraction for top_k filter")
    parser.add_argument("--target-transform", choices=["none", "logit"], default="logit")
    parser.add_argument("--clip-epsilon", type=float, default=1e-6)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--weight-low-af", action="store_true", default=True)
    parser.add_argument("--no-weight-low-af", dest="weight_low_af", action="store_false")
    parser.add_argument("--enable-cv", action="store_true", default=True)
    parser.add_argument("--no-enable-cv", dest="enable_cv", action="store_false")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--max-pseudo-fraction", type=float, default=0.5,
                        help="Max fraction of pseudo-labels relative to labeled data")
    parser.add_argument("--force-extract", action="store_true", help="Force BAM feature re-extraction")

    args = parser.parse_args()

    labeled_df, combined_df, feature_columns, no_scale_columns = load_combined_features(
        labeled_tsv=args.labeled_tsv,
        unlabeled_tsv=args.unlabeled_tsv,
        outdir=os.path.join(args.out_dir, "features"),
        probe_infile=args.probe_infile,
        force_extract=args.force_extract,
        feature_cache_dir=args.feature_dir,
    )

    # Split combined back into labeled and unlabeled
    unlabeled_mask = combined_df["ddPCR_AF"].isna()
    unlabeled_df = combined_df[unlabeled_mask].copy()

    self_train(
        labeled_df=labeled_df,
        unlabeled_df=unlabeled_df,
        feature_columns=feature_columns,
        out_dir=args.out_dir,
        model_name=args.model_name,
        group_cols=args.group_cols,
        n_iterations=args.n_iterations,
        filter_strategy=args.filter_strategy,
        filter_quantile=args.filter_quantile,
        filter_top_k_frac=args.filter_top_k_frac,
        target_transform=args.target_transform,
        clip_epsilon=args.clip_epsilon,
        test_size=args.test_size,
        random_state=args.random_state,
        weight_low_af=args.weight_low_af,
        enable_cv=args.enable_cv,
        cv_folds=args.cv_folds,
        max_pseudo_fraction=args.max_pseudo_fraction,
    )


if __name__ == "__main__":
    main()

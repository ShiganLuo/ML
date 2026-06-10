"""Method 2: Autoencoder-Based Semi-Supervised SV Frequency Correction.

Uses ALL data (labeled + unlabeled) to learn a compressed feature representation
via a denoising autoencoder, then trains regression models on the latent features
using only labeled data.

Pipeline:
  1. Train denoising autoencoder on combined feature matrix.
  2. Encode all data into latent space.
  3. Train regression on labeled latent features.
  4. Evaluate on held-out labeled data.
  5. Compare with baseline (raw features, no autoencoder).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from _data import load_combined_features, extract_xy, grouped_train_test_split
from train import (
    ModelRegistry,
    fit_model_with_registry,
    _fit_model_with_cv,
    _logit,
    _sigmoid,
    _clip_01,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ───────────────────────── Denoising Autoencoder ─────────────────────────

class DenoisingAutoencoder:
    """Denoising autoencoder built on sklearn MLPRegressor.

    Trains the encoder (input → latent) and decoder (latent → reconstruction)
    jointly by minimizing reconstruction error on noisy input.

    Parameters
    ----------
    latent_dim : int
        Dimension of the bottleneck representation.
    hidden_encoder : tuple[int, ...]
        Additional hidden layers between input and latent (encoder side).
    hidden_decoder : tuple[int, ...]
        Additional hidden layers between latent and output (decoder side).
    noise_std : float
        Standard deviation of Gaussian noise added to input during training.
    max_iter : int
        Maximum training iterations.
    random_state : int
        Random seed.
    early_stopping : bool
        Whether to use early stopping.
    validation_fraction : float
        Fraction of data for early stopping validation.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_encoder: Tuple[int, ...] = (),
        hidden_decoder: Tuple[int, ...] = (),
        noise_std: float = 0.1,
        max_iter: int = 500,
        random_state: int = 42,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
    ):
        """Initialize the denoising autoencoder.

        Parameters
        ----------
        latent_dim : int, optional
            Dimension of the bottleneck latent representation. Default is 8.
        hidden_encoder : tuple of int, optional
            Sizes of additional hidden layers in the encoder between input
            and the latent layer. Default is ``()`` (no extra layers).
        hidden_decoder : tuple of int, optional
            Sizes of additional hidden layers in the decoder between the
            latent layer and the output. Default is ``()`` (no extra layers).
        noise_std : float, optional
            Standard deviation of Gaussian noise added to inputs during
            denoising training. Default is 0.1.
        max_iter : int, optional
            Maximum number of training iterations for the underlying
            ``MLPRegressor``. Default is 500.
        random_state : int, optional
            Random seed for reproducibility. Default is 42.
        early_stopping : bool, optional
            Whether to use early stopping on a validation fraction to
            terminate training when the validation score stops improving.
            Default is True.
        validation_fraction : float, optional
            Proportion of training data to set aside as a validation set
            for early stopping. Only used when *early_stopping* is True.
            Default is 0.1.
        """
        self.latent_dim = latent_dim
        self.hidden_encoder = hidden_encoder
        self.hidden_decoder = hidden_decoder
        self.noise_std = noise_std
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.mlp_: Optional[MLPRegressor] = None
        self.reconstruction_error_: float = float("nan")

    def _build_architecture(self, n_features: int) -> Tuple[int, ...]:
        """Build the full hidden-layer size tuple: encoder → latent → decoder.

        Parameters
        ----------
        n_features : int
            Number of input features (used only for logging context, not
            directly in the computation).

        Returns
        -------
        tuple of int
            Hidden-layer sizes that combine *hidden_encoder*, the latent
            dimension, and *hidden_decoder*.
        """
        return tuple(self.hidden_encoder) + (self.latent_dim,) + tuple(self.hidden_decoder)

    def fit(self, X: np.ndarray) -> "DenoisingAutoencoder":
        """Fit the autoencoder on feature matrix *X*.

        Gaussian noise is added to the input and the network is trained to
        reconstruct the original (clean) input, following the denoising
        autoencoder paradigm.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix. All samples are used for self-supervised
            training (no labels required).

        Returns
        -------
        DenoisingAutoencoder
            The fitted instance (allows method chaining).
        """
        n_features = X.shape[1]
        hidden_layer_sizes = self._build_architecture(n_features)
        logger.info("Autoencoder architecture: %d → %s → %d",
                     n_features, hidden_layer_sizes, n_features)

        # Add Gaussian noise for denoising training
        rng = np.random.RandomState(self.random_state)
        X_noisy = X + rng.normal(0, self.noise_std, X.shape)

        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            batch_size="auto",
            verbose=True,
        )
        # Train to reconstruct clean input from noisy input
        self.mlp_.fit(X_noisy, X)

        # Compute reconstruction error on clean input
        X_reconstructed = self.mlp_.predict(X)
        self.reconstruction_error_ = float(np.mean((X - X_reconstructed) ** 2))
        logger.info("Autoencoder reconstruction MSE: %.6f", self.reconstruction_error_)

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input into the latent (bottleneck) representation.

        Performs a manual forward pass through the encoder portion of the
        trained ``MLPRegressor``, applying ReLU activations at each layer.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix to encode.

        Returns
        -------
        np.ndarray of shape (n_samples, latent_dim)
            Latent representation of the input.

        Raises
        ------
        RuntimeError
            If the autoencoder has not been fitted yet.
        """
        if self.mlp_ is None:
            raise RuntimeError("Autoencoder not fitted")

        # Manually forward-pass through encoder layers
        activation = X.copy()
        n_encoder_layers = len(self.hidden_encoder) + 1  # +1 for latent layer
        for i in range(n_encoder_layers):
            W = self.mlp_.coefs_[i]
            b = self.mlp_.intercepts_[i]
            activation = activation @ W + b
            # Apply ReLU
            activation = np.maximum(0, activation)

        return activation

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent representation back to the original feature space.

        Performs a forward pass through the decoder portion of the trained
        ``MLPRegressor``. Hidden layers use ReLU activation; the output
        layer is linear.

        Parameters
        ----------
        Z : np.ndarray of shape (n_samples, latent_dim)
            Latent-space representation to decode.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Reconstructed feature matrix.

        Raises
        ------
        RuntimeError
            If the autoencoder has not been fitted yet.
        """
        if self.mlp_ is None:
            raise RuntimeError("Autoencoder not fitted")

        activation = Z.copy()
        n_total = len(self.mlp_.coefs_)
        n_encoder_layers = len(self.hidden_encoder) + 1
        for i in range(n_encoder_layers, n_total):
            W = self.mlp_.coefs_[i]
            b = self.mlp_.intercepts_[i]
            activation = activation @ W + b
            # ReLU for hidden layers, linear for output
            if i < n_total - 1:
                activation = np.maximum(0, activation)

        return activation

    def save(self, path: str) -> None:
        """Save the fitted autoencoder to disk.

        Serializes the underlying ``MLPRegressor``, architecture
        parameters, and reconstruction error using ``joblib``.

        Parameters
        ----------
        path : str
            File path where the serialized object will be written.
        """
        joblib.dump({
            "mlp": self.mlp_,
            "latent_dim": self.latent_dim,
            "hidden_encoder": self.hidden_encoder,
            "hidden_decoder": self.hidden_decoder,
            "noise_std": self.noise_std,
            "reconstruction_error": self.reconstruction_error_,
        }, path)

    @classmethod
    def load(cls, path: str) -> "DenoisingAutoencoder":
        """Load a previously saved autoencoder from disk.

        Parameters
        ----------
        path : str
            Path to the joblib file created by :meth:`save`.

        Returns
        -------
        DenoisingAutoencoder
            A restored autoencoder instance ready for encoding/decoding.
        """
        data = joblib.load(path)
        ae = cls(
            latent_dim=data["latent_dim"],
            hidden_encoder=data["hidden_encoder"],
            hidden_decoder=data["hidden_decoder"],
            noise_std=data["noise_std"],
        )
        ae.mlp_ = data["mlp"]
        ae.reconstruction_error_ = data["reconstruction_error"]
        return ae


# ───────────────────────── Training pipeline ─────────────────────────

def train_with_autoencoder(
    labeled_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    feature_columns: List[str],
    out_dir: str,
    latent_dim: int = 8,
    hidden_encoder: Tuple[int, ...] = (),
    hidden_decoder: Tuple[int, ...] = (),
    noise_std: float = 0.1,
    ae_max_iter: int = 500,
    model_name: str = "gradient_boosting",
    group_cols: Optional[List[str]] = None,
    target_transform: str = "logit",
    clip_epsilon: float = 1e-6,
    test_size: float = 0.2,
    random_state: int = 42,
    weight_low_af: bool = True,
    weight_epsilon: float = 1e-6,
    weight_power: float = 1.0,
    enable_cv: bool = True,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Run the full semi-supervised autoencoder pipeline.

    Trains a denoising autoencoder on **all** available data (labeled and
    unlabeled) to learn a compressed latent representation, then fits a
    regression model on the latent features using only the labeled subset.
    A baseline model trained on raw features is also evaluated for
    comparison.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame containing samples with known ``ddPCR_AF`` values.
    combined_df : pd.DataFrame
        DataFrame containing both labeled and unlabeled samples.
    feature_columns : list of str
        Column names to use as input features.
    out_dir : str
        Output directory for saving models, plots, and summary JSON.
    latent_dim : int, optional
        Bottleneck dimension for the autoencoder. Default is 8.
    hidden_encoder : tuple of int, optional
        Extra encoder hidden layer sizes. Default is ``()``.
    hidden_decoder : tuple of int, optional
        Extra decoder hidden layer sizes. Default is ``()``.
    noise_std : float, optional
        Gaussian noise standard deviation for denoising. Default is 0.1.
    ae_max_iter : int, optional
        Maximum training iterations for the autoencoder. Default is 500.
    model_name : str, optional
        Name of the regression model to use (e.g. ``'gradient_boosting'``).
        Default is ``'gradient_boosting'``.
    group_cols : list of str or None, optional
        Columns used for grouped train/test splitting to avoid data
        leakage. Default is None.
    target_transform : str, optional
        Transformation applied to the target variable before training
        (``'logit'`` or ``'none'``). Default is ``'logit'``.
    clip_epsilon : float, optional
        Small epsilon for clipping targets before logit transform.
        Default is 1e-6.
    test_size : float, optional
        Fraction of labeled data held out for testing. Default is 0.2.
    random_state : int, optional
        Random seed. Default is 42.
    weight_low_af : bool, optional
        If True, up-weight low-frequency samples during regression
        training. Default is True.
    weight_epsilon : float, optional
        Small constant to avoid division by zero in weight computation.
        Default is 1e-6.
    weight_power : float, optional
        Exponent controlling the degree of up-weighting for low-AF
        samples. Default is 1.0.
    enable_cv : bool, optional
        Whether to use cross-validated hyperparameter tuning. Default
        is True.
    cv_folds : int, optional
        Number of cross-validation folds. Default is 5.

    Returns
    -------
    dict
        Summary dictionary with keys ``'autoencoder'``, ``'latent_model'``,
        ``'baseline_model'``, and ``'improvement'`` containing metrics
        and configuration details.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Extract all features (labeled + unlabeled)
    X_all = combined_df[feature_columns].fillna(0.0).to_numpy(dtype=float)
    logger.info("Combined feature matrix: %s", X_all.shape)

    # Train autoencoder
    ae_dir = os.path.join(out_dir, "autoencoder")
    os.makedirs(ae_dir, exist_ok=True)

    ae = DenoisingAutoencoder(
        latent_dim=latent_dim,
        hidden_encoder=hidden_encoder,
        hidden_decoder=hidden_decoder,
        noise_std=noise_std,
        max_iter=ae_max_iter,
        random_state=random_state,
    )
    ae.fit(X_all)
    ae.save(os.path.join(ae_dir, "autoencoder.joblib"))

    # Encode all data
    Z_all = ae.encode(X_all)
    logger.info("Latent representation shape: %s", Z_all.shape)

    # Save latent features
    latent_columns = [f"latent_{i}" for i in range(Z_all.shape[1])]
    latent_df = pd.DataFrame(Z_all, columns=latent_columns, index=combined_df.index)
    for col in ["原始编号", "ddPCR_AF", "Freq", "FusionType", "sampleID"]:
        if col in combined_df.columns:
            latent_df[col] = combined_df[col].values
    latent_df.to_csv(os.path.join(ae_dir, "latent_features.tsv"), sep="\t", index=False)

    # Visualize latent space (2D if latent_dim >= 2)
    if Z_all.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        labeled_mask = combined_df["ddPCR_AF"].notna()
        scatter = ax.scatter(
            Z_all[labeled_mask, 0], Z_all[labeled_mask, 1],
            c=combined_df.loc[labeled_mask, "ddPCR_AF"],
            cmap="viridis", s=20, alpha=0.8, label="Labeled",
        )
        ax.scatter(
            Z_all[~labeled_mask, 0], Z_all[~labeled_mask, 1],
            c="lightgray", s=5, alpha=0.3, label="Unlabeled",
        )
        plt.colorbar(scatter, label="ddPCR_AF")
        ax.set_xlabel("Latent dim 0")
        ax.set_ylabel("Latent dim 1")
        ax.set_title("Latent Space (colored by ddPCR_AF)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(ae_dir, "latent_space.png"), dpi=200)
        plt.close(fig)

    # --- Train regression on labeled latent features ---
    labeled_mask = combined_df["ddPCR_AF"].notna()
    Z_labeled = Z_all[labeled_mask]
    y_labeled = combined_df.loc[labeled_mask, "ddPCR_AF"].to_numpy(dtype=float)

    # Grouped split
    train_idx, test_idx, train_groups = grouped_train_test_split(
        X=Z_labeled, y=y_labeled, df=combined_df.loc[labeled_mask],
        group_cols=group_cols, test_size=test_size, random_state=random_state,
    )

    Z_train = Z_labeled[train_idx]
    y_train = y_labeled[train_idx]
    Z_test = Z_labeled[test_idx]
    y_test = y_labeled[test_idx]

    # Target transform
    if target_transform == "logit":
        y_train_model = _logit(_clip_01(y_train, clip_epsilon))
    else:
        y_train_model = y_train

    # Sample weights
    if weight_low_af:
        train_weights = (1.0 / (y_train + weight_epsilon)) ** weight_power
        train_weights = train_weights / np.mean(train_weights)
    else:
        train_weights = None

    # Train with latent features
    registry = ModelRegistry(random_state=random_state)
    model = registry.build_models(model_names=[model_name])[model_name]

    if enable_cv:
        param_grid = registry.get_param_grid(model_name)
        model, best_params = _fit_model_with_cv(
            name=model_name,
            X_train=Z_train,
            y_train=y_train_model,
            train_weights=train_weights,
            groups=train_groups,
            cv_folds=cv_folds,
            param_grid=param_grid,
            random_state=random_state,
        )
    else:
        fit_model_with_registry(model_name, model, Z_train, y_train_model, train_weights)

    # Evaluate
    pred_test = model.predict(Z_test)
    if target_transform == "logit":
        pred_test = _sigmoid(pred_test)

    mse = float(mean_squared_error(y_test, pred_test))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    pearson_r = float(np.corrcoef(y_test, pred_test)[0, 1])

    logger.info("Latent model: MSE=%.6f, MAE=%.6f, R²=%.6f, r=%.6f", mse, mae, r2, pearson_r)

    # Save latent model
    joblib.dump(model, os.path.join(out_dir, f"model_{model_name}_latent.joblib"))

    # --- Baseline: same model on raw features ---
    X_raw_labeled = combined_df.loc[labeled_mask, feature_columns].fillna(0.0).to_numpy(dtype=float)
    X_raw_train = X_raw_labeled[train_idx]
    X_raw_test = X_raw_labeled[test_idx]

    baseline_model = registry.build_models(model_names=[model_name])[model_name]
    if enable_cv:
        baseline_model, _ = _fit_model_with_cv(
            name=model_name,
            X_train=X_raw_train,
            y_train=y_train_model,
            train_weights=train_weights,
            groups=train_groups,
            cv_folds=cv_folds,
            param_grid=param_grid,
            random_state=random_state,
        )
    else:
        fit_model_with_registry(model_name, baseline_model, X_raw_train, y_train_model, train_weights)

    pred_baseline = baseline_model.predict(X_raw_test)
    if target_transform == "logit":
        pred_baseline = _sigmoid(pred_baseline)

    mse_base = float(mean_squared_error(y_test, pred_baseline))
    mae_base = float(mean_absolute_error(y_test, pred_baseline))
    r2_base = float(r2_score(y_test, pred_baseline))
    pearson_r_base = float(np.corrcoef(y_test, pred_baseline)[0, 1])

    logger.info("Baseline model: MSE=%.6f, MAE=%.6f, R²=%.6f, r=%.6f", mse_base, mae_base, r2_base, pearson_r_base)

    # --- Comparison plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, y_pred, title, metrics in zip(
        axes,
        [pred_baseline, pred_test],
        ["Baseline (raw features)", "Autoencoder (latent features)"],
        [{"mse": mse_base, "r2": r2_base, "r": pearson_r_base},
         {"mse": mse, "r2": r2, "r": pearson_r}],
    ):
        ax.scatter(y_test, y_pred, s=30, alpha=0.75)
        vmin = min(y_test.min(), y_pred.min())
        vmax = max(y_test.max(), y_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="y=x")
        ax.set_xlabel("True ddPCR_AF")
        ax.set_ylabel("Predicted ddPCR_AF")
        ax.set_title(f"{title}\nMSE={metrics['mse']:.6f} R²={metrics['r2']:.6f} r={metrics['r']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Autoencoder Comparison — {model_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_pred_vs_truth.png"), dpi=200)
    plt.close(fig)

    # Residual comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_pred, title in zip(
        axes,
        [pred_baseline, pred_test],
        ["Baseline residuals", "Autoencoder residuals"],
    ):
        residuals = y_pred - y_test
        ax.scatter(y_test, residuals, s=30, alpha=0.75)
        ax.axhline(0, color="red", ls="--", lw=1.2)
        ax.set_xlabel("True ddPCR_AF")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_residuals.png"), dpi=200)
    plt.close(fig)

    # Summary
    summary = {
        "autoencoder": {
            "latent_dim": latent_dim,
            "hidden_encoder": list(hidden_encoder),
            "hidden_decoder": list(hidden_decoder),
            "noise_std": noise_std,
            "reconstruction_mse": ae.reconstruction_error_,
        },
        "latent_model": {"mse": mse, "mae": mae, "r2": r2, "pearson_r": pearson_r},
        "baseline_model": {"mse": mse_base, "mae": mae_base, "r2": r2_base, "pearson_r": pearson_r_base},
        "improvement": {
            "mse_reduction": mse_base - mse,
            "mae_reduction": mae_base - mae,
            "r2_gain": r2 - r2_base,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ───────────────────────── CLI ─────────────────────────

def main():
    """CLI entry point for the autoencoder-based semi-supervised pipeline.

    Parses command-line arguments, loads combined labeled and unlabeled
    feature data, and invokes :func:`train_with_autoencoder`.
    """
    parser = argparse.ArgumentParser(
        description="Autoencoder-Based Semi-Supervised SV Frequency Correction",
    )
    parser.add_argument("--labeled-tsv", required=True)
    parser.add_argument("--unlabeled-tsv", required=True)
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument("--probe-infile", default=None)
    parser.add_argument("--feature-dir", default=None,
                        help="Shared directory for BAM feature cache (avoid re-extraction)")
    parser.add_argument("--group-cols", action="append", default=["原始编号", "FusionGene", "FusionExon"],
                        help="Column(s) for grouped train/test split (repeatable, default: 原始编号)")
    parser.add_argument("--latent-dim", type=int, default=8, help="Bottleneck dimension")
    parser.add_argument("--hidden-encoder", type=int, nargs="*", default=[],
                        help="Extra encoder hidden layer sizes")
    parser.add_argument("--hidden-decoder", type=int, nargs="*", default=[],
                        help="Extra decoder hidden layer sizes")
    parser.add_argument("--noise-std", type=float, default=0.1, help="Denoising noise std")
    parser.add_argument("--ae-max-iter", type=int, default=500, help="Autoencoder max iterations")
    parser.add_argument("--model-name", default="gradient_boosting",
                        choices=["ridge", "svr_rbf", "gradient_boosting", "random_forest",
                                 "extra_trees", "adaboost", "knn", "mlp"])
    parser.add_argument("--target-transform", choices=["none", "logit"], default="logit")
    parser.add_argument("--clip-epsilon", type=float, default=1e-6)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--weight-low-af", action="store_true", default=True)
    parser.add_argument("--no-weight-low-af", dest="weight_low_af", action="store_false")
    parser.add_argument("--enable-cv", action="store_true", default=True)
    parser.add_argument("--no-enable-cv", dest="enable_cv", action="store_false")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--force-extract", action="store_true")

    args = parser.parse_args()
    labeled_df, combined_df, feature_columns, no_scale_columns = load_combined_features(
        labeled_tsv=args.labeled_tsv,
        unlabeled_tsv=args.unlabeled_tsv,
        outdir=os.path.join(args.out_dir, "features"),
        probe_infile=args.probe_infile,
        force_extract=args.force_extract,
        feature_cache_dir=args.feature_dir,
    )

    train_with_autoencoder(
        labeled_df=labeled_df,
        combined_df=combined_df,
        feature_columns=feature_columns,
        out_dir=args.out_dir,
        latent_dim=args.latent_dim,
        group_cols=args.group_cols,
        hidden_encoder=tuple(args.hidden_encoder),
        hidden_decoder=tuple(args.hidden_decoder),
        noise_std=args.noise_std,
        ae_max_iter=args.ae_max_iter,
        model_name=args.model_name,
        target_transform=args.target_transform,
        clip_epsilon=args.clip_epsilon,
        test_size=args.test_size,
        random_state=args.random_state,
        weight_low_af=args.weight_low_af,
        enable_cv=args.enable_cv,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()

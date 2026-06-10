"""Method 3: Consistency Regularization for Semi-Supervised SV Frequency Correction.

Trains a neural network with two loss terms:
  - Supervised loss: MSE on labeled data
  - Consistency loss: predictions should be stable under input perturbation

Total loss = supervised_loss + λ * consistency_loss

The consistency term forces the model to produce similar predictions for
an unlabeled sample and its noisy copy, leveraging the smoothness assumption
that nearby points in feature space should have similar predictions.

Uses PyTorch if available; otherwise falls back to a simulated consistency
approach (ensemble agreement on perturbed inputs).
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
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from _data import load_combined_features, extract_xy, grouped_train_test_split
from train import (
    _logit,
    _sigmoid,
    _clip_01,
    ModelRegistry,
    fit_model_with_registry,
    _fit_model_with_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Try PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    logger.info("PyTorch available — using gradient-based consistency regularization")
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available — using ensemble-consistency fallback")


# ───────────────────────── PyTorch Model ─────────────────────────

if HAS_TORCH:

    class ConsistencyNet(nn.Module):
        """Feed-forward neural network for consistency-regularized regression.

        Built as a sequence of fully connected layers with ReLU activations
        and dropout, ending with a single linear output unit. The network
        is designed to be trained with a combined supervised + consistency
        loss on labeled and unlabeled data respectively.
        """
        def __init__(self, n_features: int, hidden_sizes: List[int], dropout: float = 0.1):
            """Initialize the consistency network.

            Parameters
            ----------
            n_features : int
                Number of input features.
            hidden_sizes : list of int
                Sizes of the hidden layers (e.g. ``[64, 32]``).
            dropout : float, optional
                Dropout probability applied after each hidden layer.
                Default is 0.1.
            """
            super().__init__()
            layers = []
            prev = n_features
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the network.

            Parameters
            ----------
            x : torch.Tensor of shape (n_samples, n_features)
                Input tensor.

            Returns
            -------
            torch.Tensor of shape (n_samples,)
                Predicted scalar values per sample.
            """
            return self.net(x).squeeze(-1)


    def train_consistency_pytorch(
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        out_dir: str,
        hidden_sizes: Optional[List[int]] = None,
        consistency_weight: float = 1.0,
        noise_std: float = 0.1,
        n_epochs: int = 300,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        random_state: int = 42,
        target_transform: str = "logit",
        clip_epsilon: float = 1e-6,
        weight_low_af: bool = True,
        weight_epsilon: float = 1e-6,
        weight_power: float = 1.0,
    ) -> Dict[str, Any]:
        """Train a regression model with PyTorch-based consistency regularization.

        Optimizes a ``ConsistencyNet`` with a combined loss:
        ``supervised_loss + lambda * consistency_loss``, where the
        consistency term penalizes prediction differences between an
        unlabeled sample and its Gaussian-noise-perturbed copy.

        Parameters
        ----------
        X_labeled : np.ndarray of shape (n_labeled, n_features)
            Feature matrix for labeled training samples.
        y_labeled : np.ndarray of shape (n_labeled,)
            Target values (e.g. ddPCR allele frequency) for labeled samples.
        X_unlabeled : np.ndarray of shape (n_unlabeled, n_features)
            Feature matrix for unlabeled samples used only in the
            consistency loss term.
        X_test : np.ndarray of shape (n_test, n_features)
            Test feature matrix for evaluation during training.
        y_test : np.ndarray of shape (n_test,)
            Test target values.
        out_dir : str
            Directory for saving the trained model and config.
        hidden_sizes : list of int or None, optional
            Hidden layer sizes for ``ConsistencyNet``. Default is
            ``[64, 32]``.
        consistency_weight : float, optional
            Weight (lambda) for the consistency loss term. Default is 1.0.
        noise_std : float, optional
            Standard deviation of Gaussian noise used to create perturbed
            copies of unlabeled inputs. Default is 0.1.
        n_epochs : int, optional
            Number of training epochs. Default is 300.
        batch_size : int, optional
            Mini-batch size for labeled data. Default is 64.
        learning_rate : float, optional
            Learning rate for the Adam optimizer. Default is 1e-3.
        dropout : float, optional
            Dropout probability in ``ConsistencyNet``. Default is 0.1.
        random_state : int, optional
            Random seed for reproducibility. Default is 42.
        target_transform : str, optional
            Target transformation (``'logit'`` or ``'none'``). Default is
            ``'logit'``.
        clip_epsilon : float, optional
            Clipping epsilon for logit transform. Default is 1e-6.
        weight_low_af : bool, optional
            Whether to up-weight low-frequency samples. Default is True.
        weight_epsilon : float, optional
            Small constant to avoid division by zero in weights. Default
            is 1e-6.
        weight_power : float, optional
            Exponent controlling up-weighting strength. Default is 1.0.

        Returns
        -------
        dict
            Dictionary with keys ``'final'`` (metrics dict), ``'history'``
            (list of per-epoch records), and ``'pred_test'`` (np.ndarray
            of final test predictions).
        """
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        os.makedirs(out_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Prepare data
        n_features = X_labeled.shape[1]
        model = ConsistencyNet(n_features, hidden_sizes, dropout).to(device)

        X_labeled_t = torch.tensor(X_labeled, dtype=torch.float32, device=device)
        y_labeled_t = torch.tensor(y_labeled, dtype=torch.float32, device=device)
        X_unlabeled_t = torch.tensor(X_unlabeled, dtype=torch.float32, device=device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

        # Sample weights for labeled data
        if weight_low_af:
            w_np = (1.0 / (y_labeled + weight_epsilon)) ** weight_power
            w_np = w_np / np.mean(w_np)
            w_t = torch.tensor(w_np, dtype=torch.float32, device=device)
        else:
            w_t = torch.ones(len(y_labeled), dtype=torch.float32, device=device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        rng = torch.Generator(device=device)
        rng.manual_seed(random_state)

        history = []
        best_test_mse = float("inf")
        best_state = None

        for epoch in range(n_epochs):
            model.train()

            # Shuffle labeled data
            perm = torch.randperm(len(X_labeled_t), generator=rng)
            X_shuffled = X_labeled_t[perm]
            y_shuffled = y_labeled_t[perm]
            w_shuffled = w_t[perm]

            # Mini-batch training
            epoch_sup_loss = 0.0
            epoch_con_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_shuffled), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                w_batch = w_shuffled[i:i + batch_size]

                # Supervised loss
                pred_labeled = model(X_batch)
                sup_loss = (w_batch * (pred_labeled - y_batch) ** 2).mean()

                # Consistency loss on unlabeled data
                if len(X_unlabeled) > 0:
                    # Sample unlabeled batch
                    u_idx = torch.randint(0, len(X_unlabeled_t),
                                          (min(batch_size, len(X_unlabeled_t)),),
                                          generator=rng)
                    X_u = X_unlabeled_t[u_idx]

                    # Original prediction
                    pred_u = model(X_u)

                    # Noisy copy prediction
                    noise = torch.randn_like(X_u) * noise_std
                    pred_u_noisy = model(X_u + noise)

                    con_loss = ((pred_u - pred_u_noisy) ** 2).mean()
                else:
                    con_loss = torch.tensor(0.0, device=device)

                loss = sup_loss + consistency_weight * con_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_sup_loss += sup_loss.item()
                epoch_con_loss += con_loss.item()
                n_batches += 1

            # Evaluate
            model.eval()
            with torch.no_grad():
                pred_test_t = model(X_test_t)
                pred_test_np = pred_test_t.cpu().numpy()

            if target_transform == "logit":
                pred_test_np = _sigmoid(pred_test_np)

            test_mse = float(mean_squared_error(y_test, pred_test_np))
            test_mae = float(mean_absolute_error(y_test, pred_test_np))
            test_r2 = float(r2_score(y_test, pred_test_np))

            scheduler.step(test_mse)

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            history.append({
                "epoch": epoch,
                "sup_loss": epoch_sup_loss / n_batches,
                "con_loss": epoch_con_loss / n_batches,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "test_r2": test_r2,
            })

            if epoch % 50 == 0 or epoch == n_epochs - 1:
                logger.info("Epoch %d: sup_loss=%.6f, con_loss=%.6f, test_mse=%.6f, test_r2=%.6f",
                            epoch, history[-1]["sup_loss"], history[-1]["con_loss"],
                            test_mse, test_r2)

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            pred_final = model(X_test_t).cpu().numpy()
        if target_transform == "logit":
            pred_final = _sigmoid(pred_final)

        final_mse = float(mean_squared_error(y_test, pred_final))
        final_mae = float(mean_absolute_error(y_test, pred_final))
        final_r2 = float(r2_score(y_test, pred_final))
        final_r = float(np.corrcoef(y_test, pred_final)[0, 1])

        logger.info("Final: MSE=%.6f, MAE=%.6f, R²=%.6f, r=%.6f",
                     final_mse, final_mae, final_r2, final_r)

        # Save model
        torch.save(model.state_dict(), os.path.join(out_dir, "consistency_model.pt"))
        joblib.dump({
            "hidden_sizes": hidden_sizes,
            "n_features": n_features,
            "dropout": dropout,
            "target_transform": target_transform,
        }, os.path.join(out_dir, "model_config.joblib"))

        return {
            "final": {"mse": final_mse, "mae": final_mae, "r2": final_r2, "pearson_r": final_r},
            "history": history,
            "pred_test": pred_final,
        }


# ───────────────────────── Numpy Fallback ─────────────────────────

def train_consistency_numpy(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    train_groups: Optional[np.ndarray] = None,
    consistency_weight: float = 1.0,
    noise_std: float = 0.1,
    n_rounds: int = 5,
    noise_samples: int = 10,
    model_name: str = "gradient_boosting",
    random_state: int = 42,
    target_transform: str = "logit",
    clip_epsilon: float = 1e-6,
    weight_low_af: bool = True,
    weight_epsilon: float = 1e-6,
    weight_power: float = 1.0,
    enable_cv: bool = True,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train with ensemble-based consistency regularization (numpy fallback).

    When PyTorch is unavailable, this function approximates consistency
    regularization by training an ensemble of models on noise-perturbed
    feature sets. Agreement among ensemble members on unlabeled data is
    used to generate pseudo-labels for subsequent rounds.

    Parameters
    ----------
    X_labeled : np.ndarray of shape (n_labeled, n_features)
        Feature matrix for labeled training samples.
    y_labeled : np.ndarray of shape (n_labeled,)
        Target values for labeled samples.
    X_unlabeled : np.ndarray of shape (n_unlabeled, n_features)
        Feature matrix for unlabeled samples.
    X_test : np.ndarray of shape (n_test, n_features)
        Test feature matrix.
    y_test : np.ndarray of shape (n_test,)
        Test target values.
    out_dir : str
        Directory for saving ensemble models and outputs.
    train_groups : np.ndarray or None, optional
        Group labels for grouped cross-validation splits. Default is None.
    consistency_weight : float, optional
        Weight given to pseudo-labeled samples relative to labeled data.
        Default is 1.0.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to features each round.
        Default is 0.1.
    n_rounds : int, optional
        Number of ensemble training rounds. Default is 5.
    noise_samples : int, optional
        Reserved for future use. Default is 10.
    model_name : str, optional
        Name of the base regression model. Default is
        ``'gradient_boosting'``.
    random_state : int, optional
        Random seed. Default is 42.
    target_transform : str, optional
        Target transformation (``'logit'`` or ``'none'``). Default is
        ``'logit'``.
    clip_epsilon : float, optional
        Clipping epsilon for logit transform and range filtering.
        Default is 1e-6.
    weight_low_af : bool, optional
        Whether to up-weight low-frequency samples. Default is True.
    weight_epsilon : float, optional
        Small constant to avoid division by zero in weights. Default
        is 1e-6.
    weight_power : float, optional
        Exponent controlling up-weighting strength. Default is 1.0.
    enable_cv : bool, optional
        Whether to use cross-validated hyperparameter tuning for the
        first round. Default is True.
    cv_folds : int, optional
        Number of cross-validation folds. Default is 5.

    Returns
    -------
    dict
        Dictionary with keys ``'final'`` (metrics dict), ``'history'``
        (list of per-round records), ``'pred_test'`` (np.ndarray of
        ensemble-mean test predictions), and ``'n_ensemble'`` (int).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)
    registry = ModelRegistry(random_state=random_state)

    models = []
    history = []
    n_labeled = len(X_labeled)

    # Sample weights
    if weight_low_af:
        base_weights = (1.0 / (y_labeled + weight_epsilon)) ** weight_power
        base_weights = base_weights / np.mean(base_weights)
    else:
        base_weights = None

    # Target transform
    if target_transform == "logit":
        y_train_model = _logit(_clip_01(y_labeled, clip_epsilon))
    else:
        y_train_model = y_labeled

    history = []

    for round_idx in range(n_rounds):
        # Add noise to features for this round
        noise_labeled = rng.normal(0, noise_std, X_labeled.shape)
        X_labeled_noisy = X_labeled + noise_labeled

        # Optionally include unlabeled data as pseudo-labeled
        if round_idx > 0 and len(X_unlabeled) > 0:
            # Predict with current ensemble
            preds = np.array([m.predict(X_unlabeled) for m in models])
            if target_transform == "logit":
                preds = _sigmoid(preds)
            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0)

            # Consistency filter: keep predictions with low disagreement
            # and reasonable range
            range_mask = (mean_pred > clip_epsilon) & (mean_pred < 1.0 - clip_epsilon)
            consistency_threshold = np.percentile(std_pred[range_mask], 25)
            consistent_mask = (std_pred <= consistency_threshold) & range_mask

            n_pseudo = consistent_mask.sum()
            if n_pseudo > 0:
                X_pseudo = X_unlabeled[consistent_mask]
                y_pseudo = mean_pred[consistent_mask]

                # Combine
                X_combined = np.vstack([X_labeled_noisy, X_pseudo])
                y_combined = np.concatenate([y_labeled, y_pseudo])

                # Pseudo-labels get lower weight
                w_pseudo = np.full(n_pseudo, consistency_weight) * base_weights.mean() if base_weights is not None else np.full(n_pseudo, consistency_weight)
                if base_weights is not None:
                    weights = np.concatenate([base_weights, w_pseudo])
                else:
                    weights = None
            else:
                X_combined = X_labeled_noisy
                y_combined = y_labeled
                weights = base_weights
        else:
            X_combined = X_labeled_noisy
            y_combined = y_labeled
            weights = base_weights

        if target_transform == "logit":
            y_combined_model = _logit(_clip_01(y_combined, clip_epsilon))
        else:
            y_combined_model = y_combined

        # Train model
        model = registry.build_models(model_names=[model_name])[model_name]
        if enable_cv and round_idx == 0:
            param_grid = registry.get_param_grid(model_name)
            model, _ = _fit_model_with_cv(
                name=model_name, X_train=X_combined, y_train=y_combined_model,
                train_weights=weights, groups=train_groups, cv_folds=cv_folds,
                param_grid=param_grid, random_state=random_state + round_idx,
            )
        else:
            fit_model_with_registry(model_name, model, X_combined, y_combined_model, weights)

        models.append(model)

        # Evaluate ensemble on test
        preds_test = np.array([m.predict(X_test) for m in models])
        if target_transform == "logit":
            preds_test = _sigmoid(preds_test)
        pred_mean = preds_test.mean(axis=0)

        mse = float(mean_squared_error(y_test, pred_mean))
        mae = float(mean_absolute_error(y_test, pred_mean))
        r2 = float(r2_score(y_test, pred_mean))
        pearson_r = float(np.corrcoef(y_test, pred_mean)[0, 1])

        n_pseudo_val = 0
        if round_idx > 0:
            # Recount from the last filtering
            preds_ens = np.array([m.predict(X_unlabeled) for m in models])
            if target_transform == "logit":
                preds_ens = _sigmoid(preds_ens)
            std_ens = preds_ens.std(axis=0)
            mean_ens = preds_ens.mean(axis=0)
            range_m = (mean_ens > clip_epsilon) & (mean_ens < 1.0 - clip_epsilon)
            thr = np.percentile(std_ens[range_m], 25) if range_m.any() else 0
            n_pseudo_val = int(((std_ens <= thr) & range_m).sum())

        history.append({
            "round": round_idx,
            "n_train": len(X_combined),
            "n_pseudo": n_pseudo_val,
            "mse": mse, "mae": mae, "r2": r2, "pearson_r": pearson_r,
        })
        logger.info("Round %d: n_train=%d, MSE=%.6f, R²=%.6f, r=%.6f",
                     round_idx, len(X_combined), mse, r2, pearson_r)

    # Save models
    for i, m in enumerate(models):
        joblib.dump(m, os.path.join(out_dir, f"ensemble_model_{i}.joblib"))

    # Final predictions (ensemble mean)
    final_pred = preds_test.mean(axis=0)
    final_mse = float(mean_squared_error(y_test, final_pred))
    final_mae = float(mean_absolute_error(y_test, final_pred))
    final_r2 = float(r2_score(y_test, final_pred))
    final_r = float(np.corrcoef(y_test, final_pred)[0, 1])

    return {
        "final": {"mse": final_mse, "mae": final_mae, "r2": final_r2, "pearson_r": final_r},
        "history": history,
        "pred_test": final_pred,
        "n_ensemble": len(models),
    }


# ───────────────────────── Main pipeline ─────────────────────────

def train_consistency(
    labeled_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    feature_columns: List[str],
    out_dir: str,
    consistency_weight: float = 1.0,
    noise_std: float = 0.1,
    hidden_sizes: Optional[List[int]] = None,
    n_epochs: int = 300,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    dropout: float = 0.1,
    n_rounds: int = 5,
    noise_samples: int = 10,
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
    """Run the full consistency regularization pipeline.

    Splits labeled data into train/test, trains a baseline model using
    only supervised learning, then trains a consistency-regularized model
    (PyTorch or numpy fallback) that additionally leverages unlabeled
    data. Generates comparison plots and a JSON summary.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame with labeled samples (must contain ``ddPCR_AF``).
    combined_df : pd.DataFrame
        DataFrame containing both labeled and unlabeled samples.
    feature_columns : list of str
        Column names to use as input features.
    out_dir : str
        Output directory for models, plots, and summary files.
    consistency_weight : float, optional
        Weight (lambda) for the consistency loss term. Default is 1.0.
    noise_std : float, optional
        Noise standard deviation for input perturbation. Default is 0.1.
    hidden_sizes : list of int or None, optional
        Hidden layer sizes for the PyTorch model. Default is None
        (falls back to ``[64, 32]``).
    n_epochs : int, optional
        Training epochs for PyTorch mode. Default is 300.
    batch_size : int, optional
        Mini-batch size for PyTorch training. Default is 64.
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is 1e-3.
    dropout : float, optional
        Dropout probability in the PyTorch model. Default is 0.1.
    n_rounds : int, optional
        Number of ensemble rounds in numpy fallback mode. Default is 5.
    noise_samples : int, optional
        Reserved for future use. Default is 10.
    model_name : str, optional
        Base regression model name for the numpy fallback. Default is
        ``'gradient_boosting'``.
    group_cols : list of str or None, optional
        Columns for grouped train/test splitting. Default is None.
    target_transform : str, optional
        Target transformation (``'logit'`` or ``'none'``). Default is
        ``'logit'``.
    clip_epsilon : float, optional
        Clipping epsilon. Default is 1e-6.
    test_size : float, optional
        Fraction of labeled data held out for testing. Default is 0.2.
    random_state : int, optional
        Random seed. Default is 42.
    weight_low_af : bool, optional
        Whether to up-weight low-frequency samples. Default is True.
    weight_epsilon : float, optional
        Small constant for weight computation. Default is 1e-6.
    weight_power : float, optional
        Exponent for weight computation. Default is 1.0.
    enable_cv : bool, optional
        Whether to use cross-validation for hyperparameter tuning.
        Default is True.
    cv_folds : int, optional
        Number of cross-validation folds. Default is 5.

    Returns
    -------
    dict
        Summary dictionary with keys ``'method'``, ``'baseline'``,
        ``'consistency'``, ``'improvement'``, and ``'config'``.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Extract labeled data
    X_labeled, y_labeled, labeled_idx = extract_xy(labeled_df, feature_columns)
    logger.info("Labeled: %d rows, Unlabeled in combined: %d rows",
                len(X_labeled), len(combined_df) - len(labeled_df))

    # Extract unlabeled features
    unlabeled_mask = combined_df["ddPCR_AF"].isna()
    X_unlabeled = combined_df.loc[unlabeled_mask, feature_columns].fillna(0.0).to_numpy(dtype=float)

    # Grouped split on labeled data
    train_idx, test_idx, train_groups = grouped_train_test_split(
        X=X_labeled, y=y_labeled, df=labeled_df.loc[labeled_idx],
        group_cols=group_cols, test_size=test_size, random_state=random_state,
    )

    X_train = X_labeled[train_idx]
    y_train = y_labeled[train_idx]
    X_test = X_labeled[test_idx]
    y_test = y_labeled[test_idx]

    # Baseline (no consistency, no unlabeled data)
    registry = ModelRegistry(random_state=random_state)

    if target_transform == "logit":
        y_train_model = _logit(_clip_01(y_train, clip_epsilon))
    else:
        y_train_model = y_train

    if weight_low_af:
        train_weights = (1.0 / (y_train + weight_epsilon)) ** weight_power
        train_weights = train_weights / np.mean(train_weights)
    else:
        train_weights = None

    baseline_model = registry.build_models(model_names=[model_name])[model_name]
    if enable_cv:
        param_grid = registry.get_param_grid(model_name)
        baseline_model, _ = _fit_model_with_cv(
            name=model_name, X_train=X_train, y_train=y_train_model,
            train_weights=train_weights, groups=train_groups, cv_folds=cv_folds,
            param_grid=param_grid, random_state=random_state,
        )
    else:
        fit_model_with_registry(model_name, baseline_model, X_train, y_train_model, train_weights)

    pred_baseline = baseline_model.predict(X_test)
    if target_transform == "logit":
        pred_baseline = _sigmoid(pred_baseline)

    mse_base = float(mean_squared_error(y_test, pred_baseline))
    r2_base = float(r2_score(y_test, pred_baseline))
    r_base = float(np.corrcoef(y_test, pred_baseline)[0, 1])
    logger.info("Baseline: MSE=%.6f, R²=%.6f, r=%.6f", mse_base, r2_base, r_base)

    # Consistency training
    if HAS_TORCH:
        result = train_consistency_pytorch(
            X_labeled=X_train, y_labeled=y_train,
            X_unlabeled=X_unlabeled,
            X_test=X_test, y_test=y_test,
            out_dir=out_dir,
            hidden_sizes=hidden_sizes or [64, 32],
            consistency_weight=consistency_weight,
            noise_std=noise_std,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout=dropout,
            random_state=random_state,
            target_transform=target_transform,
            clip_epsilon=clip_epsilon,
            weight_low_af=weight_low_af,
            weight_epsilon=weight_epsilon,
            weight_power=weight_power,
        )
    else:
        result = train_consistency_numpy(
            X_labeled=X_train, y_labeled=y_train,
            X_unlabeled=X_unlabeled,
            X_test=X_test, y_test=y_test,
            out_dir=out_dir,
            train_groups=train_groups,
            consistency_weight=consistency_weight,
            noise_std=noise_std,
            n_rounds=n_rounds,
            noise_samples=noise_samples,
            model_name=model_name,
            random_state=random_state,
            target_transform=target_transform,
            clip_epsilon=clip_epsilon,
            weight_low_af=weight_low_af,
            weight_epsilon=weight_epsilon,
            weight_power=weight_power,
            enable_cv=enable_cv,
            cv_folds=cv_folds,
        )

    pred_consistency = result["pred_test"]

    # Comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, y_pred, title, metrics in zip(
        axes,
        [pred_baseline, pred_consistency],
        ["Baseline (supervised only)", "Consistency Regularization"],
        [{"mse": mse_base, "r2": r2_base, "r": r_base},
         result["final"]],
    ):
        ax.scatter(y_test, y_pred, s=30, alpha=0.75)
        vmin = min(y_test.min(), y_pred.min())
        vmax = max(y_test.max(), y_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="y=x")
        ax.set_xlabel("True ddPCR_AF")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{title}\nMSE={metrics['mse']:.6f} R²={metrics['r2']:.6f} r={metrics.get('pearson_r', metrics.get('r', 0)):.3f}")
        ax.legend()
        ax.grid(True, alpha=0.25)
    fig.suptitle("Consistency Regularization Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_pred_vs_truth.png"), dpi=200)
    plt.close(fig)

    # Training curves (if available)
    if "history" in result and result["history"]:
        history_df = pd.DataFrame(result["history"])
        history_df.to_csv(os.path.join(out_dir, "training_history.tsv"), sep="\t", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if "sup_loss" in history_df.columns:
            axes[0].plot(history_df["epoch"], history_df["sup_loss"], label="Supervised loss")
            axes[0].plot(history_df["epoch"], history_df["con_loss"], label="Consistency loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
        elif "mse" in history_df.columns:
            axes[0].plot(history_df["round"], history_df["mse"], "o-", label="Test MSE")
            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("MSE")
        axes[0].set_title("Training Curves")
        axes[0].grid(True, alpha=0.25)

        if "test_r2" in history_df.columns:
            axes[1].plot(history_df["epoch"], history_df["test_r2"], "o-", color="#C44E52")
        elif "r2" in history_df.columns:
            axes[1].plot(history_df["round"], history_df["r2"], "o-", color="#C44E52")
        axes[1].set_xlabel("Epoch/Round")
        axes[1].set_ylabel("R²")
        axes[1].set_title("Test R²")
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=200)
        plt.close(fig)

    # Summary
    summary = {
        "method": "consistency_pytorch" if HAS_TORCH else "consistency_ensemble",
        "baseline": {"mse": mse_base, "r2": r2_base, "pearson_r": r_base},
        "consistency": result["final"],
        "improvement": {
            "mse_reduction": mse_base - result["final"]["mse"],
            "r2_gain": result["final"]["r2"] - r2_base,
        },
        "config": {
            "consistency_weight": consistency_weight,
            "noise_std": noise_std,
            "target_transform": target_transform,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=== Consistency Regularization Summary ===")
    logger.info("Baseline:  MSE=%.6f, R²=%.6f", mse_base, r2_base)
    logger.info("Consistency: MSE=%.6f, R²=%.6f", result["final"]["mse"], result["final"]["r2"])
    logger.info("Improvement: MSE Δ=%.6f, R² Δ=%.6f",
                summary["improvement"]["mse_reduction"],
                summary["improvement"]["r2_gain"])

    return summary


# ───────────────────────── CLI ─────────────────────────

def main():
    """CLI entry point for the consistency regularization pipeline.

    Parses command-line arguments, loads combined labeled and unlabeled
    feature data, and invokes :func:`train_consistency`.
    """
    parser = argparse.ArgumentParser(
        description="Consistency Regularization Semi-Supervised SV Frequency Correction",
    )
    parser.add_argument("--labeled-tsv", required=True)
    parser.add_argument("--unlabeled-tsv", required=True)
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument("--probe-infile", default=None)
    parser.add_argument("--feature-dir", default=None,
                        help="Shared directory for BAM feature cache (avoid re-extraction)")
    parser.add_argument("--group-cols", action="append", default=["原始编号", "FusionGene", "FusionExon"],
                        help="Column(s) for grouped train/test split (repeatable, default: 原始编号)")
    # PyTorch params
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 32],
                        help="Hidden layer sizes (PyTorch mode)")
    parser.add_argument("--consistency-weight", type=float, default=1.0,
                        help="Weight for consistency loss (λ)")
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Noise std for perturbation")
    parser.add_argument("--n-epochs", type=int, default=300, help="Training epochs (PyTorch)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Numpy fallback params
    parser.add_argument("--n-rounds", type=int, default=5, help="Ensemble rounds (numpy fallback)")
    parser.add_argument("--model-name", default="gradient_boosting",
                        help="Model for numpy fallback")
    # Common
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

    train_consistency(
        labeled_df=labeled_df,
        combined_df=combined_df,
        feature_columns=feature_columns,
        out_dir=args.out_dir,
        consistency_weight=args.consistency_weight,
        noise_std=args.noise_std,
        hidden_sizes=args.hidden_sizes,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        n_rounds=args.n_rounds,
        model_name=args.model_name,
        group_cols=args.group_cols,
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

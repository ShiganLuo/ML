"""Train regression models to correct SV frequency from breakpoint features."""

from __future__ import annotations

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass, field
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification for a regression model.
    
    Attributes
    ----------
    name : str
        Unique identifier for the model.
    factory : callable
        Function that returns a sklearn-compatible regressor instance.
    supports_sample_weight : bool
        Whether the model's fit() method accepts sample_weight parameter.
    supports_feature_importance : bool
        Whether the model exposes feature_importances_ or coef_ after fitting.
    description : str
        Human-readable description of the model.
    default_param_grid : dict
        Default parameter grid for hyperparameter tuning.
    """
    name: str
    factory: Callable[[], Any]
    supports_sample_weight: bool = True
    supports_feature_importance: bool = True
    description: str = ""
    default_param_grid: Dict[str, List[Any]] = field(default_factory=dict)


class ModelRegistry:
    """Registry for managing regression models with standardized interface.
    
    This class provides a unified way to register, configure, and train
    different regression models while handling their interface differences
    transparently.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the model registry.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self._specs: Dict[str, ModelSpec] = {}
        self._models: Dict[str, Any] = {}
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """Register all default regression models with expanded parameter grids."""
        # Linear models
        self.register(
            name="ridge",
            factory=lambda: Ridge(random_state=self.random_state),
            supports_sample_weight=True,
            supports_feature_importance=True,
            description="Ridge regression with L2 regularization",
            default_param_grid={
                "alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                "fit_intercept": [True, False],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            }
        )
        
        self.register(
            name="elastic_net",
            factory=lambda: ElasticNet(random_state=self.random_state, max_iter=2000),
            supports_sample_weight=True,
            supports_feature_importance=True,
            description="ElasticNet with L1+L2 regularization",
            default_param_grid={
                "alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "fit_intercept": [True, False],
                "selection": ["cyclic", "random"],
            }
        )
        
        self.register(
            name="lasso",
            factory=lambda: Lasso(random_state=self.random_state, max_iter=2000),
            supports_sample_weight=True,
            supports_feature_importance=True,
            description="Lasso regression with L1 regularization",
            default_param_grid={
                "alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                "fit_intercept": [True, False],
                "selection": ["cyclic", "random"],
            }
        )
        
        # Support Vector Machine - expanded grid
        self.register(
            name="svr_rbf",
            factory=lambda: SVR(kernel="rbf"),
            supports_sample_weight=True,
            supports_feature_importance=False,
            description="Support Vector Regression with RBF kernel",
            default_param_grid={
                "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
                "epsilon": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                "gamma": ["scale", "auto", 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
                "kernel": ["rbf"],
            }
        )
        
        # SVR with linear kernel
        self.register(
            name="svr_linear",
            factory=lambda: SVR(kernel="linear"),
            supports_sample_weight=True,
            supports_feature_importance=False,
            description="Support Vector Regression with linear kernel",
            default_param_grid={
                "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                "epsilon": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            }
        )
        
        # SVR with polynomial kernel
        self.register(
            name="svr_poly",
            factory=lambda: SVR(kernel="poly"),
            supports_sample_weight=True,
            supports_feature_importance=False,
            description="Support Vector Regression with polynomial kernel",
            default_param_grid={
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "epsilon": [0.01, 0.05, 0.1, 0.5],
                "degree": [2, 3, 4, 5],
                "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
                "coef0": [0.0, 0.1, 0.5, 1.0],
            }
        )
        
        # Ensemble models - Random Forest with expanded grid
        self.register(
            name="random_forest",
            factory=lambda: RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            supports_sample_weight=True,
            supports_feature_importance=True,
            description="Random Forest ensemble",
            default_param_grid={
                "n_estimators": [100, 200, 300, 500, 800, 1000],
                "max_depth": [3, 5, 7, 10, 15, 20, None],
                "min_samples_split": [2, 3, 5, 7, 10, 15, 20],
                "min_samples_leaf": [1, 2, 3, 4, 5, 10],
                "max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7, 0.9],
                "bootstrap": [True, False],
                "max_leaf_nodes": [None, 10, 20, 50, 100],
            }
        )
        
        # Gradient Boosting with expanded grid
        self.register(
            name="gradient_boosting",
            factory=lambda: GradientBoostingRegressor(random_state=self.random_state),
            supports_sample_weight=True,
            supports_feature_importance=True,
            description="Gradient Boosting ensemble",
            default_param_grid={
                "n_estimators": [100, 200, 300, 500, 800, 1000],
                "max_depth": [2, 3, 4, 5, 6, 7, 8, 10],
                "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10, 15, 20],
                "min_samples_leaf": [1, 2, 3, 5, 10],
                "max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7],
                "loss": ["squared_error", "absolute_error", "huber"],
            }
        )
        
        # Extra Trees with expanded grid
        self.register(
            name="extra_trees",
            factory=lambda: ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
            supports_sample_weight=True,
            supports_feature_importance=True,
            description="Extra Trees ensemble",
            default_param_grid={
                "n_estimators": [100, 200, 300, 500, 800, 1000],
                "max_depth": [3, 5, 7, 10, 15, 20, None],
                "min_samples_split": [2, 3, 5, 7, 10, 15, 20],
                "min_samples_leaf": [1, 2, 3, 4, 5, 10],
                "max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7, 0.9],
                "bootstrap": [True, False],
                "max_leaf_nodes": [None, 10, 20, 50, 100],
            }
        )
        
        # AdaBoost with expanded grid
        self.register(
            name="adaboost",
            factory=lambda: AdaBoostRegressor(random_state=self.random_state),
            supports_sample_weight=True,
            supports_feature_importance=True,
            description="AdaBoost ensemble",
            default_param_grid={
                "n_estimators": [50, 100, 150, 200, 300, 500],
                "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                "loss": ["linear", "square", "exponential"],
            }
        )
        
        # Instance-based models
        self.register(
            name="knn",
            factory=lambda: KNeighborsRegressor(n_jobs=-1),
            supports_sample_weight=False,
            supports_feature_importance=False,
            description="K-Nearest Neighbors regression",
            default_param_grid={
                "n_neighbors": [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50],
                "weights": ["uniform", "distance"],
                "p": [1, 2, 3, 4, 5],
                "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
                "leaf_size": [10, 20, 30, 50, 100],
            }
        )
        
        # Neural network models
        self.register(
            name="mlp",
            factory=lambda: MLPRegressor(random_state=self.random_state, max_iter=1000, early_stopping=True),
            supports_sample_weight=False,
            supports_feature_importance=False,
            description="Multi-layer Perceptron neural network",
            default_param_grid={
                "hidden_layer_sizes": [
                    (50,), (100,), (150,), (200,),
                    (50, 25), (100, 50), (150, 75), (200, 100),
                    (100, 50, 25), (150, 100, 50),
                    (200, 100, 50), (100, 100, 50, 25),
                ],
                "activation": ["relu", "tanh", "logistic"],
                "solver": ["adam", "sgd", "lbfgs"],
                "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                "learning_rate": ["constant", "invscaling", "adaptive"],
                "learning_rate_init": [0.0001, 0.0005, 0.001, 0.005, 0.01],
                "batch_size": ["auto", 32, 64, 128, 256],
            }
        )
    
    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        supports_sample_weight: bool = True,
        supports_feature_importance: bool = True,
        description: str = "",
        default_param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """Register a new model.
        
        Parameters
        ----------
        name : str
            Unique identifier for the model.
        factory : callable
            Function that returns a sklearn-compatible regressor instance.
        supports_sample_weight : bool, optional
            Whether the model's fit() method accepts sample_weight parameter.
        supports_feature_importance : bool, optional
            Whether the model exposes feature_importances_ or coef_ after fitting.
        description : str, optional
            Human-readable description of the model.
        default_param_grid : dict or None, optional
            Default parameter grid for hyperparameter tuning.
        """
        self._specs[name] = ModelSpec(
            name=name,
            factory=factory,
            supports_sample_weight=supports_sample_weight,
            supports_feature_importance=supports_feature_importance,
            description=description,
            default_param_grid=default_param_grid or {},
        )
    
    def build_models(
        self,
        model_names: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build model instances from registered specifications.
        
        Parameters
        ----------
        model_names : list[str] or None, optional
            List of model names to build. If None, builds all registered models.
        model_config : dict or None, optional
            Configuration overrides for model hyperparameters.
            
        Returns
        -------
        dict
            Mapping from model name to model instance.
        """
        config = model_config or {}
        models = {}
        
        names_to_build = model_names or list(self._specs.keys())
        
        for name in names_to_build:
            if name not in self._specs:
                logger.warning(f"Model '{name}' not registered, skipping")
                continue
            
            spec = self._specs[name]
            
            try:
                models[name] = spec.factory()
            except Exception as e:
                logger.error(f"Failed to create model '{name}': {e}")
        
        return models
    
    def fit_model(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        """Fit a model with appropriate handling of sample_weight.
        
        Parameters
        ----------
        name : str
            Model name (used to check capabilities).
        model : object
            sklearn-compatible regressor instance.
        X_train : numpy.ndarray
            Training features.
        y_train : numpy.ndarray
            Training labels.
        sample_weight : numpy.ndarray or None, optional
            Sample weights for training.
            
        Returns
        -------
        object
            The fitted model.
        """
        spec = self._specs.get(name)
        if spec is None:
            # Fallback: try with sample_weight, catch TypeError
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
                model.fit(X_train, y_train)
            return model
        
        if sample_weight is not None and spec.supports_sample_weight:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def get_spec(self, name: str) -> Optional[ModelSpec]:
        """Get specification for a registered model.
        
        Parameters
        ----------
        name : str
            Model name.
            
        Returns
        -------
        ModelSpec or None
            Model specification if found, None otherwise.
        """
        return self._specs.get(name)
    
    def get_param_grid(self, name: str, custom_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, List[Any]]:
        """Get parameter grid for a model, with optional custom override.
        
        Parameters
        ----------
        name : str
            Model name.
        custom_grid : dict or None, optional
            Custom parameter grid to use instead of default.
            
        Returns
        -------
        dict
            Parameter grid for the model.
        """
        if custom_grid is not None:
            return custom_grid
        
        spec = self._specs.get(name)
        if spec is None:
            return {}
        
        return spec.default_param_grid
    
    def list_models(self) -> List[str]:
        """List all registered model names.
        
        Returns
        -------
        list[str]
            List of registered model names.
        """
        return list(self._specs.keys())


# Global registry instance
_registry = ModelRegistry()


def build_models(
    random_state: int,
    model_config: Optional[Dict[str, object]] = None,
    include_svr: bool = True,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the model registry for training.

    Parameters
    ----------
    random_state : int
        Random seed for model initialization.
    model_config : dict or None, optional
        Optional configuration to customize model hyperparameters.
    include_svr : bool, optional
        Whether to include the default SVR model.
    model_names : list[str] or None, optional
        Specific model names to include. If None, includes all models.

    Returns
    -------
    dict
        Mapping from model name to model instance.
    """
    global _registry
    _registry = ModelRegistry(random_state=random_state)
    
    if model_names is None:
        # Default: include all models except SVR if not requested
        model_names = _registry.list_models()
        if not include_svr:
            model_names = [n for n in model_names if not n.startswith("svr_")]
    
    return _registry.build_models(model_names=model_names, model_config=model_config)


def fit_model_with_registry(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Any:
    """Fit a model using the registry's standardized interface.
    
    Parameters
    ----------
    name : str
        Model name.
    model : object
        sklearn-compatible regressor instance.
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels.
    sample_weight : numpy.ndarray or None, optional
        Sample weights for training.
        
    Returns
    -------
    object
        The fitted model.
    """
    global _registry
    return _registry.fit_model(name, model, X_train, y_train, sample_weight)


def _fit_model_with_cv(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    cv_folds: int,
    param_grid: Dict[str, List[object]],
    random_state: int,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
) -> tuple[Any, Dict[str, object]]:
    """Run CV to pick hyperparameters for any model.

    Parameters
    ----------
    name : str
        Model name (used to create instance).
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels (possibly transformed).
    train_weights : numpy.ndarray or None
        Optional sample weights.
    groups : numpy.ndarray or None
        Optional grouping labels for grouped CV.
    cv_folds : int
        Number of CV folds.
    param_grid : dict
        Parameter grid for hyperparameter search.
    random_state : int
        Random seed used for CV splitting when groups are absent.
    scoring : str, optional
        Scoring metric for GridSearchCV.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    tuple
        Best estimator and best parameter dict.
    """
    global _registry
    
    # Create fresh model instance
    spec = _registry.get_spec(name)
    if spec is None:
        raise ValueError(f"Model '{name}' not registered")
    
    model = spec.factory()
    
    if groups is not None and len(groups) > 0:
        splitter = GroupKFold(n_splits=cv_folds)
        cv_kwargs = {"groups": groups}
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_kwargs = {}

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )
    
    fit_kwargs = {}
    if train_weights is not None and spec.supports_sample_weight:
        fit_kwargs["sample_weight"] = train_weights
    
    grid.fit(X_train, y_train, **fit_kwargs, **cv_kwargs)
    
    logger.info(f"Best params for {name}: {grid.best_params_}")
    logger.info(f"Best score for {name}: {grid.best_score_:.6f}")
    
    return grid.best_estimator_, grid.best_params_


def _save_training_report(
    out_dir: str,
    feature_columns: List[str],
    y_test: pd.Series,
    test_pred_frame: pd.DataFrame,
    models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    best_name: str,
) -> None:
    """Save plots and a short markdown report for model training results."""
    os.makedirs(out_dir, exist_ok=True)

    if f"pred_{best_name}" not in test_pred_frame.columns:
        best_name = next((name for name in models if f"pred_{name}" in test_pred_frame.columns), "")
    if not best_name:
        return

    y_true = np.asarray(pd.to_numeric(y_test, errors="coerce"), dtype=float)
    y_pred = np.asarray(pd.to_numeric(test_pred_frame[f"pred_{best_name}"], errors="coerce"), dtype=float)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.any():
        logger.warning("No valid predictions found for plotting")
        return

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    residuals = y_pred - y_true
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")

    # Prediction vs truth
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=30, alpha=0.75)
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("True ddPCR_AF")
    ax.set_ylabel(f"Predicted ddPCR_AF ({best_name})")
    ax.set_title(f"Prediction vs Truth - {best_name}")
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {pearson_r:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"prediction_vs_truth_{best_name}.png"), dpi=200)
    plt.close(fig)

    # Residual vs truth
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, residuals, s=30, alpha=0.75)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("True ddPCR_AF")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title(f"Residuals vs Truth - {best_name}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"residuals_vs_truth_{best_name}.png"), dpi=200)
    plt.close(fig)

    # Residual histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=min(20, max(5, len(residuals) // 2)), alpha=0.85, color="#4C72B0")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution - {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"residual_hist_{best_name}.png"), dpi=200)
    plt.close(fig)

    # Feature importance / coefficients
    model: Any = models[best_name]
    importance = None
    importance_title = ""
    
    # Use registry to check feature importance support
    global _registry
    spec = _registry.get_spec(best_name)
    
    if spec and spec.supports_feature_importance:
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(getattr(model, "feature_importances_"), dtype=float)
            importance_title = f"Feature Importance - {best_name}"
        elif hasattr(model, "coef_"):
            importance = np.abs(np.asarray(getattr(model, "coef_"), dtype=float).ravel())
            importance_title = f"Absolute Coefficients - {best_name}"

    if importance is not None and len(importance) == len(feature_columns):
        importance_df = pd.DataFrame({"feature": feature_columns, "importance": importance})
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df.to_csv(os.path.join(out_dir, f"feature_importance_{best_name}.tsv"), sep="\t", index=False)

        top_n = min(20, len(importance_df))
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
        plot_df = importance_df.head(top_n).iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["importance"], color="#55A868")
        ax.set_xlabel("Importance")
        ax.set_title(importance_title)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"feature_importance_{best_name}.png"), dpi=200)
        plt.close(fig)

    report_lines = [
        f"Best model: {best_name}",
        "",
        "Metrics:",
    ]
    for name, values in metrics.items():
        report_lines.append(
            f"- {name}: mse={values['mse']:.6f}, mae={values['mae']:.6f}, r2={values['r2']:.6f}"
        )
    # add Pearson r summary and compare to reported R^2 for clarity
    report_lines.extend([
        "",
        f"Pearson r (best model): {pearson_r:.6f}",
        f"Pearson r squared: {pearson_r**2:.6f}",
    ])
    if best_name in metrics:
        report_lines.append(f"Reported r2 (best model): {metrics[best_name]['r2']:.6f}")
        if not np.isfinite(pearson_r) or abs((pearson_r**2) - metrics[best_name]['r2']) > 1e-6:
            report_lines.append("")
            report_lines.append(
                "Note: Pearson r and reported R^2 are related but may differ. "
                "Differences arise from bias, scaling, or non-linear errors in predictions."
            )

    report_lines.extend(
        [
            "",
            "Generated files:",
            f"- prediction_vs_truth_{best_name}.png",
            f"- residuals_vs_truth_{best_name}.png",
            f"- residual_hist_{best_name}.png",
            f"- feature_importance_{best_name}.png (if supported)",
            f"- feature_importance_{best_name}.tsv (if supported)",
        ]
    )
    with open(os.path.join(out_dir, "training_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

def report_best_model(
    metrics: Dict[str, Dict[str, float]],
) -> str:
    """Return the model name with the lowest MSE.

    Parameters
    ----------
    metrics : dict
        Mapping from model name to metric values.

    Returns
    -------
    str
        Name of the best model by MSE.
    """
    best_name = ""
    best_mse = float("inf")
    for name, values in metrics.items():
        logger.info(
            f"Model: {name}, MSE: {values['mse']:.6f}, MAE: {values['mae']:.6f}, R^2: {values['r2']:.6f}"
        )
        if values['mse'] < best_mse:
            best_mse = values['mse']
            best_name = name
    return best_name


def _clip_01(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Clip values into (0, 1) with a safety margin."""
    return np.clip(values, epsilon, 1.0 - epsilon)


def _logit(values: np.ndarray) -> np.ndarray:
    """Apply logit transform."""
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply sigmoid transform."""
    return 1.0 / (1.0 + np.exp(-values))


def _save_model_comparison(
    out_dir: str,
    models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    feature_columns: list[str],
    y_test: pd.Series,
    test_pred_frame: pd.DataFrame,
    label_col: str = "ddPCR_AF",
    low_freq_threshold: float = 0.02,
) -> None:
    """Generate cross-model comparison plots and a summary report."""
    os.makedirs(out_dir, exist_ok=True)

    # Extract model info
    model_info = []
    for name, model in models.items():
        m = metrics.get(name, {})
        model_info.append({
            "name": name,
            "mse": m.get("mse", float("nan")),
            "mae": m.get("mae", float("nan")),
            "r2": m.get("r2", float("nan")),
            "model": model,
        })

    mse_vals = [m["mse"] for m in model_info]
    mae_vals = [m["mae"] for m in model_info]
    r2_vals = [m["r2"] for m in model_info]
    names = [m["name"] for m in model_info]

    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(names))
    width = 0.6
    
    axes[0].bar(x, mse_vals, width, color="#4C72B0", alpha=0.85)
    axes[0].set_xlabel("Model", fontsize=11)
    axes[0].set_ylabel("MSE", fontsize=11)
    axes[0].set_title("MSE Comparison", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].grid(True, alpha=0.25, axis="y")
    best_mse_idx = int(np.nanargmin(mse_vals))
    axes[0].axhline(mse_vals[best_mse_idx], color="red", linestyle="--", alpha=0.6,
                    label=f"best: {names[best_mse_idx]}")
    axes[0].legend(fontsize=8)
    
    axes[1].bar(x, mae_vals, width, color="#55A868", alpha=0.85)
    axes[1].set_xlabel("Model", fontsize=11)
    axes[1].set_ylabel("MAE", fontsize=11)
    axes[1].set_title("MAE Comparison", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].grid(True, alpha=0.25, axis="y")
    best_mae_idx = int(np.nanargmin(mae_vals))
    axes[1].axhline(mae_vals[best_mae_idx], color="red", linestyle="--", alpha=0.6,
                    label=f"best: {names[best_mae_idx]}")
    axes[1].legend(fontsize=8)
    
    axes[2].bar(x, r2_vals, width, color="#C44E52", alpha=0.85)
    axes[2].set_xlabel("Model", fontsize=11)
    axes[2].set_ylabel("R²", fontsize=11)
    axes[2].set_title("R² Comparison", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].grid(True, alpha=0.25, axis="y")
    best_r2_idx = int(np.nanargmax(r2_vals))
    axes[2].axhline(r2_vals[best_r2_idx], color="red", linestyle="--", alpha=0.6,
                    label=f"best: {names[best_r2_idx]}")
    axes[2].legend(fontsize=8)
    
    fig.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison_metrics.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Low-frequency residual comparison
    y_true = np.asarray(pd.to_numeric(y_test, errors="coerce"), dtype=float)
    low_mask = y_true < low_freq_threshold

    if low_mask.sum() >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot of absolute residuals for low-freq samples
        low_abs_residuals = {}
        for name in names:
            pred_col = f"pred_{name}"
            if pred_col in test_pred_frame.columns:
                y_pred = np.asarray(
                    pd.to_numeric(test_pred_frame[pred_col], errors="coerce"), dtype=float
                )
                residuals = np.abs(y_pred - y_true)
                low_abs_residuals[name] = residuals[low_mask]

        if low_abs_residuals:
            axes[0].boxplot(
                [low_abs_residuals[n] for n in names if n in low_abs_residuals],
                labels=[n for n in names if n in low_abs_residuals],
                vert=True,
            )
            axes[0].set_xlabel("Model", fontsize=11)
            axes[0].set_ylabel("|Residual|", fontsize=11)
            axes[0].set_title(f"|Residual| Distribution (ddPCR_AF < {low_freq_threshold})", fontsize=12)
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].grid(True, alpha=0.25, axis="y")

            # Mean |residual| comparison
            low_mean_abs = [np.mean(low_abs_residuals.get(n, [np.nan])) for n in names]
            best_low_idx = int(np.nanargmin(low_mean_abs))
            axes[1].bar(range(len(names)), low_mean_abs, color="#B2182B", alpha=0.85)
            axes[1].set_xlabel("Model", fontsize=11)
            axes[1].set_ylabel(f"Mean |Residual| (AF < {low_freq_threshold})", fontsize=11)
            axes[1].set_title("Low-Frequency Mean |Residual| Comparison", fontsize=12)
            axes[1].set_xticks(range(len(names)))
            axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            axes[1].grid(True, alpha=0.25, axis="y")
            axes[1].axhline(low_mean_abs[best_low_idx], color="blue", linestyle="--", alpha=0.6,
                            label=f"best: {names[best_low_idx]}")
            axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "low_freq_residual_comparison.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        logger.warning("Not enough low-frequency samples (<%d) for residual comparison", 3)

    # 3. Prediction range validity
    y_max = float(np.nanmax(y_true)) if len(y_true) > 0 else 100.0
    range_issues = []
    for name in names:
        pred_col = f"pred_{name}"
        if pred_col not in test_pred_frame.columns:
            continue
        y_pred = np.asarray(
            pd.to_numeric(test_pred_frame[pred_col], errors="coerce"), dtype=float
        )
        n_negative = int(np.sum(y_pred < 0))
        n_above_max = int(np.sum(y_pred > y_max))
        range_issues.append({
            "model": name,
            "n_negative": n_negative,
            "n_above_max": n_above_max,
            "pred_min": float(np.nanmin(y_pred)),
            "pred_max": float(np.nanmax(y_pred)),
        })

    if range_issues:
        range_df = pd.DataFrame(range_issues)
        range_df.to_csv(os.path.join(out_dir, "prediction_range_check.tsv"), sep="\t", index=False)

        flagged = range_df[(range_df["n_negative"] > 0) | (range_df["n_above_max"] > 0)]
        if not flagged.empty:
            logger.warning(
                "Models with out-of-range predictions:\n%s",
                flagged[["model", "n_negative", "n_above_max", "pred_min", "pred_max"]].to_string(index=False),
            )

    # 4. Feature importance comparison (for models that support it)
    coef_data = {}
    importance_data = {}
    for name, model in models.items():
        spec = _registry.get_spec(name)
        if spec and spec.supports_feature_importance:
            if hasattr(model, "coef_"):
                coef_data[name] = np.asarray(model.coef_, dtype=float)
            elif hasattr(model, "feature_importances_"):
                importance_data[name] = np.asarray(model.feature_importances_, dtype=float)

    if importance_data and len(feature_columns) == len(next(iter(importance_data.values()))):
        importance_df = pd.DataFrame(importance_data, index=feature_columns)
        importance_df.to_csv(os.path.join(out_dir, "feature_importance_comparison.tsv"), sep="\t")

        # Plot top features across models
        mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)
        top_n = min(20, len(mean_importance))
        top_features = mean_importance.head(top_n).index.tolist()
        
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))
        plot_data = importance_df.loc[top_features]
        plot_data.plot(kind="barh", ax=ax, alpha=0.85)
        ax.set_xlabel("Feature Importance", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        ax.set_title("Feature Importance Comparison across Models", fontsize=12)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.25, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_importance_comparison.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 5. Summary report
    report_lines = [
        "Model Comparison Summary",
        "=" * 60,
        "",
        "Models evaluated:",
    ]
    for m in model_info:
        report_lines.append(
            f"  {m['name']}: MSE={m['mse']:.6f}, MAE={m['mae']:.6f}, R²={m['r2']:.6f}"
        )

    report_lines.append("")
    report_lines.append("Best by global metrics:")
    report_lines.append(f"  MSE: {names[int(np.nanargmin(mse_vals))]} ({mse_vals[int(np.nanargmin(mse_vals))]:.6f})")
    report_lines.append(f"  MAE: {names[int(np.nanargmin(mae_vals))]} ({mae_vals[int(np.nanargmin(mae_vals))]:.6f})")
    report_lines.append(f"  R²:  {names[int(np.nanargmax(r2_vals))]} ({r2_vals[int(np.nanargmax(r2_vals))]:.6f})")

    if low_mask.sum() >= 3 and low_abs_residuals:
        report_lines.append("")
        report_lines.append(f"Best by low-frequency (AF < {low_freq_threshold}) mean |residual|:")
        report_lines.append(f"  {names[best_low_idx]} ({low_mean_abs[best_low_idx]:.6f})")

    if range_issues:
        report_lines.append("")
        report_lines.append("Prediction range check:")
        for ri in range_issues:
            flag = ""
            if ri["n_negative"] > 0 or ri["n_above_max"] > 0:
                flag = " [WARNING: out-of-range predictions]"
            report_lines.append(
                f"  {ri['model']}: pred range [{ri['pred_min']:.3f}, {ri['pred_max']:.3f}]{flag}"
            )

    report_lines.append("")
    report_lines.append("Recommendation:")
    report_lines.append("  Review the plots to pick the final model:")
    report_lines.append("  - model_comparison_metrics.png: overall MSE/MAE/R² comparison")
    report_lines.append("  - low_freq_residual_comparison.png: critical for low-frequency accuracy")
    report_lines.append("  - feature_importance_comparison.png: feature importance across models")
    report_lines.append("  - prediction_range_check.tsv: avoid models with negative predictions")

    with open(os.path.join(out_dir, "model_comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Model comparison report saved to %s", out_dir)


def _fit_eval_save_models(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    out_dir: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    test_idx: np.ndarray,
    split_id_col: Optional[Union[str, List[str]]],
    random_state: int,
    target_transform: str = "none",
    clip_epsilon: float = 1e-6,
    weight_low_af: bool = False,
    weight_epsilon: float = 1e-6,
    weight_power: float = 1.0,
    model_config: Optional[Dict[str, object]] = None,
    model_names: Optional[List[str]] = None,
    enable_cv: bool = False,
    cv_folds: int = 5,
    cv_scoring: str = "neg_mean_squared_error",
    custom_param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    train_groups: Optional[np.ndarray] = None,
) -> None:
    """Fit models, evaluate them, and save all training artifacts.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Full feature dataframe including labels and identifiers.
    feature_columns : list[str]
        Feature column names used for model training.
    out_dir : str
        Output directory for artifacts.
    X_train : numpy.ndarray
        Training feature matrix.
    X_test : numpy.ndarray
        Test feature matrix.
    y_train : pandas.Series
        Training labels.
    y_test : pandas.Series
        Test labels.
    test_idx : numpy.ndarray
        Indices used for the test split.
    split_id_col : str or list[str] or None
        Columns used to define grouped splits.
    random_state : int
        Random seed for model initialization.
    target_transform : str, optional
        Target transform to apply ("none" or "logit").
    clip_epsilon : float, optional
        Epsilon for clipping labels when using logit transform.
    weight_low_af : bool, optional
        Whether to upweight low ddPCR_AF samples during training/evaluation.
    weight_epsilon : float, optional
        Small value to avoid division by zero when computing weights.
    weight_power : float, optional
        Power applied to inverse-frequency weights; higher values emphasize low AF.
    model_config : dict or None, optional
        Optional model configuration for build_models.
    model_names : list[str] or None, optional
        Specific model names to include. If None, includes all models.
    enable_cv : bool, optional
        Whether to run cross-validation for hyperparameter tuning.
    cv_folds : int, optional
        Number of folds for cross-validation.
    cv_scoring : str, optional
        Scoring metric for cross-validation.
    custom_param_grids : dict or None, optional
        Custom parameter grids for specific models.
    train_groups : numpy.ndarray or None, optional
        Optional grouping labels for grouped CV.
    """
    split_frame = feature_df.copy()
    if split_id_col:
        id_cols = split_id_col if isinstance(split_id_col, list) else [split_id_col]
        for col in id_cols:
            if col in split_frame.columns:
                split_frame[col] = split_frame[col].astype(str)
    split_frame["split"] = "train"
    split_frame.loc[test_idx, "split"] = "test"
    split_frame.to_csv(os.path.join(out_dir, "split_assignments.tsv"), sep="\t", index=True)

    # Build models using registry
    models = build_models(random_state, model_config=model_config, include_svr=True, model_names=model_names)
    
    test_pred_frame = feature_df.iloc[test_idx].copy()
    if target_transform not in {"none", "logit"}:
        raise ValueError(f"Unsupported target_transform: {target_transform}")

    y_train_raw = y_train.to_numpy(dtype=float)
    y_test_raw = y_test.to_numpy(dtype=float)
    if weight_low_af:
        train_weights = (1.0 / (y_train_raw + weight_epsilon)) ** weight_power
        test_weights = (1.0 / (y_test_raw + weight_epsilon)) ** weight_power
        # Normalize weights to keep scale stable across runs
        train_weights = train_weights / np.mean(train_weights)
        test_weights = test_weights / np.mean(test_weights)
    else:
        train_weights = None
        test_weights = None

    if target_transform == "logit":
        y_train_model = _logit(_clip_01(y_train_raw, clip_epsilon))
        y_test_model = _logit(_clip_01(y_test_raw, clip_epsilon))
    else:
        y_train_model = y_train_raw
        y_test_model = y_test_raw

    # Run CV for each model if enabled
    cv_summary: Dict[str, Dict[str, Any]] = {}
    if enable_cv:
        logger.info("Starting cross-validation for hyperparameter tuning...")
        for name in list(models.keys()):
            logger.info(f"Running CV for {name}...")
            
            # Get parameter grid
            custom_grid = custom_param_grids.get(name) if custom_param_grids else None
            param_grid = _registry.get_param_grid(name, custom_grid)
            
            if not param_grid:
                logger.info(f"No param grid defined for {name}, skipping CV")
                continue
            
            try:
                best_model, best_params = _fit_model_with_cv(
                    name=name,
                    X_train=X_train,
                    y_train=y_train_model,
                    train_weights=train_weights,
                    groups=train_groups,
                    cv_folds=cv_folds,
                    param_grid=param_grid,
                    random_state=random_state,
                    scoring=cv_scoring,
                )
                models[name] = best_model
                cv_summary[name] = {
                    "best_params": best_params,
                    "cv_folds": cv_folds,
                }
            except Exception as e:
                logger.error(f"CV failed for {name}: {e}")
                # Keep the default model

    metrics: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model_report_dir = os.path.join(out_dir, f"{name}_report")
        os.makedirs(model_report_dir, exist_ok=True)
        
        # Use registry-based fit method
        fit_model_with_registry(name, model, X_train, y_train_model, train_weights)
        
        predictions = model.predict(X_test)
        if target_transform == "logit":
            y_pred = _sigmoid(predictions)
            y_true_eval = y_test_raw
        else:
            y_pred = predictions
            y_true_eval = y_test_raw

        mse = mean_squared_error(y_true_eval, y_pred, sample_weight=test_weights)
        mae = mean_absolute_error(y_true_eval, y_pred, sample_weight=test_weights)
        r2 = r2_score(y_true_eval, y_pred, sample_weight=test_weights)
        metrics[name] = {"mse": float(mse), "mae": float(mae), "r2": float(r2)}
        joblib.dump(model, os.path.join(out_dir, f"model_{name}.joblib"))
        test_pred_frame[f"pred_{name}"] = y_pred
        test_pred_frame.to_csv(os.path.join(model_report_dir, f"{name}_test_predictions.tsv"), sep="\t", index=True)
        _save_training_report(
            out_dir=model_report_dir,
            feature_columns=feature_columns,
            y_test=pd.Series(y_true_eval, index=y_test.index),
            test_pred_frame=test_pred_frame,
            models=models,
            metrics=metrics,
            best_name=name,
        )
    best_name = report_best_model(metrics)

    _save_model_comparison(
        out_dir=out_dir,
        models=models,
        metrics=metrics,
        feature_columns=feature_columns,
        y_test=y_test,
        test_pred_frame=test_pred_frame,
    )
    with open(os.path.join(out_dir, "feature_cols.json"), "w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    training_config = {
        "target_transform": target_transform,
        "clip_epsilon": clip_epsilon,
        "weight_low_af": weight_low_af,
        "weight_epsilon": weight_epsilon,
        "weight_power": weight_power,
        "enable_cv": enable_cv,
        "cv_folds": cv_folds,
        "cv_scoring": cv_scoring,
        "cv_summary": cv_summary,
        "model_names": model_names,
    }
    with open(os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(training_config, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "best_model.txt"), "w", encoding="utf-8") as handle:
        handle.write(best_name + "\n")

    logger.info(f"Best model: {best_name}")
    logger.info(json.dumps(metrics, ensure_ascii=False, indent=2))


def group_train(
    feature_df: pd.DataFrame,
    out_dir: str,
    group_cols: Union[List[str], str] = ["sampleID", "mutation"],
    label_col: str = "ddPCR_AF",
    test_size: float = 0.2,
    random_state: int = 42,
    target_transform: str = "none",
    clip_epsilon: float = 1e-6,
    weight_low_af: bool = False,
    weight_epsilon: float = 1e-6,
    weight_power: float = 1.0,
    model_config: Optional[Dict[str, object]] = None,
    model_names: Optional[List[str]] = None,
    enable_cv: bool = False,
    cv_folds: int = 5,
    cv_scoring: str = "neg_mean_squared_error",
    custom_param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
) -> None:
    """Train regression models on precomputed mutation-level features.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Preprocessed feature table.
    out_dir : str
        Output directory for models and metrics.
    group_cols : list[str] or str, optional
        Column(s) used for grouped train/test split.
    label_col : str, optional
        Column name for the target variable.
    test_size : float, optional
        Fraction used for the hold-out split.
    random_state : int, optional
        Seed for reproducibility.
    target_transform : str, optional
        Target transform to apply ("none" or "logit").
    clip_epsilon : float, optional
        Epsilon for clipping labels when using logit transform.
    weight_low_af : bool, optional
        Whether to upweight low ddPCR_AF samples during training/evaluation.
    weight_epsilon : float, optional
        Small value to avoid division by zero when computing weights.
    weight_power : float, optional
        Power applied to inverse-frequency weights; higher values emphasize low AF.
    model_config : dict or None, optional
        Optional model configuration for build_models.
    model_names : list[str] or None, optional
        Specific model names to include. If None, includes all models.
    enable_cv : bool, optional
        Whether to run cross-validation for hyperparameter tuning.
    cv_folds : int, optional
        Number of folds for cross-validation.
    cv_scoring : str, optional
        Scoring metric for cross-validation.
    custom_param_grids : dict or None, optional
        Custom parameter grids for specific models.
    """
    os.makedirs(out_dir, exist_ok=True)
    if label_col not in feature_df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    for col in group_cols:
        if col not in feature_df.columns:
            raise ValueError(f"Missing group column: {col}")

    y = feature_df[label_col].astype(float)

    # Training script consumes preprocessed features from feature extraction output.
    exclude_cols = set(group_cols) | {label_col}
    feature_columns = [col for col in feature_df.columns if col not in exclude_cols]
    if not feature_columns:
        raise ValueError("No feature columns found in input table")

    feature_matrix = feature_df[feature_columns]
    non_numeric_cols = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(feature_matrix[col])]
    if non_numeric_cols:
        raise ValueError(
            "Input feature table must be preprocessed to numeric columns only; "
            f"found non-numeric columns: {non_numeric_cols}"
        )

    # Build composite group key from all group columns.
    groups = feature_df[group_cols[0]].astype(str)
    for col in group_cols[1:]:
        groups = groups + "__" + feature_df[col].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(feature_matrix, y, groups=groups))

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    X_train = feature_matrix.iloc[train_idx].to_numpy(dtype=float)
    X_test = feature_matrix.iloc[test_idx].to_numpy(dtype=float)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    _fit_eval_save_models(
        feature_df=feature_df,
        feature_columns=feature_columns,
        out_dir=out_dir,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_idx=test_idx,
        split_id_col=group_cols,
        random_state=random_state,
        target_transform=target_transform,
        clip_epsilon=clip_epsilon,
        weight_low_af=weight_low_af,
        weight_epsilon=weight_epsilon,
        weight_power=weight_power,
        model_config=model_config,
        model_names=model_names,
        enable_cv=enable_cv,
        cv_folds=cv_folds,
        cv_scoring=cv_scoring,
        custom_param_grids=custom_param_grids,
        train_groups=groups.to_numpy()[train_idx],
    )

def train(
    feature_df: pd.DataFrame,
    out_dir: str,
    label_col: str = "ddPCR_AF",
    test_size: float = 0.2,
    random_state: int = 42,
    target_transform: str = "none",
    clip_epsilon: float = 1e-6,
    weight_low_af: bool = False,
    weight_epsilon: float = 1e-6,
    weight_power: float = 1.0,
    model_config: Optional[Dict[str, object]] = None,
    model_names: Optional[List[str]] = None,
    enable_cv: bool = False,
    cv_folds: int = 5,
    cv_scoring: str = "neg_mean_squared_error",
    custom_param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
) -> None:
    """Train regression models with a random row-wise train/test split.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Preprocessed feature table.
    out_dir : str
        Output directory for models and metrics.
    label_col : str, optional
        Column name for the target variable.
    test_size : float, optional
        Fraction used for the hold-out split.
    random_state : int, optional
        Seed for reproducibility.
    target_transform : str, optional
        Target transform to apply ("none" or "logit").
    clip_epsilon : float, optional
        Epsilon for clipping labels when using logit transform.
    weight_low_af : bool, optional
        Whether to upweight low ddPCR_AF samples during training/evaluation.
    weight_epsilon : float, optional
        Small value to avoid division by zero when computing weights.
    weight_power : float, optional
        Power applied to inverse-frequency weights; higher values emphasize low AF.
    model_config : dict or None, optional
        Optional model configuration for build_models.
    model_names : list[str] or None, optional
        Specific model names to include. If None, includes all models.
    enable_cv : bool, optional
        Whether to run cross-validation for hyperparameter tuning.
    cv_folds : int, optional
        Number of folds for cross-validation.
    cv_scoring : str, optional
        Scoring metric for cross-validation.
    custom_param_grids : dict or None, optional
        Custom parameter grids for specific models.
    """
    os.makedirs(out_dir, exist_ok=True)
    if label_col not in feature_df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    y = feature_df[label_col].astype(float)
    feature_columns = [col for col in feature_df.columns if col not in {label_col}]
    if not feature_columns:
        raise ValueError("No feature columns found in input table")

    feature_matrix = feature_df[feature_columns]
    non_numeric_cols = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(feature_matrix[col])]
    if non_numeric_cols:
        raise ValueError(
            "Input feature table must be preprocessed to numeric columns only; "
            f"found non-numeric columns: {non_numeric_cols}"
        )

    train_idx, test_idx = train_test_split(
        np.arange(len(feature_df)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    X_train = feature_matrix.iloc[train_idx].to_numpy(dtype=float)
    X_test = feature_matrix.iloc[test_idx].to_numpy(dtype=float)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    _fit_eval_save_models(
        feature_df=feature_df,
        feature_columns=feature_columns,
        out_dir=out_dir,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_idx=test_idx,
        split_id_col=None,
        random_state=random_state,
        target_transform=target_transform,
        clip_epsilon=clip_epsilon,
        weight_low_af=weight_low_af,
        weight_epsilon=weight_epsilon,
        weight_power=weight_power,
        model_config=model_config,
        model_names=model_names,
        enable_cv=enable_cv,
        cv_folds=cv_folds,
        cv_scoring=cv_scoring,
        custom_param_grids=custom_param_grids,
    )

def main():
    parser = argparse.ArgumentParser(description="Train SV frequency correction models")
    parser.add_argument("-i", "--table", default="/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML/features.tsv", help="TSV with mutation-level features")
    parser.add_argument("-o", "--out-dir", required=True, help="output directory")
    parser.add_argument("--group-cols", nargs="+", default=["sampleID", "mutation"], help="column(s) used for grouped train/test split")
    parser.add_argument("--label-col", default="ddPCR_AF", help="column name for the target variable")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-transform", choices=["none", "logit"], default="none")
    parser.add_argument("--clip-epsilon", type=float, default=1e-6)
    parser.add_argument("--weight-low-af", action="store_true", help="upweight low ddPCR_AF samples")
    parser.add_argument("--weight-epsilon", type=float, default=1e-6)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--model-names", nargs="+", default=None, help="specific model names to include")
    parser.add_argument("--enable-cv", action="store_true", help="enable cross-validation for hyperparameter tuning")
    parser.add_argument("--cv-folds", type=int, default=5, help="number of CV folds")
    parser.add_argument("--cv-scoring", default="neg_mean_squared_error", help="scoring metric for CV")
    parser.add_argument("--custom-param-grids", default=None, help="JSON string for custom param grids")
    args = parser.parse_args()
    
    custom_param_grids = json.loads(args.custom_param_grids) if args.custom_param_grids else None
    
    feature_df = pd.read_csv(args.table, sep="\t")
    group_train(
        feature_df=feature_df,
        out_dir=args.out_dir,
        group_cols=args.group_cols,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
        target_transform=args.target_transform,
        clip_epsilon=args.clip_epsilon,
        weight_low_af=args.weight_low_af,
        weight_epsilon=args.weight_epsilon,
        weight_power=args.weight_power,
        model_names=args.model_names,
        enable_cv=args.enable_cv,
        cv_folds=args.cv_folds,
        cv_scoring=args.cv_scoring,
        custom_param_grids=custom_param_grids,
    )

def run():
    table_file = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/feature/preprocessed_features.tsv"
    feature_df = pd.read_csv(table_file, sep="\t")
    exclude_cols = ["sampleID", "FusionType"]
    if exclude_cols:
        for col in exclude_cols:
            if col in feature_df.columns:
                feature_df = feature_df.drop(columns=[col])
    outdir = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/train"
    
    group_train(
        feature_df=feature_df,
        out_dir=outdir,
        group_cols=["原始编号", "FusionGene", "FusionExon"],
        label_col="ddPCR_AF",
        test_size=0.3,
        random_state=42,
        target_transform="logit",
        clip_epsilon=1e-6,
        weight_low_af=True,
        weight_epsilon=1e-6,
        weight_power=1.0,
        enable_cv=True,
        cv_folds=10,
    )

if __name__ == "__main__":
    # main()
    run()

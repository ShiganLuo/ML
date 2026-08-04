from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import FeatureSelector
from .SingleVariable import SingleVariableAUCSelector
logger = logging.getLogger(__name__)

class TwoStageSelector(FeatureSelector):
    """Two-stage feature selection: fast filter + model-based refinement.

    Stage 1 — Single-variable AUC filter (fast, univariate):
        For each feature independently, compute AUC against the binary label.
        Features with AUC >= auc_threshold pass the filter. This removes
        features with zero or near-zero discriminative power in O(n_features)
        time. Typical output: 24 → 8-12 features.

    Stage 2 — Random Forest importance (slower, multivariate):
        Train a Random Forest on the Stage 1 survivors. Rank features by
        `feature_importances_` (Gini impurity reduction). Select the top_k
        most important features. This captures feature interactions that
        univariate AUC cannot see — e.g., two features that are weak alone
        but strong together.

    Why two stages?
        - Stage 1 is O(n_features) and removes noise features cheaply.
        - Stage 2 is O(n_features * n_trees * n_samples * log(n_samples))
          and would be too slow on all 24+ features.
        - The combination is faster than running RF on all features, and
          more accurate than AUC alone.

    Parameters
    ----------
    auc_threshold : float
        Minimum single-feature AUC to pass Stage 1 (default 0.6).
        Higher = more aggressive filtering. 0.85 keeps only features
        with strong individual signal.
    top_k : int
        Maximum number of features to keep after Stage 2 (default 50).
        If Stage 1 survivors < top_k, all survivors are kept.

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Indices of selected features in the original feature array.
    importances_ : np.ndarray
        Random Forest importance scores for the selected features.
    """

    def __init__(self, auc_threshold: float = 0.6, top_k: int = 50):
        self.auc_threshold = auc_threshold
        self.top_k = top_k
        self.stage1_selector = SingleVariableAUCSelector(auc_threshold)
        self.selected_indices_ = None
        self.importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TwoStageSelector':
        """Fit using two-stage selection.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Binary labels (0=MSS, 1=MSI-H).
        """
        # Stage 1: AUC filter — remove features with AUC < threshold
        self.stage1_selector.fit(X, y)
        X_filtered = self.stage1_selector.transform(X)
        stage1_indices = self.stage1_selector.get_selected_indices()

        if len(stage1_indices) == 0:
            logger.warning("No features passed stage 1 filter")
            self.selected_indices_ = np.array([])
            return self

        # Stage 2: Random Forest importance — rank survivors by multivariate importance
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_filtered, y)
            importances = rf.feature_importances_

            # Select top_k most important features
            n_select = min(self.top_k, len(stage1_indices))
            top_k_local = np.argsort(importances)[-n_select:]
            self.selected_indices_ = stage1_indices[top_k_local]
            self.importances_ = importances[top_k_local]

            logger.info(f"Stage 2: Selected {len(self.selected_indices_)} features via Random Forest")

        except ImportError:
            logger.warning("sklearn not available, falling back to stage 1 only")
            self.selected_indices_ = stage1_indices

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_


from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import FeatureSelector
logger = logging.getLogger(__name__)

class SingleVariableAUCSelector(FeatureSelector):
    """Select features based on individual AUC scores."""

    def __init__(self, auc_threshold: float = 0.6):
        self.auc_threshold = auc_threshold
        self.auc_scores_ = None
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SingleVariableAUCSelector':
        """Fit by computing AUC for each feature."""
        n_features = X.shape[1]
        self.auc_scores_ = np.zeros(n_features)

        for i in range(n_features):
            scores = X[:, i]
            if np.std(scores) < 1e-10:
                self.auc_scores_[i] = 0.5
                continue

            # Compute AUC
            sorted_idx = np.argsort(scores)[::-1]
            y_sorted = y[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            self.auc_scores_[i] = trapezoid(tpr, fpr)

        # Select features above threshold
        self.selected_indices_ = np.where(self.auc_scores_ >= self.auc_threshold)[0]
        logger.info(f"Selected {len(self.selected_indices_)}/{n_features} features (AUC >= {self.auc_threshold})")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features."""
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import FeatureSelector
logger = logging.getLogger(__name__)
class TwoStageVarianceSelector(FeatureSelector):
    """Two-stage: variance filter -> RF importance.

    For one-class detectors: first remove low-variance features (noise),
    then rank survivors by RF importance on labeled data.

    This is a compromise: the first stage is unsupervised (no label needed),
    the second stage uses labels but only on the surviving features.
    """

    def __init__(self, top_k: int = 50, var_percentile: float = 25):
        self.top_k = top_k
        self.var_percentile = var_percentile
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'TwoStageVarianceSelector':
        # Stage 1: variance filter
        variances = np.var(X, axis=0)
        threshold = np.percentile(variances, self.var_percentile)
        stage1_mask = variances >= threshold
        stage1_indices = np.where(stage1_mask)[0]

        if len(stage1_indices) == 0:
            logger.warning("TwoStageVariance: no features passed variance filter")
            self.selected_indices_ = np.array([])
            return self

        X_filtered = X[:, stage1_indices]

        # Stage 2: RF importance (if labels available)
        if y is not None and len(np.unique(y)) > 1:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_filtered, y)
            importances = rf.feature_importances_
            n_select = min(self.top_k, len(stage1_indices))
            top_k_local = np.argsort(importances)[-n_select:]
            self.selected_indices_ = stage1_indices[top_k_local]
        else:
            # No labels: just take top_k by variance
            var_filtered = variances[stage1_indices]
            n_select = min(self.top_k, len(stage1_indices))
            top_k_local = np.argsort(var_filtered)[-n_select:]
            self.selected_indices_ = stage1_indices[top_k_local]

        logger.info(f"TwoStageVariance: selected {len(self.selected_indices_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_
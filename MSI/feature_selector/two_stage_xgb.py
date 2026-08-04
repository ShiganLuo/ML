from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import FeatureSelector
from .SingleVariable import SingleVariableAUCSelector
logger = logging.getLogger(__name__)

class TwoStageXgbSelector(FeatureSelector):
    """Two-stage: AUC filter -> XGBoost importance.

    Stage 1: Single-variable AUC filter (fast noise removal).
    Stage 2: XGBoost importance ranking (captures interactions).
    """

    def __init__(self, auc_threshold: float = 0.6, top_k: int = 50):
        self.auc_threshold = auc_threshold
        self.top_k = top_k
        self.selected_indices_ = None
        self.importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TwoStageXgbSelector':
        # Stage 1: AUC filter
        stage1 = SingleVariableAUCSelector(self.auc_threshold)
        stage1.fit(X, y)
        X_filtered = stage1.transform(X)
        stage1_indices = stage1.get_selected_indices()

        if len(stage1_indices) == 0:
            logger.warning("TwoStageXgb: No features passed stage 1")
            self.selected_indices_ = np.array([])
            return self

        # Stage 2: XGBoost importance
        xgb_sel = XgbImportanceSelector(top_k=self.top_k)
        xgb_sel.fit(X_filtered, y)

        self.selected_indices_ = stage1_indices[xgb_sel.selected_indices_]
        self.importances_ = xgb_sel.importances_

        logger.info(f"TwoStageXgb selected {len(self.selected_indices_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_

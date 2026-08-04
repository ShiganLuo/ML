from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import FeatureSelector
logger = logging.getLogger(__name__)

class XgbImportanceSelector(FeatureSelector):
    """Feature selection using XGBoost feature importance (gain).

    Trains an XGBoost classifier on all features, ranks by importance,
    and selects the top_k features. This is an embedded method that
    captures feature interactions naturally.
    """

    def __init__(self, top_k: int = 50, importance_type: str = 'gain'):
        self.top_k = top_k
        self.importance_type = importance_type
        self.selected_indices_ = None
        self.importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XgbImportanceSelector':
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("xgboost not available, falling back to RF importance")
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            n_select = min(self.top_k, X.shape[1])
            self.selected_indices_ = np.argsort(importances)[-n_select:]
            self.importances_ = importances[self.selected_indices_]
            return self

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        spw = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=42,
            eval_metric='logloss', n_jobs=-1,
        )
        model.fit(X, y)

        booster = model.get_booster()
        score = booster.get_score(importance_type=self.importance_type)
        importances = np.zeros(X.shape[1])
        for key, val in score.items():
            idx = int(key[1:]) if key.startswith('f') else -1
            if 0 <= idx < X.shape[1]:
                importances[idx] = val

        n_select = min(self.top_k, X.shape[1])
        self.selected_indices_ = np.argsort(importances)[-n_select:]
        self.importances_ = importances[self.selected_indices_]

        logger.info(f"XgbImportance selected {len(self.selected_indices_)}/{X.shape[1]} features "
                     f"(top {self.top_k} by {self.importance_type})")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_

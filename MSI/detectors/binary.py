# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Binary classifier detector (XGBoost, Logistic)."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .base import Detector
import logging
logger = logging.getLogger(__name__)
class BinaryClassifierDetector(Detector):
    """Supervised binary classifier for MSI detection.

    Trains on BOTH MSS and MSI-H samples (unlike one-class detectors
    that only see MSS). Uses XGBoost if available, falls back to
    Logistic Regression with L2 regularisation.

    The score is P(MSI-H), so higher = more likely MSI-H.

    Parameters
    ----------
    method : str
        'xgboost' or 'logistic'.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, method: str = 'xgboost', random_state: int = 42):
        self.method = method
        self.random_state = random_state
        self.model_ = None
        self.scaler_ = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'BinaryClassifierDetector':
        from sklearn.preprocessing import StandardScaler

        if y_train is None:
            raise ValueError("BinaryClassifierDetector requires y_train labels")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        y = np.asarray(y_train, dtype=int)

        if self.method == 'xgboost':
            try:
                from xgboost import XGBClassifier
                # Handle class imbalance
                n_pos = y.sum()
                n_neg = len(y) - n_pos
                scale_pos_weight = n_neg / max(n_pos, 1)
                self.model_ = XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=-1,
                )
            except ImportError:
                logger.warning("xgboost not available, falling back to logistic regression")
                self.method = 'logistic'

        if self.method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.model_ = LogisticRegression(
                penalty='l2', C=1.0, solver='lbfgs',
                max_iter=1000, random_state=self.random_state,
                class_weight='balanced',
            )

        self.model_.fit(X_scaled, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

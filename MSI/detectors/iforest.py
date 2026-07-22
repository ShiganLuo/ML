# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Isolation Forest detector."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .base import Detector

class IsolationForestDetector(Detector):
    """Isolation Forest anomaly detection.

    Isolates anomalies by randomly selecting features and split values.
    Anomalies are isolated in fewer splits, yielding lower path lengths.
    Non-parametric: no Gaussian assumption, handles multi-modal distributions.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    contamination : float or 'auto'
        Expected proportion of outliers. Used for decision_function scaling.
    max_samples : int or float or 'auto'
        Samples drawn to build each tree.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 'auto',
        max_samples: float = 1.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model_ = None
        self.scaler_ = None

    def fit(self, X_train: np.ndarray) -> 'IsolationForestDetector':
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        # decision_function: lower = more anomalous; negate for consistency
        return -self.model_.decision_function(X_scaled)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""One-Class SVM detector."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .base import Detector

class OneClassSVMDetector(Detector):
    """One-Class SVM anomaly detection.

    Uses RBF kernel to learn a non-linear boundary around normal (MSS) data.
    More flexible than Mahalanobis for non-Gaussian distributions, but
    requires tuning nu and gamma.
    """

    def __init__(self, nu: float = 0.1, gamma: str = 'scale'):
        """
        Parameters
        ----------
        nu : float
            Upper bound on the fraction of training errors (0 < nu <= 1).
            Roughly corresponds to the expected proportion of outliers in training data.
        gamma : str or float
            RBF kernel coefficient. 'scale' = 1 / (n_features * X.var()).
        """
        self.nu = nu
        self.gamma = gamma
        self.model_ = None
        self.scaler_ = None

    def fit(self, X_train: np.ndarray) -> 'OneClassSVMDetector':
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        self.model_ = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)
        self.model_.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        return -self.model_.decision_function(X_scaled)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

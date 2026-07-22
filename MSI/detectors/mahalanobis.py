# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Mahalanobis distance detector."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .base import Detector

class MahalanobisDetector(Detector):
    """Mahalanobis distance based anomaly detection."""

    def __init__(self):
        self.mean_ = None
        self.cov_inv_ = None

    def fit(self, X_train: np.ndarray) -> 'MahalanobisDetector':
        self.mean_ = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularize
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def _mahal(self, x: np.ndarray) -> float:
        diff = x - self.mean_
        return np.sqrt(diff @ self.cov_inv_ @ diff)

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._mahal(x) for x in X])

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

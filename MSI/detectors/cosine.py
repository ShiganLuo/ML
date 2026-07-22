# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Cosine distance detector for MSI anomaly detection."""

from typing import Optional
import numpy as np
from .base import Detector


class CosineDetector(Detector):
    """Anomaly detection via cosine distance from training centroid.

    Computes cosine distance = 1 - cosine_similarity between each sample
    and the centroid of the training set (MSS samples).

    More robust to noise than Mahalanobis in high dimensions because it
    measures direction, not magnitude. No covariance estimation needed.

    Parameters
    ----------
    metric : str
        'cosine' (default) or 'correlation'.
        correlation: 1 - Pearson correlation with centroid.
    """

    def __init__(self, metric: str = 'cosine'):
        self.metric = metric
        self.centroid_ = None

    def fit(self, X_train: np.ndarray, y_train=None) -> 'CosineDetector':
        """Fit by computing the centroid of training samples."""
        self.centroid_ = np.mean(X_train, axis=0)
        # Avoid zero centroid
        if np.linalg.norm(self.centroid_) < 1e-10:
            self.centroid_ = np.ones(X_train.shape[1]) / X_train.shape[1]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute cosine distance to centroid for each sample."""
        scores = np.zeros(X.shape[0])
        c = self.centroid_
        c_norm = np.linalg.norm(c)

        for i in range(X.shape[0]):
            x = X[i]
            if self.metric == 'correlation':
                # Pearson correlation distance
                x_centered = x - np.mean(x)
                c_centered = c - np.mean(c)
                dot = np.dot(x_centered, c_centered)
                denom = np.linalg.norm(x_centered) * np.linalg.norm(c_centered)
            else:
                dot = np.dot(x, c)
                denom = np.linalg.norm(x) * c_norm

            if denom < 1e-10:
                scores[i] = 1.0
            else:
                cosine_sim = np.clip(dot / denom, -1.0, 1.0)
                scores[i] = 1.0 - cosine_sim

        return scores

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

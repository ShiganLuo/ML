# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Ensemble detector."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .base import Detector

class EnsembleDetector(Detector):
    """Ensemble of multiple detectors."""

    def __init__(self, detectors: List[Detector], weights: Optional[List[float]] = None):
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)

    def fit(self, X_train: np.ndarray) -> 'EnsembleDetector':
        for d in self.detectors:
            d.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        scores = []
        for d, w in zip(self.detectors, self.weights):
            s = d.score(X)
            # Normalize to [0, 1]
            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
            scores.append(s_norm * w)

        return np.sum(scores, axis=0)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

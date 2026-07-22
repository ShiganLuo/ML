# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""MSI percentage detector."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .base import Detector

class MSIPercentageDetector(Detector):
    """MSI percentage based detection (like msisensor-pro)."""

    def __init__(self, threshold_col: str = 'msi_pct'):
        self.threshold_col = threshold_col

    def fit(self, X_train: np.ndarray) -> 'MSIPercentageDetector':
        # No training needed
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # X should be a single column (msi_pct)
        return X.flatten()

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

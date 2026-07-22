# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Detector abstract base class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

class Detector(ABC):
    """Abstract base class for MSI detection."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'Detector':
        """Fit the detector.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray or None
            Training labels (1=MSI-H, 0=MSS). None for one-class detectors.
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Predict MSI status using threshold."""
        pass

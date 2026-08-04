from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import FeatureSelector
logger = logging.getLogger(__name__)

class LassoSelector(FeatureSelector):
    """Feature selection using L1 regularization."""

    def __init__(self, C: float = 0.1):
        self.C = C
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoSelector':
        """Fit using L1 logistic regression."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression(
                penalty='l1', C=self.C, solver='saga',
                max_iter=1000, random_state=42
            )
            model.fit(X_scaled, y)

            # Non-zero coefficients
            coef = np.abs(model.coef_[0])
            self.selected_indices_ = np.where(coef > 1e-6)[0]

            logger.info(f"Lasso selected {len(self.selected_indices_)} features")

        except ImportError:
            logger.warning("sklearn not available")
            self.selected_indices_ = np.arange(X.shape[1])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_
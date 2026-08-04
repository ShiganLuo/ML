from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import FeatureSelector
logger = logging.getLogger(__name__)

class VarianceSelector(FeatureSelector):
    """Unsupervised feature selection by variance.

    Selects the top_k features with highest variance. No labels needed.
    Suitable for one-class detectors (mahalanobis, cosine, ocsvm, etc.)
    where supervised AUC-based selection is inappropriate.

    Also supports IQR-based selection: features whose IQR exceeds a
    percentile threshold are kept.
    """

    def __init__(self, top_k: int = 50, method: str = 'variance'):
        self.top_k = top_k
        self.method = method  # 'variance' or 'iqr'
        self.selected_indices_ = None
        self.variances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'VarianceSelector':
        if self.method == 'iqr':
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            n_select = min(self.top_k, X.shape[1])
            self.selected_indices_ = np.argsort(iqr)[-n_select:]
            self.variances_ = iqr
        else:
            variances = np.var(X, axis=0)
            n_select = min(self.top_k, X.shape[1])
            self.selected_indices_ = np.argsort(variances)[-n_select:]
            self.variances_ = variances

        logger.info(f"VarianceSelector ({self.method}): selected {len(self.selected_indices_)}/{X.shape[1]} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_

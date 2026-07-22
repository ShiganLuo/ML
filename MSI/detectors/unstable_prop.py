# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""UnstableProportionDetector v2: depth-weighted, signal-strength scoring."""

from typing import Dict, Optional, List
import numpy as np
import logging
from .base import Detector

logger = logging.getLogger(__name__)


class UnstableProportionDetector(Detector):
    """MSI detection based on unstable locus proportions.

    Supports multiple scoring modes:
    - 'dw_prop': depth-weighted proportion (default, most robust)
    - 'raw_prop': simple proportion (original)
    - 'signal_str': signal strength (sum of excess over threshold)
    - 'ent_w_prop': entropy-weighted proportion

    Parameters
    ----------
    score_cols : list of str, optional
        Feature column names to use as score components.
    weights : list of float, optional
        Weights for each score column. If None, equal weights.
    combine : str
        How to combine scores: 'mean', 'max', or 'weighted'.
    """

    def __init__(self,
                 score_cols: Optional[List[str]] = None,
                 weights: Optional[List[float]] = None,
                 combine: str = 'weighted'):
        self.score_cols = score_cols or [
            'dw_prop_alt_0_10',
            'dw_prop_ent_0_5',
            'signal_str_alt_0_10',
        ]
        self.weights = weights or [0.4, 0.3, 0.3]
        self.combine = combine
        self.col_indices_ = None
        self.required_features = list(self.score_cols)

    def fit(self, X_train: np.ndarray, y_train=None) -> 'UnstableProportionDetector':
        return self

    def set_feature_names(self, feature_names: List[str]) -> None:
        """Map score column names to matrix column indices."""
        self.col_indices_ = []
        for col in self.score_cols:
            if col in feature_names:
                self.col_indices_.append(feature_names.index(col))
            else:
                logger.warning(f"Score column '{col}' not found in feature names")
        if not self.col_indices_:
            logger.warning("No valid score columns found, will use first column")
            self.col_indices_ = [0]

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute MSI score as weighted combination of unstable proportions."""
        if self.col_indices_ is None:
            return X.mean(axis=1)

        cols = X[:, self.col_indices_]

        if self.combine == 'mean':
            return cols.mean(axis=1)
        elif self.combine == 'max':
            return cols.max(axis=1)
        else:  # weighted
            w = np.array(self.weights[:len(self.col_indices_)])
            w = w / w.sum()
            return (cols * w).sum(axis=1)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores >= threshold, 'MSI-H', 'MSS')

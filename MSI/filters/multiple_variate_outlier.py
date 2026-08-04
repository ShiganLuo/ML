from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
from .base import SampleFilter
logger = logging.getLogger(__name__)

class MultivariateOutlierFilter(SampleFilter):
    """Filter samples whose features exceed N standard deviations from the median.

    Uses Mahalanobis-like distance (diagonal approximation) to detect
    samples that are extreme in multiple features simultaneously.
    """

    def __init__(self, n_sigma: float = 4.0):
        self.n_sigma = n_sigma

    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [c for c in features_df.columns
                        if pd.api.types.is_numeric_dtype(features_df[c])
                        and c not in {'MSI_status', 'origin', 'cancertype', 'MSI_CNC'}]
        if not numeric_cols:
            return features_df

        X = features_df[numeric_cols].fillna(0).values
        medians = np.median(X, axis=0)
        mads = np.median(np.abs(X - medians), axis=0) * 1.4826  # MAD -> std estimate
        mads[mads < 1e-10] = 1.0  # avoid division by zero

        z_scores = np.abs((X - medians) / mads)
        # A sample is outlier if ANY feature exceeds n_sigma
        outlier_mask = (z_scores > self.n_sigma).any(axis=1)
        mask = ~outlier_mask

        n_removed = outlier_mask.sum()
        logger.info(f"MultivariateOutlier: removed {n_removed}/{len(features_df)} samples "
                     f"(n_sigma={self.n_sigma})")
        return features_df[mask].copy()

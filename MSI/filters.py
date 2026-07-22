# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Sample filtering."""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)

class SampleFilter(ABC):
    """Abstract base class for sample filtering."""

    @abstractmethod
    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Filter samples, return filtered DataFrame."""
        pass


class DepthFilter(SampleFilter):
    """Filter samples by average depth."""

    def __init__(self, min_depth: float = 100):
        self.min_depth = min_depth

    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if 'mean_depth' in features_df.columns:
            mask = features_df['mean_depth'] >= self.min_depth
            logger.info(f"Depth filter: {mask.sum()}/{len(features_df)} passed")
            return features_df[mask]
        return features_df


class QualityFilter(SampleFilter):
    """Filter samples by quality metrics."""

    def __init__(self, min_loci: int = 100):
        self.min_loci = min_loci

    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if 'n_loci' in features_df.columns:
            mask = features_df['n_loci'] >= self.min_loci
            logger.info(f"Quality filter: {mask.sum()}/{len(features_df)} passed")
            return features_df[mask]
        return features_df


class CombinedFilter(SampleFilter):
    """Combine multiple filters."""

    def __init__(self, filters: List[SampleFilter]):
        self.filters = filters

    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        result = features_df
        for f in self.filters:
            result = f.filter(result)
        return result


class AnomalyFilter(SampleFilter):
    """Filter samples using Isolation Forest anomaly detection.

    Detects multivariate outliers in the feature space that may represent
    contaminated samples, sequencing artifacts, or unusual biology.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("sklearn not available, skipping AnomalyFilter")
            return features_df

        numeric_cols = [c for c in features_df.columns
                        if pd.api.types.is_numeric_dtype(features_df[c])
                        and c not in {'MSI_status', 'origin', 'cancertype', 'MSI_CNC'}]
        if not numeric_cols:
            return features_df

        X = features_df[numeric_cols].fillna(0).values
        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        labels = iso.fit_predict(X)
        mask = labels == 1
        n_removed = (~mask).sum()
        logger.info(f"AnomalyFilter: removed {n_removed}/{len(features_df)} samples "
                     f"(contamination={self.contamination})")
        return features_df[mask].copy()


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

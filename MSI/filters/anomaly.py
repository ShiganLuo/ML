from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
from .base import SampleFilter
logger = logging.getLogger(__name__)

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
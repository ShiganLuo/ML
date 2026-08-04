from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
from .base import SampleFilter
logger = logging.getLogger(__name__)

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

from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
from .base import SampleFilter
logger = logging.getLogger(__name__)

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
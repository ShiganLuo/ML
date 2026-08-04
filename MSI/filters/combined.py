from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
from .base import SampleFilter
class CombinedFilter(SampleFilter):
    """Combine multiple filters."""

    def __init__(self, filters: List[SampleFilter]):
        self.filters = filters

    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        result = features_df
        for f in self.filters:
            result = f.filter(result)
        return result

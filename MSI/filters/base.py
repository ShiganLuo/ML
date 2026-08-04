from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
class SampleFilter(ABC):
    """Abstract base class for sample filtering."""

    @abstractmethod
    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Filter samples, return filtered DataFrame."""
        pass

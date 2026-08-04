from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd

class FeatureSelector(ABC):
    """Abstract base class for feature selection."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """Fit the selector on training data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        pass

    @abstractmethod
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        pass

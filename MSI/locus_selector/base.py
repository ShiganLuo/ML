# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Locus selector abstract base class."""
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class LocusSelector(ABC):
    """Abstract base class for locus-level selection."""

    @abstractmethod
    def fit(self, locus_features: List[Dict], labels: np.ndarray) -> 'LocusSelector':
        """Fit the selector on locus-level data."""
        pass

    @abstractmethod
    def is_selected(self, locus_feat: Dict) -> bool:
        """Check if a locus should be included."""
        pass
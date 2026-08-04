from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import LocusSelector

class NullLocusSelector(LocusSelector):
    """Locus selector that passes all loci through (no filtering)."""

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'NullLocusSelector':
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        return True
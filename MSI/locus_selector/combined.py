from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import LocusSelector

class CombinedLocusSelector(LocusSelector):
    """Combine multiple locus selectors."""

    def __init__(self, selectors: List['LocusSelector']):
        self.selectors = selectors

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'CombinedLocusSelector':
        for s in self.selectors:
            s.fit(locus_data, sample_labels)
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        return all(s.is_selected(locus_feat) for s in self.selectors)

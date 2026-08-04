from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import LocusSelector
class UnitLengthLocusSelector(LocusSelector):
    """Select loci by repeat unit length."""

    def __init__(self, allowed_unit_lens: List[int] = [1, 2, 3]):
        self.allowed_unit_lens = allowed_unit_lens

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'UnitLengthLocusSelector':
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        return locus_feat.get('unit_len') in self.allowed_unit_lens
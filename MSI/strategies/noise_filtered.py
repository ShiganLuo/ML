# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""NoiseFilteredAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from .baseline import BaselineAggregation
from ..utils import _common_features, _unit_len_features

class NoiseFilteredAggregation(AggregationStrategy):
    """Baseline on noise-filtered data (low-freq alleles removed).

    Note: Requires ``repeat_dict`` in locus data. If absent, falls back
    to baseline aggregation.
    """

    def __init__(self, min_freq: float = 0.05):
        self.min_freq = min_freq
        self._baseline = BaselineAggregation()

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        if 'repeat_dict' not in lf.columns:
            return self._baseline.aggregate(lf)

        new_rows = []
        for _, row in lf.iterrows():
            counts = parse_repeat_counts(str(row.get('repeat_dict', '')))
            if not counts:
                new_rows.append(row)
                continue
            total = sum(counts.values())
            fc = {k: v for k, v in counts.items() if v / total >= self.min_freq}
            if fc and sum(fc.values()) > 0:
                row = row.copy()
                filtered_depth = sum(fc.values())
                ref_count = fc.get(int(row['repeat_times']), 0)
                row['alt_ratio'] = 1 - ref_count / filtered_depth
                probs = np.array(list(fc.values())) / filtered_depth
                row['entropy'] = float(-np.sum(probs * np.log2(probs + 1e-10)))
                new_rows.append(row)
        lf_filtered = pd.DataFrame(new_rows)
        return self._baseline.aggregate(lf_filtered)

    def get_feature_names(self) -> List[str]:
        return self._baseline.get_feature_names()

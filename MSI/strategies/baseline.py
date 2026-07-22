# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""BaselineAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class BaselineAggregation(AggregationStrategy):
    """Baseline strategy: mean-only aggregation.

    Common features plus high_alt_ratio and per-unit_len breakdown.
    """

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        f = _common_features(lf)
        f['high_alt_ratio'] = (lf['alt_ratio'] > 0.5).mean()
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

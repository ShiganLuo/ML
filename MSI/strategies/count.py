# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""CountAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class CountAggregation(AggregationStrategy):
    """Count-based: n_loci above multiple thresholds."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'n_alt_gt_0.05', 'n_alt_gt_0.1', 'n_alt_gt_0.2', 'n_alt_gt_0.5',
        'n_entropy_gt_1.5', 'n_entropy_gt_2.0',
        'ratio_alt_gt_0.1', 'ratio_entropy_gt_1.5',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        f = _common_features(lf)
        alt = lf['alt_ratio']
        ent = lf['entropy']
        f.update({
            'n_alt_gt_0.05': int((alt > 0.05).sum()),
            'n_alt_gt_0.1': int((alt > 0.1).sum()),
            'n_alt_gt_0.2': int((alt > 0.2).sum()),
            'n_alt_gt_0.5': int((alt > 0.5).sum()),
            'n_entropy_gt_1.5': int((ent > 1.5).sum()),
            'n_entropy_gt_2.0': int((ent > 2.0).sum()),
            'ratio_alt_gt_0.1': (alt > 0.1).mean(),
            'ratio_entropy_gt_1.5': (ent > 1.5).mean(),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

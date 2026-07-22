# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""WeightedAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class WeightedAggregation(AggregationStrategy):
    """Depth-weighted and unit_len-weighted features."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'depth_w_alt', 'depth_w_entropy', 'depth_w_del', 'depth_w_ins',
        'unit_w_alt', 'unit_w_entropy',
        'mean_depth', 'std_depth', 'cv_depth',
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
        depth = lf['depth']
        unit_len = lf['unit_len']

        total_depth = depth.sum()
        w_alt = (alt * depth).sum() / total_depth
        w_entropy = (ent * depth).sum() / total_depth
        w_del = (lf['del_ratio'] * depth).sum() / total_depth
        w_ins = (lf['ins_ratio'] * depth).sum() / total_depth

        unit_weight = 1.0 / (unit_len + 1e-10)
        w_alt_unit = (alt * unit_weight).sum() / unit_weight.sum()
        w_entropy_unit = (ent * unit_weight).sum() / unit_weight.sum()

        f.update({
            'high_alt_ratio': (alt > 0.5).mean(),
            'depth_w_alt': w_alt,
            'depth_w_entropy': w_entropy,
            'depth_w_del': w_del,
            'depth_w_ins': w_ins,
            'unit_w_alt': w_alt_unit,
            'unit_w_entropy': w_entropy_unit,
            'mean_depth': depth.mean(),
            'std_depth': depth.std(),
            'cv_depth': depth.std() / (depth.mean() + 1e-10),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""ShiftRichAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class ShiftRichAggregation(AggregationStrategy):
    """Rich shift statistics: std, skew, abs, directional fractions."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'std_shift', 'skew_shift', 'mean_abs_shift', 'max_abs_shift',
        'q95_abs_shift', 'frac_neg_shift', 'frac_pos_shift',
        'mean_del_ratio', 'del_ins_ratio',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        from scipy.stats import skew
        f = _common_features(lf)
        f['high_alt_ratio'] = (lf['alt_ratio'] > 0.5).mean()
        ms = lf['mean_shift']
        f.update({
            'std_shift': ms.std(),
            'skew_shift': float(skew(ms)) if len(ms) > 2 else 0,
            'mean_abs_shift': ms.abs().mean(),
            'max_abs_shift': ms.abs().max(),
            'q95_abs_shift': ms.abs().quantile(0.95),
            'frac_neg_shift': (ms < 0).mean(),
            'frac_pos_shift': (ms > 0).mean(),
            'mean_del_ratio': lf['del_ratio'].mean(),
            'del_ins_ratio': lf['del_ratio'].mean() / (lf['ins_ratio'].mean() + 1e-10),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

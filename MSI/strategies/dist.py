# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""DistAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features
from scipy.stats import skew, kurtosis

class DistAggregation(AggregationStrategy):
    """Distribution shape: std, skewness, kurtosis."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'std_alt', 'skew_alt', 'kurt_alt', 'std_entropy', 'skew_entropy',
        'std_del_ratio', 'std_ins_ratio',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        from scipy.stats import skew, kurtosis
        f = _common_features(lf)
        f['high_alt_ratio'] = (lf['alt_ratio'] > 0.5).mean()
        alt = lf['alt_ratio']
        ent = lf['entropy']
        f.update({
            'std_alt': alt.std(),
            'skew_alt': float(skew(alt)) if len(alt) > 2 else 0,
            'kurt_alt': float(kurtosis(alt)) if len(alt) > 3 else 0,
            'std_entropy': ent.std(),
            'skew_entropy': float(skew(ent)) if len(ent) > 2 else 0,
            'std_del_ratio': lf['del_ratio'].std(),
            'std_ins_ratio': lf['ins_ratio'].std(),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

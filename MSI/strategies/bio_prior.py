# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""BioPriorAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class BioPriorAggregation(AggregationStrategy):
    """Biological priors: microsatellite length, unit_len distribution."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'mean_ms_len', 'max_ms_len', 'frac_mono', 'frac_di', 'frac_tri',
        'mono_alt', 'mono_entropy',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        f = _common_features(lf)
        f['high_alt_ratio'] = (lf['alt_ratio'] > 0.5).mean()
        ms_len = lf['repeat_times'] * lf['unit_len']
        has_mono = (lf['unit_len'] == 1).any()
        f.update({
            'mean_ms_len': ms_len.mean(),
            'max_ms_len': ms_len.max(),
            'frac_mono': (lf['unit_len'] == 1).mean(),
            'frac_di': (lf['unit_len'] == 2).mean(),
            'frac_tri': (lf['unit_len'] == 3).mean(),
            'mono_alt': lf.loc[lf['unit_len'] == 1, 'alt_ratio'].mean() if has_mono else 0,
            'mono_entropy': lf.loc[lf['unit_len'] == 1, 'entropy'].mean() if has_mono else 0,
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""MonoFocusAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class MonoFocusAggregation(AggregationStrategy):
    """Focus on mono-nucleotide loci (BAT-25/26 type, most MSI-sensitive)."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'mono_n_loci', 'mono_mean_alt', 'mono_max_alt', 'mono_q90_alt',
        'mono_std_alt', 'mono_mean_entropy', 'mono_max_entropy',
        'mono_mean_del', 'mono_mean_ins',
        'mono_n_unstable_0.1', 'mono_n_unstable_0.2', 'mono_unstable_ratio',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        mono = lf[lf['unit_len'] == 1]
        if len(mono) == 0:
            return None

        f = _common_features(lf)
        f['high_alt_ratio'] = (lf['alt_ratio'] > 0.5).mean()
        alt_m = mono['alt_ratio']
        ent_m = mono['entropy']
        f.update({
            'mono_n_loci': len(mono),
            'mono_mean_alt': alt_m.mean(),
            'mono_max_alt': alt_m.max(),
            'mono_q90_alt': alt_m.quantile(0.9),
            'mono_std_alt': alt_m.std(),
            'mono_mean_entropy': ent_m.mean(),
            'mono_max_entropy': ent_m.max(),
            'mono_mean_del': mono['del_ratio'].mean(),
            'mono_mean_ins': mono['ins_ratio'].mean(),
            'mono_n_unstable_0.1': int((alt_m > 0.1).sum()),
            'mono_n_unstable_0.2': int((alt_m > 0.2).sum()),
            'mono_unstable_ratio': (alt_m > 0.1).mean(),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

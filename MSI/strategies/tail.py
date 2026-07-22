# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""TailAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class TailAggregation(AggregationStrategy):
    """Tail/extreme-value statistics: max, q90, threshold counts."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio', 'max_alt', 'q90_alt',
        'n_unstable_0.1', 'n_unstable_0.2', 'unstable_ratio_0.1', 'unstable_ratio_0.2',
        'max_entropy', 'q90_entropy',
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
            'high_alt_ratio': (alt > 0.5).mean(),
            'max_alt': alt.max(),
            'q90_alt': alt.quantile(0.9),
            'n_unstable_0.1': int((alt > 0.1).sum()),
            'n_unstable_0.2': int((alt > 0.2).sum()),
            'unstable_ratio_0.1': (alt > 0.1).mean(),
            'unstable_ratio_0.2': (alt > 0.2).mean(),
            'max_entropy': ent.max(),
            'q90_entropy': ent.quantile(0.9),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""PercentileAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class PercentileAggregation(AggregationStrategy):
    """Percentile-based statistics: q25/q50/q75/q95."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'q25_alt', 'q50_alt', 'q75_alt', 'q95_alt',
        'q25_entropy', 'q50_entropy', 'q75_entropy', 'q95_entropy',
        'q75_del_ratio', 'q75_ins_ratio',
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
            'q25_alt': alt.quantile(0.25),
            'q50_alt': alt.quantile(0.5),
            'q75_alt': alt.quantile(0.75),
            'q95_alt': alt.quantile(0.95),
            'q25_entropy': ent.quantile(0.25),
            'q50_entropy': ent.quantile(0.5),
            'q75_entropy': ent.quantile(0.75),
            'q95_entropy': ent.quantile(0.95),
            'q75_del_ratio': lf['del_ratio'].quantile(0.75),
            'q75_ins_ratio': lf['ins_ratio'].quantile(0.75),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

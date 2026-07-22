# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""InteractionAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class InteractionAggregation(AggregationStrategy):
    """Feature interactions and ratio features."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'alt_entropy_interaction', 'del_ins_ratio_mean', 'alt_shift_interaction',
        'cv_alt', 'cv_entropy', 'alt_entropy_corr',
        'msi_score_1', 'msi_score_2', 'weighted_alt',
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
        ms = lf['mean_shift']
        del_r = lf['del_ratio']
        ins_r = lf['ins_ratio']
        f.update({
            'high_alt_ratio': (alt > 0.5).mean(),
            'alt_entropy_interaction': (alt * ent).mean(),
            'del_ins_ratio_mean': del_r.mean() / (ins_r.mean() + 1e-10),
            'alt_shift_interaction': (alt * ms.abs()).mean(),
            'cv_alt': alt.std() / (alt.mean() + 1e-10),
            'cv_entropy': ent.std() / (ent.mean() + 1e-10),
            'alt_entropy_corr': float(alt.corr(ent)) if len(alt) > 2 else 0,
            'msi_score_1': ((alt > 0.1) & (ent > 1.5)).mean(),
            'msi_score_2': ((alt > 0.2) & (del_r > 0.1)).mean(),
            'weighted_alt': (alt * lf['depth']).sum() / lf['depth'].sum(),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

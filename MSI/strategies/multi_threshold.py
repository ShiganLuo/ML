# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""MultiThresholdAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class MultiThresholdAggregation(AggregationStrategy):
    """Cumulative distribution features with multiple thresholds."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'n_alt_gt_0.02', 'ratio_alt_gt_0.02', 'n_alt_gt_0.05', 'ratio_alt_gt_0.05',
        'n_alt_gt_0.1', 'ratio_alt_gt_0.1', 'n_alt_gt_0.15', 'ratio_alt_gt_0.15',
        'n_alt_gt_0.2', 'ratio_alt_gt_0.2', 'n_alt_gt_0.3', 'ratio_alt_gt_0.3',
        'n_alt_gt_0.4', 'ratio_alt_gt_0.4', 'n_alt_gt_0.5', 'ratio_alt_gt_0.5',
        'n_ent_gt_0.5', 'ratio_ent_gt_0.5', 'n_ent_gt_1.0', 'ratio_ent_gt_1.0',
        'n_ent_gt_1.5', 'ratio_ent_gt_1.5', 'n_ent_gt_2.0', 'ratio_ent_gt_2.0',
        'n_ent_gt_2.5', 'ratio_ent_gt_2.5', 'n_ent_gt_3.0', 'ratio_ent_gt_3.0',
        'n_alt_gt_median', 'n_ent_gt_median',
        'high_low_ratio_alt', 'high_low_ratio_ent',
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

        thresholds_alt = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        thresholds_ent = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        for thr in thresholds_alt:
            f[f'n_alt_gt_{thr}'] = int((alt > thr).sum())
            f[f'ratio_alt_gt_{thr}'] = (alt > thr).mean()
        for thr in thresholds_ent:
            f[f'n_ent_gt_{thr}'] = int((ent > thr).sum())
            f[f'ratio_ent_gt_{thr}'] = (ent > thr).mean()

        f['n_alt_gt_median'] = int((alt > alt.median()).sum())
        f['n_ent_gt_median'] = int((ent > ent.median()).sum())
        f['high_low_ratio_alt'] = (alt > 0.2).sum() / ((alt <= 0.2).sum() + 1)
        f['high_low_ratio_ent'] = (ent > 1.5).sum() / ((ent <= 1.5).sum() + 1)
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

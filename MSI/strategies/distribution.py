# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""DistributionAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features
from scipy.stats import skew, kurtosis, iqr

class DistributionAggregation(AggregationStrategy):
    """CV, normalized entropy, Gini coefficient."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'cv_alt', 'cv_entropy', 'norm_entropy', 'gini_alt',
        'iqr_alt', 'iqr_entropy', 'range_alt', 'range_entropy', 'mad_alt',
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

        cv_alt = alt.std() / (alt.mean() + 1e-10)
        cv_entropy = ent.std() / (ent.mean() + 1e-10)
        max_entropy = np.log2(len(lf)) if len(lf) > 1 else 1
        norm_entropy = ent.mean() / (max_entropy + 1e-10)

        sorted_alt = np.sort(alt.values)
        n = len(sorted_alt)
        gini = ((2 * np.sum(np.arange(1, n + 1) * sorted_alt) / (n * np.sum(sorted_alt)))
                - (n + 1) / n) if n > 0 and sorted_alt.sum() > 0 else 0

        f.update({
            'high_alt_ratio': (alt > 0.5).mean(),
            'cv_alt': cv_alt,
            'cv_entropy': cv_entropy,
            'norm_entropy': norm_entropy,
            'gini_alt': gini,
            'iqr_alt': alt.quantile(0.75) - alt.quantile(0.25),
            'iqr_entropy': ent.quantile(0.75) - ent.quantile(0.25),
            'range_alt': alt.max() - alt.min(),
            'range_entropy': ent.max() - ent.min(),
            'mad_alt': (alt - alt.mean()).abs().mean(),
        })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

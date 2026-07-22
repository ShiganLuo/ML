# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""AllAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features
from scipy.stats import skew, kurtosis

class AllAggregation(AggregationStrategy):
    """Combine all improvement strategies into one feature set."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio', 'max_alt', 'q90_alt',
        'n_unstable_0.1', 'n_unstable_0.2', 'unstable_ratio_0.1',
        'max_entropy', 'q90_entropy',
        'std_alt', 'skew_alt', 'kurt_alt', 'std_entropy', 'std_del_ratio',
        'q75_alt', 'q95_alt', 'q75_entropy',
        'std_shift', 'mean_abs_shift', 'del_ins_ratio',
        'mean_ms_len', 'frac_mono',
        'mono_mean_alt', 'mono_max_alt', 'mono_mean_entropy', 'mono_n_unstable_0.1',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        from scipy.stats import skew, kurtosis
        f = _common_features(lf)
        alt = lf['alt_ratio']
        ent = lf['entropy']
        ms = lf['mean_shift']
        ms_len = lf['repeat_times'] * lf['unit_len']
        mono = lf[lf['unit_len'] == 1]

        # tail
        f.update({
            'high_alt_ratio': (alt > 0.5).mean(),
            'max_alt': alt.max(),
            'q90_alt': alt.quantile(0.9),
            'n_unstable_0.1': int((alt > 0.1).sum()),
            'n_unstable_0.2': int((alt > 0.2).sum()),
            'unstable_ratio_0.1': (alt > 0.1).mean(),
            'max_entropy': ent.max(),
            'q90_entropy': ent.quantile(0.9),
        })
        # dist
        f.update({
            'std_alt': alt.std(),
            'skew_alt': float(skew(alt)) if len(alt) > 2 else 0,
            'kurt_alt': float(kurtosis(alt)) if len(alt) > 3 else 0,
            'std_entropy': ent.std(),
            'std_del_ratio': lf['del_ratio'].std(),
        })
        # percentile
        f.update({
            'q75_alt': alt.quantile(0.75),
            'q95_alt': alt.quantile(0.95),
            'q75_entropy': ent.quantile(0.75),
        })
        # shift
        f.update({
            'std_shift': ms.std(),
            'mean_abs_shift': ms.abs().mean(),
            'del_ins_ratio': lf['del_ratio'].mean() / (lf['ins_ratio'].mean() + 1e-10),
        })
        # bio
        f.update({
            'mean_ms_len': ms_len.mean(),
            'frac_mono': (lf['unit_len'] == 1).mean(),
        })
        # mono focus
        if len(mono) > 0:
            f.update({
                'mono_mean_alt': mono['alt_ratio'].mean(),
                'mono_max_alt': mono['alt_ratio'].max(),
                'mono_mean_entropy': mono['entropy'].mean(),
                'mono_n_unstable_0.1': int((mono['alt_ratio'] > 0.1).sum()),
            })
        else:
            f.update({
                'mono_mean_alt': 0, 'mono_max_alt': 0,
                'mono_mean_entropy': 0, 'mono_n_unstable_0.1': 0,
            })
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

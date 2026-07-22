# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""OptimizedAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features

class OptimizedAggregation(AggregationStrategy):
    """Best-practice combination with feature selection."""

    _FEATURE_NAMES = [
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio', 'max_alt', 'q90_alt', 'q95_alt',
        'n_unstable_0.1', 'unstable_ratio_0.1',
        'std_alt', 'cv_alt', 'iqr_alt',
        'mean_abs_shift', 'del_ins_ratio',
        'alt_entropy_interaction', 'msi_score',
        'mono_mean_alt', 'mono_max_alt', 'mono_n_unstable_0.1', 'mono_frac',
        'depth_w_alt',
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
        depth = lf['depth']
        mono = lf[lf['unit_len'] == 1]

        # Core discriminating features
        f.update({
            'high_alt_ratio': (alt > 0.5).mean(),
            'max_alt': alt.max(),
            'q90_alt': alt.quantile(0.9),
            'q95_alt': alt.quantile(0.95),
            'n_unstable_0.1': int((alt > 0.1).sum()),
            'unstable_ratio_0.1': (alt > 0.1).mean(),
        })
        # Distribution shape
        f.update({
            'std_alt': alt.std(),
            'cv_alt': alt.std() / (alt.mean() + 1e-10),
            'iqr_alt': alt.quantile(0.75) - alt.quantile(0.25),
        })
        # Shift
        f.update({
            'mean_abs_shift': ms.abs().mean(),
            'del_ins_ratio': del_r.mean() / (ins_r.mean() + 1e-10),
        })
        # Interaction
        f.update({
            'alt_entropy_interaction': (alt * ent).mean(),
            'msi_score': ((alt > 0.1) & (ent > 1.5)).mean(),
        })
        # Mono focus
        if len(mono) > 0:
            f.update({
                'mono_mean_alt': mono['alt_ratio'].mean(),
                'mono_max_alt': mono['alt_ratio'].max(),
                'mono_n_unstable_0.1': int((mono['alt_ratio'] > 0.1).sum()),
                'mono_frac': len(mono) / len(lf),
            })
        else:
            f.update({
                'mono_mean_alt': 0, 'mono_max_alt': 0,
                'mono_n_unstable_0.1': 0, 'mono_frac': 0,
            })
        # Depth-weighted
        f['depth_w_alt'] = (alt * depth).sum() / depth.sum()
        f.update(_unit_len_features(lf))
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

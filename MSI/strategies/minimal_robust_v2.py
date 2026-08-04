# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Minimal robust aggregation with tunable high_alt_ratio threshold.

Tests different thresholds for extreme instability:
- high_alt_ratio_0.6: alt > 0.6
- high_alt_ratio_0.7: alt > 0.7
- high_alt_ratio_0.8: alt > 0.8 (existing)

Based on analysis: MSI-H samples tend to have more extreme instability.
Higher thresholds may better capture true MSI-H signal.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features


class MinimalRobustV2Aggregation(AggregationStrategy):
    """Minimal robust aggregation with tunable high_alt_ratio threshold.
    
    Parameters
    ----------
    alt_threshold : float
        Threshold for high_alt_ratio feature (default: 0.7).
        Options: 0.5, 0.6, 0.7, 0.8
    """

    _FEATURE_NAMES = [
        'n_loci',
        'median_entropy',
        'depth_w_entropy',
        'median_alt',
        'high_alt_ratio',
        'depth_w_alt',
        'median_del_ratio',
        'depth_w_del',
        'prop_high_entropy',
        'prop_alt_gt_threshold_unit1',
        'unit_w_entropy',
    ]

    def __init__(self, alt_threshold: float = 0.7):
        self.alt_threshold = alt_threshold

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        
        alt = lf['alt_ratio']
        ent = lf['entropy']
        del_ratio = lf['del_ratio']
        depth = lf['depth']
        unit_len = lf['unit_len']
        
        total_depth = depth.sum()
        
        # Depth-weighted features
        w_entropy = (ent * depth).sum() / total_depth
        w_alt = (alt * depth).sum() / total_depth
        w_del = (del_ratio * depth).sum() / total_depth
        
        # Unit-weighted entropy
        unit_weight = 1.0 / (unit_len + 1e-10)
        w_entropy_unit = (ent * unit_weight).sum() / unit_weight.sum()
        
        # Distribution features
        prop_high_entropy = (ent > 2.0).mean()
        prop_alt_gt_threshold_unit1 = (alt[unit_len == 1] > self.alt_threshold).mean() if (unit_len == 1).sum() > 0 else 0
        
        f = {
            'n_loci': len(lf),
            'median_entropy': float(np.median(ent)),
            'depth_w_entropy': float(w_entropy),
            'median_alt': float(np.median(alt)),
            'high_alt_ratio': float((alt > self.alt_threshold).mean()),
            'depth_w_alt': float(w_alt),
            'median_del_ratio': float(np.median(del_ratio)),
            'depth_w_del': float(w_del),
            'prop_high_entropy': float(prop_high_entropy),
            'prop_alt_gt_threshold_unit1': float(prop_alt_gt_threshold_unit1),
            'unit_w_entropy': float(w_entropy_unit),
        }
        
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

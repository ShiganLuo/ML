# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""EnhancedWeightedAggregation - extends WeightedAggregation with distribution features.

Based on error analysis:
- FP samples (MSS misclassified as MSI-H): high instability but not MSI-H
- FN samples (MSI-H misclassified as MSS): low tumor_content dilutes signal

New features:
1. prop_alt_gt_0.8: proportion of loci with alt_ratio > 0.8 (extreme instability)
2. alt_skewness: skewness of alt_ratio distribution (distribution shape)
3. prop_alt_zero: proportion of loci with alt_ratio = 0 (LOH indicator)
4. alt_kurtosis: kurtosis of alt_ratio (tail heaviness)
5. prop_high_entropy: proportion of loci with entropy > 2.0
6. alt_range: max_alt - min_alt (spread measure)
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats
from .weighted import WeightedAggregation
from ..utils import _common_features, _unit_len_features


class EnhancedWeightedAggregation(WeightedAggregation):
    """Extended weighted aggregation with distribution shape features.
    
    Adds features that capture:
    - Extreme instability patterns (prop_alt_gt_0.8)
    - Distribution shape (skewness, kurtosis)
    - LOH indicators (prop_alt_zero)
    - High entropy loci proportion
    """

    _FEATURE_NAMES = WeightedAggregation._FEATURE_NAMES + [
        'prop_alt_gt_0.8',     # proportion of loci with alt_ratio > 0.8
        'alt_skewness',         # skewness of alt_ratio distribution
        'prop_alt_zero',        # proportion of loci with alt_ratio = 0
        'alt_kurtosis',         # kurtosis of alt_ratio (tail heaviness)
        'prop_high_entropy',    # proportion of loci with entropy > 2.0
        'alt_range',            # max_alt - min_alt (spread)
        'prop_alt_gt_0.5_unit1',  # prop of unit1 loci with alt > 0.5
        'prop_alt_gt_0.5_unit2',  # prop of unit2 loci with alt > 0.5
        'prop_alt_gt_0.5_unit3',  # prop of unit3 loci with alt > 0.5
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        
        # Get base weighted features
        f = super().aggregate(lf)
        if f is None:
            return None
        
        alt = lf['alt_ratio']
        ent = lf['entropy']
        unit_len = lf['unit_len']
        
        # 1. Extreme instability: proportion with alt_ratio > 0.8
        f['prop_alt_gt_0.8'] = (alt > 0.8).mean()
        
        # 2. Distribution shape: skewness
        # Positive skew = right tail (some loci very unstable)
        # MSI-H may have more positive skew
        if len(alt) >= 3:
            f['alt_skewness'] = float(stats.skew(alt))
        else:
            f['alt_skewness'] = 0.0
        
        # 3. LOH indicator: proportion with alt_ratio = 0
        # Could indicate loss of heterozygosity
        f['prop_alt_zero'] = (alt < 0.001).mean()
        
        # 4. Kurtosis: tail heaviness
        # High kurtosis = more extreme values
        if len(alt) >= 4:
            f['alt_kurtosis'] = float(stats.kurtosis(alt))
        else:
            f['alt_kurtosis'] = 0.0
        
        # 5. High entropy proportion
        f['prop_high_entropy'] = (ent > 2.0).mean()
        
        # 6. Range: spread of alt_ratio
        f['alt_range'] = alt.max() - alt.min()
        
        # 7. Per-unit_len extreme instability
        for ul in [1, 2, 3]:
            mask = unit_len == ul
            if mask.sum() > 0:
                f[f'prop_alt_gt_0.5_unit{ul}'] = (alt[mask] > 0.5).mean()
            else:
                f[f'prop_alt_gt_0.5_unit{ul}'] = 0.0
        
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

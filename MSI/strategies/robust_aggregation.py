# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Robust aggregation strategy with noise-resistant features.

Addresses noise sources:
1. Sequencing noise: median instead of mean, trimmed mean
2. Outliers: IQR-based features, winsorization
3. Sample quality: robust depth statistics

Key improvements over enhanced_weighted:
- median_alt, median_entropy: robust to extreme values
- trimmed_mean_alt: removes top/bottom 5% before averaging
- iqr_alt, iqr_entropy: interquartile range (spread measure)
- robust_cv: IQR/median instead of std/mean
- winsorized features: clip to 1st/99th percentile
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats
from .enhanced_weighted import EnhancedWeightedAggregation
from ..utils import _common_features, _unit_len_features


def trimmed_mean(values: np.ndarray, proportion: float = 0.05) -> float:
    """Compute trimmed mean (remove top/bottom proportion)."""
    if len(values) < 4:
        return np.mean(values)
    return float(stats.trim_mean(values, proportion))


def winsorize(values: np.ndarray, limits: tuple = (0.01, 0.01)) -> np.ndarray:
    """Winsorize values to specified percentiles."""
    return stats.mstats.winsorize(values, limits=limits)


class RobustAggregation(EnhancedWeightedAggregation):
    """Robust aggregation with noise-resistant statistics.
    
    Extends EnhancedWeightedAggregation with:
    - Median-based features (robust to outliers)
    - Trimmed mean (removes extreme values)
    - IQR-based features (robust spread measure)
    - Winsorized features (clipped extremes)
    - Robust coefficient of variation
    """

    _FEATURE_NAMES = EnhancedWeightedAggregation._FEATURE_NAMES + [
        # Median features
        'median_alt',
        'median_entropy',
        'median_del_ratio',
        'median_ins_ratio',
        
        # Trimmed mean features (5% trimming)
        'trimmed_alt',
        'trimmed_entropy',
        
        # IQR features (interquartile range)
        'iqr_alt',
        'iqr_entropy',
        'iqr_del_ratio',
        
        # Robust CV (IQR/median)
        'robust_cv_alt',
        'robust_cv_entropy',
        
        # Winsorized features (clipped to 1%/99%)
        'winsorized_mean_alt',
        'winsorized_mean_entropy',
        
        # Skewness (distribution shape)
        'skew_alt',
        'skew_entropy',
        
        # Kurtosis (tail heaviness)
        'kurtosis_alt',
        'kurtosis_entropy',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        
        # Get base enhanced_weighted features
        f = super().aggregate(lf)
        if f is None:
            return None
        
        alt = lf['alt_ratio']
        ent = lf['entropy']
        del_ratio = lf['del_ratio']
        ins_ratio = lf['ins_ratio']
        
        # Median features
        f['median_alt'] = float(np.median(alt))
        f['median_entropy'] = float(np.median(ent))
        f['median_del_ratio'] = float(np.median(del_ratio))
        f['median_ins_ratio'] = float(np.median(ins_ratio))
        
        # Trimmed mean features (remove top/bottom 5%)
        f['trimmed_alt'] = trimmed_mean(alt, 0.05)
        f['trimmed_entropy'] = trimmed_mean(ent, 0.05)
        
        # IQR features
        q1_alt, q3_alt = np.percentile(alt, [25, 75])
        q1_ent, q3_ent = np.percentile(ent, [25, 75])
        q1_del, q3_del = np.percentile(del_ratio, [25, 75])
        
        f['iqr_alt'] = float(q3_alt - q1_alt)
        f['iqr_entropy'] = float(q3_ent - q1_ent)
        f['iqr_del_ratio'] = float(q3_del - q1_del)
        
        # Robust CV (IQR/median)
        median_alt = f['median_alt']
        median_ent = f['median_entropy']
        f['robust_cv_alt'] = float(f['iqr_alt'] / (median_alt + 1e-10))
        f['robust_cv_entropy'] = float(f['iqr_entropy'] / (median_ent + 1e-10))
        
        # Winsorized features (clip to 1%/99%)
        alt_winsorized = winsorize(alt, (0.01, 0.01))
        ent_winsorized = winsorize(ent, (0.01, 0.01))
        f['winsorized_mean_alt'] = float(np.mean(alt_winsorized))
        f['winsorized_mean_entropy'] = float(np.mean(ent_winsorized))
        
        # Skewness and kurtosis (distribution shape)
        if len(alt) >= 3:
            f['skew_alt'] = float(stats.skew(alt))
            f['skew_entropy'] = float(stats.skew(ent))
        else:
            f['skew_alt'] = 0.0
            f['skew_entropy'] = 0.0
        
        if len(alt) >= 4:
            f['kurtosis_alt'] = float(stats.kurtosis(alt))
            f['kurtosis_entropy'] = float(stats.kurtosis(ent))
        else:
            f['kurtosis_alt'] = 0.0
            f['kurtosis_entropy'] = 0.0
        
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

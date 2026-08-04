# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Instance-level instability detection for MSI-H samples with few unstable loci.

Problem: Some MSI-H samples have only 1-2 highly unstable loci.
Current aggregation averages across all loci, diluting the signal.

Solution: Add per-sample features that capture "does this sample have ANY extreme instability":
- max_alt_ratio: the most unstable locus in this sample
- n_extreme_loci: count of loci with alt_ratio > threshold
- prop_any_instability: proportion of loci with any instability
- top_k_mean_alt: mean of top-k most unstable loci
- instability_concentration: how concentrated is the instability

These features help distinguish:
- MSI-H with 1-2 extreme loci (high max_alt, low n_extreme)
- MSI-H with many moderate loci (lower max_alt, high n_extreme)
- MSS with uniform low instability (low max_alt, low n_extreme)
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features


class InstabilityFocusedAggregation(AggregationStrategy):
    """Aggregation focused on detecting per-sample instability patterns.
    
    Key features for MSI-H with few unstable loci:
    - max_alt_ratio: highest alt_ratio in this sample (most unstable locus)
    - n_extreme_loci: count of loci with alt_ratio > threshold
    - prop_any_instability: proportion with alt > 0.1 (any instability)
    - top_k_mean_alt: mean of top-3 most unstable loci
    - instability_concentration: Gini coefficient of alt_ratio distribution
    """

    _FEATURE_NAMES = [
        'n_loci',
        # Core instability features (per-sample)
        'max_alt_ratio',
        'n_extreme_loci_05',
        'n_extreme_loci_08',
        'prop_any_instability',
        'top3_mean_alt',
        'top5_mean_alt',
        # Instability distribution
        'instability_gini',
        'instability_cv',
        # Standard aggregated features
        'median_alt',
        'median_entropy',
        'depth_w_alt',
        'depth_w_entropy',
        'depth_w_del',
        'high_alt_ratio',
        'prop_high_entropy',
        'unit_w_entropy',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        
        alt = lf['alt_ratio']
        ent = lf['entropy']
        del_ratio = lf['del_ratio']
        depth = lf['depth']
        unit_len = lf['unit_len']
        
        # Sort alt_ratio for top-k and Gini computation
        alt_sorted = np.sort(alt)[::-1]
        
        # Per-sample instability features
        max_alt = float(alt.max())
        n_extreme_05 = int((alt > 0.5).sum())
        n_extreme_08 = int((alt > 0.8).sum())
        prop_any_instability = float((alt > 0.1).mean())
        top3_mean = float(alt_sorted[:3].mean()) if len(alt_sorted) >= 3 else float(alt_sorted.mean())
        top5_mean = float(alt_sorted[:5].mean()) if len(alt_sorted) >= 5 else float(alt_sorted.mean())
        
        # Instability concentration (Gini coefficient)
        # High Gini = few loci have most instability (concentrated)
        # Low Gini = instability spread evenly across loci
        instability_gini = self._gini(alt)
        
        # CV of alt_ratio
        instability_cv = float(alt.std() / (alt.mean() + 1e-10))
        
        # Standard aggregated features
        total_depth = depth.sum()
        w_alt = float((alt * depth).sum() / total_depth)
        w_entropy = float((ent * depth).sum() / total_depth)
        w_del = float((del_ratio * depth).sum() / total_depth)
        
        unit_weight = 1.0 / (unit_len + 1e-10)
        w_entropy_unit = float((ent * unit_weight).sum() / unit_weight.sum())
        
        f = {
            'n_loci': len(lf),
            # Per-sample instability
            'max_alt_ratio': max_alt,
            'n_extreme_loci_05': n_extreme_05,
            'n_extreme_loci_08': n_extreme_08,
            'prop_any_instability': prop_any_instability,
            'top3_mean_alt': top3_mean,
            'top5_mean_alt': top5_mean,
            # Distribution shape
            'instability_gini': instability_gini,
            'instability_cv': instability_cv,
            # Standard features
            'median_alt': float(np.median(alt)),
            'median_entropy': float(np.median(ent)),
            'depth_w_alt': w_alt,
            'depth_w_entropy': w_entropy,
            'depth_w_del': w_del,
            'high_alt_ratio': float((alt > 0.5).mean()),
            'prop_high_entropy': float((ent > 2.0).mean()),
            'unit_w_entropy': w_entropy_unit,
        }
        
        return f
    
    def _gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient (inequality measure).
        
        0 = perfect equality (all values same)
        1 = perfect inequality (one value has all)
        
        For MSI: high Gini = few loci have most instability (concentrated)
        """
        if len(values) < 2:
            return 0.0
        
        # Sort values
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        
        # Compute Gini using formula
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n
        
        return float(max(0, min(1, gini)))

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

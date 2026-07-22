# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""UnstableLocusAggregation v2: depth-weighted, signal-strength, background-corrected."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features


class UnstableLocusAggregation(AggregationStrategy):
    """Detect unstable loci with depth weighting and signal strength.

    Improvements over v1:
    1. Depth-weighted proportion: low-depth loci contribute less
    2. Signal strength: sum of (alt_ratio - threshold), not just binary count
    3. Entropy-weighted: unstable loci with high entropy count more
    4. Per-unit_len breakdown: mono/di/tri have different noise rates
    """

    def __init__(self,
                 alt_thresholds: Optional[List[float]] = None,
                 entropy_thresholds: Optional[List[float]] = None):
        self.alt_thresholds = alt_thresholds or [0.05, 0.10, 0.15, 0.20]
        self.entropy_thresholds = entropy_thresholds or [0.3, 0.5, 0.8]
        self._feature_names = None

    def _build_feature_names(self) -> List[str]:
        names = [
            'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
            'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
            'high_alt_ratio',
        ]
        names += [
            'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
            'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
            'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
        ]
        for t in self.alt_thresholds:
            t_str = f'{t:.2f}'.replace('.', '_')
            names += [
                f'raw_prop_alt_{t_str}',
                f'dw_prop_alt_{t_str}',
                f'signal_str_alt_{t_str}',
                f'ent_w_prop_alt_{t_str}',
            ]
        for t in self.entropy_thresholds:
            t_str = f'{t:.1f}'.replace('.', '_')
            names += [
                f'raw_prop_ent_{t_str}',
                f'dw_prop_ent_{t_str}',
            ]
        names += [
            'raw_prop_combo_a010_e05', 'dw_prop_combo_a010_e05',
            'raw_prop_combo_a015_e05', 'dw_prop_combo_a015_e05',
            'mean_depth', 'std_depth', 'cv_depth',
            'depth_w_alt', 'depth_w_entropy',
            'mean_alt_unstable', 'mean_entropy_unstable',
            'unstable_entropy_ratio',
        ]
        return names

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None

        f = _common_features(lf)
        f['high_alt_ratio'] = (lf['alt_ratio'] > 0.5).mean()
        f.update(_unit_len_features(lf))

        alt = lf['alt_ratio'].values
        ent = lf['entropy'].values
        depth = lf['depth'].values
        n = len(alt)

        # Depth weights: sqrt(depth) normalized, so high-depth loci contribute more
        depth_w = np.sqrt(np.clip(depth, 1, None))
        depth_w = depth_w / depth_w.sum()

        # Basic features
        f['mean_depth'] = float(depth.mean())
        f['std_depth'] = float(depth.std())
        f['cv_depth'] = float(depth.std() / (depth.mean() + 1e-10))
        f['depth_w_alt'] = float((alt * depth_w).sum())
        f['depth_w_entropy'] = float((ent * depth_w).sum())

        # For each alt_ratio threshold: raw proportion, depth-weighted, signal strength, entropy-weighted
        for t in self.alt_thresholds:
            t_str = f'{t:.2f}'.replace('.', '_')
            mask = alt > t
            excess = np.maximum(alt - t, 0)

            f[f'raw_prop_alt_{t_str}'] = float(mask.sum() / n)
            f[f'dw_prop_alt_{t_str}'] = float(depth_w[mask].sum()) if mask.any() else 0.0
            f[f'signal_str_alt_{t_str}'] = float(excess.sum() / n)
            # Entropy-weighted: unstable loci with high entropy count more
            ent_w = ent * mask
            f[f'ent_w_prop_alt_{t_str}'] = float(ent_w.sum() / (ent.sum() + 1e-10))

        # For each entropy threshold
        for t in self.entropy_thresholds:
            t_str = f'{t:.1f}'.replace('.', '_')
            mask = ent > t
            f[f'raw_prop_ent_{t_str}'] = float(mask.sum() / n)
            f[f'dw_prop_ent_{t_str}'] = float(depth_w[mask].sum()) if mask.any() else 0.0

        # Combined criteria
        for (a, e, key) in [(0.10, 0.5, 'a010_e05'), (0.15, 0.5, 'a015_e05')]:
            mask = (alt > a) & (ent > e)
            f[f'raw_prop_combo_{key}'] = float(mask.sum() / n)
            f[f'dw_prop_combo_{key}'] = float(depth_w[mask].sum()) if mask.any() else 0.0

        # Summary stats for unstable loci
        primary_mask = alt > 0.10
        if primary_mask.sum() > 0:
            f['mean_alt_unstable'] = float(alt[primary_mask].mean())
            f['mean_entropy_unstable'] = float(ent[primary_mask].mean())
        else:
            f['mean_alt_unstable'] = 0.0
            f['mean_entropy_unstable'] = 0.0

        stable_mask = ~primary_mask
        mean_ent_stable = float(ent[stable_mask].mean()) if stable_mask.sum() > 0 else 0.0
        if mean_ent_stable > 1e-10:
            f['unstable_entropy_ratio'] = f['mean_entropy_unstable'] / mean_ent_stable
        else:
            f['unstable_entropy_ratio'] = 0.0

        return f

    def get_feature_names(self) -> List[str]:
        if self._feature_names is None:
            self._feature_names = self._build_feature_names()
        return self._feature_names

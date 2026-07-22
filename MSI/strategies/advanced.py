# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""AdvancedAggregation strategy with rich feature engineering."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features


class AdvancedAggregation(AggregationStrategy):
    """Advanced features: distribution shape, quantiles, long-tail, depth interaction, chrom-level."""

    _FEATURE_NAMES = [
        # --- common (9) ---
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        # --- per unit_len (15) ---
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
        # --- distribution shape (4) ---
        'alt_skew', 'alt_kurtosis', 'entropy_skew', 'entropy_kurtosis',
        # --- quantiles (8) ---
        'alt_p10', 'alt_p25', 'alt_p75', 'alt_p90',
        'entropy_p10', 'entropy_p25', 'entropy_p75', 'entropy_p90',
        # --- long-tail counts (5) ---
        'n_alt_gt_005', 'n_alt_gt_01', 'n_alt_gt_02', 'n_alt_gt_03', 'n_alt_gt_05',
        # --- tail ratios (3) ---
        'tail_ratio_01', 'tail_ratio_02', 'tail_ratio_05',
        # --- depth interaction (5) ---
        'depth_alt_corr', 'high_depth_alt', 'low_depth_alt',
        'high_depth_n', 'low_depth_n',
        # --- chromosome level (3) ---
        'chrom_alt_std', 'n_chroms', 'chrom_alt_range',
        # --- weighted features (6) ---
        'depth_w_alt', 'depth_w_entropy', 'depth_w_del', 'depth_w_ins',
        'unit_w_alt', 'unit_w_entropy',
        # --- depth stats (3) ---
        'mean_depth', 'std_depth', 'cv_depth',
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None

        f = _common_features(lf)
        f['high_alt_ratio'] = (lf['alt_ratio'] > 0.5).mean()
        f.update(_unit_len_features(lf))

        alt = lf['alt_ratio'].values
        ent = lf['entropy'].values
        depth = lf['depth'].values

        # --- distribution shape ---
        f['alt_skew'] = float(stats.skew(alt)) if len(alt) >= 3 else 0.0
        f['alt_kurtosis'] = float(stats.kurtosis(alt)) if len(alt) >= 4 else 0.0
        f['entropy_skew'] = float(stats.skew(ent)) if len(ent) >= 3 else 0.0
        f['entropy_kurtosis'] = float(stats.kurtosis(ent)) if len(ent) >= 4 else 0.0

        # --- quantiles ---
        f['alt_p10'] = float(np.percentile(alt, 10))
        f['alt_p25'] = float(np.percentile(alt, 25))
        f['alt_p75'] = float(np.percentile(alt, 75))
        f['alt_p90'] = float(np.percentile(alt, 90))
        f['entropy_p10'] = float(np.percentile(ent, 10))
        f['entropy_p25'] = float(np.percentile(ent, 25))
        f['entropy_p75'] = float(np.percentile(ent, 75))
        f['entropy_p90'] = float(np.percentile(ent, 90))

        # --- long-tail counts ---
        n = len(alt)
        f['n_alt_gt_005'] = int((alt > 0.05).sum())
        f['n_alt_gt_01'] = int((alt > 0.10).sum())
        f['n_alt_gt_02'] = int((alt > 0.20).sum())
        f['n_alt_gt_03'] = int((alt > 0.30).sum())
        f['n_alt_gt_05'] = int((alt > 0.50).sum())

        # --- tail ratios ---
        f['tail_ratio_01'] = f['n_alt_gt_01'] / max(n, 1)
        f['tail_ratio_02'] = f['n_alt_gt_02'] / max(n, 1)
        f['tail_ratio_05'] = f['n_alt_gt_05'] / max(n, 1)

        # --- depth interaction ---
        if n >= 2 and np.std(depth) > 1e-10 and np.std(alt) > 1e-10:
            f['depth_alt_corr'] = float(np.corrcoef(depth, alt)[0, 1])
        else:
            f['depth_alt_corr'] = 0.0
        depth_median = np.median(depth)
        high_mask = depth >= depth_median
        low_mask = ~high_mask
        f['high_depth_alt'] = float(alt[high_mask].mean()) if high_mask.sum() > 0 else 0.0
        f['low_depth_alt'] = float(alt[low_mask].mean()) if low_mask.sum() > 0 else 0.0
        f['high_depth_n'] = int(high_mask.sum())
        f['low_depth_n'] = int(low_mask.sum())

        # --- chromosome level ---
        if 'chrom' in lf.columns:
            chrom_groups = lf.groupby('chrom')['alt_ratio'].agg(['mean', 'count'])
            chrom_means = chrom_groups['mean']
            f['chrom_alt_std'] = float(chrom_means.std()) if len(chrom_means) >= 2 else 0.0
            f['n_chroms'] = int(len(chrom_means))
            f['chrom_alt_range'] = float(chrom_means.max() - chrom_means.min()) if len(chrom_means) >= 2 else 0.0
        else:
            f['chrom_alt_std'] = 0.0
            f['n_chroms'] = 0
            f['chrom_alt_range'] = 0.0

        # --- weighted features ---
        total_depth = depth.sum()
        if total_depth > 0:
            f['depth_w_alt'] = float((alt * depth).sum() / total_depth)
            f['depth_w_entropy'] = float((ent * depth).sum() / total_depth)
            f['depth_w_del'] = float((lf['del_ratio'].values * depth).sum() / total_depth)
            f['depth_w_ins'] = float((lf['ins_ratio'].values * depth).sum() / total_depth)
        else:
            f['depth_w_alt'] = 0.0
            f['depth_w_entropy'] = 0.0
            f['depth_w_del'] = 0.0
            f['depth_w_ins'] = 0.0

        unit_len = lf['unit_len'].values
        unit_weight = 1.0 / (unit_len + 1e-10)
        w_sum = unit_weight.sum()
        if w_sum > 0:
            f['unit_w_alt'] = float((alt * unit_weight).sum() / w_sum)
            f['unit_w_entropy'] = float((ent * unit_weight).sum() / w_sum)
        else:
            f['unit_w_alt'] = 0.0
            f['unit_w_entropy'] = 0.0

        # --- depth stats ---
        f['mean_depth'] = float(depth.mean())
        f['std_depth'] = float(depth.std())
        f['cv_depth'] = float(depth.std() / (depth.mean() + 1e-10))

        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

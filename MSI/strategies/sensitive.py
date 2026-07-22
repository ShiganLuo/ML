"""SensitiveAggregation: weighted + quantile + count features for higher sensitivity."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features


class SensitiveAggregation(AggregationStrategy):
    """Weighted features + quantile + count features.

    Goal: capture MSI-H samples with low-level instability that get
    diluted by mean-based features.

    Key additions over WeightedAggregation:
    - P90/P95 of alt_ratio: capture the tail of unstable loci
    - P90/P95 of entropy: capture high-entropy tail
    - Count features: n loci above threshold
    - Ratio features: count / total loci
    - Tail score: weighted sum of top-K alt_ratio values
    """

    _FEATURE_NAMES = [
        # --- from weighted (30) ---
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        'depth_w_alt', 'depth_w_entropy', 'depth_w_del', 'depth_w_ins',
        'unit_w_alt', 'unit_w_entropy',
        'mean_depth', 'std_depth', 'cv_depth',
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
        # --- quantile features (8) ---
        'alt_p90', 'alt_p95', 'entropy_p90', 'entropy_p95',
        'del_p90', 'ins_p90', 'shift_p90', 'alt_iqr',
        # --- count features (8) ---
        'n_alt_gt_005', 'n_alt_gt_01', 'n_alt_gt_02', 'n_alt_gt_03',
        'ratio_alt_gt_005', 'ratio_alt_gt_01', 'ratio_alt_gt_02', 'ratio_alt_gt_03',
        # --- tail features (4) ---
        'tail_mean_top5', 'tail_mean_top10', 'tail_sum_top5', 'tail_sum_top10',
        # --- purity-corrected placeholders (will be added if tumor_content available) ---
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
        del_r = lf['del_ratio'].values
        ins_r = lf['ins_ratio'].values
        shift = np.abs(lf['mean_shift'].values)
        n = len(alt)

        # Weighted features
        total_depth = depth.sum()
        if total_depth > 0:
            f['depth_w_alt'] = float((alt * depth).sum() / total_depth)
            f['depth_w_entropy'] = float((ent * depth).sum() / total_depth)
            f['depth_w_del'] = float((del_r * depth).sum() / total_depth)
            f['depth_w_ins'] = float((ins_r * depth).sum() / total_depth)
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

        f['mean_depth'] = float(depth.mean())
        f['std_depth'] = float(depth.std())
        f['cv_depth'] = float(depth.std() / (depth.mean() + 1e-10))

        # Quantile features
        f['alt_p90'] = float(np.percentile(alt, 90))
        f['alt_p95'] = float(np.percentile(alt, 95))
        f['entropy_p90'] = float(np.percentile(ent, 90))
        f['entropy_p95'] = float(np.percentile(ent, 95))
        f['del_p90'] = float(np.percentile(del_r, 90))
        f['ins_p90'] = float(np.percentile(ins_r, 90))
        f['shift_p90'] = float(np.percentile(shift, 90))
        f['alt_iqr'] = float(np.percentile(alt, 75) - np.percentile(alt, 25))

        # Count features
        f['n_alt_gt_005'] = int((alt > 0.05).sum())
        f['n_alt_gt_01'] = int((alt > 0.10).sum())
        f['n_alt_gt_02'] = int((alt > 0.20).sum())
        f['n_alt_gt_03'] = int((alt > 0.30).sum())
        f['ratio_alt_gt_005'] = f['n_alt_gt_005'] / max(n, 1)
        f['ratio_alt_gt_01'] = f['n_alt_gt_01'] / max(n, 1)
        f['ratio_alt_gt_02'] = f['n_alt_gt_02'] / max(n, 1)
        f['ratio_alt_gt_03'] = f['n_alt_gt_03'] / max(n, 1)

        # Tail features: mean and sum of top-K alt_ratio values
        sorted_alt = np.sort(alt)[::-1]
        f['tail_mean_top5'] = float(sorted_alt[:5].mean()) if n >= 5 else float(sorted_alt.mean())
        f['tail_mean_top10'] = float(sorted_alt[:10].mean()) if n >= 10 else float(sorted_alt.mean())
        f['tail_sum_top5'] = float(sorted_alt[:5].sum())
        f['tail_sum_top10'] = float(sorted_alt[:10].sum())

        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

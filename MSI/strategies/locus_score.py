"""LocusScoreAggregation: per-locus scoring and accumulation."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features


class LocusScoreAggregation(AggregationStrategy):
    """Per-locus instability scoring and accumulation.

    Instead of using summary statistics (mean, std), this strategy:
    1. Computes a per-locus instability score
    2. Accumulates scores across all loci
    3. Uses the distribution of locus scores as features

    Locus score = alt_ratio * (1 + entropy) * depth_weight
    This gives high weight to loci that are both high-alt and high-entropy.

    The key insight: MSI-H samples have many loci with moderate scores,
    while MSS samples have most loci with near-zero scores and a few
    noise loci with low scores.
    """

    _FEATURE_NAMES = [
        # --- basic ---
        'n_loci', 'mean_alt', 'mean_entropy', 'mean_del_ratio',
        'mean_ins_ratio', 'mean_ref_ratio', 'mean_shift', 'max_shift',
        'high_alt_ratio',
        # --- per-unit_len ---
        'alt_unit1', 'entropy_unit1', 'n_unit1', 'ins_ratio_unit1', 'del_ratio_unit1',
        'alt_unit2', 'entropy_unit2', 'n_unit2', 'ins_ratio_unit2', 'del_ratio_unit2',
        'alt_unit3', 'entropy_unit3', 'n_unit3', 'ins_ratio_unit3', 'del_ratio_unit3',
        # --- locus score accumulation ---
        'locus_score_sum', 'locus_score_mean', 'locus_score_std',
        'locus_score_max', 'locus_score_median',
        'locus_score_p90', 'locus_score_p95',
        # --- locus score counts ---
        'n_locus_score_gt_001', 'n_locus_score_gt_005', 'n_locus_score_gt_01',
        'n_locus_score_gt_02', 'n_locus_score_gt_05',
        'ratio_locus_score_gt_001', 'ratio_locus_score_gt_005', 'ratio_locus_score_gt_01',
        # --- locus score by unit_len ---
        'locus_score_mean_unit1', 'locus_score_mean_unit2', 'locus_score_mean_unit3',
        'locus_score_sum_unit1', 'locus_score_sum_unit2', 'locus_score_sum_unit3',
        # --- weighted accumulation ---
        'depth_w_locus_score', 'ent_w_locus_score',
        # --- tail accumulation ---
        'tail_score_mean_top5', 'tail_score_mean_top10',
        'tail_score_sum_top5', 'tail_score_sum_top10',
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
        unit_len = lf['unit_len'].values
        n = len(alt)

        # Per-locus instability score
        # score = alt_ratio * (1 + entropy) * sqrt(depth_weight)
        depth_w = np.sqrt(np.clip(depth, 1, None))
        depth_w = depth_w / (depth_w.sum() + 1e-10)
        locus_scores = alt * (1 + ent) * np.sqrt(depth / (depth.mean() + 1e-10))

        # Locus score accumulation
        f['locus_score_sum'] = float(locus_scores.sum())
        f['locus_score_mean'] = float(locus_scores.mean())
        f['locus_score_std'] = float(locus_scores.std()) if n > 1 else 0.0
        f['locus_score_max'] = float(locus_scores.max())
        f['locus_score_median'] = float(np.median(locus_scores))
        f['locus_score_p90'] = float(np.percentile(locus_scores, 90))
        f['locus_score_p95'] = float(np.percentile(locus_scores, 95))

        # Locus score counts
        f['n_locus_score_gt_001'] = int((locus_scores > 0.01).sum())
        f['n_locus_score_gt_005'] = int((locus_scores > 0.05).sum())
        f['n_locus_score_gt_01'] = int((locus_scores > 0.1).sum())
        f['n_locus_score_gt_02'] = int((locus_scores > 0.2).sum())
        f['n_locus_score_gt_05'] = int((locus_scores > 0.5).sum())
        f['ratio_locus_score_gt_001'] = f['n_locus_score_gt_001'] / max(n, 1)
        f['ratio_locus_score_gt_005'] = f['n_locus_score_gt_005'] / max(n, 1)
        f['ratio_locus_score_gt_01'] = f['n_locus_score_gt_01'] / max(n, 1)

        # Locus score by unit_len
        for ul in [1, 2, 3]:
            mask = unit_len == ul
            if mask.sum() > 0:
                f[f'locus_score_mean_unit{ul}'] = float(locus_scores[mask].mean())
                f[f'locus_score_sum_unit{ul}'] = float(locus_scores[mask].sum())
            else:
                f[f'locus_score_mean_unit{ul}'] = 0.0
                f[f'locus_score_sum_unit{ul}'] = 0.0

        # Weighted accumulation
        f['depth_w_locus_score'] = float((locus_scores * depth_w).sum())
        ent_w = ent / (ent.sum() + 1e-10)
        f['ent_w_locus_score'] = float((locus_scores * ent_w).sum())

        # Tail accumulation: top-K locus scores
        sorted_scores = np.sort(locus_scores)[::-1]
        f['tail_score_mean_top5'] = float(sorted_scores[:5].mean()) if n >= 5 else float(sorted_scores.mean())
        f['tail_score_mean_top10'] = float(sorted_scores[:10].mean()) if n >= 10 else float(sorted_scores.mean())
        f['tail_score_sum_top5'] = float(sorted_scores[:5].sum())
        f['tail_score_sum_top10'] = float(sorted_scores[:10].sum())

        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES

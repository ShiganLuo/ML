# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""LocusLevelAggregation aggregation strategy."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .base import AggregationStrategy
from ..utils import _common_features, _unit_len_features
from collections import Counter

class LocusLevelAggregation(AggregationStrategy):
    """Per-locus features without aggregation (pivot to columns).

    Instead of computing summary statistics across loci, each common
    locus becomes a separate feature column.  The model learns which
    specific loci are discriminative.

    Requires :meth:`fit` to be called before :meth:`aggregate` so that
    a consistent set of loci is used across all samples.

    Parameters
    ----------
    features : list of str
        Per-locus feature names to pivot (default ``['alt_ratio']``).
    min_samples : int
        Minimum number of samples a locus must appear in to be included
        (only used when fitting from data, default 10).
    """

    def __init__(self, features: Optional[List[str]] = None, min_samples: int = 10):
        self.features = features or ['alt_ratio']
        self.min_samples = min_samples
        self._common_loci: Optional[List[tuple]] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, locus_data: Dict[str, List[Dict]]) -> 'LocusLevelAggregation':
        """Learn the common locus set from training data.

        Parameters
        ----------
        locus_data : dict
            ``{sample_id: [locus_feature_dicts]}`` from extract_batch.

        Returns
        -------
        LocusLevelAggregation
            Self (for chaining).
        """
        locus_counts: Dict[tuple, int] = {}
        for loci in locus_data.values():
            for lf in loci:
                key = (lf['chrom'], lf['pos'], lf['unit_len'])
                locus_counts[key] = locus_counts.get(key, 0) + 1

        self._common_loci = sorted(
            k for k, v in locus_counts.items() if v >= self.min_samples
        )
        self._feature_names = []
        for feat_name in self.features:
            for loc in self._common_loci:
                self._feature_names.append(f'{feat_name}_{loc[0]}_{loc[1]}')

        logger.info(
            f"LocusLevelAggregation.fit: {len(self._common_loci)} loci "
            f"× {len(self.features)} features = {len(self._feature_names)} columns "
            f"(min_samples={self.min_samples})"
        )
        return self

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        """Pivot locus-level features into a flat feature dict.

        Parameters
        ----------
        lf : pd.DataFrame
            Locus-level dataframe for a single sample.

        Returns
        -------
        Optional[Dict]
            Feature dict with per-locus values, or None if not fitted.
        """
        if self._common_loci is None:
            raise RuntimeError("LocusLevelAggregation.fit() must be called before aggregate()")

        if len(self._common_loci) == 0:
            return None

        # Build O(1) lookup: (chrom, pos, unit_len) -> {feat_name: value}
        keys = list(zip(lf['chrom'], lf['pos'], lf['unit_len']))
        feat_arrays = {fn: lf[fn].values for fn in self.features}
        lookup = {}
        for i, key in enumerate(keys):
            lookup[key] = {fn: feat_arrays[fn][i] for fn in self.features}

        f: Dict[str, float] = {}
        for loc in self._common_loci:
            if loc in lookup:
                vals = lookup[loc]
                for feat_name in self.features:
                    f[f'{feat_name}_{loc[0]}_{loc[1]}'] = float(vals[feat_name])
            else:
                for feat_name in self.features:
                    f[f'{feat_name}_{loc[0]}_{loc[1]}'] = 0.0
        return f

    def get_feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise RuntimeError("LocusLevelAggregation.fit() must be called before get_feature_names()")
        return self._feature_names

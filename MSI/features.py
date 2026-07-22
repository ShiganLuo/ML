# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Feature extraction from site.txt files."""

import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import COL_CHROM, COL_POS, COL_UNIT_LEN, COL_REPEAT_TIMES, COL_REPEAT_DICT, COL_DEPTH

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from site.txt files."""

    def __init__(
            self, 
            min_depth: int = 10, 
            locus_selector: Optional['LocusSelector'] = None,
            agg_strategy: Optional['AggregationStrategy'] = None
        ):
        self.min_depth = min_depth
        self.locus_selector = locus_selector
        self.agg_strategy = agg_strategy

    def extract_locus_features(self, row: pd.Series) -> Optional[Dict]:
        """Extract features for a single locus.

        Parameters
        ----------
        row : pd.Series
            A row from site.txt with columns:
            [chrom, pos, unit_len, unit_binary, repeat_times, ...]

        Returns
        -------
        dict or None
            Locus-level features.
        """
        try:
            chrom = str(row[COL_CHROM])
            pos = int(row[COL_POS])
            repeat_times = int(row[COL_REPEAT_TIMES])
            dist_str = str(row[COL_REPEAT_DICT])
            depth = int(row[COL_DEPTH])
            unit_len = int(row[COL_UNIT_LEN])

            if depth < self.min_depth:
                return None

            # Parse distribution
            counts = {}
            for item in dist_str.split(','):
                parts = item.split(':')
                if len(parts) == 2:
                    try:
                        counts[int(parts[0])] = int(parts[1])
                    except ValueError:
                        continue

            if not counts:
                return None

            # Reference count
            ref_count = counts.get(repeat_times, 0)
            alt_ratio = 1 - ref_count / depth

            # Insertion / deletion ratios (relative to reference repeat count)
            ins_ratio = sum(v for k, v in counts.items() if k > repeat_times) / depth
            del_ratio = sum(v for k, v in counts.items() if k < repeat_times) / depth

            # Entropy
            probs = np.array(list(counts.values())) / depth
            entropy = -np.sum(probs * np.log2(probs + 1e-10))

            # Shift statistics
            shifts = np.array([k - repeat_times for k in counts.keys()])
            weights = np.array(list(counts.values())) / depth
            max_shift = np.max(np.abs(shifts))
            mean_shift = float(np.sum(shifts * weights))
            # Weighted median
            sorted_idx = np.argsort(shifts)
            cumw = np.cumsum(weights[sorted_idx])
            median_shift = float(shifts[sorted_idx[np.searchsorted(cumw, 0.5)]])

            return {
                'chrom': chrom,
                'pos': pos,
                'unit_len': unit_len,
                'repeat_times': repeat_times,
                'depth': depth,
                'alt_ratio': alt_ratio,
                'entropy': entropy,
                'max_shift': max_shift,
                'mean_shift': mean_shift,
                'median_shift': median_shift,
                'ref_ratio': ref_count / depth,
                'ins_ratio': ins_ratio,
                'del_ratio': del_ratio,
            }

        except Exception:
            return None

    def _aggregate_locus_features(self, lf: pd.DataFrame) -> Optional[Dict]:
        """Aggregate locus-level features into sample-level features.

        Uses the configured AggregationStrategy if available, otherwise
        falls back to the built-in baseline aggregation.

        Parameters
        ----------
        lf : pd.DataFrame
            DataFrame of locus-level features (one row per locus).

        Returns
        -------
        dict or None
            Sample-level aggregated features.
        """
        if self.agg_strategy is not None:
            return self.agg_strategy.aggregate(lf)

        # Fallback: built-in baseline aggregation
        if len(lf) == 0:
            return None

        features = {
            'n_loci': len(lf),
            'mean_alt': lf['alt_ratio'].mean(),
            'high_alt_ratio': (lf['alt_ratio'] > 0.5).mean(),
            'mean_entropy': lf['entropy'].mean(),
            'mean_ref_ratio': lf['ref_ratio'].mean(),
            'mean_shift': lf['mean_shift'].mean(),
            'max_shift': lf['max_shift'].max(),
            'mean_ins_ratio': lf['ins_ratio'].mean(),
            'mean_del_ratio': lf['del_ratio'].mean(),
        }

        for ul in [1, 2, 3]:
            mask = lf['unit_len'] == ul
            if mask.sum() > 0:
                features[f'alt_unit{ul}'] = lf.loc[mask, 'alt_ratio'].mean()
                features[f'entropy_unit{ul}'] = lf.loc[mask, 'entropy'].mean()
                features[f'n_unit{ul}'] = int(mask.sum())
                features[f'ins_ratio_unit{ul}'] = lf.loc[mask, 'ins_ratio'].mean()
                features[f'del_ratio_unit{ul}'] = lf.loc[mask, 'del_ratio'].mean()
            else:
                features[f'alt_unit{ul}'] = 0
                features[f'entropy_unit{ul}'] = 0
                features[f'n_unit{ul}'] = 0
                features[f'ins_ratio_unit{ul}'] = 0
                features[f'del_ratio_unit{ul}'] = 0

        return features

    def extract_sample_features(self, site_file: str) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """Extract aggregated features for a sample.

        Parameters
        ----------
        site_file : str
            Path to site.txt file.

        Returns
        -------
        tuple
            (sample_features, locus_features_list)
        """
        try:
            df = pd.read_csv(site_file, sep='\t', header=None)
            if len(df) == 0:
                return None, None

            # Extract locus features
            locus_features = []
            for _, row in df.iterrows():
                feat = self.extract_locus_features(row)
                if feat is not None:
                    # Apply locus selector if available
                    if self.locus_selector is not None:
                        if not self.locus_selector.is_selected(feat):
                            continue
                    locus_features.append(feat)

            if not locus_features:
                return None, None

            # Convert to DataFrame for easier aggregation
            lf = pd.DataFrame(locus_features)

            features = self._aggregate_locus_features(lf)
            return features, locus_features

        except Exception as e:
            logger.warning(f"Failed to extract features from {site_file}: {e}")
            return None, None

    def extract_batch(self, site_paths: np.ndarray, sample_ids: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, List[Dict]]]:
        """Extract features for multiple samples.

        Parameters
        ----------
        site_paths : np.ndarray
            Array of site.txt file paths.
        sample_ids : np.ndarray
            Array of sample IDs.

        Returns
        -------
        tuple
            (feature_matrix, locus_data_dict)
        """
        rows = []
        locus_data = {}
        total = len(site_paths)

        for i, (sid, path) in enumerate(zip(sample_ids, site_paths)):
            if not isinstance(path, str) or not os.path.isfile(path):
                continue

            feat, loci = self.extract_sample_features(path)
            if feat is not None:
                feat['sample_id'] = sid
                rows.append(feat)
            if loci is not None:
                locus_data[sid] = loci

            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i+1}/{total}...")

        df = pd.DataFrame(rows)
        if 'sample_id' in df.columns:
            df = df.set_index('sample_id')

        logger.info(f"Extracted features for {len(df)} samples")
        return df, locus_data

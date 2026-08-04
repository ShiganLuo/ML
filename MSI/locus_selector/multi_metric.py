from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import LocusSelector
logger = logging.getLogger(__name__)

class MultiMetricLocusSelector(LocusSelector):
    """Select loci if ANY metric (alt_ratio AUC, entropy AUC, shift AUC) exceeds threshold.

    More permissive than single-metric AUC: a locus with weak alt_ratio signal
    but strong entropy signal will be kept.
    """

    def __init__(self, auc_threshold: float = 0.7, min_depth: int = 20):
        self.auc_threshold = auc_threshold
        self.min_depth = min_depth
        self.selected_loci_ = None

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'MultiMetricLocusSelector':
        from scipy.integrate import trapezoid

        # Collect per-locus metrics across samples
        locus_metrics = {}  # key -> {'alt': [...], 'entropy': [...], 'shift': [...], 'labels': [...]}
        for sid, loci in locus_data.items():
            label = sample_labels.get(sid)
            if label is None:
                continue
            for feat in loci:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_metrics:
                    locus_metrics[key] = {'alt': [], 'entropy': [], 'shift': [], 'labels': []}
                m = locus_metrics[key]
                m['alt'].append(feat['alt_ratio'])
                m['entropy'].append(feat['entropy'])
                m['shift'].append(abs(feat.get('mean_shift', 0)))
                m['labels'].append(label)

        def _calc_auc(values, labels):
            if len(values) < 10 or np.std(values) < 1e-10:
                return 0.5
            sorted_idx = np.argsort(values)[::-1]
            y_sorted = np.array(labels)[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            return trapezoid(tpr, fpr)

        self.selected_loci_ = set()
        self.locus_auc_ = {}
        for key, m in locus_metrics.items():
            alt_auc = _calc_auc(m['alt'], m['labels'])
            ent_auc = _calc_auc(m['entropy'], m['labels'])
            shift_auc = _calc_auc(m['shift'], m['labels'])
            max_auc = max(alt_auc, ent_auc, shift_auc)
            self.locus_auc_[key] = {'alt': alt_auc, 'entropy': ent_auc, 'shift': shift_auc, 'max': max_auc}
            if max_auc >= self.auc_threshold:
                self.selected_loci_.add(key)

        logger.info(f"MultiMetric: {len(self.selected_loci_)}/{len(locus_metrics)} loci "
                     f"(any metric AUC >= {self.auc_threshold})")
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_

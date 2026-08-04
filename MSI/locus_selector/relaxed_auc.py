from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import LocusSelector
logger = logging.getLogger(__name__)

class RelaxedAUCSelector(LocusSelector):
    """AUC-based locus selector with lower threshold for broader coverage."""

    def __init__(self, auc_threshold: float = 0.7, min_depth: int = 20):
        self.auc_threshold = auc_threshold
        self.min_depth = min_depth
        self.selected_loci_ = None

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'RelaxedAUCSelector':
        locus_scores = {}
        for sid, loci in locus_data.items():
            label = sample_labels.get(sid)
            if label is None:
                continue
            for feat in loci:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_scores:
                    locus_scores[key] = []
                locus_scores[key].append((feat['alt_ratio'], label))

        self.locus_auc_ = {}
        for key, scores in locus_scores.items():
            if len(scores) < 10:
                continue
            values = np.array([s[0] for s in scores])
            labels = np.array([s[1] for s in scores])
            if np.std(values) < 1e-10:
                continue
            sorted_idx = np.argsort(values)[::-1]
            y_sorted = labels[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            from scipy.integrate import trapezoid
            auc = trapezoid(tpr, fpr)
            self.locus_auc_[key] = auc

        self.selected_loci_ = {k for k, v in self.locus_auc_.items() if v >= self.auc_threshold}
        logger.info(f"RelaxedAUC: {len(self.selected_loci_)}/{len(self.locus_auc_)} loci (AUC >= {self.auc_threshold})")
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_
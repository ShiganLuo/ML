from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd
from .base import LocusSelector
logger = logging.getLogger(__name__)

class AUCBasedLocusSelector(LocusSelector):
    """Select loci based on individual AUC scores."""

    def __init__(self, auc_threshold: float = 0.6, min_depth: int = 30):
        self.auc_threshold = auc_threshold
        self.min_depth = min_depth
        self.selected_loci_ = None

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'AUCBasedLocusSelector':
        """Fit by computing AUC for each locus across samples.

        Parameters
        ----------
        locus_data : dict
            {sample_id: [locus_features]} for each sample.
        sample_labels : dict
            {sample_id: 0/1} binary labels.
        """
        # Collect per-locus alt_ratios across samples
        locus_scores = {}  # locus_key -> [(alt_ratio, label)]

        for sid, loci in locus_data.items():
            label = sample_labels.get(sid)
            if label is None:
                continue
            for feat in loci:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_scores:
                    locus_scores[key] = []
                locus_scores[key].append((feat['alt_ratio'], label))

        # Compute AUC for each locus
        self.locus_auc_ = {}
        for key, scores in locus_scores.items():
            if len(scores) < 10:  # Need enough samples
                continue
            values = np.array([s[0] for s in scores])
            labels = np.array([s[1] for s in scores])

            if np.std(values) < 1e-10:
                continue

            # Simple AUC computation
            sorted_idx = np.argsort(values)[::-1]
            y_sorted = labels[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            auc = trapezoid(tpr, fpr)

            self.locus_auc_[key] = auc

        # Select loci above threshold
        self.selected_loci_ = {
            k for k, v in self.locus_auc_.items() if v >= self.auc_threshold
        }

        logger.info(f"Locus selection: {len(self.selected_loci_)}/{len(self.locus_auc_)} loci selected (AUC >= {self.auc_threshold})")

        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        """Check if a locus is selected."""
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_


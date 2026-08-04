# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Stability-based locus selection.

Instead of computing AUC on the full dataset (data leakage) or a single split
(high variance), this approach:
1. Randomly subsamples training data n_iterations times
2. Computes AUC for each locus in each subsample
3. Keeps loci that consistently exceed the threshold across subsamples

This is more robust than single-split AUC and avoids data leakage.
"""

from typing import List, Dict, Optional
import logging
import numpy as np
from scipy.integrate import trapezoid
from .base import LocusSelector

logger = logging.getLogger(__name__)


class StabilitySelectionLocusSelector(LocusSelector):
    """Stability-based locus selection using bootstrap subsampling.
    
    Parameters
    ----------
    auc_threshold : float
        Per-iteration AUC threshold for a locus to be considered selected.
    stability_threshold : float
        Proportion of iterations a locus must be selected in (0-1).
        E.g., 0.7 means locus must have AUC >= auc_threshold in >=70% of iterations.
    n_iterations : int
        Number of random subsamples to draw.
    subsample_ratio : float
        Fraction of samples to use in each subsample (0-1).
    min_depth : int
        Minimum depth for a locus to be considered.
    random_state : int
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        auc_threshold: float = 0.7,
        stability_threshold: float = 0.6,
        n_iterations: int = 50,
        subsample_ratio: float = 0.8,
        min_depth: int = 30,
        random_state: int = 42,
    ):
        self.auc_threshold = auc_threshold
        self.stability_threshold = stability_threshold
        self.n_iterations = n_iterations
        self.subsample_ratio = subsample_ratio
        self.min_depth = min_depth
        self.random_state = random_state
        self.selected_loci_ = None
        self.locus_stability_ = None
    
    def _compute_auc(self, values: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC from values and labels."""
        if len(values) < 10 or np.std(values) < 1e-10:
            return 0.5
        
        sorted_idx = np.argsort(values)[::-1]
        y_sorted = labels[sorted_idx]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        
        if tps[-1] == 0 or fps[-1] == 0:
            return 0.5
        
        tpr = np.concatenate([[0], tps / tps[-1]])
        fpr = np.concatenate([[0], fps / fps[-1]])
        return float(trapezoid(tpr, fpr))
    
    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'StabilitySelectionLocusSelector':
        """Fit using stability selection.
        
        Parameters
        ----------
        locus_data : dict
            {sample_id: [locus_features]} for each sample.
        sample_labels : dict
            {sample_id: 0/1} binary labels.
        """
        rng = np.random.RandomState(self.random_state)
        
        # Get all labeled sample IDs
        labeled_sids = [sid for sid in locus_data if sid in sample_labels]
        if len(labeled_sids) < 20:
            logger.warning(f"Too few labeled samples ({len(labeled_sids)}), falling back to AUC selector")
            from .feature_selectors import AUCBasedLocusSelector
            fallback = AUCBasedLocusSelector(auc_threshold=self.auc_threshold)
            return fallback.fit(locus_data, sample_labels)
        
        # Build per-locus data structure: locus_key -> {sample_id: (alt_ratio, label)}
        locus_sample_data = {}
        for sid in labeled_sids:
            label = sample_labels[sid]
            for feat in locus_data[sid]:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_sample_data:
                    locus_sample_data[key] = {}
                locus_sample_data[key][sid] = (feat['alt_ratio'], label)
        
        # Filter loci with too few samples
        locus_sample_data = {
            k: v for k, v in locus_sample_data.items() 
            if len(v) >= self.min_depth
        }
        
        logger.info(f"Stability selection: {len(locus_sample_data)} loci with >= {self.min_depth} samples")
        
        # Run stability selection
        locus_selection_counts = {key: 0 for key in locus_sample_data}
        n_samples = len(labeled_sids)
        subsample_size = int(n_samples * self.subsample_ratio)
        
        for iteration in range(self.n_iterations):
            # Random subsample
            subsample_indices = rng.choice(n_samples, size=subsample_size, replace=False)
            subsample_sids = set(labeled_sids[i] for i in subsample_indices)
            
            # Compute AUC for each locus on this subsample
            for key, sample_data in locus_sample_data.items():
                values = []
                labels = []
                for sid, (alt, label) in sample_data.items():
                    if sid in subsample_sids:
                        values.append(alt)
                        labels.append(label)
                
                if len(values) < 10:
                    continue
                
                auc = self._compute_auc(np.array(values), np.array(labels))
                if auc >= self.auc_threshold:
                    locus_selection_counts[key] += 1
        
        # Compute stability proportion for each locus
        self.locus_stability_ = {
            key: count / self.n_iterations 
            for key, count in locus_selection_counts.items()
        }
        
        # Select loci with stability >= threshold
        self.selected_loci_ = {
            key for key, stability in self.locus_stability_.items()
            if stability >= self.stability_threshold
        }
        
        logger.info(
            f"Stability selection: {len(self.selected_loci_)}/{len(self.locus_stability_)} loci selected "
            f"(stability >= {self.stability_threshold}, AUC >= {self.auc_threshold}, "
            f"{self.n_iterations} iterations, subsample_ratio={self.subsample_ratio})"
        )
        
        # Log stability distribution
        stabilities = list(self.locus_stability_.values())
        logger.info(f"  Stability distribution: mean={np.mean(stabilities):.3f}, "
                   f"median={np.median(stabilities):.3f}, "
                   f">=0.9: {sum(1 for s in stabilities if s >= 0.9)}, "
                   f">=0.7: {sum(1 for s in stabilities if s >= 0.7)}, "
                   f">=0.5: {sum(1 for s in stabilities if s >= 0.5)}")
        
        return self
    
    def is_selected(self, locus_feat: Dict) -> bool:
        """Check if a locus is selected."""
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_

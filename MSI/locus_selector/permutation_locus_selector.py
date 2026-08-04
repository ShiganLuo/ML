# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Permutation test based locus selection.

Instead of using a fixed AUC threshold, this approach:
1. Computes real AUC for each locus
2. Permutes labels n_permutations times to build null distribution
3. Only keeps loci whose real AUC is significantly above random (p < alpha)

This provides statistical rigor and avoids arbitrary thresholds.
"""

from typing import List, Dict, Optional
import logging
import numpy as np
from scipy.integrate import trapezoid
from .base import LocusSelector

logger = logging.getLogger(__name__)


class PermutationTestLocusSelector(LocusSelector):
    """Permutation test based locus selection.
    
    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05). Loci with p-value < alpha are selected.
    n_permutations : int
        Number of label permutations to build null distribution.
    min_depth : int
        Minimum number of samples for a locus to be considered.
    random_state : int
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 100,
        min_depth: int = 30,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.min_depth = min_depth
        self.random_state = random_state
        self.selected_loci_ = None
        self.locus_pvalues_ = None
    
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
    
    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'PermutationTestLocusSelector':
        """Fit using permutation test.
        
        Parameters
        ----------
        locus_data : dict
            {sample_id: [locus_features]} for each sample.
        sample_labels : dict
            {sample_id: 0/1} binary labels.
        """
        rng = np.random.RandomState(self.random_state)
        
        # Build per-locus data: locus_key -> [(alt_ratio, label)]
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
        
        # Filter loci with too few samples
        locus_scores = {k: v for k, v in locus_scores.items() if len(v) >= self.min_depth}
        
        logger.info(f"Permutation test: {len(locus_scores)} loci with >= {self.min_depth} samples")
        
        # Compute real AUC and build null distribution
        self.locus_pvalues_ = {}
        self.locus_auc_ = {}
        
        for key, scores in locus_scores.items():
            values = np.array([s[0] for s in scores])
            labels = np.array([s[1] for s in scores])
            
            # Real AUC
            real_auc = self._compute_auc(values, labels)
            self.locus_auc_[key] = real_auc
            
            # Null distribution via label permutation
            null_aucs = []
            for _ in range(self.n_permutations):
                permuted_labels = rng.permutation(labels)
                null_auc = self._compute_auc(values, permuted_labels)
                null_aucs.append(null_auc)
            
            null_aucs = np.array(null_aucs)
            
            # p-value: proportion of null AUCs >= real AUC
            # One-sided test: is real AUC significantly HIGH?
            p_value = (np.sum(null_aucs >= real_auc) + 1) / (self.n_permutations + 1)
            self.locus_pvalues_[key] = p_value
        
        # Select loci with p-value < alpha
        self.selected_loci_ = {
            key for key, pval in self.locus_pvalues_.items()
            if pval < self.alpha
        }
        
        # Log results
        pvalues = list(self.locus_pvalues_.values())
        logger.info(
            f"Permutation test: {len(self.selected_loci_)}/{len(self.locus_pvalues_)} loci selected "
            f"(p < {self.alpha}, {self.n_permutations} permutations)"
        )
        logger.info(f"  p-value distribution: mean={np.mean(pvalues):.4f}, "
                   f"median={np.median(pvalues):.4f}, "
                   f"<0.01: {sum(1 for p in pvalues if p < 0.01)}, "
                   f"<0.05: {sum(1 for p in pvalues if p < 0.05)}, "
                   f"<0.1: {sum(1 for p in pvalues if p < 0.1)}")
        
        # Log AUC distribution for selected vs not selected
        if self.selected_loci_:
            selected_aucs = [self.locus_auc_[k] for k in self.selected_loci_]
            not_selected = [self.locus_auc_[k] for k in self.locus_pvalues_ if k not in self.selected_loci_]
            logger.info(f"  Selected loci AUC: mean={np.mean(selected_aucs):.4f}, "
                       f"min={np.min(selected_aucs):.4f}, max={np.max(selected_aucs):.4f}")
            if not_selected:
                logger.info(f"  Not selected AUC: mean={np.mean(not_selected):.4f}, "
                           f"min={np.min(not_selected):.4f}, max={np.max(not_selected):.4f}")
        
        return self
    
    def is_selected(self, locus_feat: Dict) -> bool:
        """Check if a locus is selected."""
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_

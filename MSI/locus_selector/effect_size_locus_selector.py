# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Effect size based locus selection.

Instead of AUC, this approach uses statistical effect size measures:
- Cohen's d: standardized mean difference
- Glass's delta: uses only control group SD (for unequal variances)
- Cliff's delta: non-parametric effect size (robust to outliers)

Effect size is more interpretable than AUC and doesn't depend on sample size.
"""

from typing import List, Dict, Optional
import logging
import numpy as np
from .base import LocusSelector

logger = logging.getLogger(__name__)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size.
    
    d = (mean1 - mean2) / pooled_std
    
    Interpretation: |d| < 0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 5 or n2 < 5:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def glass_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Glass's delta effect size.
    
    delta = (mean1 - mean2) / std2
    
    Uses only control group (group2) SD. Better when variances are unequal.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 5 or n2 < 5:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std2 = np.std(group2, ddof=1)
    
    if std2 < 1e-10:
        return 0.0
    
    return (mean1 - mean2) / std2


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta effect size (non-parametric).
    
    delta = (# pairs where x1 > x2 - # pairs where x1 < x2) / (n1 * n2)
    
    Range: [-1, 1]. More robust than Cohen's d for non-normal distributions.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 5 or n2 < 5:
        return 0.0
    
    # Count dominance
    dominance = 0
    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                dominance += 1
            elif x1 < x2:
                dominance -= 1
    
    return dominance / (n1 * n2)


class EffectSizeLocusSelector(LocusSelector):
    """Effect size based locus selection.
    
    Parameters
    ----------
    effect_size_type : str
        Type of effect size: 'cohens_d', 'glass_delta', or 'cliffs_delta'.
    threshold : float
        Minimum absolute effect size to select a locus.
        For Cohen's d: 0.5 (medium) or 0.8 (large)
        For Cliff's delta: 0.33 (medium) or 0.474 (large)
    metric : str
        Which feature to use: 'alt_ratio', 'entropy', or 'max' (max of both).
    min_depth : int
        Minimum number of samples per group for a locus.
    """
    
    def __init__(
        self,
        effect_size_type: str = 'cohens_d',
        threshold: float = 0.5,
        metric: str = 'alt_ratio',
        min_depth: int = 30,
    ):
        self.effect_size_type = effect_size_type
        self.threshold = threshold
        self.metric = metric
        self.min_depth = min_depth
        self.selected_loci_ = None
        self.locus_effect_sizes_ = None
        
        # Select effect size function
        if effect_size_type == 'cohens_d':
            self._effect_size_fn = cohens_d
        elif effect_size_type == 'glass_delta':
            self._effect_size_fn = glass_delta
        elif effect_size_type == 'cliffs_delta':
            self._effect_size_fn = cliffs_delta
        else:
            raise ValueError(f"Unknown effect_size_type: {effect_size_type}")
    
    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'EffectSizeLocusSelector':
        """Fit using effect size.
        
        Parameters
        ----------
        locus_data : dict
            {sample_id: [locus_features]} for each sample.
        sample_labels : dict
            {sample_id: 0/1} binary labels.
        """
        # Build per-locus data: locus_key -> {'msi': [values], 'mss': [values]}
        locus_groups = {}
        
        for sid, loci in locus_data.items():
            label = sample_labels.get(sid)
            if label is None:
                continue
            
            group = 'msi' if label == 1 else 'mss'
            for feat in loci:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_groups:
                    locus_groups[key] = {'msi': [], 'mss': []}
                
                # Get the metric value
                if self.metric == 'max':
                    val = max(feat.get('alt_ratio', 0), feat.get('entropy', 0))
                else:
                    val = feat.get(self.metric, 0)
                
                locus_groups[key][group].append(val)
        
        # Filter loci with too few samples in each group
        locus_groups = {
            k: v for k, v in locus_groups.items()
            if len(v['msi']) >= self.min_depth and len(v['mss']) >= self.min_depth
        }
        
        logger.info(f"Effect size ({self.effect_size_type}, metric={self.metric}): "
                   f"{len(locus_groups)} loci with >= {self.min_depth} samples per group")
        
        # Compute effect size for each locus
        self.locus_effect_sizes_ = {}
        
        for key, groups in locus_groups.items():
            msi_values = np.array(groups['msi'])
            mss_values = np.array(groups['mss'])
            
            es = self._effect_size_fn(msi_values, mss_values)
            self.locus_effect_sizes_[key] = es
        
        # Select loci with |effect_size| >= threshold
        self.selected_loci_ = {
            key for key, es in self.locus_effect_sizes_.items()
            if abs(es) >= self.threshold
        }
        
        # Log results
        effect_sizes = list(self.locus_effect_sizes_.values())
        abs_effect_sizes = [abs(e) for e in effect_sizes]
        
        logger.info(
            f"Effect size selection: {len(self.selected_loci_)}/{len(self.locus_effect_sizes_)} loci selected "
            f"(|{self.effect_size_type}| >= {self.threshold})"
        )
        logger.info(f"  Effect size distribution: mean={np.mean(abs_effect_sizes):.3f}, "
                   f"median={np.median(abs_effect_sizes):.3f}, "
                   f">=0.2 (small): {sum(1 for e in abs_effect_sizes if e >= 0.2)}, "
                   f">=0.5 (medium): {sum(1 for e in abs_effect_sizes if e >= 0.5)}, "
                   f">=0.8 (large): {sum(1 for e in abs_effect_sizes if e >= 0.8)}")
        
        # Log direction
        positive = sum(1 for e in effect_sizes if e > 0)
        negative = sum(1 for e in effect_sizes if e < 0)
        logger.info(f"  Direction: {positive} positive (MSI > MSS), {negative} negative (MSI < MSS)")
        
        return self
    
    def is_selected(self, locus_feat: Dict) -> bool:
        """Check if a locus is selected."""
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Pluggable rule engine for MSI detection post-processing.

Provides rule-based overrides for ML model predictions:
- Extreme instability rules (catch MSI-H with few unstable loci)
- Cancer-type specific rules
- Confidence-based rules

Usage:
    engine = RuleEngine(rules=[ExtremeInstabilityRule(), CancerSpecificRule()])
    final_predictions = engine.apply(predictions, scores, features_df)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Rule(ABC):
    """Abstract base class for a prediction rule."""
    
    @abstractmethod
    def apply(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        features_df: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """Apply rule and return modified predictions.
        
        Parameters
        ----------
        predictions : np.ndarray
            Current predictions ('MSI-H' or 'MSS').
        scores : np.ndarray
            Model scores.
        features_df : pd.DataFrame
            Feature matrix.
        threshold : float
            Current decision threshold.
        
        Returns
        -------
        np.ndarray
            Modified predictions.
        """
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Return human-readable description of the rule."""
        pass


class ExtremeInstabilityRule(Rule):
    """Rule to catch MSI-H samples with few but extreme unstable loci.
    
    Problem: Some MSI-H samples have only 1-2 loci with alt_ratio > 0.9,
    but the model may miss them because aggregation dilutes the signal.
    
    Solution: If a sample has max_alt_ratio > threshold AND at least n_min
    extreme loci, override prediction to MSI-H.
    
    Parameters
    ----------
    alt_threshold : float
        Minimum max_alt_ratio to trigger rule (default: 0.9).
    n_min : int
        Minimum number of extreme loci required (default: 1).
    score_margin : float
        Only apply if model score is within this margin of threshold.
        Helps avoid overriding confident correct predictions.
    """
    
    def __init__(
        self,
        alt_threshold: float = 0.9,
        n_min: int = 1,
        score_margin: float = 0.2,
    ):
        self.alt_threshold = alt_threshold
        self.n_min = n_min
        self.score_margin = score_margin
    
    def apply(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        features_df: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """Apply extreme instability rule."""
        modified = predictions.copy()
        
        # Check if required features exist
        if 'max_alt_ratio' not in features_df.columns:
            logger.warning("max_alt_ratio not in features, skipping ExtremeInstabilityRule")
            return modified
        
        max_alt = features_df['max_alt_ratio'].values
        
        # Count extreme loci if available
        if 'n_extreme_loci_08' in features_df.columns:
            n_extreme = features_df['n_extreme_loci_08'].values
        elif 'n_extreme_loci_05' in features_df.columns:
            n_extreme = features_df['n_extreme_loci_05'].values
        else:
            # Fallback: use high_alt_ratio * n_loci
            n_loci = features_df.get('n_loci', pd.Series([50] * len(features_df)))
            high_alt = features_df.get('high_alt_ratio', pd.Series([0] * len(features_df)))
            n_extreme = (high_alt * n_loci).values
        
        # Apply rule: override MSS predictions where extreme instability detected
        # Only override if model is not very confident (score not too low)
        override_mask = (
            (predictions == 'MSS') &
            (max_alt > self.alt_threshold) &
            (n_extreme >= self.n_min) &
            (scores > threshold - self.score_margin)
        )
        
        n_overridden = override_mask.sum()
        if n_overridden > 0:
            modified[override_mask] = 'MSI-H'
            logger.info(f"ExtremeInstabilityRule: overridden {n_overridden} MSS -> MSI-H "
                       f"(max_alt>{self.alt_threshold}, n_extreme>={self.n_min})")
        
        return modified
    
    def describe(self) -> str:
        return (f"ExtremeInstabilityRule: if max_alt_ratio > {self.alt_threshold} "
                f"and n_extreme_loci >= {self.n_min}, predict MSI-H")


class CancerSpecificRule(Rule):
    """Cancer-type specific threshold adjustment.
    
    Some cancer types (e.g., endometrial) have higher baseline instability.
    This rule applies lower thresholds for specific cancer types.
    
    Parameters
    ----------
    cancer_thresholds : dict
        {cancer_type: threshold} for cancer-specific thresholds.
    """
    
    def __init__(self, cancer_thresholds: Optional[Dict[str, float]] = None):
        self.cancer_thresholds = cancer_thresholds or {}
    
    def apply(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        features_df: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """Apply cancer-specific thresholds."""
        modified = predictions.copy()
        
        if 'cancertype' not in features_df.columns:
            return modified
        
        n_overridden = 0
        for cancer, cancer_thr in self.cancer_thresholds.items():
            if cancer_thr >= threshold:
                continue  # Only apply if cancer threshold is lower
            
            mask = features_df['cancertype'] == cancer
            override_mask = (
                mask &
                (predictions == 'MSS') &
                (scores >= cancer_thr) &
                (scores < threshold)
            )
            
            n_overridden += override_mask.sum()
            modified[override_mask] = 'MSI-H'
        
        if n_overridden > 0:
            logger.info(f"CancerSpecificRule: overridden {n_overridden} MSS -> MSI-H "
                       f"using cancer-specific thresholds")
        
        return modified
    
    def describe(self) -> str:
        return f"CancerSpecificRule: {self.cancer_thresholds}"


class ConfidenceRule(Rule):
    """Rule based on prediction confidence.
    
    If model score is very close to threshold, mark as uncertain
    or apply different logic.
    
    Parameters
    ----------
    uncertain_margin : float
        Margin around threshold to mark as uncertain (default: 0.1).
    uncertain_label : str
        Label for uncertain predictions (default: 'MSI-H', conservative).
    """
    
    def __init__(
        self,
        uncertain_margin: float = 0.1,
        uncertain_label: str = 'MSI-H',
    ):
        self.uncertain_margin = uncertain_margin
        self.uncertain_label = uncertain_label
    
    def apply(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        features_df: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """Apply confidence-based rule."""
        modified = predictions.copy()
        
        # Find uncertain predictions (close to threshold)
        uncertain_mask = np.abs(scores - threshold) < self.uncertain_margin
        
        # For uncertain cases, apply conservative labeling
        override_mask = (
            uncertain_mask &
            (predictions == 'MSS') &
            (self.uncertain_label == 'MSI-H')
        )
        
        n_overridden = override_mask.sum()
        if n_overridden > 0:
            modified[override_mask] = 'MSI-H'
            logger.info(f"ConfidenceRule: overridden {n_overridden} uncertain MSS -> MSI-H")
        
        return modified
    
    def describe(self) -> str:
        return (f"ConfidenceRule: uncertain within ±{self.uncertain_margin} of threshold, "
                f"default to {self.uncertain_label}")


class MaxAltRatioRule(Rule):
    """Simple rule: if max_alt_ratio exceeds a hard threshold, force MSI-H.
    
    This is the simplest rule for catching MSI-H with few unstable loci.
    
    Parameters
    ----------
    hard_threshold : float
        If max_alt_ratio > this, always predict MSI-H (default: 0.95).
    """
    
    def __init__(self, hard_threshold: float = 0.95):
        self.hard_threshold = hard_threshold
    
    def apply(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        features_df: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """Apply hard threshold rule."""
        modified = predictions.copy()
        
        if 'max_alt_ratio' not in features_df.columns:
            return modified
        
        override_mask = (
            (predictions == 'MSS') &
            (features_df['max_alt_ratio'].values > self.hard_threshold)
        )
        
        n_overridden = override_mask.sum()
        if n_overridden > 0:
            modified[override_mask] = 'MSI-H'
            logger.info(f"MaxAltRatioRule: overridden {n_overridden} MSS -> MSI-H "
                       f"(max_alt > {self.hard_threshold})")
        
        return modified
    
    def describe(self) -> str:
        return f"MaxAltRatioRule: if max_alt_ratio > {self.hard_threshold}, predict MSI-H"


class RuleEngine:
    """Pluggable rule engine for post-processing predictions.
    
    Applies a chain of rules in order. Each rule can override previous decisions.
    
    Usage:
        engine = RuleEngine(rules=[
            MaxAltRatioRule(hard_threshold=0.95),
            ExtremeInstabilityRule(alt_threshold=0.9, n_min=1),
        ])
        final_predictions = engine.apply(predictions, scores, features_df, threshold)
    
    Parameters
    ----------
    rules : list of Rule
        Rules to apply in order.
    """
    
    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules = rules or []
    
    def apply(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        features_df: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """Apply all rules in order.
        
        Parameters
        ----------
        predictions : np.ndarray
            Initial model predictions.
        scores : np.ndarray
            Model scores.
        features_df : pd.DataFrame
            Feature matrix.
        threshold : float
            Decision threshold.
        
        Returns
        -------
        np.ndarray
            Final predictions after all rules applied.
        """
        result = predictions.copy()
        
        for rule in self.rules:
            result = rule.apply(result, scores, features_df, threshold)
        
        return result
    
    def describe(self) -> str:
        """Describe all rules in the engine."""
        lines = ["RuleEngine:"]
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"  {i}. {rule.describe()}")
        return "\n".join(lines)


# Predefined rule configurations
def create_conservative_engine() -> RuleEngine:
    """Create a conservative rule engine (few overrides)."""
    return RuleEngine(rules=[
        MaxAltRatioRule(hard_threshold=0.95),
    ])


def create_balanced_engine() -> RuleEngine:
    """Create a balanced rule engine."""
    return RuleEngine(rules=[
        MaxAltRatioRule(hard_threshold=0.95),
        ExtremeInstabilityRule(alt_threshold=0.9, n_min=1, score_margin=0.2),
    ])


def create_sensitive_engine() -> RuleEngine:
    """Create a sensitive rule engine (more overrides, higher sensitivity)."""
    return RuleEngine(rules=[
        MaxAltRatioRule(hard_threshold=0.9),
        ExtremeInstabilityRule(alt_threshold=0.85, n_min=1, score_margin=0.3),
    ])

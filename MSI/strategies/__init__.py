# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Aggregation strategies for MSI feature extraction."""

from typing import Dict

from .base import AggregationStrategy
from .baseline import BaselineAggregation
from .tail import TailAggregation
from .dist import DistAggregation
from .bio_prior import BioPriorAggregation
from .percentile import PercentileAggregation
from .count import CountAggregation
from .shift_rich import ShiftRichAggregation
from .noise_filtered import NoiseFilteredAggregation
from .mono_focus import MonoFocusAggregation
from .all_features import AllAggregation
from .interaction import InteractionAggregation
from .distribution import DistributionAggregation
from .multi_threshold import MultiThresholdAggregation
from .weighted import WeightedAggregation
from .optimized import OptimizedAggregation
from .locus_level import LocusLevelAggregation
from .advanced import AdvancedAggregation
from .unstable_locus import UnstableLocusAggregation
from .sensitive import SensitiveAggregation
from .locus_score import LocusScoreAggregation

AGG_STRATEGIES: Dict[str, AggregationStrategy] = {
    'baseline': BaselineAggregation(),
    'tail': TailAggregation(),
    'dist': DistAggregation(),
    'bio_prior': BioPriorAggregation(),
    'percentile': PercentileAggregation(),
    'count': CountAggregation(),
    'shift_rich': ShiftRichAggregation(),
    'noise_filtered': NoiseFilteredAggregation(),
    'mono_focus': MonoFocusAggregation(),
    'all': AllAggregation(),
    'interaction': InteractionAggregation(),
    'distribution': DistributionAggregation(),
    'multi_threshold': MultiThresholdAggregation(),
    'weighted': WeightedAggregation(),
    'optimized': OptimizedAggregation(),
    'locus_level': LocusLevelAggregation(),
    'advanced': AdvancedAggregation(),
    'unstable_locus': UnstableLocusAggregation(),
    'sensitive': SensitiveAggregation(),
    'locus_score': LocusScoreAggregation(),
}

__all__ = [
    'AggregationStrategy', 'AGG_STRATEGIES',
    'BaselineAggregation', 'TailAggregation', 'DistAggregation',
    'BioPriorAggregation', 'PercentileAggregation', 'CountAggregation',
    'ShiftRichAggregation', 'NoiseFilteredAggregation', 'MonoFocusAggregation',
    'AllAggregation', 'InteractionAggregation', 'DistributionAggregation',
    'MultiThresholdAggregation', 'WeightedAggregation', 'OptimizedAggregation',
    'LocusLevelAggregation', 'AdvancedAggregation', 'UnstableLocusAggregation',
    'SensitiveAggregation', 'LocusScoreAggregation',
]

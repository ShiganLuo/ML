# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Modular MSI detection framework with pluggable components."""

# Constants
from .constants import COL_CHROM, COL_POS, COL_UNIT_LEN, COL_REPEAT_TIMES, COL_REPEAT_DICT, COL_DEPTH

# Utilities
from .utils import _common_features, _unit_len_features, parse_repeat_counts, compute_roc, find_best_threshold, evaluate

# Features
from .features import FeatureExtractor

# Selectors
from .feature_selectors import (
    LocusSelector, AUCBasedLocusSelector, UnitLengthLocusSelector,
    CombinedLocusSelector, NullLocusSelector, RelaxedAUCSelector, MultiMetricLocusSelector,
    FeatureSelector, SingleVariableAUCSelector, TwoStageSelector, LassoSelector,
    XgbImportanceSelector, TwoStageXgbSelector,
)

# Filters
from .filters import SampleFilter, DepthFilter, QualityFilter, CombinedFilter, AnomalyFilter, MultivariateOutlierFilter

# Strategies
from .strategies import (
    AggregationStrategy, AGG_STRATEGIES,
    BaselineAggregation, TailAggregation, DistAggregation,
    BioPriorAggregation, PercentileAggregation, CountAggregation,
    ShiftRichAggregation, NoiseFilteredAggregation, MonoFocusAggregation,
    AllAggregation, InteractionAggregation, DistributionAggregation,
    MultiThresholdAggregation, WeightedAggregation, OptimizedAggregation,
    LocusLevelAggregation, AdvancedAggregation, UnstableLocusAggregation,
    SensitiveAggregation, LocusScoreAggregation,
)

# Detectors
from .detectors import (
    Detector, MahalanobisDetector, MSIPercentageDetector,
    OneClassSVMDetector, IsolationForestDetector, OCLRDetector,
    BinaryClassifierDetector, EnsembleDetector, UnstableProportionDetector,
    CosineDetector,
)

# Pipeline
from .pipeline import MSIDetectionPipeline

__all__ = [
    # Constants
    'COL_CHROM', 'COL_POS', 'COL_UNIT_LEN', 'COL_REPEAT_TIMES', 'COL_REPEAT_DICT', 'COL_DEPTH',
    # Utils
    '_common_features', '_unit_len_features', 'parse_repeat_counts',
    'compute_roc', 'find_best_threshold', 'evaluate',
    # Features
    'FeatureExtractor',
    # Selectors
    'LocusSelector', 'AUCBasedLocusSelector', 'UnitLengthLocusSelector',
    'CombinedLocusSelector', 'NullLocusSelector', 'RelaxedAUCSelector', 'MultiMetricLocusSelector',
    'FeatureSelector', 'SingleVariableAUCSelector', 'TwoStageSelector', 'LassoSelector',
    'XgbImportanceSelector', 'TwoStageXgbSelector',
    # Filters
    'SampleFilter', 'DepthFilter', 'QualityFilter', 'CombinedFilter', 'AnomalyFilter', 'MultivariateOutlierFilter',
    # Strategies
    'AggregationStrategy', 'AGG_STRATEGIES',
    'BaselineAggregation', 'TailAggregation', 'DistAggregation',
    'BioPriorAggregation', 'PercentileAggregation', 'CountAggregation',
    'ShiftRichAggregation', 'NoiseFilteredAggregation', 'MonoFocusAggregation',
    'AllAggregation', 'InteractionAggregation', 'DistributionAggregation',
    'MultiThresholdAggregation', 'WeightedAggregation', 'OptimizedAggregation',
    'LocusLevelAggregation', 'AdvancedAggregation', 'UnstableLocusAggregation',
    'SensitiveAggregation', 'LocusScoreAggregation',
    # Detectors
    'Detector', 'MahalanobisDetector', 'MSIPercentageDetector',
    'OneClassSVMDetector', 'IsolationForestDetector', 'OCLRDetector',
    'BinaryClassifierDetector', 'EnsembleDetector', 'UnstableProportionDetector',
    'CosineDetector',
    # Pipeline
    'MSIDetectionPipeline',
]

# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Detector implementations for MSI detection."""

from .base import Detector
from .mahalanobis import MahalanobisDetector
from .msi_pct import MSIPercentageDetector
from .ocsvm import OneClassSVMDetector
from .iforest import IsolationForestDetector
from .oclr import OCLRDetector
from .binary import BinaryClassifierDetector
from .ensemble import EnsembleDetector
from .unstable_prop import UnstableProportionDetector
from .cosine import CosineDetector

__all__ = [
    'Detector',
    'MahalanobisDetector',
    'MSIPercentageDetector',
    'OneClassSVMDetector',
    'IsolationForestDetector',
    'OCLRDetector',
    'BinaryClassifierDetector',
    'EnsembleDetector',
    'UnstableProportionDetector',
    'CosineDetector',
]

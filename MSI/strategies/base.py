# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""AggregationStrategy abstract base class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import re

import pandas as pd

class AggregationStrategy(ABC):
    """Abstract base class for sample-level feature aggregation.

    Each strategy defines how locus-level features are aggregated into
    a single feature vector per sample. Strategies differ in which
    statistics they compute and how they weight different signal types.

    Subclasses must implement :meth:`aggregate` and :meth:`get_feature_names`.
    """

    @abstractmethod
    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        """Aggregate locus-level features into sample-level features.

        Parameters
        ----------
        lf : pd.DataFrame
            Locus-level feature dataframe for a single sample.

        Returns
        -------
        Optional[Dict]
            Feature dictionary, or None if aggregation is not possible.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names produced by this strategy.

        Returns
        -------
        list of str
            Feature names in the order they appear in the output dict.
        """
        pass

    def get_name(self) -> str:
        """Return human-readable strategy name.

        Returns
        -------
        str
            Class name in snake_case.
        """
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', type(self).__name__).lower()
        if name.endswith('_aggregation'):
            name = name[:-len('_aggregation')]
        return name

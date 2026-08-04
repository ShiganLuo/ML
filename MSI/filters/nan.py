from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import pandas as pd
import numpy as np
from .base import SampleFilter
logger = logging.getLogger(__name__)

class NaNFilter(SampleFilter):
    """Fill NaN/inf in numeric feature columns with 0.

    Runs early (before feature selection) on ALL numeric columns.
    Non-numeric columns (MSI_status, cancertype, etc.) and metadata
    columns listed in ``exclude`` are ignored.

    Parameters
    ----------
    exclude : set[str] | None
        Column names to skip when checking for NaN (e.g. metadata).
        Defaults to {'MSI_status', 'cancertype', 'MSI_CNC', 'chrom',
        'sample_id', 'origin'}.
    """

    _DEFAULT_EXCLUDE = frozenset({
        'MSI_status', 'cancertype', 'MSI_CNC',
        'chrom', 'sample_id', 'origin',
    })

    def __init__(self, exclude: Optional[set] = None):
        self.exclude = exclude or set(self._DEFAULT_EXCLUDE)

    def filter(self, features_df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            c for c in features_df.columns
            if c not in self.exclude
            and pd.api.types.is_numeric_dtype(features_df[c])
        ]
        if not numeric_cols:
            return features_df

        sub = features_df[numeric_cols]
        bad = sub.isna() | ~np.isfinite(sub.values)
        n_affected = bad.any(axis=1).sum()
        if n_affected > 0:
            col_counts = bad.sum()
            col_counts = col_counts[col_counts > 0].sort_values(ascending=False)
            detail = ', '.join(f'{c}={n}' for c, n in col_counts.head(5).items())
            logger.info(
                f"NaNFilter: filled {n_affected}/{len(features_df)} samples "
                f"with 0 ({detail}{'...' if len(col_counts) > 5 else ''})"
            )
        features_df[numeric_cols] = sub.fillna(0).replace([np.inf, -np.inf], 0)
        return features_df

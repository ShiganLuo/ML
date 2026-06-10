"""Shared data loading utilities for semi-supervised SV frequency correction.

Loads labeled (ddPCR) and unlabeled feature tables, combines them for
preprocessing, and returns train-ready matrices.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from features import parser_table, preprocess_features, load_preprocessing

logger = logging.getLogger(__name__)

# Columns produced by parser_table() that are not model features.
META_COLUMNS = [
    "原始编号", "FusionGene", "FusionExon", "sampleID",
    "ddPCR_AF", "Freq", "FusionType",
]

# Default extra_keep_cols for parser_table().
DEFAULT_EXTRA_KEEP = ["原始编号", "FusionGene", "FusionExon", "sampleID",
                       "ddPCR_AF", "Freq", "FusionType"]


def _extract_or_load_features(
    tsv_path: str,
    outdir: str,
    prefix: str,
    probe_infile: Optional[str],
    bam_columns: List[str],
    extra_keep_cols: List[str],
    force_extract: bool = False,
) -> pd.DataFrame:
    """Extract BAM features from a TSV file or load from a cached copy.

    Parses the input TSV with ``parser_table`` to extract BAM-derived
    features.  If a cached result already exists on disk, it is loaded
    directly instead of re-extracting.  Missing columns listed in
    ``extra_keep_cols`` that are absent from the input TSV are added as
    NaN so that downstream ``concat`` operations succeed.

    Parameters
    ----------
    tsv_path : str
        Path to the input feature TSV file.
    outdir : str
        Directory where the cached feature file is stored.
    prefix : str
        Prefix for the cache filename (e.g. ``"labeled"`` or ``"unlabeled"``).
    probe_infile : str or None
        Path to the probe BED file passed to ``parser_table``.
    bam_columns : list of str
        Column names in the TSV that identify BAM path and breakpoint
        positions (e.g. ``["BamPath", "Pos1", "Pos2"]``).
    extra_keep_cols : list of str
        Additional non-feature columns to retain from the TSV.
    force_extract : bool, optional
        If True, ignore any cached file and re-extract features.
        Default is False.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing BAM features plus the requested metadata
        columns.  Missing meta columns are filled with NaN.
    """
    cache_path = os.path.join(outdir, f"{prefix}_raw_features.tsv")
    if not force_extract and os.path.exists(cache_path):
        logger.info("Loading cached features from %s", cache_path)
        return pd.read_csv(cache_path, sep="\t")
    logger.info("Extracting BAM features for %s (%s) ...", prefix, tsv_path)
    # Filter extra_keep_cols to only those present in the input file
    # (e.g. unlabeled TSV has no ddPCR_AF column)
    header = pd.read_csv(tsv_path, sep="\t", nrows=0)
    available_cols = set(header.columns)
    valid_keep_cols = [c for c in extra_keep_cols if c in available_cols]
    missing_cols = [c for c in extra_keep_cols if c not in available_cols]
    if missing_cols:
        logger.info("Columns not in %s (will be added as NaN): %s", tsv_path, missing_cols)
    df = parser_table(
        infile=tsv_path,
        probe_infile=probe_infile,
        outdir=None,
        extra_keep_cols=valid_keep_cols,
        bam_columns=bam_columns,
    )
    # Add missing columns as NaN so downstream combining works
    for col in missing_cols:
        df[col] = np.nan
    # Log extracted columns for debugging
    bam_feature_cols = [c for c in df.columns if c not in extra_keep_cols and not c.startswith("sv_type_")]
    logger.info("Extracted %d BAM feature columns for %s: %s",
                len(bam_feature_cols), prefix, bam_feature_cols[:10] if bam_feature_cols else "(none)")
    df.to_csv(cache_path, sep="\t", index=False)
    return df


def load_combined_features(
    labeled_tsv: str,
    unlabeled_tsv: str,
    outdir: str,
    probe_infile: Optional[str] = None,
    bam_columns: Optional[List[str]] = None,
    extra_keep_cols: Optional[List[str]] = None,
    force_extract: bool = False,
    feature_cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Load and preprocess labeled + unlabeled data together.

    Reads labeled (ddPCR-validated) and unlabeled feature tables,
    extracts BAM features via :func:`_extract_or_load_features`,
    combines them, and applies preprocessing (StandardScaler,
    VarianceThreshold, and correlated-feature filtering) using
    :func:`preprocess_features`.  The combined data is used for
    semi-supervised learning while only the labeled subset retains the
    ``ddPCR_AF`` target column.

    Parameters
    ----------
    labeled_tsv : str
        Path to the labeled (ddPCR) feature TSV file.
    unlabeled_tsv : str
        Path to the unlabeled feature TSV file.
    outdir : str
        Output directory for preprocessing artifacts (scalers, selectors).
    probe_infile : str or None, optional
        Probe BED file forwarded to ``parser_table``.  Default is None.
    bam_columns : list of str or None, optional
        Column names for BAM path and breakpoint positions.  Defaults to
        ``["BamPath", "Pos1", "Pos2"]``.
    extra_keep_cols : list of str or None, optional
        Metadata columns to preserve.  Defaults to
        ``DEFAULT_EXTRA_KEEP``.
    force_extract : bool, optional
        Force re-extraction of BAM features even if a cache exists.
        Default is False.
    feature_cache_dir : str or None, optional
        Shared directory for BAM feature extraction cache.  When set,
        features are extracted/loaded from this directory instead of
        ``outdir/labeled/`` and ``outdir/unlabeled/``.  This allows
        multiple scripts to share the same extracted features.
        Preprocessing results still go to ``outdir``.

    Returns
    -------
    labeled_df : DataFrame
        Preprocessed labeled rows (with ddPCR_AF).
    combined_df : DataFrame
        Preprocessed combined rows (labeled + unlabeled).
    feature_columns : list[str]
        Numeric feature columns used for modeling.
    no_scale_columns : list[str]
        Columns to skip during StandardScaler (sv_type one-hot).
    """
    if bam_columns is None:
        bam_columns = ["BamPath", "Pos1", "Pos2"]
    if extra_keep_cols is None:
        extra_keep_cols = list(DEFAULT_EXTRA_KEEP)

    os.makedirs(outdir, exist_ok=True)

    # --- BAM feature extraction (shared or per-script) ---
    if feature_cache_dir is not None:
        labeled_feat_dir = os.path.join(feature_cache_dir, "labeled")
        unlabeled_feat_dir = os.path.join(feature_cache_dir, "unlabeled")
    else:
        labeled_feat_dir = os.path.join(outdir, "labeled")
        unlabeled_feat_dir = os.path.join(outdir, "unlabeled")
    os.makedirs(labeled_feat_dir, exist_ok=True)
    os.makedirs(unlabeled_feat_dir, exist_ok=True)

    extra_keep_cols_labeled = [c for c in extra_keep_cols if c in pd.read_csv(labeled_tsv, sep="\t", nrows=0).columns]
    extra_keep_cols_unlabeled = [c for c in extra_keep_cols if c in pd.read_csv(unlabeled_tsv, sep="\t", nrows=0).columns]
    df_labeled = _extract_or_load_features(
        labeled_tsv, labeled_feat_dir, "labeled",
        probe_infile, bam_columns, extra_keep_cols_labeled, force_extract,
    )
    df_unlabeled = _extract_or_load_features(
        unlabeled_tsv, unlabeled_feat_dir, "unlabeled",
        probe_infile, bam_columns, extra_keep_cols_unlabeled, force_extract,
    )

    # Add is_labeled flag before combining
    df_labeled = df_labeled.copy()
    df_unlabeled = df_unlabeled.copy()
    df_labeled["_is_labeled"] = True
    df_unlabeled["_is_labeled"] = False

    # Fill missing meta columns in unlabeled data
    for col in extra_keep_cols:
        if col not in df_unlabeled.columns:
            df_unlabeled[col] = np.nan

    # Align columns
    common_cols = [c for c in df_labeled.columns if c in df_unlabeled.columns]
    df_labeled = df_labeled[common_cols]
    df_unlabeled = df_unlabeled[common_cols]

    combined = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

    # --- Identify feature columns ---
    exclude = set(META_COLUMNS) | {"_is_labeled"}
    sv_type_cols = [c for c in combined.columns if c.startswith("sv_type_")]
    feature_columns = [
        c for c in combined.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(combined[c])
    ]
    no_scale_columns = [c for c in feature_columns if c in sv_type_cols]
    numeric_feature_columns = [c for c in feature_columns if c not in no_scale_columns]

    logger.info("Features: %d total, %d numeric, %d no-scale",
                len(feature_columns), len(numeric_feature_columns), len(no_scale_columns))
    if not numeric_feature_columns:
        logger.warning("No numeric feature columns found! Only sv_type one-hot columns will be used.")
        logger.warning("This likely means BAM feature extraction failed (BAM files not accessible?).")
        logger.warning("Check that BAM paths in the TSV are valid and pysam can read them.")

    # --- Preprocess (fit on combined data, results go to outdir) ---
    keep_cols = [c for c in combined.columns if c not in feature_columns]
    combined_processed, diagnostics = preprocess_features(
        combined,
        feature_columns=numeric_feature_columns,
        outdir=outdir,
        keep_columns=keep_cols + no_scale_columns,
    )

    # Split back
    labeled_mask = combined_processed["_is_labeled"] == True  # noqa: E712
    labeled_processed = combined_processed[labeled_mask].copy()
    labeled_processed.drop(columns=["_is_labeled"], inplace=True, errors="ignore")

    combined_processed.drop(columns=["_is_labeled"], inplace=True, errors="ignore")

    # Final feature columns after preprocessing
    # Directly read from the processed DataFrame (preprocess_features already
    # dropped variance-filtered and correlated features from the returned DF).
    non_feature = set(META_COLUMNS) | {"_is_labeled"}
    final_features = [
        c for c in combined_processed.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(combined_processed[c])
    ]

    logger.info("Labeled rows: %d, Combined rows: %d", len(labeled_processed), len(combined_processed))
    logger.info("Final feature columns (%d): %s", len(final_features), final_features)

    return labeled_processed, combined_processed, final_features, no_scale_columns


def grouped_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    group_cols: Optional[List[str]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Grouped train/test split with NaN-group safety.

    Performs a group-aware train/test split using
    :class:`sklearn.model_selection.GroupShuffleSplit`.  Rows with NaN
    in any grouping column are excluded from the split and automatically
    placed in the training set, since they may still carry valid labels.

    Parameters
    ----------
    X, y : ndarray
        Feature matrix and label array.
    df : DataFrame
        Source dataframe (must share index with X/y).
    group_cols : list[str] or None
        Columns to group by.  None defaults to ``["原始编号"]``.
    test_size : float
        Fraction of groups reserved for the test set.  Default is 0.2.
    random_state : int, optional
        Random seed for reproducibility.  Default is 42.

    Returns
    -------
    train_idx, test_idx : ndarray
        Integer arrays of row indices into *X*/*y* for training and
        testing, respectively.
    train_groups : ndarray or None
        Composite group labels corresponding to the training rows,
        suitable for use with ``GroupKFold`` during cross-validation.
        None if no valid group columns are found.
    """
    if group_cols is None:
        group_cols = ["原始编号"]
    available = [c for c in group_cols if c in df.columns]
    if not available:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(X)), test_size=test_size, random_state=random_state)
        return tr, te, None

    # Build composite group key
    groups = df[available[0]].astype(str)
    for col in available[1:]:
        groups = groups + "__" + df[col].astype(str)

    # NaN-safe: rows with NaN group keys go to training, not test
    valid_mask = df[available].notna().all(axis=1)
    if not valid_mask.all():
        n_bad = int((~valid_mask).sum())
        logger.warning("%d rows have NaN in group columns, added to training set", n_bad)

    groups_valid = groups[valid_mask]
    valid_idx = np.where(valid_mask.to_numpy())[0]
    nan_idx = np.where(~valid_mask.to_numpy())[0]

    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr, te = next(splitter.split(X[valid_idx], y[valid_idx], groups=groups_valid))

    train_idx = np.concatenate([valid_idx[tr], nan_idx])
    test_idx = valid_idx[te]
    train_groups = groups.to_numpy()[train_idx]

    return train_idx, test_idx, train_groups


def extract_xy(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_col: str = "ddPCR_AF",
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """Extract feature matrix and label vector from a labeled DataFrame.

    Selects the specified feature columns and label column, drops rows
    where the label is NaN, and returns NumPy arrays ready for model
    training.  Remaining NaN values in the feature matrix are filled
    with 0.0.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame containing both feature and label columns.
    feature_columns : list of str
        Column names to use as the feature matrix.
    label_col : str, optional
        Name of the target/label column.  Default is ``"ddPCR_AF"``.

    Returns
    -------
    X : numpy.ndarray
        2-D float64 feature array of shape ``(n_valid, n_features)``.
    y : numpy.ndarray
        1-D float64 label array of shape ``(n_valid,)``.
    index : pandas.Index
        Index of the rows retained (i.e. those with a non-NaN label).
    """
    valid = df[label_col].notna()
    df_valid = df[valid]
    X = df_valid[feature_columns].fillna(0.0).to_numpy(dtype=float)
    y = df_valid[label_col].to_numpy(dtype=float)
    return X, y, df_valid.index

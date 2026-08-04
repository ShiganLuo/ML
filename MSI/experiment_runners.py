# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Reusable experiment runners for MSI detection.

Two patterns:
  - train_and_eval: train on dataset A, evaluate on dataset B (cross-reagent)
  - self_eval: train+evaluate on same dataset (pipeline does internal split)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .pipeline import MSIDetectionPipeline
from .metrics import calc_metrics, find_youden_threshold
from compare_features import _build_pipeline_components, _make_detector

logger = logging.getLogger(__name__)


def build_pipeline(strat_cfg: Dict, det_cfg: Dict) -> MSIDetectionPipeline:
    """Build a complete MSIDetectionPipeline from strategy + detector configs.

    Deduplicates the 3-step pattern: _build_pipeline_components → _make_detector → MSIDetectionPipeline.

    Parameters
    ----------
    strat_cfg : dict
        Strategy config dict (from compare_features.json).
    det_cfg : dict
        Detector config dict.

    Returns
    -------
    MSIDetectionPipeline
        Ready-to-use pipeline.
    """
    fe, locus_sel, feat_sel, sf, train_filter, req_feats, rule_engine = \
        _build_pipeline_components(strat_cfg)
    det = _make_detector(det_cfg)
    return MSIDetectionPipeline(
        feature_extractor=fe,
        locus_selector=locus_sel,
        feature_selector=feat_sel,
        sample_filter=sf,
        detector=det,
        train_filter=train_filter,
        required_features=req_feats,
        rule_engine=rule_engine,
    )


def extract_features(
    fe, sf, meta: pd.DataFrame,
    site_col: str = "site_feature",
    feature_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Extract + filter features from metadata.

    Parameters
    ----------
    fe : FeatureExtractor
    sf : SampleFilter
    meta : pd.DataFrame
        Sample metadata with site file paths and labels.
    site_col : str
        Column name for site file paths.
    feature_logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame
        Feature matrix with MSI_status column.
    """
    if feature_logger:
        feature_logger.info(f"Extracting features from {len(meta)} samples (col={site_col})")
    features, _ = fe.extract_batch(meta[site_col].values, meta.index.values)
    join_cols = [c for c in ["MSI_real", "cancertype", "tumor_content", "TMB_status"] if c in meta.columns]
    features = features.join(meta[join_cols], how="inner")
    if "MSI_real" in features.columns:
        features.rename(columns={"MSI_real": "MSI_status"}, inplace=True)
    features = sf.filter(features)
    if feature_logger:
        feature_logger.info(f"After filter: {len(features)} samples")
    return features


def train_and_eval(
    pipeline: MSIDetectionPipeline,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    run_cfg: Dict,
    site_col_train: str = "site_feature",
    site_col_test: str = "site_feature",
    cache_dir: Optional[str] = None,
    exp_logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Train on train_meta, evaluate on separate test_meta.

    Use when train and test are different datasets (e.g. v3→v4).

    Parameters
    ----------
    pipeline : MSIDetectionPipeline
        Configured pipeline (from build_pipeline).
    train_meta : pd.DataFrame
        Training metadata.
    test_meta : pd.DataFrame
        Test metadata (separate dataset).
    run_cfg : dict
        Pipeline run parameters (n_sigma, test_size, threshold_method, cv_folds).
    site_col_train : str
        Site file column for training data.
    site_col_test : str
        Site file column for test data.
    cache_dir : str, optional
        Cache directory for feature extraction.
    exp_logger : logging.Logger, optional

    Returns
    -------
    dict
        train_result, threshold, youden_threshold, selected_cols,
        n_features_used, n_features_missing, test_metrics, youden_metrics,
        test_features, test_scores, y_true.
    """
    # Train
    if exp_logger:
        exp_logger.info(f"Training on {len(train_meta)} samples...")
    train_res = pipeline.run(
        train_meta,
        n_sigma=run_cfg.get("n_sigma", 3.0),
        site_file_col=site_col_train,
        test_size=run_cfg.get("test_size", 0.2),
        cache_dir=cache_dir,
        msi_col="MSI_real",
        threshold_method=run_cfg.get("threshold_method", "cv"),
        cv_folds=run_cfg.get("cv_folds", 5),
    )
    threshold = train_res["threshold"]
    selected_cols = train_res["selected_cols"]
    if exp_logger:
        exp_logger.info(f"Threshold={threshold:.4f}, features={len(selected_cols)}")

    # Evaluate on separate test set
    if exp_logger:
        exp_logger.info(f"Evaluating on {len(test_meta)} samples...")
    test_features = extract_features(
        pipeline.feature_extractor, pipeline.sample_filter,
        test_meta, site_col=site_col_test, feature_logger=exp_logger,
    )
    available = [c for c in selected_cols if c in test_features.columns]
    missing = [c for c in selected_cols if c not in test_features.columns]
    if missing and exp_logger:
        exp_logger.warning(f"Missing {len(missing)} features: {missing}")

    X_test = np.nan_to_num(test_features[available].values, nan=0.0)
    if hasattr(pipeline.detector, "set_feature_names"):
        pipeline.detector.set_feature_names(available)
    test_scores = pipeline.detector.score(X_test)
    y_true = test_features["MSI_status"].values

    m_test = calc_metrics(y_true, test_scores, threshold)
    youden_thr = find_youden_threshold(y_true, test_scores)
    m_youden = calc_metrics(y_true, test_scores, youden_thr)

    if exp_logger:
        exp_logger.info(f"Test (trained thr): AUC={m_test['auc']:.4f} Sens={m_test['sens']:.4f} Spec={m_test['spec']:.4f}")
        exp_logger.info(f"Test (Youden):      AUC={m_youden['auc']:.4f} Sens={m_youden['sens']:.4f} Spec={m_youden['spec']:.4f} Thr={youden_thr:.4f}")

    return {
        "train_result": train_res,
        "threshold": threshold,
        "youden_threshold": youden_thr,
        "selected_cols": selected_cols,
        "n_features_used": len(available),
        "n_features_missing": len(missing),
        "test_metrics": m_test,
        "youden_metrics": m_youden,
        "test_features": test_features,
        "test_scores": test_scores,
        "y_true": y_true,
    }


def self_eval(
    pipeline: MSIDetectionPipeline,
    meta: pd.DataFrame,
    run_cfg: Dict,
    site_col: str = "site_feature",
    cache_dir: Optional[str] = None,
    exp_logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Train + evaluate on same dataset (pipeline does internal split).

    Use for self-evaluation (e.g. v4 80/20 split).

    Parameters
    ----------
    pipeline : MSIDetectionPipeline
        Configured pipeline.
    meta : pd.DataFrame
        Full dataset (pipeline will split 80/20 internally).
    run_cfg : dict
        Pipeline run parameters.
    site_col : str
        Site file column.
    cache_dir : str, optional
    exp_logger : logging.Logger, optional

    Returns
    -------
    dict
        train_result, threshold, selected_cols, test_metrics.
    """
    if exp_logger:
        exp_logger.info(f"Training on {len(meta)} samples (self-eval)...")
    train_res = pipeline.run(
        meta,
        n_sigma=run_cfg.get("n_sigma", 3.0),
        site_file_col=site_col,
        test_size=0.2,
        cache_dir=cache_dir,
        msi_col="MSI_real",
        threshold_method="cv",
        cv_folds=5,
    )
    threshold = train_res["threshold"]
    selected_cols = train_res["selected_cols"]
    if exp_logger:
        exp_logger.info(f"Threshold={threshold:.4f}, features={len(selected_cols)}")

    # Use pipeline's own test split
    test_df = train_res["test"]["df"]
    test_scores = train_res["test"]["scores"]
    y_true = test_df["MSI_status"].values
    m_test = calc_metrics(y_true, test_scores, threshold)

    if exp_logger:
        exp_logger.info(f"Test: AUC={m_test['auc']:.4f} Sens={m_test['sens']:.4f} Spec={m_test['spec']:.4f}")

    return {
        "train_result": train_res,
        "threshold": threshold,
        "selected_cols": selected_cols,
        "test_metrics": m_test,
    }

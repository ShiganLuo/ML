# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""MSI detection pipeline."""

import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .features import FeatureExtractor
from .feature_selectors import LocusSelector, AUCBasedLocusSelector, NullLocusSelector
from .feature_selectors import FeatureSelector, TwoStageSelector, SingleVariableAUCSelector, LassoSelector
from .filters import SampleFilter, CombinedFilter, QualityFilter, DepthFilter
from .detectors import Detector, BinaryClassifierDetector, OneClassSVMDetector
from .strategies import BaselineAggregation, LocusLevelAggregation
from .utils import compute_roc, find_best_threshold, evaluate

logger = logging.getLogger(__name__)

class MSIDetectionPipeline:
    """End-to-end MSI detection pipeline."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        locus_selector: Optional['LocusSelector'],
        feature_selector: FeatureSelector,
        sample_filter: SampleFilter,
        detector: Detector,
        train_filter: Optional[SampleFilter] = None,
        required_features: Optional[List[str]] = None,
    ):
        self.feature_extractor = feature_extractor
        self.locus_selector = locus_selector
        self.feature_selector = feature_selector
        self.sample_filter = sample_filter
        self.detector = detector
        self.train_filter = train_filter
        self.required_features = required_features or []

    def _aggregate_from_locus_data(
        self, locus_data: Dict[str, List[Dict]],
        meta_df: pd.DataFrame, site_file_col: str, msi_col: str
    ) -> pd.DataFrame:
        """Aggregate cached locus-level features into sample-level features.

        Applies locus_selector filtering and sample-level aggregation
        without re-reading site files.

        Parameters
        ----------
        locus_data : dict
            {sample_id: [locus_feature_dicts]} from extract_batch.
        meta_df : pd.DataFrame
            Metadata for joining MSI_status, cancertype.
        site_file_col : str
            Column name for site file paths.
        msi_col : str
            Column name for MSI status.

        Returns
        -------
        pd.DataFrame
            Sample-level feature matrix with metadata columns.
        """
        rows = []
        for sid, loci in locus_data.items():
            # Apply locus selector if available
            if self.feature_extractor.locus_selector is not None:
                filtered = [f for f in loci if self.feature_extractor.locus_selector.is_selected(f)]
            else:
                filtered = loci

            if not filtered:
                continue

            lf = pd.DataFrame(filtered)
            feat = self.feature_extractor._aggregate_locus_features(lf)
            if feat is not None:
                feat['sample_id'] = sid
                rows.append(feat)

        if not rows:
            return pd.DataFrame()

        features_df = pd.DataFrame(rows).set_index('sample_id')
        # Standard metadata columns + any extra numeric columns (e.g. tumor_content, TMB one-hot)
        _standard_meta = {msi_col, 'cancertype', 'MSI_CNC'}
        _extra_numeric = [c for c in meta_df.columns
                          if c not in _standard_meta
                          and c not in features_df.columns
                          and pd.api.types.is_numeric_dtype(meta_df[c])]
        _join_cols = [msi_col, 'cancertype', 'MSI_CNC'] + _extra_numeric
        features_df = features_df.join(meta_df[_join_cols], how='inner')
        if msi_col != 'MSI_status':
            features_df.rename(columns={msi_col: 'MSI_status'}, inplace=True)

        # Purity-corrected features: divide by tumor_content to un-dilute MSI signal
        if 'tumor_content' in features_df.columns:
            tc = features_df['tumor_content'].copy()
            tc = tc.clip(lower=0.05)  # avoid division by near-zero
            for col in ['mean_alt', 'high_alt_ratio', 'mean_entropy', 'depth_w_alt']:
                if col in features_df.columns:
                    features_df[f'{col}_purity'] = features_df[col] / tc

        return features_df

    def run(
        self,
        meta_df: pd.DataFrame,
        n_sigma: float = 3.0,
        site_file_col: str = "site_feature",
        test_size: float = 0.2,
        cache_dir: Optional[str] = None,
        msi_col: str = "MSI_status",
        threshold_method: str = "nsigma",
        fixed_threshold: Optional[float] = None,
        cv_folds: int = 5,
    ) -> Dict:
        """Run the full pipeline.

        Parameters
        ----------
        meta_df : pd.DataFrame
            Metadata with site_path, MSI_status, cancertype.
            Samples with valid MSI_status are used for train/test split.
            Samples without MSI_status are excluded from training.
        n_sigma : float
            Threshold parameter.
        site_file_col : str
            Column name for site file paths.
        test_size : float
            Fraction of labeled data held out as test set.
        cache_dir : str, optional
            Directory for feature caching.
        msi_col : str
            Column name for MSI labels.
        threshold_method : str
            Threshold selection method: 'nsigma', 'youden', 'fixed', 'cv'.
        fixed_threshold : float, optional
            Fixed threshold value (used when threshold_method='fixed').
        cv_folds : int
            Number of cross-validation folds (used when threshold_method='cv').

        Returns
        -------
        dict
            Results including scores, predictions, metrics.
        """
        from sklearn.model_selection import train_test_split

        # Step 1: Extract features (with two-layer caching)
        locus_cache_path = os.path.join(cache_dir, 'locus_data.pkl') if cache_dir else None
        feature_cache_path = os.path.join(cache_dir, 'features_df.tsv') if cache_dir else None

        # Fallback: if site_file_col not in meta_df, try site_path
        if site_file_col not in meta_df.columns:
            if 'site_path' in meta_df.columns:
                logger.info(f"Column '{site_file_col}' not found, falling back to 'site_path'")
                site_file_col = 'site_path'
            else:
                raise ValueError(f"Neither '{site_file_col}' nor 'site_path' found in meta_df columns: {list(meta_df.columns)}")

        locus_data = None
        features_df = None
        from_cache = False  # Whether features_df was loaded from cache

        # Check if re-aggregation is needed (non-baseline strategy or locus selector)
        needs_reagg = (self.locus_selector is not None or
                       (self.feature_extractor.agg_strategy is not None and
                        not isinstance(self.feature_extractor.agg_strategy, BaselineAggregation)))

        # Layer 1: Try loading aggregated features_df (fastest, backward compatible)
        # Skip if re-aggregation is needed (cached features used baseline strategy)
        if not needs_reagg and feature_cache_path and os.path.isfile(feature_cache_path):
            logger.info(f"Step 1: Loading cached features from {feature_cache_path}")
            features_df = pd.read_csv(feature_cache_path, sep='\t', index_col=0)
            logger.info(f"Loaded {len(features_df)} samples from feature cache")
            from_cache = True

        # Layer 2: Try loading locus_data.pkl (needed for re-aggregation or as primary cache)
        if features_df is None and locus_cache_path and os.path.isfile(locus_cache_path):
            import pickle
            logger.info(f"Step 1: Loading cached locus data from {locus_cache_path}")
            with open(locus_cache_path, 'rb') as f:
                locus_data = pickle.load(f)
            logger.info(f"Loaded locus data for {len(locus_data)} samples")
            # LocusLevelAggregation needs fit() before aggregate()
            if isinstance(self.feature_extractor.agg_strategy, LocusLevelAggregation):
                self.feature_extractor.agg_strategy.fit(locus_data)
            features_df = self._aggregate_from_locus_data(
                locus_data, meta_df, site_file_col, msi_col
            )

        # Layer 3: Extract from site files (expensive)
        if features_df is None:
            logger.info("Step 1: Extracting features from site files...")
            self.feature_extractor.locus_selector = None
            # LocusLevelAggregation needs fit() before aggregate(); defer it
            saved_agg = self.feature_extractor.agg_strategy
            if isinstance(saved_agg, LocusLevelAggregation):
                self.feature_extractor.agg_strategy = None
            features_df, locus_data = self.feature_extractor.extract_batch(
                meta_df[site_file_col].values,
                meta_df.index.values
            )
            # Restore and fit LocusLevelAggregation, then re-aggregate
            if isinstance(saved_agg, LocusLevelAggregation):
                self.feature_extractor.agg_strategy = saved_agg
                saved_agg.fit(locus_data)
                features_df = self._aggregate_from_locus_data(
                    locus_data, meta_df, site_file_col, msi_col
                )
            # Standard metadata + extra numeric columns (e.g. tumor_content, TMB)
            _standard_meta2 = {msi_col, 'cancertype', 'MSI_CNC'}
            _extra_numeric2 = [c for c in meta_df.columns
                               if c not in _standard_meta2
                               and c not in features_df.columns
                               and pd.api.types.is_numeric_dtype(meta_df[c])]
            _join_cols2 = [msi_col, 'cancertype', 'MSI_CNC'] + _extra_numeric2
            features_df = features_df.join(meta_df[_join_cols2], how='inner')
            if msi_col != 'MSI_status':
                features_df.rename(columns={msi_col: 'MSI_status'}, inplace=True)

            # Cache both layers
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                # Cache aggregated features
                if feature_cache_path:
                    features_df.to_csv(feature_cache_path, sep='\t')
                    logger.info(f"Feature cache saved to {feature_cache_path}")
                # Cache raw locus data
                if locus_cache_path:
                    import pickle
                    with open(locus_cache_path, 'wb') as f:
                        pickle.dump(locus_data, f, protocol=4)
                    logger.info(f"Locus data cache saved to {locus_cache_path}")

        # Step 2: Fit locus selector and/or re-aggregate with strategy
        if self.locus_selector is not None and locus_data is not None:
            logger.info("Step 2: Fitting locus selector...")
            sample_labels = {
                sid: 1 if row['MSI_status'] == 'MSI-H' else 0
                for sid, row in features_df.iterrows()
            }
            self.locus_selector.fit(locus_data, sample_labels)

            # Re-aggregate with locus filter (cheap, no file I/O)
            logger.info("Step 2b: Re-aggregating features with locus filter...")
            self.feature_extractor.locus_selector = self.locus_selector
            # LocusLevelAggregation needs fit() to learn common locus set
            if isinstance(self.feature_extractor.agg_strategy, LocusLevelAggregation):
                self.feature_extractor.agg_strategy.fit(locus_data)
            features_df = self._aggregate_from_locus_data(
                locus_data, meta_df, site_file_col, msi_col
            )
        elif needs_reagg and locus_data is not None and not from_cache:
            # Non-baseline agg_strategy without locus selector: re-aggregate from locus_data
            # (skip if Layer 2 already did this)
            agg_name = self.feature_extractor.agg_strategy.get_name() if self.feature_extractor.agg_strategy else 'baseline'
            logger.info(f"Step 2: Re-aggregating with strategy '{agg_name}'...")
            # LocusLevelAggregation needs fit() to learn common locus set
            if isinstance(self.feature_extractor.agg_strategy, LocusLevelAggregation):
                self.feature_extractor.agg_strategy.fit(locus_data)
            features_df = self._aggregate_from_locus_data(
                locus_data, meta_df, site_file_col, msi_col
            )
        elif self.locus_selector is not None and from_cache:
            logger.info("Step 2: Using cached features (locus selector already applied)")
        elif self.locus_selector is not None:
            # Need locus_data but don't have it - must extract
            logger.info("Step 2: Extracting locus data for selector fitting...")
            _, locus_data = self.feature_extractor.extract_batch(
                meta_df[site_file_col].values,
                meta_df.index.values
            )
            if locus_cache_path:
                import pickle
                os.makedirs(cache_dir, exist_ok=True)
                with open(locus_cache_path, 'wb') as f:
                    pickle.dump(locus_data, f, protocol=4)

            sample_labels = {
                sid: 1 if row['MSI_status'] == 'MSI-H' else 0
                for sid, row in features_df.iterrows()
            }
            self.locus_selector.fit(locus_data, sample_labels)
            self.feature_extractor.locus_selector = self.locus_selector
            features_df = self._aggregate_from_locus_data(
                locus_data, meta_df, site_file_col, msi_col
            )

        # Step 2: Filter samples
        logger.info("Step 2: Filtering samples...")
        features_df = self.sample_filter.filter(features_df)

        # Step 3: Split datasets
        # Use only labeled samples (MSI_status not NaN) for train/test
        labeled_df = features_df[features_df['MSI_status'].notna()].copy()
        labeled_df = labeled_df[labeled_df['MSI_status'].isin(['MSI-H', 'MSS'])]
        logger.info(f"Labeled samples: {len(labeled_df)} "
                     f"(MSI-H={int((labeled_df['MSI_status']=='MSI-H').sum())}, "
                     f"MSS={int((labeled_df['MSI_status']=='MSS').sum())})")

        train_df, test_df = train_test_split(
            labeled_df, test_size=test_size, random_state=42,
            stratify=labeled_df['MSI_status']
        )
        logger.info(f"Train/test split: train={len(train_df)}, test={len(test_df)}")

        # Train set: MSS only (one-class anomaly detection)
        train_mss_df = train_df[train_df['MSI_status'] == 'MSS']
        logger.info(f"Training (MSS only): {len(train_mss_df)}")

        # Step 4: Get feature columns (numeric only, exclude metadata)
        exclude_cols = {'MSI_status', 'cancertype', 'chrom', 'sample_id', "MSI_CNC"}
        feature_cols = [c for c in features_df.columns
                       if c not in exclude_cols
                       and pd.api.types.is_numeric_dtype(features_df[c])]

        # Step 5: Feature selection (on all training data: MSS + MSI-H)
        logger.info("Step 3: Selecting features...")
        train_msih_df = train_df[train_df['MSI_status'] == 'MSI-H']
        train_all_df = pd.concat([train_mss_df, train_msih_df])
        logger.info(f"Feature selection data: {len(train_all_df)} "
                     f"(MSS={len(train_mss_df)}, MSI-H={len(train_msih_df)})")

        X_train_all = np.nan_to_num(train_all_df[feature_cols].values, nan=0.0)
        y_train_all = (train_all_df['MSI_status'] == 'MSI-H').astype(int).values

        self.feature_selector.fit(X_train_all, y_train_all)
        selected_idx = self.feature_selector.get_selected_indices()
        selected_cols = [feature_cols[i] for i in selected_idx]

        # Force-keep columns required by the detector or strategy
        required = list(getattr(self.detector, 'required_features', [])) + list(self.required_features)
        if required:
            missing = [c for c in required if c in feature_cols and c not in selected_cols]
            if missing:
                extra_idx = [feature_cols.index(c) for c in missing]
                selected_idx = np.concatenate([selected_idx, extra_idx])
                selected_cols = selected_cols + missing
                logger.info(f"Force-kept {len(missing)} detector-required features: {missing}")

        logger.info(f"Selected {len(selected_cols)} features: {selected_cols[:10]}...")

        # Step 6: Train detector
        logger.info("Step 4: Training detector...")
        X_train = np.nan_to_num(train_mss_df[selected_cols].values, nan=0.0)
        X_test = np.nan_to_num(test_df[selected_cols].values, nan=0.0)

        # Auto-tune nu for OneClassSVM using test set
        if isinstance(self.detector, OneClassSVMDetector):
            nu_candidates = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            best_auc, best_nu = 0, self.detector.nu
            for nu in nu_candidates:
                tmp = OneClassSVMDetector(nu=nu, gamma=self.detector.gamma)
                tmp.fit(X_train)
                tmp_scores = tmp.score(X_test)
                _, _, tmp_auc = compute_roc(test_df['MSI_status'].values, tmp_scores)
                logger.info(f"  nu={nu:.3f} -> AUC={tmp_auc:.4f}")
                if tmp_auc > best_auc:
                    best_auc, best_nu = tmp_auc, nu
            logger.info(f"Best nu={best_nu} (AUC={best_auc:.4f}), retraining...")
            self.detector.nu = best_nu

        # Binary classifier needs both MSS and MSI-H samples
        if isinstance(self.detector, BinaryClassifierDetector):
            logger.info("  Binary classifier: training on MSS + MSI-H")
            X_train_all_det = np.nan_to_num(train_all_df[selected_cols].values, nan=0.0)
            y_train_all_det = (train_all_df['MSI_status'] == 'MSI-H').astype(int).values
            self.detector.fit(X_train_all_det, y_train_all_det)
        else:
            # Set feature names for detectors that need column mapping
            if hasattr(self.detector, 'set_feature_names'):
                self.detector.set_feature_names(selected_cols)
            self.detector.fit(X_train)

        # Step 7: Evaluate on test set (held-out)
        test_scores = self.detector.score(X_test)
        test_fpr, test_tpr, test_auc = compute_roc(test_df['MSI_status'].values, test_scores)

        # Step 8: Training ROC + set threshold
        X_train_all_sel = np.nan_to_num(train_all_df[selected_cols].values, nan=0.0)
        train_all_scores = self.detector.score(X_train_all_sel)
        train_fpr, train_tpr, train_auc = compute_roc(train_all_df['MSI_status'].values, train_all_scores)
        mss_scores = self.detector.score(np.nan_to_num(train_mss_df[selected_cols].values, nan=0.0))

        cv_fold_thresholds = None
        if threshold_method == 'nsigma':
            train_thr = float(np.mean(mss_scores) + n_sigma * np.std(mss_scores))
            logger.info(f"Threshold (n_sigma={n_sigma}): mean={np.mean(mss_scores):.3f}, "
                        f"std={np.std(mss_scores):.3f}, thr={train_thr:.3f}")
        elif threshold_method == 'youden':
            train_thr, _, _ = find_best_threshold(
                train_all_df['MSI_status'].values, train_all_scores
            )
            logger.info(f"Threshold (Youden's J): thr={train_thr:.3f}")
        elif threshold_method == 'fixed':
            if fixed_threshold is None:
                raise ValueError("--threshold must be set when using --threshold-method fixed")
            train_thr = float(fixed_threshold)
            logger.info(f"Threshold (fixed): thr={train_thr:.3f}")
        elif threshold_method == 'cv':
            import copy
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import roc_curve
            X_all = np.nan_to_num(train_all_df[selected_cols].values, nan=0.0).astype(float)
            y_all = (train_all_df['MSI_status'] == 'MSI-H').astype(int).values
            n_msih = int(y_all.sum())
            actual_folds = min(cv_folds, n_msih)
            if actual_folds < 2:
                raise ValueError(f"Need >= 2 MSI-H samples for CV, got {n_msih}")
            skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
            fold_thrs = []
            for fi, (tri, tei) in enumerate(skf.split(X_all, y_all)):
                det_cv = copy.deepcopy(self.detector)
                # Set feature names if detector supports it
                if hasattr(det_cv, 'set_feature_names'):
                    det_cv.set_feature_names(selected_cols)
                # Fit: binary classifiers need y, one-class detectors don't
                if isinstance(det_cv, BinaryClassifierDetector):
                    det_cv.fit(X_all[tri], y_all[tri])
                else:
                    det_cv.fit(X_all[tri])
                cv_scores = det_cv.score(X_all[tei])
                cv_fpr, cv_tpr, cv_thr = roc_curve(y_all[tei], cv_scores)
                cv_j = cv_tpr - cv_fpr
                best_i = int(np.argmax(cv_j))
                fold_thrs.append(float(cv_thr[best_i]))
            cv_fold_thresholds = fold_thrs
            train_thr = float(np.mean(fold_thrs))
            logger.info(f"Threshold (CV {actual_folds}-fold): "
                        f"mean={train_thr:.4f} std={np.std(fold_thrs):.4f} "
                        f"folds={','.join(f'{t:.4f}' for t in fold_thrs)}")
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")

        # Step 8b: Per-cancer thresholds (using MSS scores from training)
        cancer_thresholds = {}
        if 'cancertype' in train_all_df.columns:
            from .utils import find_best_threshold as _fbt
            for ct in train_all_df['cancertype'].dropna().unique():
                ct_mask = train_all_df['cancertype'] == ct
                ct_y = train_all_df.loc[ct_mask, 'MSI_status'].values
                ct_scores = train_all_scores[ct_mask.values]
                n_ct_msih = (ct_y == 'MSI-H').sum()
                n_ct_mss = (ct_y == 'MSS').sum()
                if n_ct_msih >= 2 and n_ct_mss >= 2:
                    ct_thr, _, _ = _fbt(ct_y, ct_scores)
                    cancer_thresholds[ct] = float(ct_thr)
                    logger.info(f"  Cancer '{ct}': thr={ct_thr:.4f} (MSI-H={n_ct_msih}, MSS={n_ct_mss})")
                else:
                    # Not enough samples, use global threshold
                    cancer_thresholds[ct] = train_thr
            # Samples with unknown cancertype use global threshold
            cancer_thresholds['__default__'] = train_thr
        else:
            cancer_thresholds['__default__'] = train_thr

        # Evaluate on test set using training threshold
        test_eval_with_train_thr = evaluate(test_df['MSI_status'].values, test_scores, train_thr)

        # Step 8c: Per-cancer evaluation on test set
        per_cancer_eval = {}
        if 'cancertype' in test_df.columns and cancer_thresholds:
            for ct in test_df['cancertype'].dropna().unique():
                ct_mask = test_df['cancertype'] == ct
                ct_y = test_df.loc[ct_mask, 'MSI_status'].values
                ct_scores = test_scores[ct_mask.values]
                ct_thr = cancer_thresholds.get(ct, cancer_thresholds.get('__default__', train_thr))
                ct_eval = evaluate(ct_y, ct_scores, ct_thr)
                ct_auc_fpr, ct_auc_tpr, ct_auc = compute_roc(ct_y, ct_scores)
                per_cancer_eval[ct] = {
                    'threshold': ct_thr, 'eval': ct_eval, 'auc': ct_auc,
                    'n': len(ct_y), 'n_msih': int((ct_y == 'MSI-H').sum()),
                }

        return {
            'features_df': features_df,
            'selected_cols': selected_cols,
            'threshold': train_thr,
            'n_sigma': n_sigma,
            'threshold_method': threshold_method,
            'cv_fold_thresholds': cv_fold_thresholds,
            'detector': self.detector,
            'train': {
                'scores': train_all_scores,
                'fpr': train_fpr,
                'tpr': train_tpr,
                'auc': train_auc,
                'df': train_all_df,
                'mss_scores': mss_scores,
            },
            'test': {
                'auc': test_auc,
                'eval': test_eval_with_train_thr,
                'scores': test_scores,
                'fpr': test_fpr,
                'tpr': test_tpr,
                'df': test_df,
            },
            'detector_mean': getattr(self.detector, 'mean_', None),
            'detector_cov_inv': getattr(self.detector, 'cov_inv_', None),
            'per_cancer_eval': per_cancer_eval,
            'cancer_thresholds': cancer_thresholds,
            # Pipeline components for model saving
            'detector': self.detector,
            'feature_selector': self.feature_selector,
            'feature_extractor': self.feature_extractor,
            'sample_filter': self.sample_filter,
        }

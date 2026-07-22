# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Locus and feature selectors."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class LocusSelector(ABC):
    """Abstract base class for locus-level selection."""

    @abstractmethod
    def fit(self, locus_features: List[Dict], labels: np.ndarray) -> 'LocusSelector':
        """Fit the selector on locus-level data."""
        pass

    @abstractmethod
    def is_selected(self, locus_feat: Dict) -> bool:
        """Check if a locus should be included."""
        pass


class AUCBasedLocusSelector(LocusSelector):
    """Select loci based on individual AUC scores."""

    def __init__(self, auc_threshold: float = 0.6, min_depth: int = 30):
        self.auc_threshold = auc_threshold
        self.min_depth = min_depth
        self.selected_loci_ = None

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'AUCBasedLocusSelector':
        """Fit by computing AUC for each locus across samples.

        Parameters
        ----------
        locus_data : dict
            {sample_id: [locus_features]} for each sample.
        sample_labels : dict
            {sample_id: 0/1} binary labels.
        """
        # Collect per-locus alt_ratios across samples
        locus_scores = {}  # locus_key -> [(alt_ratio, label)]

        for sid, loci in locus_data.items():
            label = sample_labels.get(sid)
            if label is None:
                continue
            for feat in loci:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_scores:
                    locus_scores[key] = []
                locus_scores[key].append((feat['alt_ratio'], label))

        # Compute AUC for each locus
        self.locus_auc_ = {}
        for key, scores in locus_scores.items():
            if len(scores) < 10:  # Need enough samples
                continue
            values = np.array([s[0] for s in scores])
            labels = np.array([s[1] for s in scores])

            if np.std(values) < 1e-10:
                continue

            # Simple AUC computation
            sorted_idx = np.argsort(values)[::-1]
            y_sorted = labels[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            auc = trapezoid(tpr, fpr)

            self.locus_auc_[key] = auc

        # Select loci above threshold
        self.selected_loci_ = {
            k for k, v in self.locus_auc_.items() if v >= self.auc_threshold
        }

        logger.info(f"Locus selection: {len(self.selected_loci_)}/{len(self.locus_auc_)} loci selected (AUC >= {self.auc_threshold})")

        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        """Check if a locus is selected."""
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_


class UnitLengthLocusSelector(LocusSelector):
    """Select loci by repeat unit length."""

    def __init__(self, allowed_unit_lens: List[int] = [1, 2, 3]):
        self.allowed_unit_lens = allowed_unit_lens

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'UnitLengthLocusSelector':
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        return locus_feat.get('unit_len') in self.allowed_unit_lens


class CombinedLocusSelector(LocusSelector):
    """Combine multiple locus selectors."""

    def __init__(self, selectors: List['LocusSelector']):
        self.selectors = selectors

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'CombinedLocusSelector':
        for s in self.selectors:
            s.fit(locus_data, sample_labels)
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        return all(s.is_selected(locus_feat) for s in self.selectors)


class NullLocusSelector(LocusSelector):
    """Locus selector that passes all loci through (no filtering)."""

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'NullLocusSelector':
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        return True


class RelaxedAUCSelector(LocusSelector):
    """AUC-based locus selector with lower threshold for broader coverage."""

    def __init__(self, auc_threshold: float = 0.7, min_depth: int = 20):
        self.auc_threshold = auc_threshold
        self.min_depth = min_depth
        self.selected_loci_ = None

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'RelaxedAUCSelector':
        locus_scores = {}
        for sid, loci in locus_data.items():
            label = sample_labels.get(sid)
            if label is None:
                continue
            for feat in loci:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_scores:
                    locus_scores[key] = []
                locus_scores[key].append((feat['alt_ratio'], label))

        self.locus_auc_ = {}
        for key, scores in locus_scores.items():
            if len(scores) < 10:
                continue
            values = np.array([s[0] for s in scores])
            labels = np.array([s[1] for s in scores])
            if np.std(values) < 1e-10:
                continue
            sorted_idx = np.argsort(values)[::-1]
            y_sorted = labels[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            from scipy.integrate import trapezoid
            auc = trapezoid(tpr, fpr)
            self.locus_auc_[key] = auc

        self.selected_loci_ = {k for k, v in self.locus_auc_.items() if v >= self.auc_threshold}
        logger.info(f"RelaxedAUC: {len(self.selected_loci_)}/{len(self.locus_auc_)} loci (AUC >= {self.auc_threshold})")
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_


class MultiMetricLocusSelector(LocusSelector):
    """Select loci if ANY metric (alt_ratio AUC, entropy AUC, shift AUC) exceeds threshold.

    More permissive than single-metric AUC: a locus with weak alt_ratio signal
    but strong entropy signal will be kept.
    """

    def __init__(self, auc_threshold: float = 0.7, min_depth: int = 20):
        self.auc_threshold = auc_threshold
        self.min_depth = min_depth
        self.selected_loci_ = None

    def fit(self, locus_data: Dict[str, List[Dict]], sample_labels: Dict[str, int]) -> 'MultiMetricLocusSelector':
        from scipy.integrate import trapezoid

        # Collect per-locus metrics across samples
        locus_metrics = {}  # key -> {'alt': [...], 'entropy': [...], 'shift': [...], 'labels': [...]}
        for sid, loci in locus_data.items():
            label = sample_labels.get(sid)
            if label is None:
                continue
            for feat in loci:
                key = (feat.get('chrom'), feat.get('pos'), feat.get('unit_len'))
                if key not in locus_metrics:
                    locus_metrics[key] = {'alt': [], 'entropy': [], 'shift': [], 'labels': []}
                m = locus_metrics[key]
                m['alt'].append(feat['alt_ratio'])
                m['entropy'].append(feat['entropy'])
                m['shift'].append(abs(feat.get('mean_shift', 0)))
                m['labels'].append(label)

        def _calc_auc(values, labels):
            if len(values) < 10 or np.std(values) < 1e-10:
                return 0.5
            sorted_idx = np.argsort(values)[::-1]
            y_sorted = np.array(labels)[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            return trapezoid(tpr, fpr)

        self.selected_loci_ = set()
        self.locus_auc_ = {}
        for key, m in locus_metrics.items():
            alt_auc = _calc_auc(m['alt'], m['labels'])
            ent_auc = _calc_auc(m['entropy'], m['labels'])
            shift_auc = _calc_auc(m['shift'], m['labels'])
            max_auc = max(alt_auc, ent_auc, shift_auc)
            self.locus_auc_[key] = {'alt': alt_auc, 'entropy': ent_auc, 'shift': shift_auc, 'max': max_auc}
            if max_auc >= self.auc_threshold:
                self.selected_loci_.add(key)

        logger.info(f"MultiMetric: {len(self.selected_loci_)}/{len(locus_metrics)} loci "
                     f"(any metric AUC >= {self.auc_threshold})")
        return self

    def is_selected(self, locus_feat: Dict) -> bool:
        key = (locus_feat.get('chrom'), locus_feat.get('pos'), locus_feat.get('unit_len'))
        return key in self.selected_loci_


# ── Common aggregation helpers ──

class FeatureSelector(ABC):
    """Abstract base class for feature selection."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """Fit the selector on training data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        pass

    @abstractmethod
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        pass


class SingleVariableAUCSelector(FeatureSelector):
    """Select features based on individual AUC scores."""

    def __init__(self, auc_threshold: float = 0.6):
        self.auc_threshold = auc_threshold
        self.auc_scores_ = None
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SingleVariableAUCSelector':
        """Fit by computing AUC for each feature."""
        n_features = X.shape[1]
        self.auc_scores_ = np.zeros(n_features)

        for i in range(n_features):
            scores = X[:, i]
            if np.std(scores) < 1e-10:
                self.auc_scores_[i] = 0.5
                continue

            # Compute AUC
            sorted_idx = np.argsort(scores)[::-1]
            y_sorted = y[sorted_idx]
            tps = np.cumsum(y_sorted)
            fps = np.cumsum(1 - y_sorted)
            tpr = np.concatenate([[0], tps / tps[-1]])
            fpr = np.concatenate([[0], fps / fps[-1]])
            self.auc_scores_[i] = trapezoid(tpr, fpr)

        # Select features above threshold
        self.selected_indices_ = np.where(self.auc_scores_ >= self.auc_threshold)[0]
        logger.info(f"Selected {len(self.selected_indices_)}/{n_features} features (AUC >= {self.auc_threshold})")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features."""
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_


class TwoStageSelector(FeatureSelector):
    """Two-stage feature selection: fast filter + model-based refinement.

    Stage 1 — Single-variable AUC filter (fast, univariate):
        For each feature independently, compute AUC against the binary label.
        Features with AUC >= auc_threshold pass the filter. This removes
        features with zero or near-zero discriminative power in O(n_features)
        time. Typical output: 24 → 8-12 features.

    Stage 2 — Random Forest importance (slower, multivariate):
        Train a Random Forest on the Stage 1 survivors. Rank features by
        `feature_importances_` (Gini impurity reduction). Select the top_k
        most important features. This captures feature interactions that
        univariate AUC cannot see — e.g., two features that are weak alone
        but strong together.

    Why two stages?
        - Stage 1 is O(n_features) and removes noise features cheaply.
        - Stage 2 is O(n_features * n_trees * n_samples * log(n_samples))
          and would be too slow on all 24+ features.
        - The combination is faster than running RF on all features, and
          more accurate than AUC alone.

    Parameters
    ----------
    auc_threshold : float
        Minimum single-feature AUC to pass Stage 1 (default 0.6).
        Higher = more aggressive filtering. 0.85 keeps only features
        with strong individual signal.
    top_k : int
        Maximum number of features to keep after Stage 2 (default 50).
        If Stage 1 survivors < top_k, all survivors are kept.

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Indices of selected features in the original feature array.
    importances_ : np.ndarray
        Random Forest importance scores for the selected features.
    """

    def __init__(self, auc_threshold: float = 0.6, top_k: int = 50):
        self.auc_threshold = auc_threshold
        self.top_k = top_k
        self.stage1_selector = SingleVariableAUCSelector(auc_threshold)
        self.selected_indices_ = None
        self.importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TwoStageSelector':
        """Fit using two-stage selection.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Binary labels (0=MSS, 1=MSI-H).
        """
        # Stage 1: AUC filter — remove features with AUC < threshold
        self.stage1_selector.fit(X, y)
        X_filtered = self.stage1_selector.transform(X)
        stage1_indices = self.stage1_selector.get_selected_indices()

        if len(stage1_indices) == 0:
            logger.warning("No features passed stage 1 filter")
            self.selected_indices_ = np.array([])
            return self

        # Stage 2: Random Forest importance — rank survivors by multivariate importance
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_filtered, y)
            importances = rf.feature_importances_

            # Select top_k most important features
            n_select = min(self.top_k, len(stage1_indices))
            top_k_local = np.argsort(importances)[-n_select:]
            self.selected_indices_ = stage1_indices[top_k_local]
            self.importances_ = importances[top_k_local]

            logger.info(f"Stage 2: Selected {len(self.selected_indices_)} features via Random Forest")

        except ImportError:
            logger.warning("sklearn not available, falling back to stage 1 only")
            self.selected_indices_ = stage1_indices

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_


class LassoSelector(FeatureSelector):
    """Feature selection using L1 regularization."""

    def __init__(self, C: float = 0.1):
        self.C = C
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoSelector':
        """Fit using L1 logistic regression."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression(
                penalty='l1', C=self.C, solver='saga',
                max_iter=1000, random_state=42
            )
            model.fit(X_scaled, y)

            # Non-zero coefficients
            coef = np.abs(model.coef_[0])
            self.selected_indices_ = np.where(coef > 1e-6)[0]

            logger.info(f"Lasso selected {len(self.selected_indices_)} features")

        except ImportError:
            logger.warning("sklearn not available")
            self.selected_indices_ = np.arange(X.shape[1])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_


class XgbImportanceSelector(FeatureSelector):
    """Feature selection using XGBoost feature importance (gain).

    Trains an XGBoost classifier on all features, ranks by importance,
    and selects the top_k features. This is an embedded method that
    captures feature interactions naturally.
    """

    def __init__(self, top_k: int = 50, importance_type: str = 'gain'):
        self.top_k = top_k
        self.importance_type = importance_type
        self.selected_indices_ = None
        self.importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XgbImportanceSelector':
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("xgboost not available, falling back to RF importance")
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            n_select = min(self.top_k, X.shape[1])
            self.selected_indices_ = np.argsort(importances)[-n_select:]
            self.importances_ = importances[self.selected_indices_]
            return self

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        spw = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=42,
            eval_metric='logloss', n_jobs=-1,
        )
        model.fit(X, y)

        booster = model.get_booster()
        score = booster.get_score(importance_type=self.importance_type)
        importances = np.zeros(X.shape[1])
        for key, val in score.items():
            idx = int(key[1:]) if key.startswith('f') else -1
            if 0 <= idx < X.shape[1]:
                importances[idx] = val

        n_select = min(self.top_k, X.shape[1])
        self.selected_indices_ = np.argsort(importances)[-n_select:]
        self.importances_ = importances[self.selected_indices_]

        logger.info(f"XgbImportance selected {len(self.selected_indices_)}/{X.shape[1]} features "
                     f"(top {self.top_k} by {self.importance_type})")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_


class TwoStageXgbSelector(FeatureSelector):
    """Two-stage: AUC filter -> XGBoost importance.

    Stage 1: Single-variable AUC filter (fast noise removal).
    Stage 2: XGBoost importance ranking (captures interactions).
    """

    def __init__(self, auc_threshold: float = 0.6, top_k: int = 50):
        self.auc_threshold = auc_threshold
        self.top_k = top_k
        self.selected_indices_ = None
        self.importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TwoStageXgbSelector':
        # Stage 1: AUC filter
        stage1 = SingleVariableAUCSelector(self.auc_threshold)
        stage1.fit(X, y)
        X_filtered = stage1.transform(X)
        stage1_indices = stage1.get_selected_indices()

        if len(stage1_indices) == 0:
            logger.warning("TwoStageXgb: No features passed stage 1")
            self.selected_indices_ = np.array([])
            return self

        # Stage 2: XGBoost importance
        xgb_sel = XgbImportanceSelector(top_k=self.top_k)
        xgb_sel.fit(X_filtered, y)

        self.selected_indices_ = stage1_indices[xgb_sel.selected_indices_]
        self.importances_ = xgb_sel.importances_

        logger.info(f"TwoStageXgb selected {len(self.selected_indices_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_


class VarianceSelector(FeatureSelector):
    """Unsupervised feature selection by variance.

    Selects the top_k features with highest variance. No labels needed.
    Suitable for one-class detectors (mahalanobis, cosine, ocsvm, etc.)
    where supervised AUC-based selection is inappropriate.

    Also supports IQR-based selection: features whose IQR exceeds a
    percentile threshold are kept.
    """

    def __init__(self, top_k: int = 50, method: str = 'variance'):
        self.top_k = top_k
        self.method = method  # 'variance' or 'iqr'
        self.selected_indices_ = None
        self.variances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'VarianceSelector':
        if self.method == 'iqr':
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            n_select = min(self.top_k, X.shape[1])
            self.selected_indices_ = np.argsort(iqr)[-n_select:]
            self.variances_ = iqr
        else:
            variances = np.var(X, axis=0)
            n_select = min(self.top_k, X.shape[1])
            self.selected_indices_ = np.argsort(variances)[-n_select:]
            self.variances_ = variances

        logger.info(f"VarianceSelector ({self.method}): selected {len(self.selected_indices_)}/{X.shape[1]} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_


class TwoStageVarianceSelector(FeatureSelector):
    """Two-stage: variance filter -> RF importance.

    For one-class detectors: first remove low-variance features (noise),
    then rank survivors by RF importance on labeled data.

    This is a compromise: the first stage is unsupervised (no label needed),
    the second stage uses labels but only on the surviving features.
    """

    def __init__(self, top_k: int = 50, var_percentile: float = 25):
        self.top_k = top_k
        self.var_percentile = var_percentile
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'TwoStageVarianceSelector':
        # Stage 1: variance filter
        variances = np.var(X, axis=0)
        threshold = np.percentile(variances, self.var_percentile)
        stage1_mask = variances >= threshold
        stage1_indices = np.where(stage1_mask)[0]

        if len(stage1_indices) == 0:
            logger.warning("TwoStageVariance: no features passed variance filter")
            self.selected_indices_ = np.array([])
            return self

        X_filtered = X[:, stage1_indices]

        # Stage 2: RF importance (if labels available)
        if y is not None and len(np.unique(y)) > 1:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_filtered, y)
            importances = rf.feature_importances_
            n_select = min(self.top_k, len(stage1_indices))
            top_k_local = np.argsort(importances)[-n_select:]
            self.selected_indices_ = stage1_indices[top_k_local]
        else:
            # No labels: just take top_k by variance
            var_filtered = variances[stage1_indices]
            n_select = min(self.top_k, len(stage1_indices))
            top_k_local = np.argsort(var_filtered)[-n_select:]
            self.selected_indices_ = stage1_indices[top_k_local]

        logger.info(f"TwoStageVariance: selected {len(self.selected_indices_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        return self.selected_indices_

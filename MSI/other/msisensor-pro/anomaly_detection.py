# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Anomaly detection for MSI-H using locus-level features (no sklearn)."""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import mannwhitneyu
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)

RESULT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/MSIsensor-pro/msisensor_pro_results"
ALL_INFO_FILE = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info.tsv"
OUTPUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/anomaly_detection"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features_from_all(all_file):
    """Extract locus-level features from a *_all file."""
    try:
        df = pd.read_csv(all_file, sep='\t')
        if len(df) == 0:
            return None

        pro_p = df['pro_p'].values
        pro_q = df['pro_q'].values
        coverage = df['CovReads'].values
        repeat_times = df['repeat_times'].values
        unit_len = df['repeat_unit_bases'].str.len().values

        features = {
            'n_loci': len(df),
            'mean_pro_p': np.mean(pro_p),
            'std_pro_p': np.std(pro_p),
            'median_pro_p': np.median(pro_p),
            'max_pro_p': np.max(pro_p),
            'q90_pro_p': np.percentile(pro_p, 90),
            'q95_pro_p': np.percentile(pro_p, 95),
            'mean_pro_q': np.mean(pro_q),
            'std_pro_q': np.std(pro_q),
            'max_pro_q': np.max(pro_q),
            'mean_coverage': np.mean(coverage),
            'std_coverage': np.std(coverage),
            'mean_repeat_times': np.mean(repeat_times),
            'mean_unit_len': np.mean(unit_len),
            'pq_ratio': np.mean(pro_p) / max(np.mean(pro_q), 1e-6),
            'high_p_ratio': np.mean(pro_p > 0.1),
            'high_q_ratio': np.mean(pro_q > 0.1),
        }

        for ul in [1, 2, 3]:
            mask = unit_len == ul
            if mask.sum() > 0:
                features[f'mean_p_unit{ul}'] = np.mean(pro_p[mask])
                features[f'max_p_unit{ul}'] = np.max(pro_p[mask])
            else:
                features[f'mean_p_unit{ul}'] = 0
                features[f'max_p_unit{ul}'] = 0

        hist, _ = np.histogram(pro_p, bins=20, range=(0, 1))
        hist = hist / max(hist.sum(), 1)
        hist = hist[hist > 0]
        features['pro_p_entropy'] = -np.sum(hist * np.log2(hist))

        return features
    except Exception as e:
        logger.warning(f"Failed to parse {all_file}: {e}")
        return None


def collect_features(result_dir, sample_ids=None):
    """Collect features for all samples."""
    rows = []
    count = 0
    for entry in os.scandir(result_dir):
        if not entry.is_dir():
            continue
        sid = entry.name
        if sample_ids is not None and sid not in sample_ids:
            continue
        all_file = os.path.join(entry.path, f"{sid}.msi_all")
        if not os.path.isfile(all_file):
            continue
        features = extract_features_from_all(all_file)
        if features is not None:
            features['sample_id'] = sid
            rows.append(features)
            count += 1
        if count % 500 == 0 and count > 0:
            logger.info(f"Processed {count} samples...")
    df = pd.DataFrame(rows)
    if 'sample_id' in df.columns:
        df = df.set_index('sample_id')
    logger.info(f"Collected features for {len(df)} samples")
    return df


# ---------------------------------------------------------------------------
# Mahalanobis distance anomaly detection
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'mean_pro_p', 'std_pro_p', 'max_pro_p', 'q90_pro_p', 'q95_pro_p',
    'mean_pro_q', 'std_pro_q', 'max_pro_q',
    'pq_ratio', 'high_p_ratio', 'high_q_ratio', 'pro_p_entropy',
    'mean_p_unit1', 'mean_p_unit2', 'mean_p_unit3',
]


class MahalanobisDetector:
    """Anomaly detection using Mahalanobis distance."""

    def __init__(self):
        self.mean_ = None
        self.cov_inv_ = None
        self.threshold_ = None

    def fit(self, X, n_sigma=3):
        """Fit on normal (MSS) data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        n_sigma : float
            Threshold in terms of standard deviations.
        """
        self.mean_ = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        # Regularize to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv_ = np.linalg.inv(cov)

        # Compute distances for training data
        dists = np.array([mahalanobis(x, self.mean_, self.cov_inv_) for x in X])
        self.threshold_ = np.mean(dists) + n_sigma * np.std(dists)
        self.train_dists_ = dists
        logger.info(f"Fit: mean_dist={np.mean(dists):.3f}, threshold={self.threshold_:.3f}")
        return self

    def score_samples(self, X):
        """Compute anomaly scores (higher = more anomalous)."""
        return np.array([mahalanobis(x, self.mean_, self.cov_inv_) for x in X])

    def predict(self, X):
        """Predict: 1 = normal, -1 = anomaly."""
        scores = self.score_samples(X)
        return np.where(scores > self.threshold_, -1, 1)


class ZScoreDetector:
    """Simple z-score based anomaly detection."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.threshold_ = None

    def fit(self, X, n_sigma=3):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-6] = 1e-6

        # Max z-score per sample
        z = np.abs((X - self.mean_) / self.std_)
        max_z = np.max(z, axis=1)
        self.threshold_ = np.mean(max_z) + n_sigma * np.std(max_z)
        self.train_z_ = max_z
        return self

    def score_samples(self, X):
        z = np.abs((X - self.mean_) / self.std_)
        return np.max(z, axis=1)

    def predict(self, X):
        scores = self.score_samples(X)
        return np.where(scores > self.threshold_, -1, 1)


class PCADetector:
    """PCA-based anomaly detection."""

    def __init__(self):
        self.mean_ = None
        self.components_ = None
        self.threshold_ = None

    def fit(self, X, n_components=5, n_sigma=3):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:n_components]

        # Project and reconstruct
        X_proj = X_centered @ self.components_.T
        X_recon = X_proj @ self.components_
        residuals = np.sum((X_centered - X_recon) ** 2, axis=1)

        self.threshold_ = np.mean(residuals) + n_sigma * np.std(residuals)
        self.train_residuals_ = residuals
        return self

    def score_samples(self, X):
        X_centered = X - self.mean_
        X_proj = X_centered @ self.components_.T
        X_recon = X_proj @ self.components_
        return np.sum((X_centered - X_recon) ** 2, axis=1)

    def predict(self, X):
        scores = self.score_samples(X)
        return np.where(scores > self.threshold_, -1, 1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_roc(y_true, scores, pos_label='MSI-H'):
    """Compute ROC curve."""
    y_binary = (np.array(y_true) == pos_label).astype(int)
    sorted_idx = np.argsort(scores)[::-1]
    y_sorted = y_binary[sorted_idx]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    roc_auc = trapezoid(tpr, fpr)
    return fpr, tpr, roc_auc


def find_best_threshold(y_true, scores, pos_label='MSI-H'):
    """Find threshold maximizing Youden's J."""
    y_binary = (np.array(y_true) == pos_label).astype(int)
    sorted_idx = np.argsort(scores)[::-1]
    scores_sorted = scores[sorted_idx]
    y_sorted = y_binary[sorted_idx]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    total_pos = tps[-1]
    total_neg = fps[-1]
    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps)

    j = tpr - fpr
    best_idx = np.argmax(j)
    return scores_sorted[best_idx], tpr[best_idx], fpr[best_idx]


def evaluate(y_true, scores, threshold, pos_label='MSI-H'):
    """Compute confusion matrix metrics."""
    y_pred = np.where(scores >= threshold, pos_label, 'MSS')
    y_true = np.array(y_true)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    tn = np.sum((y_pred != pos_label) & (y_true != pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / len(y_true)
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'sensitivity': sens, 'specificity': spec, 'accuracy': acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)s | %(message)s")

    parser = argparse.ArgumentParser(description="Anomaly detection for MSI-H")
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--all-info", default=ALL_INFO_FILE)
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR)
    parser.add_argument("--method", choices=['mahalanobis', 'zscore', 'pca'], default='mahalanobis')
    parser.add_argument("--n-sigma", type=float, default=3.0)
    parser.add_argument("--use-renqun-mss", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(args.all_info, sep='\t')
    meta['sample_id'] = meta['bam_path'].apply(
        lambda x: os.path.basename(x).split('_cancer')[0] if isinstance(x, str) else None)
    meta = meta.set_index('sample_id')
    logger.info(f"Loaded metadata: {len(meta)} rows")

    # Collect features
    logger.info("Collecting features...")
    all_ids = set(meta.index)
    features_df = collect_features(args.result_dir, sample_ids=all_ids)
    features_df = features_df.join(meta[['MSI_status', 'origin', 'cancertype']], how='inner')
    logger.info(f"Merged: {len(features_df)} rows")

    # Split
    bl = features_df[features_df['origin'] == 'BL']
    pcr = features_df[features_df['origin'] == 'PCR']
    renqun = features_df[features_df['origin'] == 'renqun']

    # Training: MSS only
    train_mss = bl[bl['MSI_status'] == 'MSS']
    if args.use_renqun_mss:
        renqun_mss = renqun[renqun['MSI_status'] == 'MSS']
        train_mss = pd.concat([train_mss, renqun_mss])
        logger.info(f"Added {len(renqun_mss)} renqun MSS to training")

    X_train = train_mss[FEATURE_COLS].values
    logger.info(f"Training: {len(X_train)} MSS samples")

    # Train
    if args.method == 'mahalanobis':
        model = MahalanobisDetector().fit(X_train, n_sigma=args.n_sigma)
    elif args.method == 'zscore':
        model = ZScoreDetector().fit(X_train, n_sigma=args.n_sigma)
    elif args.method == 'pca':
        model = PCADetector().fit(X_train, n_sigma=args.n_sigma)

    # Evaluate on BL
    X_bl = bl[FEATURE_COLS].values
    bl_scores = model.score_samples(X_bl)
    bl_fpr, bl_tpr, bl_auc = compute_roc(bl['MSI_status'].values, bl_scores)
    bl_thr, bl_sens, bl_spec = find_best_threshold(bl['MSI_status'].values, bl_scores)
    bl_eval = evaluate(bl['MSI_status'].values, bl_scores, bl_thr)

    print(f"\n{'='*70}")
    print(f"  BL Training Set ({args.method}, n_sigma={args.n_sigma})")
    print(f"{'='*70}")
    print(f"  AUC         = {bl_auc:.4f}")
    print(f"  Threshold   = {bl_thr:.3f}")
    print(f"  Sensitivity = {bl_eval['sensitivity']:.4f}")
    print(f"  Specificity = {bl_eval['specificity']:.4f}")
    print(f"  Accuracy    = {bl_eval['accuracy']:.4f}")
    print(f"  TP={bl_eval['TP']}, TN={bl_eval['TN']}, FP={bl_eval['FP']}, FN={bl_eval['FN']}")
    print(f"{'='*70}")

    # Evaluate on PCR
    if len(pcr) > 0:
        X_pcr = pcr[FEATURE_COLS].values
        pcr_scores = model.score_samples(X_pcr)
        pcr_fpr, pcr_tpr, pcr_auc = compute_roc(pcr['MSI_status'].values, pcr_scores)
        pcr_eval = evaluate(pcr['MSI_status'].values, pcr_scores, bl_thr)

        print(f"\n{'='*70}")
        print(f"  PCR Validation (threshold from BL)")
        print(f"{'='*70}")
        print(f"  AUC         = {pcr_auc:.4f}")
        print(f"  Sensitivity = {pcr_eval['sensitivity']:.4f}")
        print(f"  Specificity = {pcr_eval['specificity']:.4f}")
        print(f"  TP={pcr_eval['TP']}, TN={pcr_eval['TN']}, FP={pcr_eval['FP']}, FN={pcr_eval['FN']}")
        print(f"{'='*70}")

    # Predict renqun
    X_renqun = renqun[FEATURE_COLS].values
    renqun_scores = model.score_samples(X_renqun)
    renqun_pred = np.where(renqun_scores >= bl_thr, 'MSI-H', 'MSS')

    n_msih = (renqun_pred == 'MSI-H').sum()
    n_mss = (renqun_pred == 'MSS').sum()
    print(f"\n{'='*70}")
    print(f"  Renqun Prediction (threshold={bl_thr:.3f})")
    print(f"{'='*70}")
    print(f"  Total: {len(renqun)}")
    print(f"  MSI-H: {n_msih} ({n_msih/len(renqun)*100:.1f}%)")
    print(f"  MSS:   {n_mss} ({n_mss/len(renqun)*100:.1f}%)")

    # Per cancer type
    for ct in sorted(renqun['cancertype'].unique()):
        mask = renqun['cancertype'] == ct
        n = mask.sum()
        n_h = (renqun_pred[mask] == 'MSI-H').sum()
        print(f"  {ct:15s}: {n:>5d} samples, MSI-H={n_h} ({n_h/n*100:.1f}%)")
    print(f"{'='*70}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC
    ax = axes[0]
    ax.plot(bl_fpr, bl_tpr, 'b-', lw=2, label=f"BL AUC={bl_auc:.3f}")
    if len(pcr) > 0:
        ax.plot(pcr_fpr, pcr_tpr, 'r-', lw=2, label=f"PCR AUC={pcr_auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'{args.method} ROC')
    ax.legend(loc='lower right')

    # Score distribution (BL)
    ax = axes[1]
    mss_scores = bl_scores[bl['MSI_status'] == 'MSS']
    msih_scores = bl_scores[bl['MSI_status'] == 'MSI-H']
    ax.hist(mss_scores, bins=40, alpha=0.5, color='blue', label='MSS')
    ax.hist(msih_scores, bins=40, alpha=0.5, color='red', label='MSI-H')
    ax.axvline(bl_thr, color='green', linestyle='--', lw=2, label=f'Thr={bl_thr:.2f}')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Count')
    ax.set_title('BL Score Distribution')
    ax.legend()

    # Renqun distribution
    ax = axes[2]
    ax.hist(renqun_scores[renqun_pred == 'MSS'], bins=40, alpha=0.5, color='blue', label='Pred MSS')
    ax.hist(renqun_scores[renqun_pred == 'MSI-H'], bins=40, alpha=0.5, color='red', label='Pred MSI-H')
    ax.axvline(bl_thr, color='green', linestyle='--', lw=2, label=f'Thr={bl_thr:.2f}')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Count')
    ax.set_title('Renqun Score Distribution')
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(args.output_dir, f'{args.method}_results.png')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outpath}")

    # Save
    renqun_out = renqun[['MSI_status', 'cancertype']].copy()
    renqun_out['anomaly_score'] = renqun_scores
    renqun_out['predicted_status'] = renqun_pred
    outpath = os.path.join(args.output_dir, f'{args.method}_predictions.tsv')
    renqun_out.to_csv(outpath, sep='\t')
    logger.info(f"Saved: {outpath}")

    # Feature analysis
    if args.method == 'mahalanobis':
        print(f"\n{'='*70}")
        print("  Feature Contribution (distance components)")
        print(f"{'='*70}")
        comp = np.abs(model.mean_)
        for i in np.argsort(-comp)[:10]:
            print(f"  {FEATURE_COLS[i]:25s} {comp[i]:.4f}")
        print(f"{'='*70}")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

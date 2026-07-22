# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""CLI entry point for MSI detection."""

import os
import sys
import argparse
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from .features import FeatureExtractor
from .feature_selectors import AUCBasedLocusSelector, TwoStageSelector, SingleVariableAUCSelector, NullLocusSelector
from .filters import CombinedFilter, QualityFilter, DepthFilter
from .detectors import (
    BinaryClassifierDetector, MahalanobisDetector,
    MSIPercentageDetector, OneClassSVMDetector,
    IsolationForestDetector, OCLRDetector,
)
from .strategies import AGG_STRATEGIES
from .pipeline import MSIDetectionPipeline

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments.

    Subcommands
    -----------
    train : Train anomaly detection model.
    predict : Predict MSI status for new samples.
    """
    parser = argparse.ArgumentParser(description="MSI Anomaly Detection")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train anomaly detection model")
    p_train.add_argument("--all-info", required=True, help="Path to all_info.tsv")
    p_train.add_argument("--output-dir", "-o", required=True, help="Output directory")
    p_train.add_argument("--min-depth", type=int, default=10, help="Min depth for locus")
    p_train.add_argument("--locus-auc-threshold", type=float, default=0.6, help="AUC threshold for locus selection")
    p_train.add_argument("--feature-auc-threshold", type=float, default=0.6, help="AUC threshold for feature selection")
    p_train.add_argument("--top-k", type=int, default=50, help="Top K features to select")
    p_train.add_argument("--n-sigma", type=float, default=3.0, help="Threshold = mean + n*std")
    p_train.add_argument("--threshold-method", choices=['nsigma', 'youden', 'fixed', 'cv'],
                         default='nsigma', help="Threshold method (default: nsigma)")
    p_train.add_argument("--threshold", type=float, default=None,
                         help="Fixed threshold value (only for --threshold-method fixed)")
    p_train.add_argument("--cv-folds", type=int, default=5,
                         help="Number of CV folds for --threshold-method cv (default: 5)")
    p_train.add_argument("--selector", choices=['auc', 'twostage', 'lasso'], default='twostage')
    p_train.add_argument("--locus-selector", choices=['none', 'auc', 'unitlen'], default='auc')
    p_train.add_argument("--detector", choices=['mahalanobis', 'ocsvm', 'iforest', 'oclr', 'xgboost', 'logistic', 'ensemble'], default='mahalanobis')
    p_train.add_argument("--agg-strategy", choices=list(AGG_STRATEGIES.keys()), default='baseline',
                         help="Aggregation strategy for locus->sample features (default: baseline)")
    p_train.add_argument("--use-renqun-mss", action="store_true", default=False,
                         help="Include renqun MSS samples in training")
    p_train.add_argument("--site-file-col", choices=["site_path","site_feature"], default="site_feature",
                         dest="site_file_col", help="Column in all_info.tsv for site file paths")
    p_train.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of BL data held out as test set (default: 0.2)")
    p_train.add_argument("--cache-dir", default=None,
                         help="Directory to cache extracted features (skip re-extraction)")
    p_train.add_argument("--MSI-col", default="MSI_real", dest="msi_col",
                         help="Column name for MSI status in all_info.tsv (default: MSI_status)")
    # --- predict ---
    p_pred = sub.add_parser("predict", help="Predict MSI status")
    p_pred.add_argument("--model-dir", required=True, help="Path to trained model directory")
    p_pred.add_argument("--samples", nargs="+", required=True,
                        help="Sample site files or directory")
    p_pred.add_argument("--output", "-o", required=True, help="Output TSV path")
    p_pred.add_argument("--cutoff", type=float, default=None, help="MSI score cutoff")

    return parser.parse_args()

def _resolve_sample_paths(samples_arg):
    """Expand directory or glob patterns to a list of site file paths."""
    import glob as _glob
    paths = []
    for item in samples_arg:
        if os.path.isdir(item):
            paths.extend(sorted(_glob.glob(os.path.join(item, "*.site.txt"))))
        elif '*' in item or '?' in item:
            paths.extend(sorted(_glob.glob(item)))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            logger.warning(f"Skipping invalid path: {item}")
    return paths

def train_model(args):
    """Train anomaly detection model."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(args.all_info, sep='\t')
    meta['sample_id'] = meta[args.site_file_col].apply(
        lambda x: os.path.basename(x).split('_cancer')[0] if isinstance(x, str) else None)
    meta = meta.set_index('sample_id')
    logger.info(f"Loaded: {len(meta)} rows")

    # Initialize components
    if args.agg_strategy == 'locus_level':
        agg_strategy = LocusLevelAggregation()
    elif args.agg_strategy == 'baseline':
        agg_strategy = BaselineAggregation()
    else:
        agg_strategy = AGG_STRATEGIES.get(args.agg_strategy, BaselineAggregation())
    feature_extractor = FeatureExtractor(min_depth=args.min_depth, agg_strategy=agg_strategy)

    # Locus selector
    locus_selector = None
    if args.locus_selector == 'auc':
        locus_selector = AUCBasedLocusSelector(auc_threshold=args.locus_auc_threshold)
    elif args.locus_selector == 'unitlen':
        locus_selector = UnitLengthLocusSelector(allowed_unit_lens=[1, 2, 3])

    # Feature selector
    if args.selector == 'auc':
        feature_selector = SingleVariableAUCSelector(auc_threshold=args.feature_auc_threshold)
    elif args.selector == 'twostage':
        feature_selector = TwoStageSelector(auc_threshold=args.feature_auc_threshold, top_k=args.top_k)
    elif args.selector == 'lasso':
        feature_selector = LassoSelector(C=0.1)

    sample_filter = CombinedFilter([QualityFilter(min_loci=50)])

    if args.detector == 'mahalanobis':
        detector = MahalanobisDetector()
    elif args.detector == 'ocsvm':
        detector = OneClassSVMDetector()
    elif args.detector == 'iforest':
        detector = IsolationForestDetector()
    elif args.detector == 'oclr':
        detector = OCLRDetector()
    elif args.detector == 'xgboost':
        detector = BinaryClassifierDetector(method='xgboost')
    elif args.detector == 'logistic':
        detector = BinaryClassifierDetector(method='logistic')
    elif args.detector == 'ensemble':
        detector = EnsembleDetector([MahalanobisDetector()])

    # Create pipeline
    pipeline = MSIDetectionPipeline(
        feature_extractor=feature_extractor,
        locus_selector=locus_selector,
        feature_selector=feature_selector,
        sample_filter=sample_filter,
        detector=detector
    )

    # Run pipeline
    results = pipeline.run(
        meta,
        use_renqun_mss=args.use_renqun_mss,
        n_sigma=args.n_sigma,
        site_file_col=args.site_file_col,
        test_size=args.test_size,
        cache_dir=args.cache_dir,
        msi_col=args.msi_col,
        threshold_method=args.threshold_method,
        fixed_threshold=args.threshold,
        cv_folds=args.cv_folds,
    )

    # Print results
    _print_results(results, args.output_dir)

    # Build locus selector config for model saving
    locus_selector_config = None
    if args.locus_selector == 'auc':
        locus_selector_config = {'type': 'auc', 'auc_threshold': args.locus_auc_threshold}
    elif args.locus_selector == 'unitlen':
        locus_selector_config = {'type': 'unitlen', 'allowed_unit_lens': [1, 2, 3]}

    # Save model (for later prediction)
    _save_model(results, args.output_dir, min_depth=args.min_depth,
                locus_selector_config=locus_selector_config, msi_col=args.msi_col,
                detector_type=type(detector).__name__)

    return results

def _print_results(results, output_dir):
    """Print and save results."""
    threshold = results['threshold']
    test = results['test']
    val = results['val']
    predict = results['predict']

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Threshold: {threshold:.3f} (method={results.get('threshold_method', '?')}, n_sigma={results.get('n_sigma', '?')})")
    lines.append(f"{'='*70}")

    # Train AUC for overfitting diagnosis
    lines.append(f"\n  Train AUC = {results['train']['auc']:.4f} (biased, for reference only)")

    lines.append(f"\n  BL Test Set (held-out)")
    lines.append(f"  AUC   = {test['auc']:.4f}")
    lines.append(f"  Sens  = {test['eval']['sens']:.4f}")
    lines.append(f"  Spec  = {test['eval']['spec']:.4f}")
    lines.append(f"  Acc   = {test['eval']['acc']:.4f}")
    lines.append(f"  TP={test['eval']['TP']} TN={test['eval']['TN']} FP={test['eval']['FP']} FN={test['eval']['FN']}")
    test_df = test['df'].copy()
    test_df['predicted'] = np.where(test['scores'] >= threshold, 'MSI-H', 'MSS')
    if 'cancertype' in test_df.columns:
        lines.append(f"  --- By cancer type ---")
        for ct in sorted(test_df['cancertype'].unique()):
            mask = test_df['cancertype'] == ct
            n = mask.sum()
            n_tp = ((test_df.loc[mask, 'predicted'] == 'MSI-H') & (test_df.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
            n_fp = ((test_df.loc[mask, 'predicted'] == 'MSI-H') & (test_df.loc[mask, 'MSI_status'] == 'MSS')).sum()
            n_fn = ((test_df.loc[mask, 'predicted'] == 'MSS') & (test_df.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
            n_tn = ((test_df.loc[mask, 'predicted'] == 'MSS') & (test_df.loc[mask, 'MSI_status'] == 'MSS')).sum()
            lines.append(f"  {str(ct):15s}: n={n:>4d} TP={n_tp} FP={n_fp} FN={n_fn} TN={n_tn}")

    if val:
        lines.append(f"\n  PCR Validation (independent)")
        lines.append(f"  AUC   = {val['auc']:.4f}")
        lines.append(f"  Sens  = {val['eval']['sens']:.4f}")
        lines.append(f"  Spec  = {val['eval']['spec']:.4f}")
        lines.append(f"  Acc   = {val['eval']['acc']:.4f}")
        lines.append(f"  TP={val['eval']['TP']} TN={val['eval']['TN']} FP={val['eval']['FP']} FN={val['eval']['FN']}")
        val_df = val['df'].copy()
        val_df['predicted'] = np.where(val['scores'] >= threshold, 'MSI-H', 'MSS')
        if 'cancertype' in val_df.columns:
            lines.append(f"  --- By cancer type ---")
            for ct in sorted(val_df['cancertype'].unique()):
                mask = val_df['cancertype'] == ct
                n = mask.sum()
                n_tp = ((val_df.loc[mask, 'predicted'] == 'MSI-H') & (val_df.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
                n_fp = ((val_df.loc[mask, 'predicted'] == 'MSI-H') & (val_df.loc[mask, 'MSI_status'] == 'MSS')).sum()
                n_fn = ((val_df.loc[mask, 'predicted'] == 'MSS') & (val_df.loc[mask, 'MSI_status'] == 'MSI-H')).sum()
                n_tn = ((val_df.loc[mask, 'predicted'] == 'MSS') & (val_df.loc[mask, 'MSI_status'] == 'MSS')).sum()
                lines.append(f"  {str(ct):15s}: n={n:>4d} TP={n_tp} FP={n_fp} FN={n_fn} TN={n_tn}")

    # Renqun prediction
    predict_df = predict['df'].copy()
    predict_df['score'] = predict['scores']
    predict_df['predicted'] = predict['labels']

    lines.append(f"\n  Renqun Prediction")
    lines.append(f"  Total: {len(predict_df)}")
    lines.append(f"  MSI-H: {(predict_df['predicted']=='MSI-H').sum()} ({(predict_df['predicted']=='MSI-H').mean()*100:.1f}%)")
    for ct in sorted(predict_df['cancertype'].unique()):
        mask = predict_df['cancertype'] == ct
        n = mask.sum()
        n_h = (predict_df.loc[mask, 'predicted'] == 'MSI-H').sum()
        lines.append(f"  {str(ct):15s}: {n:>5d}, MSI-H={n_h} ({n_h/n*100:.1f}%)")
    lines.append(f"{'='*70}")

    # Feature importance
    lines.append(f"\n  Selected Features ({len(results['selected_cols'])}):")
    for col in results['selected_cols'][:15]:
        lines.append(f"    {col}")
    lines.append(f"{'='*70}")

    # Print to stdout
    for line in lines:
        print(line)

    # Write report file
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"Report saved to {report_path}")

    # Save predictions
    predict_df[['MSI_status', 'cancertype', 'score', 'predicted']].to_csv(
        os.path.join(output_dir, 'predictions.tsv'), sep='\t'
    )

    # Save BL train scores
    train_df = results['train']['df']
    cols = ['MSI_status']
    if 'cancertype' in train_df.columns:
        cols.append('cancertype')
    bl_train_out = train_df[cols].copy()
    bl_train_out['score'] = results['train']['scores']
    bl_train_out['predicted'] = np.where(
        results['train']['scores'] >= results['threshold'], 'MSI-H', 'MSS'
    )
    bl_train_out.to_csv(os.path.join(output_dir, 'bl_train_scores.tsv'), sep='\t')

    # Save BL test scores
    test_df = results['test']['df']
    cols = ['MSI_status']
    if 'cancertype' in test_df.columns:
        cols.append('cancertype')
    bl_test_out = test_df[cols].copy()
    bl_test_out['score'] = results['test']['scores']
    bl_test_out['predicted'] = np.where(
        results['test']['scores'] >= results['threshold'], 'MSI-H', 'MSS'
    )
    bl_test_out.to_csv(os.path.join(output_dir, 'bl_test_scores.tsv'), sep='\t')

    # Save PCR val scores
    if results['val'] is not None:
        val_df = results['val']['df']
        cols = ['MSI_status']
        if 'cancertype' in val_df.columns:
            cols.append('cancertype')
        val_out = val_df[cols].copy()
        val_out['score'] = results['val']['scores']
        val_out['predicted'] = np.where(
            results['val']['scores'] >= results['threshold'], 'MSI-H', 'MSS'
        )
        val_out.to_csv(os.path.join(output_dir, 'pcr_val_scores.tsv'), sep='\t')

    # Save misclassified samples
    misclassified = []
    for name, df_out in [('bl_test', bl_test_out),
                         ('pcr_val', val_out if results['val'] is not None else None)]:
        if df_out is None:
            continue
        wrong = df_out[df_out['MSI_status'] != df_out['predicted']]
        if len(wrong) > 0:
            wrong = wrong.copy()
            wrong['dataset'] = name
            misclassified.append(wrong)

    if misclassified:
        mc_df = pd.concat(misclassified)
        mc_path = os.path.join(output_dir, 'misclassified.tsv')
        mc_df.to_csv(mc_path, sep='\t')
        logger.info(f"Misclassified samples: {len(mc_df)} -> {mc_path}")
    else:
        logger.info("No misclassified samples")

    # Plot
    _plot_results(results, output_dir)

def _save_model(results, output_dir, min_depth: int = 10,
                locus_selector_config: Optional[Dict] = None, msi_col: str = 'MSI_status',
                detector_type: str = 'MahalanobisDetector'):
    """Save model parameters for later prediction."""
    import json
    import pickle

    model_info = {
        'selected_cols': results['selected_cols'],
        'threshold': float(results['threshold']),
        'n_sigma': float(results.get('n_sigma', 3.0)),
        'min_depth': min_depth,
        'msi_col': msi_col,
        'locus_selector': locus_selector_config,
        'detector_type': detector_type,
    }

    # Save detector-specific parameters
    detector = results.get('detector')
    if detector is not None:
        detector_path = os.path.join(output_dir, 'detector.pkl')
        with open(detector_path, 'wb') as f:
            pickle.dump(detector, f)
        model_info['detector_pkl'] = 'detector.pkl'
        logger.info(f"Detector saved to {detector_path}")

    # Backward compatibility: save Mahalanobis params in JSON
    if detector_type == 'MahalanobisDetector':
        model_info['mean'] = [float(x) for x in results['detector_mean']] if results.get('detector_mean') is not None else None
        model_info['cov_inv'] = [[float(x) for x in row] for row in results['detector_cov_inv']] if results.get('detector_cov_inv') is not None else None

    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Model saved to {output_dir}")

def _plot_results(results, output_dir):
    """Plot ROC and score distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    threshold = results['threshold']
    test = results['test']
    predict = results['predict']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ROC (BL test + train + PCR val)
    ax = axes[0]
    ax.plot(results['train']['fpr'], results['train']['tpr'], 'b--', lw=1.5, alpha=0.6,
            label=f"Train AUC={results['train']['auc']:.3f} (biased)")
    ax.plot(test['fpr'], test['tpr'], 'r-', lw=2, label=f"BL test AUC={test['auc']:.3f}")
    if results['val'] is not None:
        ax.plot(results['val']['fpr'], results['val']['tpr'], 'g-.',
                lw=1.5, alpha=0.8, label=f"PCR val AUC={results['val']['auc']:.3f}")
    ax.plot([0,1],[0,1],'k--', lw=0.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC')
    ax.legend()

    # BL test scores
    ax = axes[1]
    test_df = test['df']
    mss_mask = test_df['MSI_status'] == 'MSS'
    msih_mask = test_df['MSI_status'] == 'MSI-H'
    ax.hist(test['scores'][mss_mask.values], bins=40, alpha=0.5, color='blue', label='MSS')
    ax.hist(test['scores'][msih_mask.values], bins=40, alpha=0.5, color='red', label='MSI-H')
    ax.axvline(threshold, color='green', linestyle='--', label=f'Thr={threshold:.2f}')
    ax.legend()
    ax.set_title('BL Test Scores')

    # PCR val scores
    ax = axes[2]
    if results['val'] is not None:
        val_meta = results['val']['df']
        val_scores = results['val']['scores']
        mss_mask = val_meta['MSI_status'] == 'MSS'
        msih_mask = val_meta['MSI_status'] == 'MSI-H'
        ax.hist(val_scores[mss_mask.values], bins=40, alpha=0.5, color='blue', label='MSS')
        ax.hist(val_scores[msih_mask.values], bins=40, alpha=0.5, color='red', label='MSI-H')
        ax.axvline(threshold, color='green', linestyle='--', label=f'Thr={threshold:.2f}')
        ax.legend()
        ax.set_title(f"PCR Val (AUC={results['val']['auc']:.3f})")
    else:
        ax.set_title('PCR Val (N/A)')

    # Renqun scores
    ax = axes[3]
    predict_df = predict['df'].copy()
    predict_df['score'] = predict['scores']
    predict_df['predicted'] = predict['labels']
    ax.hist(predict_df.loc[predict_df['predicted']=='MSS', 'score'], bins=40, alpha=0.5, color='blue', label='Pred MSS')
    ax.hist(predict_df.loc[predict_df['predicted']=='MSI-H', 'score'], bins=40, alpha=0.5, color='red', label='Pred MSI-H')
    ax.axvline(threshold, color='green', linestyle='--', label=f'Thr={threshold:.2f}')
    ax.legend()
    ax.set_title('Renqun Scores')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'results.png'), dpi=150)
    plt.close()

def predict_samples(args):
    """Predict MSI status for new samples."""
    import json
    import pickle

    # Load model
    model_path = os.path.join(args.model_dir, 'model.json')
    with open(model_path) as f:
        model_info = json.load(f)

    selected_cols = model_info['selected_cols']
    threshold = model_info['threshold']
    if args.cutoff is not None:
        threshold = args.cutoff

    # Reconstruct feature extractor with training config
    min_depth = model_info.get('min_depth', 10)
    locus_selector = None
    ls_config = model_info.get('locus_selector')
    if ls_config is not None:
        if ls_config['type'] == 'auc':
            locus_selector = AUCBasedLocusSelector(auc_threshold=ls_config['auc_threshold'])
        elif ls_config['type'] == 'unitlen':
            locus_selector = UnitLengthLocusSelector(allowed_unit_lens=ls_config['allowed_unit_lens'])

    feature_extractor = FeatureExtractor(min_depth=min_depth, locus_selector=locus_selector)
    logger.info(f"Loaded model: min_depth={min_depth}, locus_selector={ls_config}, threshold={threshold:.3f}")

    # Load detector
    detector_type = model_info.get('detector_type', 'MahalanobisDetector')
    detector_pkl = model_info.get('detector_pkl')

    if detector_pkl is not None:
        # Load from pickle (preferred, supports all detector types)
        detector_path = os.path.join(args.model_dir, detector_pkl)
        with open(detector_path, 'rb') as f:
            detector = pickle.load(f)
        logger.info(f"Loaded detector from {detector_path} (type={detector_type})")
    else:
        # Backward compatibility: reconstruct Mahalanobis from JSON
        if detector_type == 'MahalanobisDetector':
            detector = MahalanobisDetector()
            detector.mean_ = np.array(model_info['mean'])
            detector.cov_inv_ = np.array(model_info['cov_inv'])
            logger.info("Loaded Mahalanobis detector from JSON params")
        else:
            logger.error(f"Cannot load detector type '{detector_type}' without detector.pkl")
            sys.exit(1)

    # Load samples
    sample_files = _resolve_sample_paths(args.samples)
    if not sample_files:
        logger.error("No sample files found")
        sys.exit(1)

    logger.info(f"Predicting {len(sample_files)} samples...")

    # Extract features
    rows = []
    for fpath in sample_files:
        feat, _ = feature_extractor.extract_sample_features(fpath)
        if feat is not None:
            feat['sample_id'] = os.path.basename(fpath).split('.')[0]
            rows.append(feat)

    if not rows:
        logger.error("No features extracted")
        sys.exit(1)

    df = pd.DataFrame(rows).set_index('sample_id')

    # Predict
    X = np.nan_to_num(df[selected_cols].values, nan=0.0).astype(float)
    scores = detector.score(X)
    labels = np.where(scores >= threshold, 'MSI-H', 'MSS')

    # Output
    result_df = pd.DataFrame({
        'sample_id': df.index,
        'score': scores,
        'predicted': labels,
    })
    result_df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Predictions saved to {args.output}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Prediction Summary")
    print(f"{'='*50}")
    print(f"  Detector: {detector_type}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Total: {len(result_df)}")
    print(f"  MSI-H: {(labels=='MSI-H').sum()} ({(labels=='MSI-H').mean()*100:.1f}%)")
    print(f"  MSS:   {(labels=='MSS').sum()} ({(labels=='MSS').mean()*100:.1f}%)")
    print(f"{'='*50}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    args = parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict_samples(args)


if __name__ == "__main__":
    main()

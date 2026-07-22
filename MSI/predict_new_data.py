#!/usr/bin/env python3
"""Predict MSI status on new data.

Two modes:
  1. Load saved model:  --model-dir path/to/model/
  2. Train + predict:   --config config.json --route route_name

Usage:
    # Mode 1: Use saved model (fast, no retraining)
    python predict_new_data.py --data v4_newsequencing.tsv \
        --model-dir output/.../xgboost/advanced/

    # Mode 2: Train fresh and predict
    python predict_new_data.py --data v4_newsequencing.tsv \
        --config config/compare_features.json --route advanced_xgb
"""

import os
import sys
import json
import pickle
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from MSI import MSIDetectionPipeline, compute_roc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def load_model(model_dir: str) -> dict:
    """Load saved model artifacts from a directory."""
    model_json = os.path.join(model_dir, 'model.json')
    if not os.path.isfile(model_json):
        raise FileNotFoundError(f"model.json not found in {model_dir}")

    with open(model_json) as f:
        info = json.load(f)

    artifacts = {'info': info}

    # Load detector
    if 'detector_pkl' in info:
        with open(os.path.join(model_dir, info['detector_pkl']), 'rb') as f:
            artifacts['detector'] = pickle.load(f)

    # Load feature extractor
    if 'feature_extractor_pkl' in info:
        with open(os.path.join(model_dir, info['feature_extractor_pkl']), 'rb') as f:
            artifacts['feature_extractor'] = pickle.load(f)

    # Load feature selector
    if 'feature_selector_pkl' in info:
        with open(os.path.join(model_dir, info['feature_selector_pkl']), 'rb') as f:
            artifacts['feature_selector'] = pickle.load(f)

    # Load sample filter
    if 'sample_filter_pkl' in info:
        with open(os.path.join(model_dir, info['sample_filter_pkl']), 'rb') as f:
            artifacts['sample_filter'] = pickle.load(f)

    return artifacts


def extract_and_predict(new_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Extract features from new data and predict using loaded model."""
    info = artifacts['info']
    selected_cols = info['selected_cols']
    threshold = info['threshold']

    fe = artifacts.get('feature_extractor')
    det = artifacts.get('detector')
    sf = artifacts.get('sample_filter')

    if fe is None or det is None:
        raise ValueError("Model must contain feature_extractor and detector")

    # Determine site file column
    site_col = 'site_path' if 'site_path' in new_df.columns else 'site_feature'
    logger.info(f"Using '{site_col}' for feature extraction")

    # Extract features
    new_features, _ = fe.extract_batch(
        new_df[site_col].values,
        new_df.index.values,
    )

    # Join metadata
    _join_cols = [c for c in ['MSI_status', 'MSI_real', 'cancertype', 'MSI_CNC',
                               'tumor_content', 'TMB_H', 'TMB_L', 'TMB_U']
                  if c in new_df.columns]
    new_features = new_features.join(new_df[_join_cols], how='inner')

    if 'MSI_real' in new_features.columns and 'MSI_status' not in new_features.columns:
        new_features.rename(columns={'MSI_real': 'MSI_status'}, inplace=True)

    # Apply sample filter
    if sf is not None:
        new_features = sf.filter(new_features)
    logger.info(f"After filter: {len(new_features)} samples")

    # Select features
    available_cols = [c for c in selected_cols if c in new_features.columns]
    missing = [c for c in selected_cols if c not in new_features.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features: {missing[:5]}...")

    X_new = np.nan_to_num(new_features[available_cols].values, nan=0.0)

    # Set feature names for detectors that need it
    if hasattr(det, 'set_feature_names'):
        det.set_feature_names(available_cols)

    # Score and predict
    scores = det.score(X_new)
    predictions = np.where(scores >= threshold, 'MSI-H', 'MSS')

    new_features['score'] = scores
    new_features['predicted'] = predictions

    return new_features


def evaluate_predictions(new_features: pd.DataFrame, info: dict) -> None:
    """Print evaluation metrics if ground truth is available."""
    threshold = info['threshold']

    if 'MSI_status' not in new_features.columns:
        logger.info("No MSI_status column, skipping evaluation")
        return

    y_true = new_features['MSI_status'].values
    scores = new_features['score'].values
    predictions = new_features['predicted'].values

    tp = int(((predictions == 'MSI-H') & (y_true == 'MSI-H')).sum())
    fp = int(((predictions == 'MSI-H') & (y_true == 'MSS')).sum())
    fn = int(((predictions == 'MSS') & (y_true == 'MSI-H')).sum())
    tn = int(((predictions == 'MSS') & (y_true == 'MSS')).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / len(y_true)
    fpr, tpr, auc_val = compute_roc(y_true, scores)

    print(f"\n{'='*60}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Samples: {len(new_features)}")
    print(f"  AUC   = {auc_val:.4f}")
    print(f"  Sens  = {sens:.4f} ({tp}/{tp+fn})")
    print(f"  Spec  = {spec:.4f} ({tn}/{tn+fp})")
    print(f"  Acc   = {acc:.4f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"{'='*60}")

    # Per-cancer
    if 'cancertype' in new_features.columns:
        print(f"\n  --- By cancer type ---")
        for ct in sorted(new_features['cancertype'].dropna().unique()):
            mask = new_features['cancertype'] == ct
            sub = new_features[mask]
            n_tp = ((sub['predicted'] == 'MSI-H') & (sub['MSI_status'] == 'MSI-H')).sum()
            n_fp = ((sub['predicted'] == 'MSI-H') & (sub['MSI_status'] == 'MSS')).sum()
            n_fn = ((sub['predicted'] == 'MSS') & (sub['MSI_status'] == 'MSI-H')).sum()
            n_tn = ((sub['predicted'] == 'MSS') & (sub['MSI_status'] == 'MSS')).sum()
            n_ct = len(sub)
            ct_sens = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else '-'
            ct_spec = n_tn / (n_tn + n_fp) if (n_tn + n_fp) > 0 else '-'
            s = f"{ct_sens:.3f}" if isinstance(ct_sens, float) else ct_sens
            p = f"{ct_spec:.3f}" if isinstance(ct_spec, float) else ct_spec
            print(f"  {str(ct):15s}: n={n_ct:>4d} TP={n_tp} FP={n_fp} FN={n_fn} TN={n_tn} "
                  f"Sens={s} Spec={p}")


def main():
    parser = argparse.ArgumentParser(description="Predict MSI on new data")
    parser.add_argument('--data', required=True, help='TSV with sample_id, MSI_real, site_path')
    parser.add_argument('--model-dir', default=None, help='Saved model directory (mode 1)')
    parser.add_argument('--config', default=None, help='Config JSON (mode 2)')
    parser.add_argument('--route', default=None, help='Route name (mode 2)')
    parser.add_argument('--output', '-o', default=None, help='Output TSV path')
    args = parser.parse_args()

    # Load new data
    new_df = pd.read_csv(args.data, sep='\t')
    logger.info(f"Loaded {len(new_df)} samples")

    # Rename columns
    col_map = {}
    if '样本编号' in new_df.columns:
        col_map['样本编号'] = 'sample_id'
    if col_map:
        new_df.rename(columns=col_map, inplace=True)
    if 'sample_id' in new_df.columns:
        new_df['sample_id'] = new_df['sample_id'].astype(str)
        new_df = new_df.set_index('sample_id')
    if 'cancertype' not in new_df.columns:
        new_df['cancertype'] = 'unknown'

    if args.model_dir:
        # Mode 1: Load saved model
        logger.info(f"Loading model from {args.model_dir}")
        artifacts = load_model(args.model_dir)
        info = artifacts['info']
        logger.info(f"Model: threshold={info['threshold']:.4f}, features={len(info['selected_cols'])}")

        new_features = extract_and_predict(new_df, artifacts)
        evaluate_predictions(new_features, info)

        # Save
        output_path = args.output or os.path.join(args.model_dir, 'predictions.tsv')
        out_cols = [c for c in ['MSI_status', 'cancertype', 'score', 'predicted'] if c in new_features.columns]
        new_features[out_cols].to_csv(output_path, sep='\t')
        logger.info(f"Saved to {output_path}")

    elif args.config and args.route:
        # Mode 2: Train fresh and predict
        from compare_features import load_config, _build_pipeline_components, _make_detector

        route_configs = load_config(args.config, route_names=[args.route])
        cfg = route_configs[args.route]

        strat_cfg = cfg['strategies'][0]
        det_cfg = cfg['detectors'][0]

        fe, locus_sel, feat_sel, sf, train_filter, required_features = _build_pipeline_components(strat_cfg)
        det = _make_detector(det_cfg)

        pipeline = MSIDetectionPipeline(
            feature_extractor=fe, locus_selector=locus_sel,
            feature_selector=feat_sel, sample_filter=sf,
            detector=det, train_filter=train_filter,
            required_features=required_features,
        )

        # Load training data
        train_meta = pd.read_csv(cfg['all_info'], sep='\t')
        train_meta['sample_id'] = train_meta['site_feature'].apply(
            lambda x: os.path.basename(x).split('_cancer')[0] if isinstance(x, str) else None)
        train_meta = train_meta.set_index('sample_id')

        # Merge all_info_v2
        all_info_v2 = cfg.get('all_info_v2')
        if all_info_v2 and os.path.isfile(all_info_v2):
            v2 = pd.read_csv(all_info_v2, sep='\t')
            v2 = v2.set_index('sample_id')
            if 'tumor_content' in v2.columns:
                train_meta['tumor_content'] = pd.to_numeric(v2['tumor_content'], errors='coerce')
            if 'TMB_status' in v2.columns:
                tmb = v2['TMB_status'].reindex(train_meta.index)
                for cat in ['TMB-H', 'TMB-L', 'TMB-U']:
                    col = cat.replace("-", "_")
                    train_meta[col] = (tmb == cat).astype(float)
                    train_meta.loc[tmb.isna(), col] = np.nan

        # Train
        run_cfg = cfg.get('pipeline', {})
        results = pipeline.run(
            train_meta,
            n_sigma=run_cfg.get('n_sigma', 3.0),
            site_file_col=run_cfg.get('site_file_col', 'site_feature'),
            test_size=run_cfg.get('test_size', 0.2),
            cache_dir=cache_dir,
            msi_col=run_cfg.get('msi_col', 'MSI_real'),
            threshold_method=run_cfg.get('threshold_method', 'cv'),
            cv_folds=run_cfg.get('cv_folds', 5),
        )

        # Build artifacts dict from pipeline results
        artifacts = {
            'info': {
                'selected_cols': results['selected_cols'],
                'threshold': float(results['threshold']),
                'threshold_method': results.get('threshold_method', '?'),
                'n_sigma': float(results.get('n_sigma', 0)),
            },
            'detector': results['detector'],
            'feature_extractor': results['feature_extractor'],
            'feature_selector': results['feature_selector'],
            'sample_filter': results['sample_filter'],
        }

        new_features = extract_and_predict(new_df, artifacts)
        evaluate_predictions(new_features, artifacts['info'])

        # Save
        output_path = args.output or os.path.join(
            cfg.get('output_dir', '.'), f'predictions_{args.route}.tsv')
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        out_cols = [c for c in ['MSI_status', 'cancertype', 'score', 'predicted'] if c in new_features.columns]
        new_features[out_cols].to_csv(output_path, sep='\t')
        logger.info(f"Saved to {output_path}")

    else:
        parser.error("Either --model-dir or (--config + --route) is required")


if __name__ == '__main__':
    main()

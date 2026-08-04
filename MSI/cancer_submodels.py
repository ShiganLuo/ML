#!/usr/bin/env python3
"""Cancer-type sub-model approach for MSI detection.

Instead of training one global model, train separate models for each cancer type:
1. 结直肠癌: largest sample size, dedicated model
2. 子宫内膜癌: second largest, dedicated model
3. 其他癌种: combined model (胃癌、胰腺癌、肝胆癌、肺癌等)

This allows each model to learn cancer-specific patterns.

Usage:
    python cancer_submodels.py
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from MSI import MSIDetectionPipeline, compute_roc, evaluate, find_best_threshold
from compare_features import load_config, _build_pipeline_components, _make_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

CONFIG = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/config/compare_features.json"
OUTPUT_DIR = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/compare/cancer_submodels"


def train_submodel(meta_df: pd.DataFrame, cancer_type: str, cfg: dict, strat_cfg: dict, det_cfg: dict):
    """Train a model for a specific cancer type."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training sub-model for: {cancer_type}")
    logger.info(f"{'='*60}")
    
    # Filter by cancer type
    if cancer_type == '其他':
        # Combine all other cancer types
        major_types = ['结直肠癌', '子宫内膜癌']
        sub_meta = meta_df[~meta_df['cancertype'].isin(major_types)].copy()
    else:
        sub_meta = meta_df[meta_df['cancertype'] == cancer_type].copy()
    
    logger.info(f"Samples: {len(sub_meta)}")
    if 'MSI_status' in sub_meta.columns:
        msih_count = (sub_meta['MSI_status'] == 'MSI-H').sum()
        mss_count = (sub_meta['MSI_status'] == 'MSS').sum()
        logger.info(f"  MSI-H: {msih_count}, MSS: {mss_count}")
    
    if len(sub_meta) < 100:
        logger.warning(f"Too few samples ({len(sub_meta)}) for {cancer_type}, skipping")
        return None
    
    # Build pipeline
    fe, locus_sel, feat_sel, sf, train_filter, required_features = _build_pipeline_components(strat_cfg)
    det = _make_detector(det_cfg)
    
    pipeline = MSIDetectionPipeline(
        feature_extractor=fe, locus_selector=locus_sel,
        feature_selector=feat_sel, sample_filter=sf,
        detector=det, train_filter=train_filter,
        required_features=required_features,
    )
    
    # Run pipeline
    run_cfg = cfg.get('pipeline', {})
    results = pipeline.run(
        sub_meta,
        n_sigma=run_cfg.get('n_sigma', 3.0),
        site_file_col=run_cfg.get('site_file_col', 'site_feature'),
        test_size=run_cfg.get('test_size', 0.2),
        cache_dir=cfg.get('cache_dir', '/tmp/msi_cache'),
        msi_col=run_cfg.get('msi_col', 'MSI_real'),
        threshold_method=run_cfg.get('threshold_method', 'cv'),
        cv_folds=run_cfg.get('cv_folds', 5),
    )
    
    return {
        'cancer_type': cancer_type,
        'n_samples': len(sub_meta),
        'pipeline': pipeline,
        'results': results,
    }


def main():
    """Main entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load config
    cfg = load_config(CONFIG, route_name='xgboost_robust')['xgboost_robust']
    strat_cfg = cfg['strategies'][0]
    det_cfg = cfg['detectors'][0]
    
    # Load metadata
    meta = pd.read_csv(cfg['all_info'], sep='\t')
    meta['sample_id'] = meta['site_feature'].apply(
        lambda x: os.path.basename(x).split('_cancer')[0] if isinstance(x, str) else None)
    meta = meta.set_index('sample_id')
    
    # Get cancer type distribution
    logger.info("Cancer type distribution:")
    cancer_counts = meta['cancertype'].value_counts()
    for ct, count in cancer_counts.items():
        logger.info(f"  {ct}: {count}")
    
    # Define sub-models
    cancer_types = ['结直肠癌', '子宫内膜癌', '其他']
    
    # Train sub-models
    submodels = {}
    for ct in cancer_types:
        result = train_submodel(meta, ct, cfg, strat_cfg, det_cfg)
        if result is not None:
            submodels[ct] = result
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUB-MODEL SUMMARY")
    logger.info(f"{'='*60}")
    
    for ct, model_info in submodels.items():
        res = model_info['results']
        test = res['test']
        logger.info(f"\n{ct} (n={model_info['n_samples']}):")
        logger.info(f"  AUC:  {test['auc']:.4f}")
        logger.info(f"  Sens: {test['eval']['sens']:.4f}")
        logger.info(f"  Spec: {test['eval']['spec']:.4f}")
        logger.info(f"  Threshold: {res['threshold']:.4f}")
        logger.info(f"  Features: {len(res['selected_cols'])}")
    
    # Save results
    summary_rows = []
    for ct, model_info in submodels.items():
        res = model_info['results']
        test = res['test']
        summary_rows.append({
            'cancer_type': ct,
            'n_samples': model_info['n_samples'],
            'auc': test['auc'],
            'sensitivity': test['eval']['sens'],
            'specificity': test['eval']['spec'],
            'threshold': res['threshold'],
            'n_features': len(res['selected_cols']),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'submodel_summary.tsv')
    summary_df.to_csv(summary_path, sep='\t', index=False)
    logger.info(f"\nSaved summary: {summary_path}")


if __name__ == '__main__':
    main()
